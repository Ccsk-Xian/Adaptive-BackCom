
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_registry



def build_cov_from_x(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, S, N, 2)
       B: batch size
       S: num_pilot + 1
       N: 每符号采样点数
       最后一维 2: IQ (real, imag)

    返回:
    cov_input: (B, 2, S, S)
               通道 0: 协方差实部
               通道 1: 协方差虚部
    """
    # x -> 复数 (B, S, N)
    xr = x[..., 0]
    xi = x[..., 1]
    x_complex = torch.complex(xr, xi)  # (B, S, N)

    B, S, N = x_complex.shape

    # 样本协方差矩阵: R = 1/N * X X^H
    # 这里把 S 当作 “M 根虚拟天线”：导频 + 当前符号
    X = x_complex                       # (B, S, N)
    R = (X @ X.conj().transpose(1, 2)) / N   # (B, S, S)

    R_real = R.real.unsqueeze(1)        # (B, 1, S, S)
    R_imag = R.imag.unsqueeze(1)        # (B, 1, S, S)
    cov_input = torch.cat([R_real, R_imag], dim=1)  # (B, 2, S, S)
    return cov_input

class IQBranch(nn.Module):
    """
    IQ 分支：基本沿用 PureCNN2D 结构，只是输出一个特征向量而不是直接分类。
    输入:  x.shape = (B, S, N, 2)
    输出:  feat.shape = (B, 128)
    """
    def __init__(self, in_channels: int = 2, feat_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim

        self.features = nn.Sequential(
            # [B, 2, S, N] -> [B, 32, S, N]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2

            # [B, 32, S, N/2] -> [B, 64, S, N/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

            # [B, 64, S, N/4] -> [B, 128, S, N/4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 自适应池化到 1×1，与 S、N 解耦
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 变成一个 feat_dim 的特征向量
        self.proj = nn.Sequential(
            nn.Flatten(),             # [B, 128]
            nn.Dropout(0.5),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, N, 2] -> [B, 2, S, N]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(x)
        x = self.global_pool(x)
        feat = self.proj(x)  # (B, feat_dim)
        return feat

class CMBranch(nn.Module):
    """
    协方差分支 CMNet backbone:
    输入:  cov_input.shape = (B, 2, S, S)
    输出:  feat.shape = (B, 128)
    """
    def __init__(self, S: int, feat_dim: int = 128):
        """
        S: 协方差矩阵大小 (num_pilot + 1)
        """
        super().__init__()
        self.S = S
        self.feat_dim = feat_dim

        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )

        # 这里用 2x2 pooling（比之前的 (1,2) 更通用）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 用一个 dummy tensor 自动算 flatten 后的尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, 2, S, S)
            h = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy)))))
            flat_dim = h.view(1, -1).size(1)

        self.backbone_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

    def forward(self, cov_input: torch.Tensor) -> torch.Tensor:
        """
        cov_input: (B, 2, S, S)
        """
        x = F.relu(self.conv1(cov_input))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        feat = self.backbone_fc(x)  # (B, feat_dim)
        return feat
    
class FusionIQCovNet(nn.Module):
    """
    IQ 直接输入 + 协方差 CMNet 融合模型

    输入:  x.shape = (B, S, N, 2)
    输出:  logits.shape = (B, num_classes)
    """
    def __init__(self,
                 num_pilot: int = 8,
                 num_classes: int = 2,
                 feat_dim: int = 128):
        super().__init__()
        self.S = num_pilot + 1

        # 两个分支
        self.iq_branch = IQBranch(in_channels=2, feat_dim=feat_dim)
        self.cov_branch = CMBranch(S=self.S, feat_dim=feat_dim)

        # 融合后的分类头
        self.classifier = nn.Linear(feat_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, N, 2)
        """
        # 分支 A: 直接 IQ 特征
        feat_iq = self.iq_branch(x)  # (B, feat_dim)

        # 分支 B: 协方差特征
        cov_input = build_cov_from_x(x)        # (B, 2, S, S)
        feat_cov = self.cov_branch(cov_input)  # (B, feat_dim)

        # 融合
        feat = torch.cat([feat_iq, feat_cov], dim=1)  # (B, 2*feat_dim)

        logits = self.classifier(feat)  # (B, num_classes)
        return logits
      
model_registry.register_model("CNN_150_fusion", FusionIQCovNet)