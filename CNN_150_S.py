
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_registry




import torch
import torch.nn as nn
import torch.nn.functional as F


class CMNet(nn.Module):
    """
    CMNet: Strict implementation of Table I (Input 10*10×2 sample covariance matrix)
    """
    def __init__(self):
        super(CMNet, self).__init__()

        # C1: 32 filters, 3x3, input channels=2
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=32, kernel_size=3, padding=1
        )

        # C2: 32 filters, 3x3, input channels=32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )

        # S1: MaxPooling 2×2
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))  # 固定输出大小
        # Flatten 后尺寸：32 × 4 × 4 = 512
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 128)
        # F1: 128 × 512
        # 56 13MB
        # self.fc1 = nn.Linear(112896, 128)
        # 48 10MB
        # self.fc1 = nn.Linear(82944, 128)
        # 32 4.5MB
        # self.fc1 = nn.Linear(36864, 128)
        # 24  2.5MB
        # self.fc1 = nn.Linear(20736, 128)
        # 16  1.1MB
        # self.fc1 = nn.Linear(9216, 128)
        # 8 0.29MB
        # self.fc1 = nn.Linear(2304, 128)
        # 4 0.08MB
        # self.fc1 = nn.Linear(576, 128)
        # self.fc1 = nn.Linear(320, 128)

        self.dropout2 = nn.Dropout(p=0.25)

        # F2: 2 × 128
        self.fc2 = nn.Linear(128, 2)   # 输出 2 维 score vector


        # self.fc1 = nn.Linear(2, 64)

        # self.fc2 = nn.Linear(64, 2)   # 输出 2 维 score vector

    def forward(self, x):
        """
        x: (B, 2, 8, 8)
        """
        # C1 + ReLU
        # print(x.shape)
        # B,S,C,M,M = x.shape
        # x = x.reshape(B*S,C,M,M)

        x = F.relu(self.conv1(x))
        # C2 + ReLU
        x = F.relu(self.conv2(x))
        # S1 Pooling
        # x = self.pool(x)   # 变成 (B, 32, 4, 4)
        x = self.adapt(x)   # 变成 (B, 32, 4, 4)
        
        # Flatten (C3)
        x = x.view(x.size(0), -1)  # (B, 512)

        # D1 + FC1 + ReLU
        x = self.dropout1(x)

        
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        # D2 + FC2
        x = self.dropout2(x)
        # x = x.view(x.size(0), -1)  # (B, 128)
        # print(x.shape)
        logits = self.fc2(x)

        # Softmax 不在模型内部做，让 loss 更稳
        return logits

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)

    def test_statistic(self, x, eps=1e-12):
        """
        论文里的检测统计量:
           T(y) = P(H1|y) / P(H0|y)
        假设输出的 index 0 对应 H1, index 1 对应 H0
        """
        probs = self.predict_proba(x)
        p_H1 = probs[:, 0]
        p_H0 = probs[:, 1]
        return p_H1 / (p_H0 + eps), probs


class CMNet_MIMO(nn.Module):
    """
    CMNet: Strict implementation of Table I (Input 10*10×2 sample covariance matrix)
    """
    def __init__(self):
        super(CMNet_MIMO, self).__init__()

        # C1: 32 filters, 3x3, input channels=2
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=32, kernel_size=3, padding=1
        )

        # C2: 32 filters, 3x3, input channels=32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )

        # S1: MaxPooling 2×2
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))  # 固定输出大小
        # Flatten 后尺寸：32 × 4 × 4 = 512
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 128)
        # F1: 128 × 512
        # 56 13MB
        # self.fc1 = nn.Linear(112896, 128)
        # 48 10MB
        # self.fc1 = nn.Linear(82944, 128)
        # 32 4.5MB
        # self.fc1 = nn.Linear(36864, 128)
        # 24  2.5MB
        # self.fc1 = nn.Linear(20736, 128)
        # 16  1.1MB
        # self.fc1 = nn.Linear(9216, 128)
        # 8 0.29MB
        # self.fc1 = nn.Linear(2304, 128)
        # 4 0.08MB
        # self.fc1 = nn.Linear(576, 128)
        # self.fc1 = nn.Linear(320, 128)

        self.dropout2 = nn.Dropout(p=0.25)

        # F2: 2 × 128
        self.fc2 = nn.Linear(128, 2)   # 输出 2 维 score vector


        # self.fc1 = nn.Linear(2, 64)

        # self.fc2 = nn.Linear(64, 2)   # 输出 2 维 score vector


        self.final_cov = nn.Sequential(
        
        nn.Conv1d(1,5,kernel_size=7,padding=3),
        nn.BatchNorm1d(5),
        nn.ReLU(inplace=True),

        nn.Conv1d(5,1,kernel_size=7,padding=3),


        )
    def forward(self, x):
        """
        x: (B, 2, 8, 8)
        """
        # C1 + ReLU
        # print(x.shape)
        B,S,C,M,M = x.shape
        x = x.reshape(B*S,C,M,M)

        x = F.relu(self.conv1(x))
        # C2 + ReLU
        x = F.relu(self.conv2(x))
        # S1 Pooling
        # x = self.pool(x)   # 变成 (B, 32, 4, 4)
        x = self.adapt(x)   # 变成 (B, 32, 4, 4)
        
        # Flatten (C3)
        x = x.view(B,S, -1)  # (B, 512)

        x = self.final_cov(x)
        x = x.view(B,-1)
        # data_emb = x[:, -1, :]              # (B, width) 最后一段当 data
        # pilot_embs = x[:, :-1, :]           # (B, S-1, width)

        # # pilots 汇聚（最简单先用 mean；你也可以换 attention pooling）
        # pilot_mean = pilot_embs.mean(dim=1) if pilot_embs.numel() > 0 else torch.zeros_like(data_emb)

        # x = torch.cat([
        #     data_emb,
        #     pilot_mean,

        # ], dim=-1)

        # # D1 + FC1 + ReLU
        # x = self.dropout1(x)

        
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        # D2 + FC2
        x = self.dropout2(x)
        # x = x.view(x.size(0), -1)  # (B, 128)
        # print(x.shape)
        logits = self.fc2(x)

        # Softmax 不在模型内部做，让 loss 更稳
        return logits



class CMNet_Dir(nn.Module):
    """
    CMNet: Strict implementation of Table I (Input 10*10×2 sample covariance matrix)
    """
    def __init__(self):
        super(CMNet_Dir, self).__init__()

        # C1: 32 filters, 3x3, input channels=2
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=32, kernel_size=3, padding=1
        )

        # C2: 32 filters, 3x3, input channels=32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )

        # S1: MaxPooling 2×2
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))  # 固定输出大小
        # Flatten 后尺寸：32 × 4 × 4 = 512
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 128)
        # F1: 128 × 512
        # 56 13MB
        # self.fc1 = nn.Linear(112896, 128)
        # 48 10MB
        # self.fc1 = nn.Linear(82944, 128)
        # 32 4.5MB
        # self.fc1 = nn.Linear(36864, 128)
        # 24  2.5MB
        # self.fc1 = nn.Linear(20736, 128)
        # 16  1.1MB
        # self.fc1 = nn.Linear(9216, 128)
        # 8 0.29MB
        # self.fc1 = nn.Linear(2304, 128)
        # 4 0.08MB
        # self.fc1 = nn.Linear(576, 128)
        # self.fc1 = nn.Linear(320, 128)

        self.dropout2 = nn.Dropout(p=0.25)

        # F2: 2 × 128
        self.fc2 = nn.Linear(128, 2)   # 输出 2 维 score vector

        self.new_fc1 = nn.Linear(512, 128)
       

        # F2: 2 × 128
        self.new_fc2 = nn.Linear(128, 2)   # 输出 2 维 score vector

        # 默认用旧 classifier 推理（你也可以切换）
        self.use_new_head = False



    def forward(self, x):
        """
        x: (B, 2, 8, 8)
        """
        # C1 + ReLU
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # C2 + ReLU
        x = F.relu(self.conv2(x))
        # S1 Pooling
        # x = self.pool(x)   # 变成 (B, 32, 4, 4)
        x = self.adapt(x)   # 变成 (B, 32, 4, 4)
        
        # Flatten (C3)
        x = x.view(x.size(0), -1)  # (B, 512)
        if self.use_new_head:
            # D1 + FC1 + ReLU
            x = self.dropout1(x)
            x = F.relu(self.fc1(x))

            # D2 + FC2
            x = self.dropout2(x)
            logits = self.fc2(x)
        else:

            x = F.relu(self.new_fc1(x))

            logits = self.new_fc2(x)
            
        evidence = F.softplus(logits)  # e >= 0
        alpha = evidence + 1.0                  # Dirichlet 参数
        
        return alpha
        # Softmax 不在模型内部做，让 loss 更稳
        # return logits

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)

    def test_statistic(self, x, eps=1e-12):
        """
        论文里的检测统计量:
           T(y) = P(H1|y) / P(H0|y)
        假设输出的 index 0 对应 H1, index 1 对应 H0
        """
        probs = self.predict_proba(x)
        p_H1 = probs[:, 0]
        p_H0 = probs[:, 1]
        return p_H1 / (p_H0 + eps), probs

class CMNetBig(nn.Module):
    """
    CMNet: Strict implementation of Table I (Input 10*10×2 sample covariance matrix)
    """
    def __init__(self):
        super(CMNetBig, self).__init__()

        # C1: 32 filters, 3x3, input channels=2
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn1_1 = nn.BatchNorm2d(64)
        # C2: 32 filters, 3x3, input channels=32
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )

        # S1: MaxPooling 2×2
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

        # Flatten 后尺寸：32 × 4 × 4 = 512
        self.dropout1 = nn.Dropout(p=0.5)

        # F1: 128 × 512
        self.fc1 = nn.Linear(576*2, 128)
        # self.fc1 = nn.Linear(640, 128)

        self.dropout2 = nn.Dropout(p=0.25)

        # F2: 2 × 128
        self.fc2 = nn.Linear(128, 2)   # 输出 2 维 score vector

    def forward(self, x):
        """
        x: (B, 2, 8, 8)
        """
        # C1 + ReLU
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        # C2 + ReLU
        x = F.relu(self.conv2(x))
        # S1 Pooling
        x = self.pool(x)   # 变成 (B, 32, 4, 4)

        # Flatten (C3)
        x = x.view(x.size(0), -1)  # (B, 512)

        # D1 + FC1 + ReLU
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))

        # D2 + FC2
        x = self.dropout2(x)
        logits = self.fc2(x)

        # Softmax 不在模型内部做，让 loss 更稳
        return logits


# ================================================================
# 1. 提取 11 / 00 / 10 / 01 与 data 的相关性特征 (B, 32)
# ================================================================
def build_pattern_features(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 10, N, 2)
       0-7: 8 pilots, bits = 11001001
       8-9: data symbols

    输出:
       pattern_feat: (B, 32)
    """
    B, S, N, _ = x.shape
    assert S == 10, f"Expect S=10, got {S}"

    # 复数化
    xr = x[..., 0]
    xi = x[..., 1]
    xc = torch.complex(xr, xi)  # (B, 10, N)

    d1 = xc[:, 8, :]  # (B, N)
    d2 = xc[:, 9, :]

    def pair_corr(i, j):
        p1 = xc[:, i, :]  # (B, N)
        p2 = xc[:, j, :]

        # 2×2 相关矩阵
        c11 = (d1 * p1.conj()).mean(dim=-1)
        c12 = (d1 * p2.conj()).mean(dim=-1)
        c21 = (d2 * p1.conj()).mean(dim=-1)
        c22 = (d2 * p2.conj()).mean(dim=-1)

        C_real = torch.stack([c11.real, c12.real, c21.real, c22.real], dim=-1)
        C_imag = torch.stack([c11.imag, c12.imag, c21.imag, c22.imag], dim=-1)
        return torch.cat([C_real, C_imag], dim=-1)  # (B,8)

    f11 = pair_corr(0, 1)   # (B,8)
    f00 = pair_corr(2, 3)
    f10 = pair_corr(4, 5)
    f01 = pair_corr(6, 7)

    return torch.cat([f11, f00, f10, f01], dim=-1)  # (B, 32)


# ================================================================
# 2. CNN backbone：处理协方差矩阵 (B,2,10,10) → cov_feat
# ================================================================
class CMNetBackbone(nn.Module):
    def __init__(self, in_size=5, out_dim=128):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2)  # 10x10 → 5x5

        # 自动算 flatten 大小
        with torch.no_grad():
            dummy = torch.zeros(1, 2, in_size, in_size)
            h = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy)))))
            flat_dim = h.view(1, -1).size(1)  # 一般是 32*5*5 = 800

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)  # (B, out_dim)


# ================================================================
# 3. 最终模型： CNN(cov) + Pattern Features → concat → 分类
# ================================================================
class CMNetPatternFusion(nn.Module):
    def __init__(self, in_size=5, num_classes=2):
        super().__init__()

        self.cov_backbone = CMNetBackbone(in_size=in_size, out_dim=128)

        # pattern_feat 恒为 32 维
        self.fc_out = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, cov_input, raw_x):
        """
        cov_input: (B,2,10,10)   # old SCM
        raw_x:     (B,10,N,2)    # 用于 pattern feature
        """
        # print(cov_input.shape)
        cov_feat = self.cov_backbone(cov_input)              # (B,128)
        pattern_feat = build_pattern_features(raw_x)         # (B,32)

        feat = torch.cat([cov_feat, pattern_feat], dim=-1)   # (B,160)
        return self.fc_out(feat)   


class SISOModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),          # (B,2,1,1) -> (B,2)
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x = scm_process_siso(x)   # (B,2,1,1)
        return self.net(x)


class SCMSegmentMLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AttnPool1D(nn.Module):
    """
    x: (B, T, d) -> (B, d)
    """
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)   # (B, T)
        return (w.unsqueeze(-1) * x).sum(dim=1)               # (B, d)


class EvenOddPilotContext(nn.Module):
    """
    pilots 按偶/奇分组，分别 pooling 得到 ctx0/ctx1，再做 gated fusion。
    你如果不想把 ctx0/ctx1 融成一个 ctx，这个模块也会把 ctx0/ctx1 返回出来。
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.pool0 = AttnPool1D(d)
        self.pool1 = AttnPool1D(d)
        self.gate = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

    def _safe_pool(self, x: torch.Tensor, pool: nn.Module, ref: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return torch.zeros_like(ref)
        return pool(x)

    def forward(self, pilots: torch.Tensor, data_ref: torch.Tensor):
        # pilots: (B, P, d), data_ref: (B, d)
        B, P, d = pilots.shape
        idx = torch.arange(P, device=pilots.device)

        mask0 = (idx % 2 == 0)
        mask1 = ~mask0

        p0 = pilots[:, mask0, :] if mask0.any().item() else pilots[:, :0, :]
        p1 = pilots[:, mask1, :] if mask1.any().item() else pilots[:, :0, :]

        ctx0 = self._safe_pool(p0, self.pool0, data_ref)      # (B, d)
        ctx1 = self._safe_pool(p1, self.pool1, data_ref)      # (B, d)

        g = torch.sigmoid(self.gate(torch.cat([ctx0, ctx1, data_ref], dim=-1)))  # (B, 1)
        ctx = g * ctx1 + (1.0 - g) * ctx0
        return ctx, ctx0, ctx1


class FiLM1D(nn.Module):
    """
    用 pilot context 调制 data sequence feature
    x:   (B, T, d)
    ctx: (B, d)
    """
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, 2 * d)
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(ctx).chunk(2, dim=-1)  # (B, d), (B, d)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class FusionHead(nn.Module):
    """
    融合 [data, ctx, data-ctx, data*ctx]
    """
    def __init__(self, d: int, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes)
        )

    def forward(self, data: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([data, ctx, data - ctx, data * ctx], dim=-1)
        return self.net(feat)


class DualContextHead(nn.Module):
    """
    不把 ctx0/ctx1 压成一个 ctx，而是都保留下来。
    这对应你代码里“如果不用 gate，把两组都留下呢？”这个想法。
    """
    def __init__(self, d: int, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes)
        )

    def forward(self, data: torch.Tensor, ctx0: torch.Tensor, ctx1: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([data, ctx0, ctx1, data * ctx0, data * ctx1], dim=-1)
        return self.net(feat)

class SCMTokenMixerDetector(nn.Module):
    """
    适合单天线下退化后的 SCM/vector-SCM。
    如果你是 MIMO + matrix SCM，再把 embed 改成 2D CNN 就行。
    """
    def __init__(
        self,
        scm_dim: int = 33,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed = SCMSegmentMLP(scm_dim, width, dropout)

        mixer_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(mixer_layer, num_layers=layers)
        self.pilot_ctx = EvenOddPilotContext(width, dropout=dropout)
        self.cls = FusionHead(width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F) or (B, S)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        emb = self.embed(x)          # (B, S, d)
        emb = self.mixer(emb)

        data = emb[:, -1, :]
        pilots = emb[:, :-1, :]
        ctx, ctx0, ctx1 = self.pilot_ctx(pilots, data)
        return self.cls(data, ctx)


# model_registry.register_model("CNN_150_S", CMNet)
# model_registry.register_model("CNN_150_S", SCMTokenMixerDetector)
model_registry.register_model("CNN_150_S", SISOModel)

if __name__ == "__main__":
    model = CMNet()
    total_params2 = sum(p.numel() for p in model.parameters())
    trainable_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数量: {total_params2:,}")
    print(f"可训练参数数量: {trainable_params2:,}")
