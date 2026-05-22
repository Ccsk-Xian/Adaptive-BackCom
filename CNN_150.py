
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_registry


# class PureCNN2D(nn.Module):
#     """
#     输入: x.shape = (B, S, N, 2)
#           S = num_pilot + 1, N = 每符号采样点数
#     输出: logits.shape = (B, 2)  # 二分类
#     """
#     def __init__(self, num_classes: int = 2, in_channels: int = 2):
#         super().__init__()

#         # 特征提取部分：全卷积 + 池化
#         self.features = nn.Sequential(
#             # [B, 2, S, N] -> [B, 32, S, N]
#             nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             # 只在 N 维上做 pooling，保留符号维 S
#             nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2

#             # [B, 32, S, N/2] -> [B, 64, S, N/2]
#             nn.Conv2d(32, 64, kernel_size=5, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

#             # [B, 64, S, N/4] -> [B, 128, S, N/4]
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )

#         # 自适应池化，和 N、S 的具体数值解耦
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]

#         # 分类头
#         self.classifier = nn.Sequential(
#             nn.Flatten(),                         # [B, 128]
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes),          # [B, 2]
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, S, N, 2] -> [B, 2, S, N]
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.features(x)
#         x = self.global_pool(x)
#         x = self.classifier(x)
#         return x

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


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            # Conv2,
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)

# 内存占用 34⋅S⋅N
class PureCNN2D(nn.Module):
   
    def __init__(self, num_classes: int = 2, in_channels: int = 2):
        super().__init__()
        # 1原本 (pool不影响，不会提升) 2没norm  3去掉pool，stride在cov那儿。4是2+3  5是没pool  
        # 特征提取部分：全卷积 + 池化
        # self.features = nn.Sequential(
        #     # [B, 2, S, N] -> [B, 32, S, N]
        #     nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     # 只在 N 维上做 pooling，保留符号维 S
        #     nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2

        #     # [B, 32, S, N/2] -> [B, 64, S, N/2]
        #     nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

        #     # [B, 64, S, N/4] -> [B, 128, S, N/4]
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),

        #     # # [B, 64, S, N/4] -> [B, 128, S, N/4]
        #     # nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),
        # )

        self.features = nn.Sequential(
            # [B, 2, S, N] -> [B, 32, S, N]
            # nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # 只在 N 维上做 pooling，保留符号维 S
            InvertedResidual(in_channels, 8, 2, 6),
            # nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2



            # [B, 32, S, N/2] -> [B, 64, S, N/2]
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            InvertedResidual(8, 16, 1, 6),
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

            # [B, 64, S, N/4] -> [B, 128, S, N/4]
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            InvertedResidual(16, 32, 1, 6),


            # [B, 64, S, N/4] -> [B, 128, S, N/4]
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        # 自适应池化，和 N、S 的具体数值解耦
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # [B, 128]
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(16, num_classes),        
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, N, 2] -> [B, 2, S, N]
        
        x = x.permute(0, 3, 1, 2).contiguous()
        # B, S, M, N, C = x.shape
        # x = x.permute(0, 1, 4, 2, 3)         # (B, S, C, M, N)
        # x = x.reshape(B, S * C, M, N)        # (B, S*C, M, N)


        # B, S, M, N, C = x.shape
        # x = x.permute(0, 2, 4, 1, 3)         # (B, M, C, S, N)
        # x = x.reshape(B, M * C, S, N)        # (B, M*C, S, N)
        # print(x.shape)
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class PureCNN2D_combine(nn.Module):
   
    def __init__(self, num_classes: int = 2, in_channels: int = 2):
        super().__init__()
        # 1原本 (pool不影响，不会提升) 2没norm  3去掉pool，stride在cov那儿。4是2+3  5是没pool  
        # 特征提取部分：全卷积 + 池化
        # self.features = nn.Sequential(
        #     # [B, 2, S, N] -> [B, 32, S, N]
        #     nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        #     # 只在 N 维上做 pooling，保留符号维 S
        #     nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2

        #     # [B, 32, S, N/2] -> [B, 64, S, N/2]
        #     nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

        #     # [B, 64, S, N/4] -> [B, 128, S, N/4]
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),

        #     # # [B, 64, S, N/4] -> [B, 128, S, N/4]
        #     # nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),
        # )

        self.features = nn.Sequential(
            # [B, 2, S, N] -> [B, 32, S, N]
            # nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # 只在 N 维上做 pooling，保留符号维 S
            InvertedResidual(in_channels, 8, 2, 6),
            # nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2



            # [B, 32, S, N/2] -> [B, 64, S, N/2]
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            InvertedResidual(8, 16, 1, 6),
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

            # [B, 64, S, N/4] -> [B, 128, S, N/4]
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            InvertedResidual(16, 32, 1, 6),


            # [B, 64, S, N/4] -> [B, 128, S, N/4]
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        # 自适应池化，和 N、S 的具体数值解耦
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]


        self.enc = FastSegmentEncoder(width=32, dropout=0.1)
        self.pool0 = CrossAttentionLayer(32)
        self.pool1 = CrossAttentionLayer(32)

        self.cls = nn.Sequential(
            nn.Linear(4*32, 32),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # [B, 128]
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(16, num_classes),        
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, N, 2] -> [B, 2, S, N]
        B, S, N, C = x.shape
        x_seg = x.permute(0, 1, 3, 2).contiguous().view(B * S, 2, N)
        x = x.permute(0, 3, 1, 2).contiguous()

        

        

        # (B*S,d) -> (B,S,d)
        emb = self.enc(x_seg).view(B, S, -1)



        data = emb[:, -1, :]       # (B,d)
        pilots = emb[:, :-1, :]    # (B,P,d)  P=S-1
        # 导频数目
        P = 32

        # pilots 偶/奇分组：pilot 序号 0..P-1
        idx0 = torch.arange(0, P, 3, device=pilots.device)  # even -> bit 0（按你约定）
        idx1 = torch.arange(1, P, 3, device=pilots.device)  # odd  -> bit 1

        if idx0.numel() > 0:
            p0 = pilots.index_select(dim=1, index=idx0)     # (B,P0,d)

            ctx0 = self.pool0(data.unsqueeze(1),p0).squeeze(1)
        else:
            ctx0 = torch.zeros_like(data)

        if idx1.numel() > 0:
            p1 = pilots.index_select(dim=1, index=idx1)     # (B,P1,d)

            ctx1 = self.pool1(data.unsqueeze(1),p1).squeeze(1)

        else:
            ctx1 = torch.zeros_like(data)

        x = self.features(x)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        # print(x.shape)
        feat = torch.cat([x,data, ctx1,ctx0], dim=-1)  # (B,3d)

        return self.cls(feat)

        
        # x = self.classifier(x)
        # return x

class PureCNN2D_mimo(nn.Module):
   
    def __init__(self, num_classes: int = 2, in_channels: int = 2):
        super().__init__()
        # 1原本 (pool不影响，不会提升) 2没norm  3去掉pool，stride在cov那儿。4是2+3  5是没pool  
        # 特征提取部分：全卷积 + 池化
        self.features = nn.Sequential(
            # [B, 2, S, N] -> [B, 32, S, N]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 只在 N 维上做 pooling，保留符号维 S
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

            # # [B, 64, S, N/4] -> [B, 128, S, N/4]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.final_cov = nn.Sequential(
        
        nn.Conv1d(1,8,kernel_size=7,padding=3),
        nn.BatchNorm1d(8),
        nn.ReLU(inplace=True),

        nn.Conv1d(8,1,kernel_size=7,padding=3),


        )

        # self.features = nn.Sequential(
        #     # [B, 2, S, N] -> [B, 32, S, N]
        #     nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     # 只在 N 维上做 pooling，保留符号维 S
        #     nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2



        #     # [B, 32, S, N/2] -> [B, 64, S, N/2]
        #     # nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(64),
        #     # nn.ReLU(inplace=True),
        #     InvertedResidual(32, 64, 1, 6),
        #     nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

        #     # [B, 64, S, N/4] -> [B, 128, S, N/4]
        #     # nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),
        #     InvertedResidual(64, 128, 1, 6),


        #     # # [B, 64, S, N/4] -> [B, 128, S, N/4]
        #     # nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(inplace=True),
        # )

        # 自适应池化，和 N、S 的具体数值解耦
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # [B, 128]
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),        
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, M, N, 2] -> # x: [B*S, M, N, 2]
        B, S, M, N, C=x.shape
        x = x.reshape(B * S, M, N, C)

        # x: [B*S, M, N, 2] -> [B*S, 2, M, N]
        
        x = x.permute(0, 3, 1, 2).contiguous()
        # print(x.shape)
        x = self.features(x)
        x = self.global_pool(x)
        # [B*S,128]
        x = x.view(B, S, -1) 
        # data_emb = x[:, -1, :]              # (B, width) 最后一段当 data
        # pilot_embs = x[:, :-1, :]           # (B, S-1, width)

        # # pilots 汇聚（最简单先用 mean；你也可以换 attention pooling）
        # pilot_mean = pilot_embs.mean(dim=1) if pilot_embs.numel() > 0 else torch.zeros_like(data_emb)

        # x = torch.cat([
        #     data_emb,
        #     pilot_mean,

        # ], dim=-1)
        x = self.final_cov(x)
        x = x.view(B,-1)

        x = self.classifier(x)
        return x


# 不确定性度量
class PureCNN2D_Dirichlet(nn.Module):
    def __init__(self, num_classes=2, in_channels=2):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),   # 输出 evidence logits
        )

        # ✅ 新增：新的 classifier（用于二次训练）
        self.new_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # 默认用旧 classifier 推理（你也可以切换）
        self.use_new_head = False

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(x)
        x = self.global_pool(x)

        if self.use_new_head:
            evidence_logits = self.new_classifier(x)
        else:
            evidence_logits = self.classifier(x)
        evidence = F.softplus(evidence_logits)  # e >= 0
        alpha = evidence + 1.0                  # Dirichlet 参数
        
        return alpha
        
# -------------------------
# 小工具：简单1D卷积块
# -------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, cin, cout, k=5, s=1, p=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm1d(cout),
            nn.SiLU(),
            nn.Conv1d(cout, cout, kernel_size=k, stride=1, padding=p),
            nn.BatchNorm1d(cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# 同步网络：SyncNet
# 输入 x: (B, num_pilot+1, N, 2)
# 输出：
#   logits: (B, num_classes)
#   delta : (B, 1)  （细偏移 residual）
# -------------------------
class SyncNet(nn.Module):
    def __init__(
        self,
        num_pilot=32-4,
        N=150,
        num_classes=23,          # 例如 [-20..20] => 41
        pilot_feat=64,
        data_feat=64,
        corr_feat=32,
        use_corr=True
    ):
        super().__init__()
        self.num_pilot = num_pilot
        self.N = N
        self.num_classes = num_classes
        self.use_corr = use_corr

        # Pilot encoder：输入通道 = num_pilot * 2 （把 num_pilot 个导频符号沿通道拼起来）
        self.pilot_enc = nn.Sequential(
            ConvBlock1D(num_pilot * 2, pilot_feat, k=7, p=3),
            nn.MaxPool1d(2),
            ConvBlock1D(pilot_feat, pilot_feat, k=5, p=2),
            nn.AdaptiveAvgPool1d(1),
        )

        # Data encoder：输入通道 = 2
        self.data_enc = nn.Sequential(
            ConvBlock1D(2, data_feat, k=7, p=3),
            nn.MaxPool1d(2),
            ConvBlock1D(data_feat, data_feat, k=5, p=2),
            nn.AdaptiveAvgPool1d(1),
        )

        # Corr encoder：简单做一个“导频与数据符号的相关/能量差异”特征
        # 这里我们不需要知道导频模板，用 pilot 与 data 的互相关（粗略）也能提供对齐线索
        if use_corr:
            self.corr_enc = nn.Sequential(
                ConvBlock1D(4, corr_feat, k=5, p=2),  # 输入4通道: corr_real/corr_imag/|corr|/energy_diff
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            self.corr_enc = None

        # Feature fusion
        fusion_dim = pilot_feat + data_feat + (corr_feat if use_corr else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        # Heads
        self.head_cls = nn.Linear(128, num_classes)
        self.head_reg = nn.Linear(128, 1)  # residual

    def forward(self, x):
        """
        x: (B, num_pilot+1, N, 2)
        """
        B, S, N, C = x.shape
        # assert S == self.num_pilot + 1
        # assert N == self.N
        # assert C == 2

        # 1) Pilot branch
        pilots = x[:, :self.num_pilot]  # (B, num_pilot, N, 2)
        # reshape to (B, num_pilot*2, N)
        pilots = pilots.permute(0, 1, 3, 2).contiguous()  # (B, num_pilot, 2, N)
        pilots = pilots.view(B, self.num_pilot * 2, N)    # (B, num_pilot*2, N)

        f_p = self.pilot_enc(pilots).squeeze(-1)          # (B, pilot_feat)

        # 2) Data branch
        data = x[:, self.num_pilot]                       # (B, N, 2)
        data = data.permute(0, 2, 1).contiguous()         # (B, 2, N)
        f_d = self.data_enc(data).squeeze(-1)             # (B, data_feat)

        # 3) Corr branch (optional)
        if self.use_corr:
            # 取最后一个导频符号作为参考（也可以改成所有导频平均）
            pilot_last = x[:, self.num_pilot - 1].permute(0, 2, 1)  # (B,2,N)
            # 简单互相关（不做滑动，只做点积相关）：(B,)
            # real/imag/abs
            corr = torch.sum(pilot_last * data, dim=-1)             # (B,2)  -> 对应 I,Q 相关
            corr_real = corr[:, 0:1]
            corr_imag = corr[:, 1:2]
            corr_abs = torch.sqrt(corr_real**2 + corr_imag**2 + 1e-8)

            # 能量差（对齐错位时能量分布会变）
            e_p = torch.mean(pilot_last**2, dim=(1,2), keepdim=True)  # (B,1,1)
            e_d = torch.mean(data**2, dim=(1,2), keepdim=True)        # (B,1,1)
            e_diff = (e_d - e_p).view(B, 1)

            # 把这些标量扩展成 length=1 的 1D “序列”给 corr_enc
            corr_feat = torch.cat([corr_real, corr_imag, corr_abs, e_diff], dim=1)  # (B,4)
            corr_feat = corr_feat.unsqueeze(-1)                                     # (B,4,1)
            f_c = self.corr_enc(corr_feat).squeeze(-1)                              # (B,corr_feat)
            feat = torch.cat([f_p, f_d, f_c], dim=1)
        else:
            feat = torch.cat([f_p, f_d], dim=1)

        # 4) Fusion + Heads
        h = self.fusion(feat)
        logits = self.head_cls(h)
        delta = torch.tanh(self.head_reg(h)) * 0.5  # 输出限制在 [-0.5, 0.5]

        return logits


class PureCNN2D_big(nn.Module):
    """
    输入: x.shape = (B, S, N, 2)
          S = num_pilot + 1, N = 每符号采样点数
    输出: logits.shape = (B, 2)  # 二分类
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 2):
        super().__init__()
        width = 2
        # 特征提取部分：全卷积 + 池化
        self.features = nn.Sequential(
            # [B, 2, S, N] -> [B, 32, S, N]
            nn.Conv2d(in_channels, 32 *width, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 *width),
            nn.ReLU(inplace=True),
            # 只在 N 维上做 pooling，保留符号维 S
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2

            # [B, 32, S, N/2] -> [B, 64, S, N/2]
            nn.Conv2d(32 *width, 64*width, kernel_size=3, padding=1),
            nn.BatchNorm2d(64*width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

            # [B, 64, S, N/4] -> [B, 128, S, N/4]
            nn.Conv2d(64*width, 128*width, kernel_size=3, padding=1),
            nn.BatchNorm2d(128*width),
            nn.ReLU(inplace=True),
        )

        # 自适应池化，和 N、S 的具体数值解耦
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # [B, 128]
            nn.Dropout(0.5),
            nn.Linear(128*width, 64*width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64*width, num_classes),          # [B, 2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, N, 2] -> [B, 2, S, N]
        
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class PureCNN2D_fusion(nn.Module):
    """
    输入: x.shape = (B, S, N, 2)
          S = num_pilot + 1, N = 每符号采样点数
    输出: logits.shape = (B, 2)  # 二分类
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 2):
        super().__init__()

        # 特征提取部分：全卷积 + 池化
        self.features = nn.Sequential(
            # [B, 2, S, N] -> [B, 32, S, N]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 只在 N 维上做 pooling，保留符号维 S
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

        # 自适应池化，和 N、S 的具体数值解耦
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]

        # 分类头
        self.classifier = nn.Sequential(
                                     # [B, 128]
            nn.Dropout(0.25),
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),          # [B, 2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, N, 2] -> [B, 2, S, N]
        pattern_feat = build_pattern_features(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.features(x)
        x = self.global_pool(x).view(x.size(0), -1)
       
        x = torch.cat([x, pattern_feat], dim=-1)
        feat = self.classifier(x)
        
        return feat

class PureCNN2D_GLRT(nn.Module):
    """
    输入: x.shape = (B, S, N, 2)
          S = num_pilot + 1, N = 每符号采样点数
    输出: logits.shape = (B, num_classes)  # 默认二分类

    在原 PureCNN2D 基础上，显式加入:
      - 每个符号能量 + data 与导频平均能量的差
      - data 与每个 pilot 的相关 / 归一化相关
    """
    def __init__(self, num_pilot: int = 4,
                 num_classes: int = 2,
                 in_channels: int = 2):
        super().__init__()
        self.num_pilot = num_pilot
        self.num_classes = num_classes

        # ========= CNN 特征提取部分：和你原来的基本一样 =========
        self.features = nn.Sequential(
            # [B, 2, S, N] -> [B, 32, S, N]
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 只在 N 维上做 pooling，保留符号维 S
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/2

            # [B, 32, S, N/2] -> [B, 64, S, N/2]
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),   # N -> N/4

            # [B, 64, S, N/4] -> [B, 128, S, N/4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 自适应池化，和 N、S 的具体数值解耦
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]

        # ========= 额外的 GLRT 风格统计特征 =========
        # energy_feat: 4(导频) + 1(data) + 1(diff) = 6
        # cov_feat   : 4(corr) + 4(corr_norm)     = 8
        extra_dim = 7 + 8

        # 分类头：输入维度从 128 -> 128 + extra_dim
        self.classifier = nn.Sequential(
            nn.Linear(128 + extra_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),          # [B, num_classes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, N, 2]
        """
        B, S, N, C = x.shape
        assert C == 2, "最后一维必须是 IQ=2"
        assert S == self.num_pilot + 1, f"S 应该等于 num_pilot+1, 但现在是 {S}"

        # ========= 1. CNN 分支 =========
        # 先保存原始 x，用于后面算能量 / 相关统计
        x_orig = x

        # [B, S, N, 2] -> [B, 2, S, N]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.features(x)
        x = self.global_pool(x)          # [B, 128, 1, 1]
        feat_cnn = x.view(B, -1)         # [B, 128]

        # ========= 2. 显式统计特征 (GLRT 相关) =========
        # ---- 2.1 能量特征 ----
        # x_orig: [B, S, N, 2]
        amp2 = x_orig[..., 0]**2 + x_orig[..., 1]**2   # [B, S, N]
        E_sym = amp2.mean(dim=-1)                      # [B, S]  每个符号平均能量

        # S = num_pilot + 1: 前 num_pilot 个是导频，最后一个是 data
        E_pilots_1 = E_sym[:, :2]           # [B, num_pilot]
        E_pilots_0 = E_sym[:, 2:self.num_pilot]
        E_data   = E_sym[:, self.num_pilot]            # [B]
        E_pilot_mean_1 = E_pilots_1.mean(dim=1, keepdim=True)   # [B, 1]
        E_pilot_mean_0 = E_pilots_0.mean(dim=1, keepdim=True)   # [B, 1]
        E_diff_1 = E_data.unsqueeze(1) - E_pilot_mean_1          # [B, 1]
        E_diff_0 = E_data.unsqueeze(1) - E_pilot_mean_0          # [B, 1]

        # [B, 4 + 1 + 1] = [B, 7]
        energy_feat = torch.cat(
            [E_pilots_1, E_pilots_0,E_data.unsqueeze(1), E_diff_1,E_diff_0],
            dim=1
        )

        # ---- 2.2 pilot–data 相关 / 归一化相关 ----
        pilots = x_orig[:, :self.num_pilot, :, :]      # [B, num_pilot, N, 2]
        data   = x_orig[:, self.num_pilot, :, :]       # [B, N, 2]

        # 展平成 2N 维实向量
        pilots_flat = pilots.reshape(B, self.num_pilot, -1)  # [B, num_pilot, 2N]
        data_flat   = data.reshape(B, -1)                    # [B, 2N]

        # 非归一化相关: <v_data, v_pilot_k>
        corr = (pilots_flat * data_flat.unsqueeze(1)).sum(dim=-1)  # [B, num_pilot]

        # 归一化相关系数
        data_energy_vec  = (data_flat**2).sum(dim=-1, keepdim=True)  # [B, 1]
        pilot_energy_vec = (pilots_flat**2).sum(dim=-1)              # [B, num_pilot]
        corr_norm = corr / (torch.sqrt(pilot_energy_vec * data_energy_vec + 1e-8))  # [B, num_pilot]

        # [B, 4 + 4] = [B, 8]
        cov_feat = torch.cat([corr, corr_norm], dim=1)

        # 最终的额外特征
        extra_feat = torch.cat([energy_feat, cov_feat], dim=1)  # [B, 14]

        # ========= 3. 融合 CNN 特征 + 显式统计特征 =========
        feat_all = torch.cat([feat_cnn, extra_feat], dim=1)     # [B, 128 + 14]
        logits = self.classifier(feat_all)                      # [B, num_classes]

        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F

class BackscatterCNN2D(nn.Module):

    """
    适配 FramePilotSymbolDataset:
      x: [B, S, N, 2],  S = num_pilot + 1
      y: 标量 0/1

    结构:
      - Pilot 分支: 用 4 个导频估计等效信道特征
      - Data  分支: 用当前数据符号提取瞬时回波特征
      - 融合后做二分类
    """
    def __init__(self, num_pilot: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_pilot = num_pilot
        self.num_classes = num_classes

        # -------- Pilot 分支：4 个导频一起卷积，提取“信道特征” --------
        # 输入形状: [B, 2, num_pilot, N]
        self.pilot_feat = nn.Sequential(
            # 卷积核在符号维直接覆盖 4 个导频，在时间维用 7 点卷积
            # 近似相当于一种“非线性信道估计”
            nn.Conv2d(2, 32, kernel_size=(num_pilot, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 之后只在时间维上进一步提取特征
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pilot_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, 1, 1] -> [B, 64]

        # -------- Data 分支：当前数据符号的时域特征 --------
        # 输入形状: [B, 2, 1, N]
        self.data_feat = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(1, 2)),   # 只在 N 上降采样: N -> N/2

            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.data_pool = nn.AdaptiveAvgPool2d((1, 1))    # [B, 64, 1, 1] -> [B, 64]

        # -------- 分类头：把“信道特征 + 数据特征”做融合 --------
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, N, 2],  S = num_pilot + 1
        """
        B, S, N, C = x.shape
        assert C == 2, "最后一维必须是 IQ 两个通道"
        assert S == self.num_pilot + 1, f"S 应为 num_pilot + 1, 但得到 {S}"

        # 拆分导频和当前数据符号
        pilots = x[:, :self.num_pilot, :, :]     # [B, 4, N, 2]
        data   = x[:, self.num_pilot, :, :]      # [B, N, 2]

        # ---- Pilot 分支 ----
        # [B, 4, N, 2] -> [B, 2, 4, N]
        pilots = pilots.permute(0, 3, 1, 2).contiguous()
        p = self.pilot_feat(pilots)              # [B, 64, 1, N]
        p = self.pilot_pool(p)                   # [B, 64, 1, 1]
        p = p.view(B, -1)                        # [B, 64]

        # ---- Data 分支 ----
        # data: [B, N, 2] -> [B, 2, 1, N]
        data = data.permute(0, 2, 1).unsqueeze(2).contiguous()
        d = self.data_feat(data)                 # [B, 64, 1, N']
        d = self.data_pool(d)                    # [B, 64, 1, 1]
        d = d.view(B, -1)                        # [B, 64]

        # ---- 融合 + 分类 ----
        feat = torch.cat([p, d], dim=1)          # [B, 128]
        logits = self.classifier(feat)           # [B, num_classes]
        return logits

class ResidualBlock2D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, 
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        return self.relu(out + x)

# ------------- 主网络：2D 回波残差网络 -----------------
class Res2DBackscatterNet(nn.Module):
    def __init__(self, base_channels=32, num_classes=2):
        super().__init__()

        # 输入：2 通道（实部、虚部）
        self.stem = nn.Sequential(
            nn.Conv2d(2, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # 三个残差块（2D + dilation）
        self.layer1 = ResidualBlock2D(base_channels, kernel_size=5, dilation=1)
        self.layer2 = ResidualBlock2D(base_channels, kernel_size=5, dilation=2)
        self.layer3 = ResidualBlock2D(base_channels, kernel_size=5, dilation=4)

        # 下采样
        self.down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU()
        )

        # 再来一个 2D 残差块
        self.layer4 = ResidualBlock2D(base_channels * 2, kernel_size=3, dilation=1)

        # 全局池化 -> [B, C, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(base_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 2, H, W]   # 2 通道：实部 & 虚部
        """
        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.down(out)
        out = self.layer4(out)

        # 全局平均池化
        out = self.global_pool(out)   # [B, C, 1, 1]
        out = out.view(out.size(0), -1)  # 展平 [B, C]

        logits = self.fc(out)
        return logits
    

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMNet(nn.Module):
    """
    CMNet: Strict implementation of Table I (Input 5x5×2 sample covariance matrix)
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
        # self.pool = nn.MaxPool2d(kernel_size=(1, 2)),

        # Flatten 后尺寸：32 × 4 × 4 = 512
        self.dropout1 = nn.Dropout(p=0.5)

        # F1: 128 × 512
        self.fc1 = nn.Linear(800, 128)

        self.dropout2 = nn.Dropout(p=0.25)

        # F2: 2 × 128
        self.fc2 = nn.Linear(128, 2)   # 输出 2 维 score vector

    def forward(self, x):
        """
        x: (B, 2, 8, 8)
        """
        # C1 + ReLU
        x = F.relu(self.conv1(x))
        # C2 + ReLU
        x = F.relu(self.conv2(x))
        # S1 Pooling
        # x = self.pool(x)   # 变成 (B, 32, 4, 4)

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

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        # self.net = nn.Sequential(
        #     nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad, bias=False),
        #     nn.BatchNorm1d(out_ch),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=dilation, padding=pad, bias=False),
        #     nn.BatchNorm1d(out_ch),
        # )

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, in_ch*6, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_ch*6),
            nn.ReLU(),
            nn.Conv1d(in_ch*6, in_ch*6, kernel_size=k, dilation=dilation, padding=pad,groups=in_ch*6, bias=False),
            nn.BatchNorm1d(in_ch*6),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Conv1d(in_ch*6, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))

class TCNConcatDetector(nn.Module):
    """
    输入: x (B, S, N, 2)
    输出: logits (B, 2)
    """
    def __init__(self, width=64, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LazyConv1d(width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU()
        )
        self.blocks = nn.Sequential(
            ConvBlock1D(width, width, k=7, dilation=1, dropout=dropout),
            ConvBlock1D(width, width, k=7, dilation=2, dropout=dropout),
            ConvBlock1D(width, width, k=7, dilation=4, dropout=dropout),
            ConvBlock1D(width, width, k=7, dilation=8, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (B, C, L) -> (B, C, 1)
            nn.Flatten(),              # -> (B, C)
            nn.Linear(width, 2)
        )

    def forward(self, x):
        # x: (B, S, N, 2) -> (B, 2, S*N)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 2, S, N)
        B, C, S, N = x.shape
        x = x.view(B, C*S,  N)                 # (B, 2S, N)

        x = self.stem(x)
        x = self.blocks(x)
        logits = self.head(x)
        return logits

class SegmentEncoder(nn.Module):
    """把单个段 (2, N) 编码成 embedding 向量"""
    def __init__(self, width=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            ConvBlock1D(width, width, k=7, dilation=1, dropout=dropout),
            ConvBlock1D(width, width, k=7, dilation=2, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),   # -> (B, width)
        )

    def forward(self, seg):  # seg: (B, 2, N)
        return self.net(seg) # (B, width)

class PilotCondDetector(nn.Module):
    """
    输入: x (B, S, N, 2), 默认最后一个 segment 是 data，其余是 pilot/辅助
    输出: logits (B, 2)
    """
    def __init__(self, width=64, dropout=0.1):
        super().__init__()
        self.enc = SegmentEncoder(width=width, dropout=dropout)

        # 融合： [data_emb, pilot_mean, data - pilot_mean, data * pilot_mean]
        self.head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 2)
        )

    def forward(self, x):
        # x: (B, S, N, 2) -> (B, S, 2, N)
        x = x.permute(0, 1, 3, 2).contiguous()
        B, S, C, N = x.shape

        # 展平 batch*segment，复用同一个 encoder
        x_flat = x.view(B * S, C, N)          # (B*S, 2, N)
        emb = self.enc(x_flat)                # (B*S, width)
        emb = emb.view(B, S, -1)              # (B, S, width)

        data_emb = emb[:, -1, :]              # (B, width) 最后一段当 data
        pilot_embs = emb[:, :-1, :]           # (B, S-1, width)

        # pilots 汇聚（最简单先用 mean；你也可以换 attention pooling）
        pilot_mean = pilot_embs.mean(dim=1) if pilot_embs.numel() > 0 else torch.zeros_like(data_emb)

        feat = torch.cat([
            data_emb,
            pilot_mean,

        ], dim=-1)

        logits = self.head(feat)
        return logits

# class FastSegmentEncoder(nn.Module):
#     """
#     段内共享 encoder：把单个 segment 的 IQ 序列 (2, N) 编成向量 (d)
#     """
#     def __init__(self, width: int = 64, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             # nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
#             # nn.BatchNorm1d(64),
#             # nn.GELU(),
#             # # nn.ReLU(inplace=True),
#             # nn.Dropout(dropout),

#             nn.Conv1d(2, width, kernel_size=7, padding=3, bias=False,stride=2),
#             nn.BatchNorm1d(width),
#             # nn.GELU(),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),

#             nn.Conv1d(width, width, kernel_size=7, padding=3, bias=False,stride=2),
#             nn.BatchNorm1d(width),
#             # nn.GELU(),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),

#             # nn.AdaptiveAvgPool1d(1),
#             nn.AvgPool1d(kernel_size=38),
#             nn.Flatten(start_dim=1),  # -> (B, width)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (Bseg, 2, N)
#         return self.net(x)  # (Bseg, width)


class FastSegmentEncoder2D_1(nn.Module):
    def __init__(self, width: int = 32, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, width, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((33, 1)),
        )

    def forward(self, x):
        # x: (B, 2, 33, 150)
        # print(x.shape)
        B = x.shape[0]

        y = self.net(x)              # (B, 32, 33, 1)
        # print(y.shape)
        y = y.reshape(B, 32, 33)     # (B, 32, 33)
        # print(y.shape)
        y = y.permute(0, 2, 1)       # (B, 33, 32)
        # print(y.shape)

        return y


class AttnPoolPilots(nn.Module):
    """
    用可学习 query 对 pilots 做 attention pooling（比简单 mean 更强）
    pilots: (B, P, d)
    return: (B, d)
    """
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.scale = d ** -0.5

    def forward(self, pilots: torch.Tensor) -> torch.Tensor:
        # (B,P,d) · (d,) -> (B,P)
        score = (pilots * self.q).sum(dim=-1) * self.scale
        w = torch.softmax(score, dim=1)  # (B,P)
        ctx = (w.unsqueeze(-1) * pilots).sum(dim=1)  # (B,d)
        return ctx

# class CrossAttentionLayer(nn.Module):
#     """简单的交叉注意力层"""
#     def __init__(self, d_model):
#         super().__init__()
#         self.q_proj = nn.Linear(d_model, d_model)  # 为序列A生成Q
#         self.k_proj = nn.Linear(d_model, d_model)  # 为序列B生成K  
#         self.v_proj = nn.Linear(d_model, d_model)  # 为序列B生成V
        
#     def forward(self, seq_a, seq_b):
#         """
#         seq_a: (B, L_a, d) - 查询序列
#         seq_b: (B, L_b, d) - 键值序列
#         returns: (B, L_a, d) - 序列A关注序列B后的结果
#         """
#         Q = self.q_proj(seq_a)  # (B, L_a, d)
#         K = self.k_proj(seq_b)  # (B, L_b, d)
#         V = self.v_proj(seq_b)  # (B, L_b, d)
        
#         # 计算注意力
#         scores = torch.bmm(Q, K.transpose(1, 2))  # (B, L_a, L_b)
#         scores = scores / (Q.size(-1) ** 0.5)
#         attn_weights = F.softmax(scores, dim=-1)  # (B, L_a, L_b)
        
#         # 加权求和
#         output = torch.bmm(attn_weights, V)  # (B, L_a, d)
#         return output





class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 预计算缩放因子
        self.scale = 1.0 / (d_model ** 0.5)

    def forward(self, seq_a, seq_b):

        Q = self.q_proj(seq_a)
        K = self.k_proj(seq_b)
        V = self.v_proj(seq_b)

        # transpose尽量简单化
        KT = K.permute(0, 2, 1)

        # attention
        scores = torch.bmm(Q, KT)

        # 避免 Pow / Div
        scores = scores * self.scale

        # softmax
        attn = F.softmax(scores, dim=2)

        # output
        output = torch.bmm(attn, V)

        return output

class FastPilotDetector_EvenOdd(nn.Module):
    """
    改进点：
    1) pilots 按偶/奇分组（对应 bit=0/1），分别做 attention pooling 得到 ctx0/ctx1
    2) 用门控融合 ctx0/ctx1（避免 0/1 pilots 混平均抵消）
    3) 分类 head 用更强的融合特征： [data, ctx, data-ctx, data*ctx]

    输入: x (B, S, N, 2)   约定最后一段是 data，前 S-1 段是 pilots
    输出: logits (B, 2)
    """
    def __init__(self, width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.enc = FastSegmentEncoder2D_1(width=width, dropout=dropout)
        # self.enc_pilots = FastSegmentEncoder(width=width, dropout=dropout)

        # 偶数组/奇数组 pilots 各自做 attention pooling
        # self.pool0 = AttnPoolPilots(width)
        # self.pool1 = AttnPoolPilots(width)
        self.pool0 = CrossAttentionLayer(width)
        self.pool1 = CrossAttentionLayer(width)
        # self.pool0 = nn.AdaptiveAvgPool1d(width)
        # self.pool1 = nn.AdaptiveAvgPool1d(width)

        # 门控：根据 ctx0/ctx1 决定更信哪一组（也能自适应融合）
        # self.gate = nn.Sequential(
        #     nn.Linear(width * 3, width),
        #     # nn.GELU(),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(width, 1),
        #     nn.Sigmoid()
        # )

        # 分类头：用更丰富的融合特征（通常比单纯 concat 更稳）
        self.cls = nn.Sequential(
            nn.Linear(3*width, width),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, N, 2)
        # B, S, N, C = x.shape
        # assert C == 2, "最后一维必须是 IQ=2"
        # assert S >= 2, "S 必须 >= 2（至少 1 个 pilot + 1 个 data）"

        # (B,S,N,2) -> (B*S,2,N)
        # x_seg = x.permute(0, 1, 3, 2).contiguous().view(33, 2, 150)
        # x_seg = x.permute(0, 1, 3, 2).contiguous().view(33, 2, 150)

        # (B*S,2,N) -> (B,S,d)
        # emb = self.enc(x_seg).view(1, 33, 32)
        # print(x.shape)
        # print(x.shape)
        emb = self.enc(x)
        # print(emb.shape)
        # x_seg_data = x[:,-1,:,:].unsqueeze(1).permute(0, 1, 3, 2).contiguous().view(B, 2, N)
        # x_seg_pilots = x[:,:-1,:,:].permute(0, 1, 3, 2).contiguous().view(B*(S-1), 2, N)
        # emb_data = self.enc(x_seg_data)
        # emb_pilots = self.enc_pilots(x_seg_pilots).view(B, (S-1), -1)
        # data = emb_data
        # pilots = emb_pilots

        # data = emb[:, 32:33, :]       # (B,d)
        # pilots = emb[:, 0:32, :]    # (B,P,d)  P=S-1

        idx_data = torch.tensor([32], dtype=torch.long, device=emb.device)
        data = torch.index_select(emb, dim=1, index=idx_data)

        # pilots = emb[:, 0:32, :]
        # idx_pilots = torch.arange(0, 32, dtype=torch.long, device=emb.device)
        # pilots = torch.index_select(emb, dim=1, index=idx_pilots)  # (B,32,d)

        # P = pilots.size(1)

        # pilots 偶/奇分组：pilot 序号 0..P-1
        # idx0 = torch.arange(0, P, 2, device=pilots.device)  # even -> bit 0（按你约定）
        # idx1 = torch.arange(1, P, 2, device=pilots.device)  # odd  -> bit 1
        # p0 = pilots[:, 0::2, :]
        # p1 = pilots[:, 1::2, :]
        # p0 = torch.cat([
        #     pilots[:, 0:1, :],
        #     pilots[:, 2:3, :],
        #     pilots[:, 4:5, :],
        #     pilots[:, 6:7, :],
        #     pilots[:, 9:10, :],
        #     pilots[:, 11:12, :],
        #     pilots[:, 13:14, :],
        #     pilots[:, 15:16, :],
        #     pilots[:, 18:19, :],
        #     pilots[:, 22:23, :],
        #     pilots[:, 24:25, :],
        #     pilots[:, 26:27, :],
        #     pilots[:, 29:30, :],
        #     pilots[:, 31:32, :],
        #     pilots[:, 34:35, :],
        # ], dim=1)

        # p1 = torch.cat([
        #     pilots[:, 1:2, :],
        #     pilots[:, 3:4, :],
        #     pilots[:, 5:6, :],
        #     pilots[:, 7:8, :],
        #     pilots[:, 10:11, :],
        #     pilots[:, 12:13, :],
        #     pilots[:, 14:15, :],
        #     pilots[:, 16:17, :],
        #     pilots[:, 20:21, :],
        #     pilots[:, 23:24, :],
        #     pilots[:, 25:26, :],
        #     pilots[:, 27:28, :],
        #     pilots[:, 30:31, :],
        #     pilots[:, 32:33, :],
        #     pilots[:, 35:36, :],
        # ], dim=1)
        idx0 = torch.tensor(
        [0, 2, 4, 6,8,10, 12, 14, 16,18,20,22,24,26,28,30],
        dtype=torch.long,
        device=emb.device,
        )

        idx1 = torch.tensor(
            [1, 3, 5, 7, 9,11,13,15,17,19,21,23,25,27,29,31],
            dtype=torch.long,
            device=emb.device,
        )

        p0 = torch.index_select(emb, dim=1, index=idx0)
        p1 = torch.index_select(emb, dim=1, index=idx1)


        ctx0 = self.pool0(data, p0).squeeze(1)
        ctx1 = self.pool1(data, p1).squeeze(1)

        data0 = data.squeeze(1)
        # ctx0 = self.pool0(p0)
        # ctx1 = self.pool1(p1)
        # print(p0.shape)
        # ctx0 = torch.mean(p0, dim=1)
        # ctx1 = torch.mean(p1, dim=1)
        # print(ctx0.shape)
        # print(data.shape)
        # idx_first = torch.tensor([0], dtype=torch.long, device=data.device)

        # data0 = torch.index_select(data, dim=1, index=idx_first).reshape(1, 32)
        # ctx10 = torch.index_select(ctx1, dim=1, index=idx_first).reshape(B, 32)
        # ctx00 = torch.index_select(ctx0, dim=1, index=idx_first).reshape(B, 32)

        # feat = torch.cat([data0, ctx10, ctx00], dim=1)
        feat = torch.cat([data0, ctx1, ctx0], dim=1)

        # feat = torch.cat([
        #     data[:,0,:],
        #     ctx1[:,0,:],
        #     ctx0[:,0,:]
        # ], dim=-1)
        # if idx0.numel() > 0:
        #     # p0 = pilots.index_select(dim=1, index=idx0)     # (B,P0,d)
        #     # ctx0 = self.pool0(p0)                           # (B,d)
        #     # ctx0 = p0.mean(dim=1)
        #     ctx0 = self.pool0(data.unsqueeze(1),p0).squeeze(1)
        # else:
        #     ctx0 = torch.zeros_like(data)

        # if idx1.numel() > 0:
        #     # p1 = pilots.index_select(dim=1, index=idx1)     # (B,P1,d)
        #     # ctx1 = self.pool1(p1)                           # (B,d)
        #     # ctx1 = p1.mean(dim=1)
        #     ctx1 = self.pool1(data.unsqueeze(1),p1).squeeze(1)
            # print(ctx1.shape)
        # else:
        #     ctx1 = torch.zeros_like(data)

        # 门控融合 --这里感觉得有data，如果我这里不用这个门控，把他们俩都留下呢？
        # g = self.gate(torch.cat([ctx0, ctx1,data], dim=-1))       # (B,1)
        # ctx = g * ctx1 + (1.0 - g) * ctx0                    # (B,d)
        # ctx = ctx1+ctx0
        # ctx = ctx1
        # 丰富融合特征
        # feat = torch.cat([data, ctx, data - ctx, data * ctx], dim=-1)  # (B,4d)
        # feat = torch.cat([data, ctx], dim=-1)  # (B,4d)
        # feat = torch.cat([data, ctx,data - ctx,], dim=-1)  # (B,4d)
        # feat = torch.cat([data, ctx,data * ctx,], dim=-1)  # (B,4d)
        # feat = data
        # feat = torch.cat([data, ctx1,ctx0], dim=-1)  # (B,3d)

        return self.cls(feat)

# class FastSegmentEncoder2D(nn.Module):
#     """
#     输入: (Bseg, 2, M, N)
#     输出: (Bseg, width)
#     """
#     def __init__(self, width: int = 64, dropout: float = 0.1, in_channels: int = 2):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1, 2)),  # 只池化 N

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1, 2)),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (Bseg,128,1,1)
#         self.proj = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(dropout),
#             nn.Linear(128, width),
#             nn.GELU(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.pool(x)
#         x = self.proj(x)
#         return x

class FastPilotDetector_EvenOdd_MIMO(nn.Module):
    """
    MIMO版：输入 (B, S, M, N, 2)
    约定最后一段是 data，前 S-1 段是 pilots
    """
    def __init__(self, width: int = 64, dropout: float = 0.1, in_channels: int = 2):
        super().__init__()

        # 2D encoder：把 (2, M, N) 编成一个向量 (d=width)
        self.enc = FastSegmentEncoder2D(width=width, dropout=dropout, in_channels=in_channels)

        # 偶/奇 pilots 分别池化
        self.pool0 = AttnPoolPilots(width)
        self.pool1 = AttnPoolPilots(width)

        # gate 融合 ctx0/ctx1（你原来是 concat(ctx0,ctx1,data) -> g）
        self.gate = nn.Sequential(
            nn.Linear(width * 3, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
            nn.Sigmoid()
        )

        self.final_cov = nn.Sequential(
        
        nn.Conv1d(5,8,kernel_size=7,padding=3),
        nn.BatchNorm1d(8),
        nn.ReLU(inplace=True),

        nn.Conv1d(8,1,kernel_size=7,padding=3),


        )
        self.classifier = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 2)
        )
        # 分类头：这里沿用你现在用的 feat=[data, ctx] -> 2*width
        self.cls = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,S,M,N,2)
        B, S, M, N, C = x.shape
        assert C == 2, "最后一维必须是 IQ=2"
        assert S >= 2, "S 必须 >= 2（至少 1 个 pilot + 1 个 data）"

        # (B,S,M,N,2) -> (B*S,2,M,N)
        x_seg = x.reshape(B * S, M, N, C).permute(0, 3, 1, 2).contiguous()

        # (B*S,d) -> (B,S,d)
        emb = self.enc(x_seg).view(B, S, -1)

        x = self.final_cov(emb)
        x = x.view(B,-1)

        x = self.classifier(x)
        return x

        # data = emb[:, -1, :]       # (B,d)
        # pilots = emb[:, :-1, :]    # (B,P,d)
        # P = pilots.size(1)

        # idx0 = torch.arange(0, P, 2, device=pilots.device)
        # idx1 = torch.arange(1, P, 2, device=pilots.device)

        # if idx0.numel() > 0:
        #     p0 = pilots.index_select(dim=1, index=idx0)
        #     ctx0 = self.pool0(p0)
        # else:
        #     ctx0 = torch.zeros_like(data)

        # if idx1.numel() > 0:
        #     p1 = pilots.index_select(dim=1, index=idx1)
        #     ctx1 = self.pool1(p1)
        # else:
        #     ctx1 = torch.zeros_like(data)

        # g = self.gate(torch.cat([ctx0, ctx1, data], dim=-1))  # (B,1)
        # ctx = g * ctx1 + (1.0 - g) * ctx0

        # feat = torch.cat([data, ctx], dim=-1)  # (B,2d)
        # return self.cls(feat)


# --gpt
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
    
# =========================
# Model 1: TCN + FiLM
# =========================

class DilatedResBlock1D(nn.Module):
    def __init__(self, d: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(d, d, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


class TCNSegmentEncoder(nn.Module):
    """
    输入单个 segment: (Bseg, 2, N)
    输出:
      seq: (Bseg, N, d)
      emb: (Bseg, d)
    """
    def __init__(self, in_ch: int = 2, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            DilatedResBlock1D(width, kernel_size=5, dilation=1, dropout=dropout),
            DilatedResBlock1D(width, kernel_size=5, dilation=2, dropout=dropout),
            DilatedResBlock1D(width, kernel_size=5, dilation=4, dropout=dropout),
            DilatedResBlock1D(width, kernel_size=5, dilation=8, dropout=dropout),
        ])
        self.pool = AttnPool1D(width)

    def forward(self, x: torch.Tensor):
        h = self.stem(x)                   # (Bseg, d, N)
        for blk in self.blocks:
            h = blk(h)
        seq = h.transpose(1, 2)           # (Bseg, N, d)
        emb = self.pool(seq)              # (Bseg, d)
        return seq, emb


class TCNFiLMPilotDetector(nn.Module):
    """
    推荐优先尝试。
    输入:  x (B, S, N, 2)
    输出: logits (B, 2)
    """
    def __init__(self, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = TCNSegmentEncoder(in_ch=2, width=width, dropout=dropout)
        self.pilot_ctx = EvenOddPilotContext(width, dropout=dropout)
        self.film = FiLM1D(width, dropout=dropout)

        self.post = nn.Sequential(
            DilatedResBlock1D(width, kernel_size=3, dilation=1, dropout=dropout),
            DilatedResBlock1D(width, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.data_pool = AttnPool1D(width)
        self.cls = FusionHead(width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, N, 2)
        B, S, N, C = x.shape
        assert C == 2 and S >= 2

        x_seg = x.permute(0, 1, 3, 2).contiguous().view(B * S, C, N)
        seq, emb = self.enc(x_seg)                            # (B*S,N,d), (B*S,d)

        d = emb.size(-1)
        seq = seq.view(B, S, N, d)
        emb = emb.view(B, S, d)

        data_seq = seq[:, -1, :, :]                           # (B, N, d)
        data_emb = emb[:, -1, :]                              # (B, d)
        pilots = emb[:, :-1, :]                               # (B, P, d)

        ctx, ctx0, ctx1 = self.pilot_ctx(pilots, data_emb)
        data_seq = self.film(data_seq, ctx)
        data_seq = self.post(data_seq.transpose(1, 2)).transpose(1, 2)

        data_final = self.data_pool(data_seq)
        return self.cls(data_final, ctx)

class GRUSegmentEncoder(nn.Module):
    """
    x: (Bseg, 2, N)
    -> seq: (Bseg, N, d)
    -> emb: (Bseg, d)
    """
    def __init__(
        self,
        in_ch: int = 2,
        width: int = 64,
        rnn_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, width)
        self.gru = nn.GRU(
            input_size=width,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(2 * rnn_hidden, width)
        self.pool = AttnPool1D(width)

    def forward(self, x: torch.Tensor):
        # (Bseg, 2, N) -> (Bseg, N, 2)
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        h, _ = self.gru(h)
        h = self.out_proj(h)
        emb = self.pool(h)
        return h, emb


class IQRNNCrossPilotDetector(nn.Module):
    """
    这版不把 ctx0/ctx1 压成一个向量，而是都保留给分类头。
    """
    def __init__(
        self,
        width: int = 64,
        rnn_hidden: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.enc = GRUSegmentEncoder(
            in_ch=2,
            width=width,
            rnn_hidden=rnn_hidden,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pilot_ctx = EvenOddPilotContext(width, dropout=dropout)
        self.cross = nn.MultiheadAttention(width, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(width)
        self.data_pool = AttnPool1D(width)
        self.cls = DualContextHead(width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, N, C = x.shape
        assert C == 2 and S >= 2

        x_seg = x.permute(0, 1, 3, 2).contiguous().view(B * S, C, N)
        seq, emb = self.enc(x_seg)

        d = emb.size(-1)
        seq = seq.view(B, S, N, d)
        emb = emb.view(B, S, d)

        data_seq = seq[:, -1, :, :]          # (B, N, d)
        data_emb = emb[:, -1, :]             # (B, d)
        pilots = emb[:, :-1, :]              # (B, P, d)

        ctx, ctx0, ctx1 = self.pilot_ctx(pilots, data_emb)

        # data_seq 作为 query，对 pilot tokens 做 cross-attention
        attn_out, _ = self.cross(data_seq, pilots, pilots)
        data_seq = self.norm(data_seq + attn_out + ctx.unsqueeze(1))

        data_final = self.data_pool(data_seq)
        return self.cls(data_final, ctx0, ctx1) 

class SegmentTransformerEncoder(nn.Module):
    """
    段内 encoder：先 conv stem，再小型 transformer
    """
    def __init__(
        self,
        in_ch: int = 2,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, width))
        self.pool = AttnPool1D(width)

    def forward(self, x: torch.Tensor):
        # x: (Bseg, 2, N)
        h = self.stem(x).transpose(1, 2)     # (Bseg, N, d)
        T = h.size(1)
        if T > self.pos.size(1):
            raise ValueError(f"N={T} exceeds max_len={self.pos.size(1)}")
        h = h + self.pos[:, :T, :]
        h = self.blocks(h)
        emb = self.pool(h)
        return h, emb


class HierarchicalTransformerPilotDetector(nn.Module):
    """
    Transformer 建议用这一类：层次化，而不是直接吃整条 raw IQ。
    """
    def __init__(
        self,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.enc = SegmentTransformerEncoder(
            in_ch=2,
            width=width,
            layers=layers,
            heads=heads,
            dropout=dropout,
            max_len=max_len,
        )
        self.pilot_ctx = EvenOddPilotContext(width, dropout=dropout)
        self.seg_cross = nn.MultiheadAttention(width, num_heads=heads, batch_first=True, dropout=dropout)
        self.seg_ln = nn.LayerNorm(width)
        self.cls = FusionHead(width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, N, C = x.shape
        assert C == 2 and S >= 2

        x_seg = x.permute(0, 1, 3, 2).contiguous().view(B * S, C, N)
        seq, emb = self.enc(x_seg)

        d = emb.size(-1)
        emb = emb.view(B, S, d)

        data_emb = emb[:, -1, :]             # (B, d)
        pilots = emb[:, :-1, :]              # (B, P, d)

        ctx, ctx0, ctx1 = self.pilot_ctx(pilots, data_emb)

        # data token 对 pilot tokens 做 cross-attention
        data_tok = data_emb.unsqueeze(1)     # (B,1,d)
        seg_attn, _ = self.seg_cross(data_tok, pilots, pilots)
        data_emb = self.seg_ln(data_tok + seg_attn).squeeze(1)

        return self.cls(data_emb, ctx)




# model_registry.register_model("CNN_150", TCNFiLMPilotDetector)
# model_registry.register_model("CNN_150", IQRNNCrossPilotDetector)
# model_registry.register_model("CNN_150", HierarchicalTransformerPilotDetector)
model_registry.register_model("CNN_150", FastPilotDetector_EvenOdd)
# model_registry.register_model("CNN_150", PureCNN2D)
# model_registry.register_model("CNN_150", PureCNN2D_combine)
# from thop import profile, clever_format
# if __name__ == "__main__":
#     # 99KB
#     model = FastPilotDetector_EvenOdd()
#     # model.load_state_dict(torch.load('/root/signal/block/150/weight/CNN_150pure_best_model1_1010_32d10.pt'))
#     # model1.eval()
    
#     total_params2 = sum(p.numel() for p in model.parameters())
#     trainable_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f"总参数数量: {total_params2:,}")
#     print(f"可训练参数数量: {trainable_params2:,}")


import torch
from thop import profile, clever_format

if __name__ == "__main__":
    model = PureCNN2D()
    model.eval()

    # 按你的 forward: x.shape = (B, S, N, 2)
    dummy = torch.randn(1, 48, 150, 2)   # 这里把 S,N 改成你的实际输入尺寸

    macs, params = profile(model, inputs=(dummy,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    print(f"MACs: {macs}")
    print(f"Params: {params}")