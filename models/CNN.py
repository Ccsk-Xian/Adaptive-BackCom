
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_registry

# class ConvFrameNet(nn.Module):
#     def __init__(self, input_len=75):
#         super().__init__()
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # ↓长度减半
#         )
#         # 计算池化后的长度（input_len // 4）
#         self.fc = nn.Sequential(
#             nn.Linear(16 , 32),
            
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 2)
#         )

#     def forward(self, x):  # x: [B, 2, N]
#         x = x.permute(0,2,1)
#         print(x.shape)
#         x = self.conv_net(x)
#         x = x.view(x.size(0), -1)
#         x = x.view(x.size(0), -1)  # 展平
#         return self.fc(x)


# class ConvFrameNet(nn.Module):
#     def __init__(self, input_dim=75):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),

#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 2)
#         )

#     def forward(self, x): return self.net(x)


# class ConvFrameNet(nn.Module):
#     def __init__(self, input_len=75):
#         super().__init__()
#         self.conv_net = nn.Sequential(
#             nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),  # ↓长度减半
#             nn.Conv1d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),  # 再减半
#         )
#         # 计算池化后的长度（input_len // 4）
#         self.fc = nn.Sequential(
#             nn.Linear(32 * (input_len // 4), 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)
#         )

#     def forward(self, x):  # x: [B, 2, N]
#         x = x.permute(0,2,1)
#         x = self.conv_net(x)
#         x = x.view(x.size(0), -1)  # 展平
#         return self.fc(x)

class ConvFrameNet(nn.Module):
    def __init__(self, input_len=70):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=26, padding=13),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # ↓长度减半
            nn.Conv1d(16, 32, kernel_size=10, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 再减半
        )
        # 计算池化后的长度（input_len // 4）
        self.fc = nn.Sequential(
            nn.Linear(32 * ((input_len+3) // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):  # x: [B, 2, N]
        x = x.permute(0,2,1)
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x: [B, C, L]
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


class Res1DBackscatterNet(nn.Module):
    def __init__(self, input_len=70, base_channels=32, num_classes=2):
        super().__init__()
        self.input_len = input_len

        # 输入：2 个通道（实部、虚部）
        self.stem = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )

        # 一堆残差块，可以适当加 dilation 扩大感受野
        self.layer1 = ResidualBlock1D(base_channels, kernel_size=5, dilation=1)
        self.layer2 = ResidualBlock1D(base_channels, kernel_size=5, dilation=2)
        self.layer3 = ResidualBlock1D(base_channels, kernel_size=5, dilation=4)

        # 可选：再加一点下采样，提取更粗糙特征
        self.down = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU()
        )
        self.layer4 = ResidualBlock1D(base_channels * 2, kernel_size=3, dilation=1)

        # 全局平均池化，避免依赖具体 N
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # -> [B, C, 1]

        self.fc = nn.Sequential(
            nn.Linear(base_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 2, N]  实部/虚部为两个通道
        """
        # 不要 permute，按照 Conv1d 的格式 [B, C, L] 来组织
        # x = x.permute(0,2,1)
        B, S, N, _ = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, 2, S * N)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.down(out)
        out = self.layer4(out)

        # 全局平均池化
        out = self.global_pool(out)  # [B, C, 1]
        out = out.squeeze(-1)       # [B, C]

        logits = self.fc(out)       # [B, num_classes]
        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DTransformer(nn.Module):
    """
    输入:  x (B, S, N, 2)
           - S: 4 个导频 + 1 个数据符号，作为通道
           - (N,2) 展成长度 2N 的一维序列

    流程:
      - reshape 到 (B, C=S, L=2N)
      - 1D CNN 提取时域特征
      - Transformer 在时间维 (L') 上建模
      - 池化后做二分类
    """

    def __init__(self,
                 S: int=5,
                 N: int=70,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.S = S
        self.N = N
        self.L_in = 2 * N

        # 1D CNN：输入 (B, C=S, L=2N)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=S, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # Transformer 在时间维上操作：
        #   输入: (B, L', d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 分类头
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: (B, S, N, 2)
        """
        B, S, N, C = x.shape
        assert S == self.S and N == self.N and C == 2

        # (B, S, N, 2) → (B, S, 2N)
        x = x.reshape(B, S, 2 * N)

        # Conv1d 期望 (B, C, L)
        x = x  # (B, C=S, L=2N)
        x = self.cnn(x)          # (B, d_model, L')

        # Transformer 要的是 (B, L', d_model)
        x = x.permute(0, 2, 1)   # (B, L', d_model)

        x = self.transformer(x)  # (B, L', d_model)

        # 时间维上池化（也可以取最后若干位置）
        x = x.mean(dim=1)        # (B, d_model)

        logits = self.cls(x)     # (B, num_classes)
        return logits
    

class CNN2DTransformer(nn.Module):
    """
    输入:  x (B, S, N, 2)
           - S 作为通道
           - (N,2) 变成 2×N 的“图像”
    流程:
      - reshape → (B, C=S, H=2, W=N)
      - 2D CNN 提特征
      - 展平成 (B, L', d_model)，L' = H'*W'
      - Transformer 在 patch 序列上建模
      - 池化分类
    """

    def __init__(self,
                 S: int=5,
                 N: int=70,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.S = S
        self.N = N

        # 2D CNN: 输入 (B, C=S, H=2, W=N)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=S, out_channels=64, kernel_size=(2, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )

        # 注意：CNN 输出形状为 (B, d_model, H', W')
        # 我们将 H'×W' 展成 L'，作为 Transformer 的序列长度

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: (B, S, N, 2)
        """
        B, S, N, C = x.shape
        assert S == self.S and N == self.N and C == 2

        # (B, S, N, 2) → (B, S, 2, N)
        x = x.permute(0, 1, 3, 2)      # (B, S, 2, N)

        # (B, C=S, H=2, W=N)
        x = self.cnn(x)                # (B, d_model, H', W')

        B, d_model, H, W = x.shape
        # 展成序列 (B, L', d_model)
        x = x.view(B, d_model, H * W).permute(0, 2, 1)  # (B, L', d_model)

        x = self.transformer(x)        # (B, L', d_model)

        # 对所有 patch 池化（也可以只取某些位置）
        x = x.mean(dim=1)              # (B, d_model)

        logits = self.cls(x)           # (B, num_classes)
        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN1D(nn.Module):
    """
    TinyML 友好的 1D CNN，用于二分类
    输入:  x (B, S, N, 2)
    输出:  logits (B, 2)
    """

    def __init__(self, S: int=1, N: int=150, num_classes: int = 2):
        super().__init__()
        self.S = S
        self.N = N
        L_in = 2 * N

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=S, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)   # (B, 32, L) -> (B, 32, 1)

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, S, N, 2)
        B, S, N, C = x.shape
        assert S == self.S and N == self.N and C == 2

        # (B, S, N, 2) -> (B, S, 2N)
        x = x.reshape(B, S, 2 * N)          # 把 N×IQ 合并为时间维

        # Conv1d: (B, C=S, L=2N)
        x = self.conv_block(x)              # (B, 32, L')

        x = self.gap(x).squeeze(-1)         # (B, 32)

        logits = self.fc(x)                 # (B, num_classes)
        return logits
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN2D(nn.Module):
    """
    TinyML 友好的 2D CNN，用于二分类
    输入:
        x: (B, S, N, 2)
           - S: 符号数（导频 + 数据符号），作为 channel
           - (N,2) 被视为 H=2, W=N 的小图像

    输出:
        logits (B, 2)
    """

    def __init__(self, S: int=5, N: int=70, num_classes: int = 2):
        super().__init__()
        self.S = S
        self.N = N

        # Conv2d 输入形状： (B, C=S, H=2, W=N)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=S, out_channels=16,
                      kernel_size=(2, 5), padding=(0, 2)),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
        )

        # Global Average Pooling: (B, 32, H', W') → (B, 32)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, S, N, 2)
        B, S, N, C = x.shape
        assert S == self.S and N == self.N and C == 2

        # (B, S, N, 2) → (B, S, 2, N)
        x = x.permute(0, 1, 3, 2)    # 把 2 放到 H=2，把 N 放到 W=N

        # Conv2d: (B, C=S, H=2, W=N)
        x = self.conv_block(x)

        # GAP: (B, 32, 1, 1) → (B, 32)
        x = self.gap(x).view(B, -1)

        logits = self.fc(x)   # (B, num_classes)
        return logits
model_registry.register_model("CNN", TinyCNN1D)