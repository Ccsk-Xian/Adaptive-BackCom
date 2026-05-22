import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_registry


class DepthwiseSeparableConv(nn.Module):
    """
    2D深度可分离卷积：
      - depthwise 卷积（groups=in_ch）
      - pointwise 卷积（1x1卷积）
    使用 padding="same" 自动保证输出尺寸与输入一致（步长固定为1）。
    """

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(DepthwiseSeparableConv, self).__init__()
        # 保证 kernel_size 为 tuple 格式
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1)
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=self.stride,
            padding="same",
            groups=in_ch,
            bias=True,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ResidualBlock2D(nn.Module):
    """
    残差块：
      结构： x -> DSConv -> ReLU -> Dropout -> DSConv -> ReLU -> Dropout -> + x -> ReLU
    其中 DSConv 均使用 DepthwiseSeparableConv，步长固定为1，且均采用 padding="same"。
    """

    def __init__(self, channels, dropout_rate=0.0, kernel_size=(2, 3)):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size)
        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = F.relu(out, inplace=False)
        out = self.dropout(out)
        out = self.conv2(out)
        out = F.relu(out, inplace=False)
        out = self.dropout(out)
        out = out + identity
        out = F.relu(out, inplace=False)
        return out


class MoE(nn.Module):
    """
    Expert1 多分类模型：
      - 通过参数可配置：
          * 初始卷积核大小（kernel_size）
          * 隐藏层大小（hidden_channels）
          * 残差块数量（num_residual_blocks）
      - 输入形状 (batch_size, seq_len, 2)，在 forward 中调整为 (batch_size, 1, 2, seq_len)
      - 卷积、残差块后展平，再接全连接层输出类别数
      - 在初始化时，通过一次前向传播动态计算全连接层的输入维度
    """

    def __init__(self):
        super(MoE, self).__init__()
        # 从配置中读取参数
        kernel_size = (2,7)
        num_residual_blocks = 4
        hidden_channels = 64
        dropout_rate = 0.0
        seq_len = 75
        self.num_classes = 2

        self.initial_conv = DepthwiseSeparableConv(
            in_ch=1, out_ch=hidden_channels, kernel_size=kernel_size
        )

        # 使用指定数量的残差块构建特征提取模块
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock2D(
                    hidden_channels, dropout_rate=dropout_rate, kernel_size=kernel_size
                )
                for _ in range(num_residual_blocks)
            ]
        )

        # Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)

        # 动态计算全连接层输入的特征维度
        dummy_input = torch.zeros(1, 1, 2, seq_len)  # 输入形状 (1, 1, 2, seq_len)
        dummy_out = self.initial_conv(dummy_input)
        dummy_out = F.relu(dummy_out, inplace=False)
        dummy_out = self.residual_blocks(dummy_out)
        flattened_size = dummy_out.view(1, -1).shape[1]
        self.fc = nn.Linear(flattened_size, self.num_classes)

    def forward(self, x):
        # 输入 x: (batch_size, seq_len, 2) -> 调整为 (batch_size, 1, 2, seq_len)
        # print(x.shape)
        x = x.permute(0, 2, 1).unsqueeze(1)
        out = self.initial_conv(x)
        out = F.relu(out, inplace=False)
        out = self.residual_blocks(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


model_registry.register_model("MoE", MoE)
