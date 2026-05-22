# models/LSTM2.py
import torch
import torch.nn as nn
from models import model_registry


class LSTM2(nn.Module):

    def __init__(self):
        super(LSTM2, self).__init__()
        self.num_classes = 2
        self.dropout = 0.0
        self.depth = 2
        self.hidden_size = 128
        self.features = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_size,
            num_layers=self.depth,
            batch_first=True,
            dropout=self.dropout,
        )
        
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x, _ = self.features(x)
        x = self.classifier(x[:, -1, :])

        return x


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.num_classes = 2
        self.hidden_size = 128

        
        # 1. 先用卷积提取M个序列间的空间关系
        # self.conv = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size=(3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2)),
        #     nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((M // 2, seq_len))  # 降维
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
        )
        
        # 2. LSTM处理时序
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.0
        )
        
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
    
    def forward(self, x):
        # x.shape: (batch_size, M, seq_len, 2)
        
        # 1. 调整维度以适应Conv2d: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)  # (batch, 2, M, seq_len)
        
        # 2. 卷积处理
        x = self.conv(x)  # (batch, 32, M//2, seq_len)
        
        # 3. 准备LSTM输入
        # 将通道维度转为特征维度
        x = x.permute(0, 2, 3, 1)  # (batch, M, seq_len, 32)
        batch_size, M2, seq_len, channels = x.shape
        
        # 4. 处理每个降维后的序列
        # 方法1: 合并所有序列
        x = x.reshape(batch_size, -1, channels)  # (batch, M2*seq_len, 32)
        
        # 方法2: 分别处理每个序列（可选）
        # 这里使用方法1
        
        # 5. LSTM处理
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步
        
        # 6. 分类
        x = self.classifier(x)
        return x

model_registry.register_model("LSTM2", ConvLSTM)
