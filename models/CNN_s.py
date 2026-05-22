
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


class ConvFrameNet_s(nn.Module):
    def __init__(self, input_len=75):
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
            nn.Linear(64, 7)
        )

    def forward(self, x):  # x: [B, 2, N]
        x = x.permute(0,2,1)
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)
    
model_registry.register_model("CNN_s", ConvFrameNet_s)