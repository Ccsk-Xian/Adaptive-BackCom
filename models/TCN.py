import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_registry

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=(kernel_size - 1) * dilation,
                      dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)
    
class TCNClassifier(nn.Module):
    def __init__(self, input_size=2, seq_len=75, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(input_size, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 64, kernel_size=3, dilation=2),
            TCNBlock(64, 64, kernel_size=3, dilation=4),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: [B, T, 2]
        x = x.permute(0, 2, 1)  # → [B, 2, T]
        x = self.net(x)         # → [B, C, T]
        return self.fc(x)
    
model_registry.register_model("TCN", TCNClassifier)