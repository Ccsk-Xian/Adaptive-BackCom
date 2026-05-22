# models/MCFormer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_registry


class MCFormer(nn.Module):
    def __init__(self):
        super(MCFormer, self).__init__()
        self.frame_length = 75
        self.num_classes = 2
        self.fea_dim = 32

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=self.fea_dim, kernel_size=27, padding="same"
            ),
            nn.SELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fea_dim,
            nhead=4,
            dim_feedforward=self.fea_dim,
            batch_first=True,
        )
        self.tnn = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(4 * self.fea_dim, 128),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, 2, T]
        x = self.cnn(x)  # [B, fea_dim, T]

        x = torch.squeeze(x, dim=2)  # [B, fea_dim]

        x = torch.transpose(x, 1, 2)  # [B, 1, fea_dim]

        # Transformer Encoder -> [B, 1, fea_dim]
        x = self.tnn(x)

        # 取前 4 个时刻 -> [B, 4, fea_dim] -> reshape
        x = x[:, :4, :]
        x = torch.reshape(x, [-1, 4 * self.fea_dim])

        x = self.classifier(x)

        return x


# 注册模型
model_registry.register_model("MCFormer", MCFormer)
