# models/GRU2.py
import torch
import torch.nn as nn
from models import model_registry


class GRU2(nn.Module):

    def __init__(self):
        super(GRU2, self).__init__()
        # self.num_classes = config["num_classes"]
        self.num_classes = 2
        # self.dropout = config.get("dropout", 0.0)
        self.dropout = 0.0
        self.depth = 1
        self.hidden_size = 64
        self.features = nn.GRU(
            input_size=2,
            hidden_size=self.hidden_size,
            num_layers=self.depth,
            batch_first=True,
            dropout=self.dropout,
        )

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        for name, param in self.features.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        x, _ = self.features(x)
        x = self.classifier(x[:, -1, :])

        return x


model_registry.register_model("GRU2", GRU2)
