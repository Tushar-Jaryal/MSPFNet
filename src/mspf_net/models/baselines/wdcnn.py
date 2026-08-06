from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ClassifierHead, ConvBNAct, ensure_bcl


class WDCNN(BaseBaselineModel):
    model_name = "wdcnn"

    def __init__(self, in_channels: int, num_classes: int, width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, width, 64, stride=8, act="relu"),
            ConvBNAct(width, width * 2, 16, stride=2, act="relu"),
            ConvBNAct(width * 2, width * 4, 8, stride=2, act="relu"),
            ConvBNAct(width * 4, width * 4, 4, stride=2, act="relu"),
            ConvBNAct(width * 4, width * 8, 3, stride=2, act="relu"),
        )
        self.head = ClassifierHead(width * 8, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        return self.head(self.features(x))
