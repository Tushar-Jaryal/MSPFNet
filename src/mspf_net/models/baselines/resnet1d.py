from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ClassifierHead, ConvBNAct, ensure_bcl


class ResNetBasicBlock1D(nn.Module):
    """
    Standard ResNet-18 basic block adapted for 1-D signals.

    Two Conv-BN-ReLU layers (k=7) with an identity shortcut — matches the
    plan description: 'ResNet-18 adapted to 1D, k=7 convolutions, skip
    connections. ~1.8M params'.
    """

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)          # identity skip connection


class ResNet1D(BaseBaselineModel):
    """
    ResNet-18 style 1-D backbone.

    Default (width=128, n_blocks=8): 8 basic blocks × 2 × 128×128×7
    ≈ 1.84M parameters, matching the architecture plan target of ~1.8M.
    """

    model_name = "resnet1d"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 128,
        n_blocks: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = ConvBNAct(in_channels, width, kernel_size=15, act="relu")
        self.blocks = nn.Sequential(
            *[ResNetBasicBlock1D(width, kernel_size=7) for _ in range(n_blocks)]
        )
        self.head = ClassifierHead(width, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
