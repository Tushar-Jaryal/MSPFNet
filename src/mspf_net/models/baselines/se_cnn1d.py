from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ClassifierHead, ConvBNAct, SqueezeExcite1D, ensure_bcl


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size, act="silu"),
            ConvBNAct(channels, channels, kernel_size, act="silu"),
        )
        self.se = SqueezeExcite1D(channels, ratio=4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return x + self.drop(out)


class SECNN1D(BaseBaselineModel):
    """
    Compact SE-enhanced convolutional baseline.

    Intended as a stronger channel-attention baseline than plain CNN stacks
    without changing the evaluation contract.
    """

    model_name = "se_cnn1d"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 96,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = ConvBNAct(in_channels, width, kernel_size=15, act="silu")
        self.blocks = nn.Sequential(*[SEBlock1D(width, kernel_size=7, dropout=dropout) for _ in range(n_blocks)])
        self.head = ClassifierHead(width, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
