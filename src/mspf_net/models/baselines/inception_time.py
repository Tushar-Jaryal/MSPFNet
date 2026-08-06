from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ClassifierHead, ConvBNAct, ensure_bcl


class InceptionModule1D(nn.Module):
    """Inception module (thesis Sec. 3.6): bottleneck reduce, parallel odd-kernel branches, residual skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Bottleneck: reduce input before the three conv branches
        self.reduce = ConvBNAct(in_ch, out_ch, 1)

        # Parallel conv branches (k=3, 9, 19 as in thesis §3.6)
        self.branch_k3  = ConvBNAct(out_ch, out_ch, 3)
        self.branch_k9  = ConvBNAct(out_ch, out_ch, 9)
        self.branch_k19 = ConvBNAct(out_ch, out_ch, 19)

        # Fuse three branches → out_ch
        self.fuse = ConvBNAct(out_ch * 3, out_ch, 1)

        # Residual projection when channel dims differ
        self.residual = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branches = [
            self.branch_k3(reduced),
            self.branch_k9(reduced),
            self.branch_k19(reduced),
        ]
        # All odd kernels → identical output lengths; no truncation needed
        y = self.fuse(torch.cat(branches, dim=1))
        return self.act(self.bn(y + self.residual(x)))


class InceptionTime(BaseBaselineModel):
    model_name = "inception_time"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 96,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = [InceptionModule1D(in_channels, width)]
        blocks.extend(InceptionModule1D(width, width) for _ in range(n_blocks - 1))
        self.backbone = nn.Sequential(*blocks)
        self.head = ClassifierHead(width, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        x = self.backbone(x)
        return self.head(x)
