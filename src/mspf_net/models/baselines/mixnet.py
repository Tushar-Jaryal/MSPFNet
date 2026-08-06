from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ClassifierHead, SqueezeExcite1D, ensure_bcl


class MixConv1D(nn.Module):
    """
    Split channels across multiple depthwise kernel sizes, matching the core
    mixed-kernel idea behind MixNet blocks.
    """

    def __init__(self, channels: int, kernels: tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        groups = len(kernels)
        splits = [channels // groups] * groups
        for i in range(channels % groups):
            splits[i] += 1
        self.splits = splits
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(split, split, kernel_size=k, padding=k // 2, groups=split, bias=False)
                for split, k in zip(splits, kernels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.splits, dim=1)
        out = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
        return torch.cat(out, dim=1)


class MixNetBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expansion: int = 4,
        kernels: tuple[int, ...] = (3, 5, 7),
        se_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = in_ch * expansion
        self.use_residual = in_ch == out_ch

        self.expand = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
        )
        self.mixconv = MixConv1D(hidden, kernels=kernels)
        self.mix_bn = nn.BatchNorm1d(hidden)
        self.mix_act = nn.SiLU(inplace=True)
        self.se = SqueezeExcite1D(hidden, ratio=se_ratio)
        self.project = nn.Sequential(
            nn.Conv1d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.expand(x)
        x = self.mix_act(self.mix_bn(self.mixconv(x)))
        x = self.se(x)
        x = self.project(x)
        x = self.drop(x)
        if self.use_residual:
            x = x + residual
        return x


class MixNet(BaseBaselineModel):
    model_name = "mixnet"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.1,
        expansion: int = 4,
    ):
        super().__init__()
        stage_widths = [width, width, width * 2, width * 2][: max(n_blocks, 1)]
        stem_out = stage_widths[0]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_out, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(stem_out),
            nn.SiLU(inplace=True),
        )

        blocks = []
        in_ch = stem_out
        kernel_schedule = [
            (3, 5),
            (3, 5, 7),
            (3, 5, 7),
            (5, 7, 9),
        ]
        for i in range(n_blocks):
            out_ch = stage_widths[min(i, len(stage_widths) - 1)]
            kernels = kernel_schedule[min(i, len(kernel_schedule) - 1)]
            blocks.append(
                MixNetBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    expansion=expansion,
                    kernels=kernels,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = ClassifierHead(in_ch, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
