from __future__ import annotations

import math
from abc import ABC

import torch
import torch.nn as nn


class BaseBaselineModel(nn.Module, ABC):
    """
    Small shared base used by baseline models.

    It gives us a common place for lightweight metadata and parameter reporting
    without changing the training loop contract.
    """

    model_name: str = "baseline"

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        act: str = "gelu",
    ):
        padding = kernel_size // 2
        activation: nn.Module
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "silu":
            activation = nn.SiLU(inplace=True)
        else:
            activation = nn.GELU()
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            activation,
        )


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(channels // ratio, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, hidden, kernel_size, groups=hidden, act="silu"),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ClassifierHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttentionPool(nn.Module):
    """
    Pool the sequence before self-attention so memory stays practical on 2048 windows.
    """

    def __init__(self, channels: int, n_heads: int = 4, attn_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.attn_len = attn_len
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        pooled = nn.functional.adaptive_avg_pool1d(x, self.attn_len)
        pooled = pooled.transpose(1, 2).contiguous()  # (B, T, C)
        attn_out, _ = self.attn(pooled, pooled, pooled, need_weights=False)
        return self.norm(pooled + attn_out).transpose(1, 2).contiguous()


class InceptionBranch1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: tuple[int, ...]):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNAct(in_ch, out_ch, k) for k in kernels])
        self.fuse = ConvBNAct(out_ch * len(kernels), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


def ensure_bcl(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x
