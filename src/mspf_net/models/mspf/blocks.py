"""Shared 1D/2D blocks and attention helpers for MSPF-Net."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBnRelu1D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, stride: int = 1) -> None:
        padding = ((kernel_size - 1) // 2) * dilation
        super().__init__(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class ConvBnRelu2D(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ) -> None:
        padding = tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Stage1ChannelExtractor(nn.Sequential):
    def __init__(self, out_channels: int = 64, kernel_size: int = 7) -> None:
        super().__init__(
            nn.Conv1d(1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class Spectral1DBranch(nn.Sequential):
    def __init__(self, in_ch: int = 64, out_ch: int = 128, kernel_size: int = 7) -> None:
        super().__init__(
            ConvBnRelu1D(in_ch, out_ch, kernel_size=kernel_size),
            ConvBnRelu1D(out_ch, out_ch, kernel_size=kernel_size),
        )


class EarlyChannelMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.enabled = int(channels) > 1
        if self.enabled:
            self.mix = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.mix = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return x + self.mix(x)


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x).squeeze(-1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class ChannelAttentionPool(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.fc(x.reshape(-1, x.size(-1))).reshape(x.size(0), x.size(1), 1)
        weights = scores.softmax(dim=1)
        return (x * weights).sum(dim=1)


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)
