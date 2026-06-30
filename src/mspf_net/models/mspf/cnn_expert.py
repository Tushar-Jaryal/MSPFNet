"""Non-periodic and CNN experts for MSPF-Net MoE routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ChannelAttentionPool, ConvBnRelu1D, Spectral1DBranch, TemporalAttentionPool
from .tf_utils import cwt_scalogram


class CnnExpert(nn.Module):
    """Compact 1D CNN with temporal attention pooling, then channel pooling."""

    def __init__(
        self,
        stem_channels: int,
        feat_dim: int,
        se_reduction: int = 16,
        channel_pooling: str = "attention",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem_channels = int(stem_channels)
        self.feat_dim = int(feat_dim)
        self.channel_pooling = str(channel_pooling).lower()

        mid = max(self.stem_channels, self.feat_dim // 4)
        self.trunk = nn.Sequential(
            ConvBnRelu1D(self.stem_channels, mid, kernel_size=7),
            ConvBnRelu1D(mid, mid, kernel_size=5, dilation=2),
            ConvBnRelu1D(mid, self.feat_dim, kernel_size=3, dilation=4),
        )
        self.temporal_pool = TemporalAttentionPool(self.feat_dim, reduction=se_reduction)
        self.channel_pool = (
            ChannelAttentionPool(self.feat_dim, reduction=se_reduction)
            if self.channel_pooling == "attention"
            else None
        )
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def _aggregate_channels(self, pooled: torch.Tensor, batch_size: int, channels: int) -> torch.Tensor:
        pooled = pooled.reshape(batch_size, channels, -1)
        if pooled.shape[1] == 1:
            return pooled.squeeze(1)
        if self.channel_pool is None:
            return pooled.mean(dim=1)
        return self.channel_pool(pooled)

    def forward(
        self,
        feat_bc: torch.Tensor,
        batch_size: int,
        channels: int,
    ) -> tuple[torch.Tensor, float]:
        x = self.trunk(feat_bc)
        pooled = self.temporal_pool(x)
        pooled = self._aggregate_channels(pooled, batch_size=batch_size, channels=channels)
        pooled = self.dropout(pooled)
        return pooled, 1.0


class SpectralExpert(nn.Module):
    """1D conv on CWT bands (spectral path without period folding or 2D scalogram CNN)."""

    def __init__(
        self,
        stem_channels: int,
        branch_channels: int,
        feat_dim: int,
        cwt_num_bands: int,
        cwt_kernel_size: int,
        se_reduction: int = 16,
        channel_pooling: str = "attention",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cwt_num_bands = int(cwt_num_bands)
        self.cwt_kernel_size = int(cwt_kernel_size)
        self.feat_dim = int(feat_dim)
        self.channel_pooling = str(channel_pooling).lower()
        self.stem_proj = nn.Conv1d(int(stem_channels), 1, kernel_size=1, bias=False)
        self.spectral = Spectral1DBranch(self.cwt_num_bands, int(branch_channels))
        self.temporal_pool = TemporalAttentionPool(int(branch_channels), reduction=se_reduction)
        self.proj = (
            nn.Identity()
            if int(branch_channels) == self.feat_dim
            else nn.Linear(int(branch_channels), self.feat_dim)
        )
        self.channel_pool = (
            ChannelAttentionPool(self.feat_dim, reduction=se_reduction)
            if self.channel_pooling == "attention"
            else None
        )
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def _aggregate_channels(self, pooled: torch.Tensor, batch_size: int, channels: int) -> torch.Tensor:
        pooled = pooled.reshape(batch_size, channels, -1)
        if pooled.shape[1] == 1:
            return pooled.squeeze(1)
        if self.channel_pool is None:
            return pooled.mean(dim=1)
        return self.channel_pool(pooled)

    def forward(
        self,
        feat_bc: torch.Tensor,
        batch_size: int,
        channels: int,
    ) -> tuple[torch.Tensor, float]:
        x = self.stem_proj(feat_bc)
        scalogram, _, confidence = cwt_scalogram(
            x,
            num_bands=self.cwt_num_bands,
            cwt_kernel_size=self.cwt_kernel_size,
        )
        x1d = self.spectral(scalogram)
        pooled = self.temporal_pool(x1d)
        pooled = self.proj(pooled)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        pooled = self._aggregate_channels(pooled, batch_size=batch_size, channels=channels)
        pooled = self.dropout(pooled)
        return pooled, float(confidence)

