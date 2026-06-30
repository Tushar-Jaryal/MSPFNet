"""Nonstationary path: RDC convs, downsampled BiGRU, temporal attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ChannelAttentionPool, ConvBnRelu1D, TemporalAttentionPool


class NonstationaryPath(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        feat_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        downsample_stride: int = 4,
        se_reduction: int = 16,
        channel_pooling: str = "attention",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem_channels = int(stem_channels)
        self.feat_dim = int(feat_dim)
        self.hidden_size = int(hidden_size)
        self.downsample_stride = max(1, int(downsample_stride))
        mid = max(self.stem_channels, self.hidden_size // 2)
        self.rdc = nn.Sequential(
            ConvBnRelu1D(self.stem_channels, mid, kernel_size=7),
            ConvBnRelu1D(mid, self.stem_channels, kernel_size=5),
            ConvBnRelu1D(
                self.stem_channels,
                self.stem_channels,
                kernel_size=5,
                stride=self.downsample_stride,
            ),
        )
        self.bigru = nn.GRU(
            input_size=self.stem_channels,
            hidden_size=self.hidden_size,
            num_layers=max(1, int(num_layers)),
            batch_first=True,
            bidirectional=True,
        )
        gru_out = self.hidden_size * 2
        self.temporal_pool = TemporalAttentionPool(gru_out, reduction=se_reduction)
        self.proj = nn.Linear(gru_out, self.feat_dim)
        self.channel_pooling = str(channel_pooling).lower()
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
    ) -> torch.Tensor:
        seq = self.rdc(feat_bc)
        seq = seq.transpose(1, 2)
        gru_out, _ = self.bigru(seq)
        gru_out = gru_out.transpose(1, 2)
        pooled = self.temporal_pool(gru_out)
        pooled = self.proj(pooled)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        pooled = self._aggregate_channels(pooled, batch_size=batch_size, channels=channels)
        return self.dropout(pooled)
