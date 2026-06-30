"""Slim time-frequency path: CWT scalogram + compact 2D CNN (no period folding)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ChannelAttentionPool, ConvBnRelu2D, SEBlock1D
from .tf_utils import cwt_scalogram


class _Inception2D(nn.Module):
    """Multi-kernel 2D convs on a single scalogram (multiscale in TF space)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        branch = max(out_ch // 3, 8)
        self.fine = ConvBnRelu2D(in_ch, branch, kernel_size=(3, 3))
        self.medium = ConvBnRelu2D(in_ch, branch, kernel_size=(5, 3))
        self.wide = ConvBnRelu2D(in_ch, branch, kernel_size=(3, 3), dilation=(2, 2))
        self.out_ch = branch * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.fine(x), self.medium(x), self.wide(x)], dim=1)


class TimeFrequencyPath(nn.Module):
    """
    Period-free spectral path for MSPF-Net slim.

    Addresses limitations of:
    - pure 1D CNNs (explicit time-frequency structure),
    - period-aware folding (fixed TF layout instead of estimated period stacks),
  while staying lighter than full Transformers.
    """

    def __init__(
        self,
        stem_channels: int,
        branch_channels: int,
        feat_dim: int,
        se_reduction: int,
        cwt_num_bands: int,
        cwt_kernel_size: int,
        scalogram_freq_bins: int,
        scalogram_time_bins: int,
        channel_pooling: str,
        dropout: float,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.stem_channels = int(stem_channels)
        self.branch_channels = int(branch_channels)
        self.feat_dim = int(feat_dim)
        self.cwt_num_bands = int(cwt_num_bands)
        self.cwt_kernel_size = int(cwt_kernel_size)
        self.freq_bins = max(8, int(scalogram_freq_bins))
        self.time_bins = max(16, int(scalogram_time_bins))
        self.channel_pooling = str(channel_pooling).lower()
        self.dropout_p = float(dropout)

        self.stem_proj = nn.Conv1d(self.stem_channels, 1, kernel_size=1, bias=False)
        self.tf_encoder = _Inception2D(in_ch=self.cwt_num_bands, out_ch=self.branch_channels)
        enc_out = self.tf_encoder.out_ch
        self.tf_head = nn.Sequential(
            ConvBnRelu2D(enc_out, self.branch_channels, kernel_size=(3, 3)),
            nn.AdaptiveAvgPool2d(1),
        )
        self.se = SEBlock1D(self.branch_channels, reduction=se_reduction) if use_se else nn.Identity()
        self.proj = (
            nn.Identity()
            if self.branch_channels == self.feat_dim
            else nn.Linear(self.branch_channels, self.feat_dim)
        )
        self.channel_pool = (
            ChannelAttentionPool(self.feat_dim, reduction=se_reduction)
            if self.channel_pooling == "attention"
            else None
        )
        self.dropout = nn.Dropout(p=self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()

    def _resize_scalogram(self, x: torch.Tensor) -> torch.Tensor:
        """Resize (B, n_bands, L) to 4D (B, n_bands, freq_bins, time_bins)."""
        if x.ndim != 3:
            raise ValueError(f"Expected scalogram (B, bands, L), got {tuple(x.shape)}")
        x4d = x.unsqueeze(2)
        if x4d.shape[-2:] == (self.freq_bins, self.time_bins):
            return x4d
        if x.device.type == "mps":
            return F.interpolate(
                x4d,
                size=(self.freq_bins, self.time_bins),
                mode="bilinear",
                align_corners=False,
            )
        return F.adaptive_avg_pool2d(x4d, (self.freq_bins, self.time_bins))

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
        x2d = self._resize_scalogram(scalogram)
        x2d = self.tf_encoder(x2d)
        x2d = self.tf_head(x2d).flatten(1)
        x2d = self.se(x2d.unsqueeze(-1)).squeeze(-1)
        pooled = self.proj(x2d)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        pooled = self._aggregate_channels(pooled, batch_size=batch_size, channels=channels)
        pooled = self.dropout(pooled)
        return pooled, float(confidence)
