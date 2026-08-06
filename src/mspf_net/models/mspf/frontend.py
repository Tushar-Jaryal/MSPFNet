"""Shared front-end: sensor fusion, FIR bank, and wavelet residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBnRelu1D, EarlyChannelMixer, Stage1ChannelExtractor


class SensorFusionStem(nn.Module):
    """Multichannel fusion before per-channel stem extraction."""

    def __init__(self, channels: int, mode: str = "residual_attention") -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).lower()
        if self.channels <= 1:
            self.enabled = False
            self.mix = nn.Identity()
            self.gate = None
            return
        self.enabled = True
        if self.mode == "simple_mixer":
            self.mix = EarlyChannelMixer(self.channels)
            self.gate = None
        else:
            self.mix = nn.Sequential(
                nn.Conv1d(self.channels, self.channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(self.channels),
                nn.ReLU(inplace=True),
            )
            hidden = max(self.channels // 4, 4)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(self.channels, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.channels, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.mode == "simple_mixer":
            return self.mix(x)
        mixed = x + self.mix(x)
        if self.gate is None:
            return mixed
        scale = self.gate(x).unsqueeze(-1)
        return mixed * scale + x * (1.0 - scale)


class FIRFeatureBank(nn.Module):
    """Learnable band-pass FIR filters per input channel."""

    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        kernel_size: int = 31,
        init_mode: str = "delta",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.kernel_size = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1
        self.out_channels = self.channels * self.num_bands
        weight = torch.zeros(self.channels, self.num_bands, 1, self.kernel_size)
        center = self.kernel_size // 2
        if str(init_mode).lower() == "delta":
            weight[:, :, 0, center] = 1.0
        else:
            nn.init.kaiming_normal_(weight, nonlinearity="relu")
        self.filters = nn.Parameter(weight)
        self.norm = nn.BatchNorm1d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, C, L), got {tuple(x.shape)}")
        b, c, _length = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        weight = self.filters.reshape(self.channels * self.num_bands, 1, self.kernel_size)
        out = F.conv1d(x, weight, padding=self.kernel_size // 2, groups=self.channels)
        out = self.norm(out)
        return out


class ResidualWaveletBranch(nn.Module):
    """Haar-style multi-level detail residuals added back to the signal."""

    def __init__(self, channels: int, levels: int = 2) -> None:
        super().__init__()
        self.channels = int(channels)
        self.levels = max(1, int(levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.zeros_like(x)
        current = x
        for _ in range(self.levels):
            if current.shape[-1] < 4:
                break
            length = current.shape[-1]
            even = current[..., 0::2]
            odd = current[..., 1::2]
            approx = (even + odd) * 0.5
            detail = (even - odd) * 0.5
            detail_up = detail.repeat_interleave(2, dim=-1)
            if detail_up.shape[-1] > length:
                detail_up = detail_up[..., :length]
            elif detail_up.shape[-1] < length:
                detail_up = F.pad(detail_up, (0, length - detail_up.shape[-1]))
            residual[..., :length] = residual[..., :length] + detail_up
            current = approx
        return x + residual


class SharedFrontend(nn.Module):
    """Produces per-channel stem features (B*C, stem_ch, L)."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 64,
        use_early_channel_mixer: bool = True,
        use_sensor_fusion_stem: bool = False,
        sensor_fusion_mode: str = "residual_attention",
        use_fir_frontend: bool = False,
        fir_num_bands: int = 4,
        fir_kernel_size: int = 31,
        fir_init_mode: str = "delta",
        use_wavelet_residual: bool = False,
        wavelet_levels: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem_channels = int(stem_channels)
        self.use_sensor_fusion_stem = bool(use_sensor_fusion_stem)
        self.use_fir_frontend = bool(use_fir_frontend)
        self.use_wavelet_residual = bool(use_wavelet_residual)

        if self.use_sensor_fusion_stem:
            self.channel_mixer = SensorFusionStem(self.in_channels, mode=sensor_fusion_mode)
        elif use_early_channel_mixer:
            self.channel_mixer = EarlyChannelMixer(self.in_channels)
        else:
            self.channel_mixer = nn.Identity()

        self.wavelet = ResidualWaveletBranch(self.in_channels, levels=wavelet_levels) if self.use_wavelet_residual else None
        self.fir = (
            FIRFeatureBank(
                self.in_channels,
                num_bands=fir_num_bands,
                kernel_size=fir_kernel_size,
                init_mode=fir_init_mode,
            )
            if self.use_fir_frontend
            else None
        )
        stem_in = self.in_channels
        if self.fir is not None:
            stem_in = self.fir.out_channels
        self.stage1 = Stage1ChannelExtractor(out_channels=self.stem_channels, kernel_size=7)
        self._stem_in = stem_in

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, L), got {tuple(x.shape)}")
        batch_size, channels, length = x.shape
        x = self.channel_mixer(x)
        if self.wavelet is not None:
            x = self.wavelet(x)
        if self.fir is not None:
            x = self.fir(x)
            channels = x.shape[1]
        feat_bc = self.stage1(x.reshape(batch_size * channels, 1, length))
        return feat_bc, batch_size, channels
