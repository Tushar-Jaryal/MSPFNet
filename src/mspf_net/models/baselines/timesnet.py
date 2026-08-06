from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseBaselineModel


def fft_for_period(x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [B, T, C]
    x = x.contiguous()
    fft_input = x.detach().contiguous().to("cpu") if x.device.type == "mps" else x
    xf = torch.fft.rfft(fft_input, dim=1)
    amplitude = xf.abs().mean(dim=(0, 2))
    if amplitude.numel() <= 1:
        periods = torch.tensor([x.size(1)], device=x.device, dtype=torch.long)
        batch_weights = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        return periods, batch_weights
    amplitude[0] = 0
    k = min(top_k, amplitude.numel() - 1)
    top = torch.topk(amplitude, k=k)
    frequency_indices = top.indices.clamp_min(1)
    periods = (x.size(1) // frequency_indices).clamp_min(1)
    batch_weights = xf.abs().mean(dim=2)[:, frequency_indices]
    return periods.to(x.device), batch_weights.to(x.device)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.position_dropout(self.value_embedding(x))


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6) -> None:
        super().__init__()
        kernels = list(range(1, 2 * num_kernels, 2))
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel, kernel),
                    padding=(kernel // 2, kernel // 2),
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for kernel in self.kernels:
            value = kernel(x).contiguous()
            out = value if out is None else out + value
        return (out / float(len(self.kernels))).contiguous()


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, top_k: int, num_kernels: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        periods, period_weights = fft_for_period(x, self.top_k)
        outputs = []

        for period in periods.tolist():
            period = max(1, int(period))
            padded_length = int(math.ceil(seq_len / period) * period)
            if padded_length > seq_len:
                padding = torch.zeros(
                    batch,
                    padded_length - seq_len,
                    channels,
                    device=x.device,
                    dtype=x.dtype,
                )
                series = torch.cat([x, padding], dim=1)
            else:
                series = x

            series = series.contiguous().reshape(batch, padded_length // period, period, channels)
            series = series.permute(0, 3, 1, 2).contiguous()
            series = self.conv(series)
            series = series.permute(0, 2, 3, 1).contiguous().reshape(batch, padded_length, channels)
            outputs.append(series[:, :seq_len, :].contiguous())

        weights = F.softmax(period_weights, dim=1)
        aggregated = None
        for idx, value in enumerate(outputs):
            weighted = value * weights[:, idx].reshape(batch, 1, 1)
            aggregated = weighted if aggregated is None else aggregated + weighted
        aggregated = aggregated.contiguous()
        return (aggregated + x).contiguous()


class TimesNet(BaseBaselineModel):
    model_name = "timesnet"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 64,
        d_ff: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,  # kept for config compatibility
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = DataEmbedding(in_channels, d_model, dropout)
        self.blocks = nn.ModuleList(
            [TimesBlock(d_model, d_ff, top_k=top_k, num_kernels=num_kernels) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # loader supplies (B, C, L); reference TimesNet expects (B, T, C)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"TimesNet expects a 2D/3D tensor, got shape {tuple(x.shape)}")

        x = x.transpose(1, 2).contiguous()
        x = self.embedding(x).contiguous()
        for block in self.blocks:
            x = self.layer_norm(block(x).contiguous()).contiguous()
        x = F.gelu(x)
        x = self.dropout(x).contiguous()
        x = x.contiguous().reshape(x.shape[0], -1)
        return self.projection(x)
