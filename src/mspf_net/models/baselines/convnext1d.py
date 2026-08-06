from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ensure_bcl


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm1D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pw1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        if self.gamma is not None:
            x = x * self.gamma
        x = x.transpose(1, 2).contiguous()
        return residual + self.drop_path(x)


class ConvNeXtStage1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, drop_rates: list[float]):
        super().__init__()
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                LayerNorm1D(in_ch),
                nn.Conv1d(in_ch, out_ch, kernel_size=2, stride=2),
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = (
            nn.Sequential(*[ConvNeXtBlock1D(out_ch, drop_path=rate) for rate in drop_rates[:depth]])
            if depth > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.blocks(x)


class ConvNeXt1D(BaseBaselineModel):
    model_name = "convnext1d"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        stage_count = 4
        base = n_blocks // stage_count
        remainder = n_blocks % stage_count
        depths = [base + (1 if i < remainder else 0) for i in range(stage_count)]
        dims = [width, width * 2, width * 4, width * 4]

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm1D(dims[0]),
        )

        total_blocks = sum(depths)
        drop_rates = torch.linspace(0, dropout, total_blocks).tolist()
        offset = 0
        stages = []
        in_ch = dims[0]
        for depth, dim in zip(depths, dims):
            stage_rates = drop_rates[offset : offset + depth]
            stages.append(ConvNeXtStage1D(in_ch, dim, depth, stage_rates))
            in_ch = dim
            offset += depth
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean(dim=-1)
        x = self.norm(x)
        return self.head(x)
