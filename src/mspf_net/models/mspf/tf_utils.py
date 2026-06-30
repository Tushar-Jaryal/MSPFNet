"""Time-frequency helpers shared by slim and MoE MSPF paths."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cwt_kernel_bank(
    length: int,
    num_bands: int,
    cwt_kernel_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Morlet-style wavelets at log-spaced scales for the given sequence length."""
    if num_bands <= 0:
        raise ValueError(f"num_bands must be positive, got {num_bands}")
    low = max(2.0, float(length // 64))
    high = max(4.0, float(length // 8))
    periods = torch.logspace(
        math.log10(low),
        math.log10(high),
        steps=num_bands,
        device=device,
    )

    max_period = int(periods.max().item())
    max_kernel = cwt_kernel_size if cwt_kernel_size % 2 == 1 else cwt_kernel_size + 1
    kernel_size = max(17, min(length if length % 2 == 1 else length - 1, max_period * 4 + 1, max_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    t = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size // 2)
    kernels = []
    for period in periods[:num_bands]:
        sigma = period.clamp_min(2.0) * 0.6
        carrier = torch.cos(2.0 * math.pi * t / period.clamp_min(2.0))
        envelope = torch.exp(-0.5 * (t / sigma) ** 2)
        wavelet = carrier * envelope
        wavelet = wavelet - wavelet.mean()
        wavelet = wavelet / wavelet.norm(p=2).clamp_min(1e-6)
        kernels.append(wavelet)
    return torch.stack(kernels, dim=0)


def cwt_scalogram(
    feat_bc: torch.Tensor,
    num_bands: int,
    cwt_kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build a CWT scalogram per batch row.

    Returns
    -------
    scalogram : (B*, num_bands, L) log-compressed magnitude
    band_energy : (B*, num_bands) mean energy per scale
    confidence : float in [0, 1], high when one scale dominates
    """
    orig_dtype = feat_bc.dtype
    x = feat_bc.float()
    bc, channels, length = x.shape
    kernels = cwt_kernel_bank(length, num_bands, cwt_kernel_size, x.device)
    n_bands = kernels.size(0)
    weight = kernels[:, None, :].repeat(channels, 1, 1)
    cwt = F.conv1d(x, weight, padding=kernels.size(-1) // 2, groups=channels)
    cwt = cwt.reshape(bc, channels, n_bands, length)
    band_energy = cwt.square().mean(dim=(1, 3))
    scalogram = cwt.abs().mean(dim=1)
    scalogram = torch.log1p(scalogram).to(orig_dtype)
    confidence = _band_confidence(band_energy)
    return scalogram, band_energy, confidence


def _band_confidence(band_energy: torch.Tensor) -> float:
    scores = band_energy.float().mean(dim=0).clamp_min(1e-8)
    if scores.numel() <= 1:
        return 1.0
    weights = scores / scores.sum()
    entropy = -(weights * weights.log()).sum() / math.log(scores.numel())
    return float((1.0 - entropy).clamp(0.0, 1.0).item())
