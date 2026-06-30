from __future__ import annotations

import numpy as np


def _safe_std(x: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.maximum(np.var(x, axis=axis), 1e-12))


def _skewness(x: np.ndarray, axis=None) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = _safe_std(x, axis=axis)
    centered = x - mean
    third = np.mean(centered ** 3, axis=axis)
    return third / np.maximum(std ** 3, 1e-12)


def _kurtosis(x: np.ndarray, axis=None) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = _safe_std(x, axis=axis)
    centered = x - mean
    fourth = np.mean(centered ** 4, axis=axis)
    return fourth / np.maximum(std ** 4, 1e-12)


def extract_rf_features(windows: np.ndarray) -> np.ndarray:
    """
    Hand-crafted window features for the RandomForest baseline.

    Input shape: (N, C, L)
    Output shape: (N, C * F)
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3:
        raise ValueError(f"Expected windows with shape (N, C, L); got {x.shape}")

    mean = np.mean(x, axis=2)
    std = _safe_std(x, axis=2)
    rms = np.sqrt(np.mean(np.square(x), axis=2))
    peak = np.max(np.abs(x), axis=2)
    p2p = np.ptp(x, axis=2)
    crest = peak / np.maximum(rms, 1e-12)
    skew = _skewness(x, axis=2)
    kurt = _kurtosis(x, axis=2)
    energy = np.mean(np.square(x), axis=2)

    spectrum = np.abs(np.fft.rfft(x, axis=2))
    spec_mean = np.mean(spectrum, axis=2)
    spec_std = _safe_std(spectrum, axis=2)
    dom_bin = np.argmax(spectrum[..., 1:], axis=2) + 1 if spectrum.shape[2] > 1 else np.zeros_like(mean)

    feature_blocks = [
        mean,
        std,
        rms,
        peak,
        p2p,
        crest,
        skew,
        kurt,
        energy,
        spec_mean,
        spec_std,
        dom_bin.astype(np.float64),
    ]
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)
