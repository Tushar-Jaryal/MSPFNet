from __future__ import annotations

import warnings
from pathlib import Path
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import matplotlib
matplotlib.use("Agg")                   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, LogFormatter

# ─── Colour palette (one per fault-code prefix) ───────────────────────────────
PALETTE = {
    "NOR":  "#2ecc71",
    "BRG_IR": "#e74c3c",
    "BRG_OR": "#e67e22",
    "BRG_BF": "#9b59b6",
    "BRG_CF": "#c0392b",
    "BRG_IO": "#f39c12",
    "GBX_BT": "#3498db",
    "GBX_MT": "#1abc9c",
    "GBX_RC": "#2980b9",
    "GBX_WR": "#16a085",
    "GBX_PT": "#8e44ad",
    "GBX_EG": "#27ae60",
    "GBX_CF": "#7f8c8d",
    "GBX_CK": "#e91e63",
    "MIX_TI": "#ff5722",
    "MIX_TO": "#795548",
    "MIX_II": "#607d8b",
    "MIX_IS": "#009688",
    "MIX_IB": "#673ab7",
    "MIX_OI": "#ff9800",
    "MIX_OS": "#4caf50",
    "MIX_OB": "#f44336",
}
DEFAULT_COLOR = "#95a5a6"
TEXT_DARK = "#1f2933"
TEXT_MID = "#52606d"
BOX_FACE = (1.0, 1.0, 1.0, 0.82)
BOX_EDGE = "#d9e2ec"

# Minimum printed text size (pt) after LaTeX rescales the figure down to page width.
MIN_FONT_SIZE = 9
TEXTWIDTH_IN = 5.771653543307086  # A4 minus 3.17cm L/R margins (thesis/main.tex geometry)
TEXTHEIGHT_IN = 9.692913385826772  # A4 minus 2.54cm T/B margins (thesis/main.tex geometry)

# Per-figure-family printed widths, matching thesis/chapter3.tex placements (grid/balance/heatmap).
GRID_PRINT_WIDTH_IN = TEXTWIDTH_IN
BALANCE_PRINT_WIDTH_IN = TEXTWIDTH_IN
# Both figures are now normal portrait floats instead of rotated sidewaysfigures.
SEPARABILITY_PRINT_WIDTH_IN = TEXTWIDTH_IN
CROSS_DATASET_PRINT_WIDTH_IN = TEXTWIDTH_IN


# Raster figures are drawn natively at their printed width, so dpi maps directly to print resolution.
TARGET_PRINT_DPI = 400.0


def save_dpi_for_print(fig, print_width_in: float,
                       target_print_dpi: float = TARGET_PRINT_DPI) -> float:
    """dpi for savefig that yields ~target_print_dpi at the printed size."""
    canvas_w = float(fig.get_size_inches()[0])
    if canvas_w <= 0:
        return target_print_dpi
    return max(150.0, min(600.0, target_print_dpi * print_width_in / canvas_w))


def _font_for_print(fig_width_in: float, print_width_in: float, target_pt: float = MIN_FONT_SIZE) -> float:
    """Source matplotlib fontsize so this figure's PRINTED text is >= target_pt."""
    return max(target_pt, target_pt * fig_width_in / print_width_in)

from mspf_net.constants import FS_NATIVE as FS_MAP  # single source of truth


# Signal loading

def load_signal(path: Path | str, ds_id: int, channel: int = 0) -> np.ndarray:
    """Load a raw vibration signal from any supported dataset format as a 1-D float64 array."""
    path = Path(path)
    ds_family = 3 if int(ds_id) in (31, 39) else int(ds_id)
    ext  = path.suffix.lower()

    if ext in (".mat",):
        return _load_mat(path, ds_family, channel)
    if ext == ".csv":
        return _load_csv(path, ds_family, channel)
    if ext == ".txt":
        return _load_txt(path, channel, ds_id=ds_family)
    raise ValueError(f"Unsupported extension '{ext}' for {path.name}")


def load_signal_multichannel(path: Path | str, ds_id: int) -> np.ndarray:
    """
    Load a raw vibration signal while preserving all detected signal channels.
    Returns an array shaped (n_samples, n_channels).
    """
    path = Path(path)
    ds_family = 3 if int(ds_id) in (31, 39) else int(ds_id)
    ext = path.suffix.lower()

    if ext in (".mat",):
        return _load_mat_multi(path, ds_family)
    if ext == ".csv":
        return _load_csv_multi(path, ds_family)
    if ext == ".txt":
        return _load_txt_multi(path, ds_family)
    raise ValueError(f"Unsupported extension '{ext}' for {path.name}")


def _load_mat(path: Path, ds_id: int, channel: int) -> np.ndarray:
    """Load .mat file — handles both MATLAB v5 and HDF5 (v7.3)."""
    if not HAS_SCIPY:
        raise ImportError("scipy required: pip install scipy")

    MAT_KEYS = {
        1: ["Channel_1", "ch1", "X", "data", "signal", "acc"],
        2: ["Signal", "signal", "data"],
        3: ["X097_DE_time", "X048_DE_time", "DE_time", "data"],
        7: ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mat = sio.loadmat(str(path))
        except Exception:
            try:
                import h5py
                with h5py.File(str(path), "r") as f:
                    data_keys = [k for k in f.keys() if not k.startswith("#")]
                    arr = np.array(f[data_keys[0]])
                    return _to_1d(arr, channel)
            except Exception as e:
                raise IOError(f"Cannot load {path.name}: {e}")

    data_keys = [k for k in mat if not k.startswith("_")]
    hints = MAT_KEYS.get(ds_id, [])
    for key in hints:
        if key in mat:
            return _to_1d(mat[key], channel)

    # Dataset-specific pattern fallback before generic "largest numeric array".
    if ds_id == 3:
        for key in data_keys:
            if key.endswith("_DE_time"):
                return _to_1d(mat[key], channel)
    if ds_id == 7:
        an_keys = sorted(
            [k for k in data_keys if re.fullmatch(r"AN\d+", k)],
            key=lambda x: int(x[2:]),
        )
        if an_keys:
            chosen = an_keys[min(channel, len(an_keys) - 1)]
            return _to_1d(mat[chosen], channel=0)

    # Fallback: largest numeric array
    best, best_size = None, 0
    for k in data_keys:
        v = mat[k]
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.size > best_size:
            best, best_size = mat[k], v.size
    if best is not None:
        return _to_1d(best, channel)
    raise ValueError(f"No numeric array found in {path.name}")


def _load_mat_multi(path: Path, ds_id: int) -> np.ndarray:
    """Load .mat file and preserve all available signal channels."""
    if not HAS_SCIPY:
        raise ImportError("scipy required: pip install scipy")

    MAT_KEYS = {
        1: ["Channel_1", "ch1", "X", "data", "signal", "acc"],
        2: ["Signal", "signal", "data"],
        3: ["X097_DE_time", "X048_DE_time", "DE_time", "data"],
        7: ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mat = sio.loadmat(str(path))
        except Exception:
            try:
                import h5py
                with h5py.File(str(path), "r") as f:
                    data_keys = [k for k in f.keys() if not k.startswith("#")]
                    arr = np.array(f[data_keys[0]])
                    return _to_2d(arr)
            except Exception as e:
                raise IOError(f"Cannot load {path.name}: {e}")

    data_keys = [k for k in mat if not k.startswith("_")]

    # D7 stores vibration channels as separate AN3..AN10 arrays; stack them before the generic hint lookup.
    if ds_id == 7:
        an_keys = sorted(
            [k for k in data_keys if re.fullmatch(r"AN\d+", k)],
            key=lambda x: int(x[2:]),
        )
        if an_keys:
            cols = [_to_1d(mat[k], channel=0) for k in an_keys]
            min_len = min(len(c) for c in cols)
            return np.stack([c[:min_len] for c in cols], axis=1)

    hints = MAT_KEYS.get(ds_id, [])
    for key in hints:
        if key in mat:
            return _to_2d(mat[key])

    best, best_size = None, 0
    for k in data_keys:
        v = mat[k]
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.size > best_size:
            best, best_size = mat[k], v.size
    if best is not None:
        return _to_2d(best)
    raise ValueError(f"No numeric array found in {path.name}")


def _load_csv(path: Path, ds_id: int, channel: int) -> np.ndarray:
    """Robust CSV loader: tries multiple encodings/separators, picks the highest-variance channel."""
    ENCODINGS  = ["utf-8", "gbk", "gb2312", "latin-1", "cp1252"]
    SEPARATORS = [",", None, "\t", ";", r"\s+"]  # None = pandas sniff

    best_df    = None
    best_score = 0

    for enc in ENCODINGS:
        for sep in SEPARATORS:
            try:
                kwargs = dict(header=None, on_bad_lines="skip",
                              encoding=enc, low_memory=False)
                if sep is None:
                    kwargs["sep"] = ","
                elif sep == r"\s+":
                    kwargs["sep"] = sep
                    kwargs["engine"] = "python"
                else:
                    kwargs["sep"] = sep

                df_try = pd.read_csv(path, **kwargs)
                df_num = df_try.apply(pd.to_numeric, errors="coerce")
                df_num = _trim_numeric_block(df_num)
                score  = df_num.size
                if score > best_score:
                    best_score = score
                    best_df    = df_num
                    if df_num.shape[1] >= 3:  # good enough — stop early
                        break
            except Exception:
                continue
        if best_df is not None and best_df.shape[1] >= 3:
            break

    if best_df is None or best_score < 10:
        raise IOError(f"Cannot parse any numeric columns from {path.name}")

    df = best_df.fillna(0.0)

    arr = df.values.astype(float)
    if arr.size == 0:
        raise ValueError(f"No numeric data in {path.name}")

    # ── Multi-column signal selection ─────────────────────────────────────────
    if arr.ndim == 2 and arr.shape[1] > 1:
        candidates = np.arange(arr.shape[1])

        # Pass 1: drop monotonically increasing columns (time / sample-index axes)
        not_mono = np.array([not _is_monotonic(arr[:, c]) for c in candidates])
        if not_mono.any():
            candidates = candidates[not_mono]

        # Drop all-non-negative columns (time/index axes) -- vibration signals are zero-centred AC signals.
        is_ac = np.array(
            [arr[:, c].min() < 0 or arr[:, c].max() <= 10 for c in candidates]
        )
        if is_ac.any():
            candidates = candidates[is_ac]

        # Pass 3: pick highest-variance column (default) or requested channel
        variances  = np.array([np.var(arr[:, c]) for c in candidates])
        chosen_col = candidates[np.argmax(variances)] if channel == 0 else \
                     candidates[min(channel, len(candidates) - 1)]
        return arr[:, chosen_col]

    return arr.flatten()


def _load_csv_multi(path: Path, ds_id: int) -> np.ndarray:
    """Robust CSV loader that preserves all detected signal channels."""
    ENCODINGS = ["utf-8", "gbk", "gb2312", "latin-1", "cp1252"]
    SEPARATORS = [",", None, "\t", ";", r"\s+"]

    best_df = None
    best_score = 0

    for enc in ENCODINGS:
        for sep in SEPARATORS:
            try:
                kwargs = dict(header=None, on_bad_lines="skip", encoding=enc, low_memory=False)
                if sep is None:
                    kwargs["sep"] = ","
                elif sep == r"\s+":
                    kwargs["sep"] = sep
                    kwargs["engine"] = "python"
                else:
                    kwargs["sep"] = sep

                df_try = pd.read_csv(path, **kwargs)
                df_num = df_try.apply(pd.to_numeric, errors="coerce")
                df_num = _trim_numeric_block(df_num)
                score = df_num.size
                if score > best_score:
                    best_score = score
                    best_df = df_num
                    if df_num.shape[1] >= 3:
                        break
            except Exception:
                continue
        if best_df is not None and best_df.shape[1] >= 3:
            break

    if best_df is None or best_score < 10:
        raise IOError(f"Cannot parse any numeric columns from {path.name}")

    arr = best_df.fillna(0.0).values.astype(float)
    return _select_signal_matrix(arr, ds_id=ds_id)


def _trim_numeric_block(df_num: pd.DataFrame) -> pd.DataFrame:
    """Remove sparse metadata/header rows and keep the dense numeric signal block."""
    df_num = df_num.dropna(how="all", axis=1)
    if df_num.empty:
        return df_num

    min_numeric = 2 if df_num.shape[1] > 1 else 1
    row_counts = df_num.notna().sum(axis=1)
    dense_rows = row_counts >= min_numeric
    if dense_rows.any():
        first_dense = dense_rows.idxmax()
        df_num = df_num.loc[first_dense:]
        dense_rows = df_num.notna().sum(axis=1) >= min_numeric
        if dense_rows.any():
            df_num = df_num.loc[dense_rows]

    return df_num.dropna(how="all", axis=1).dropna(how="all", axis=0)


def _is_monotonic(col: np.ndarray, tol: float = 0.99) -> bool:
    """Return True if col is nearly monotonically increasing (time/index axis)."""
    if len(col) < 4:
        return False
    diffs = np.diff(col)
    frac_positive = np.mean(diffs > 0)
    return frac_positive >= tol


def _load_txt(path: Path, channel: int = 0, ds_id: int = 0) -> np.ndarray:
    """Robust TXT loader for datasets with metadata headers followed by tabular numeric data."""
    # D8 header is stable and the fault-discriminating sensor is ch2.
    if ds_id == 8:
        try:
            arr = np.loadtxt(str(path), skiprows=18, delimiter="\t")
            return arr[:, 2].astype(np.float64)
        except Exception as e:
            raise IOError(f"D8 loader failed for {path.name}: {e}")

    try:
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    rows.append([float(x) for x in parts])
                except Exception:
                    continue
        if not rows:
            raise ValueError(f"No numeric block found in {path.name}")
        arr = np.asarray(rows, dtype=float)
    except Exception:
        try:
            arr = np.loadtxt(str(path))
        except Exception:
            arr = np.loadtxt(str(path), skiprows=1)

    if np.ndim(arr) == 2 and arr.shape[1] > 1:
        candidates = np.arange(arr.shape[1])
        not_mono = np.array([not _is_monotonic(arr[:, c]) for c in candidates])
        if not_mono.any():
            candidates = candidates[not_mono]
        if len(candidates) == 0:
            candidates = np.arange(arr.shape[1])
        chosen = candidates[min(channel, len(candidates) - 1)]
        return arr[:, chosen].astype(float).flatten()
    return np.asarray(arr, dtype=float).flatten()


def _load_txt_multi(path: Path, ds_id: int = 0) -> np.ndarray:
    """TXT loader that preserves all detected signal channels."""
    if ds_id == 8:
        try:
            arr = np.loadtxt(str(path), skiprows=18, delimiter="\t")
            return arr[:, 1:5].astype(np.float64)
        except Exception as e:
            raise IOError(f"D8 loader failed for {path.name}: {e}")

    try:
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    rows.append([float(x) for x in parts])
                except Exception:
                    continue
        if not rows:
            raise ValueError(f"No numeric block found in {path.name}")
        arr = np.asarray(rows, dtype=float)
    except Exception:
        try:
            arr = np.loadtxt(str(path))
        except Exception:
            arr = np.loadtxt(str(path), skiprows=1)

    return _select_signal_matrix(arr, ds_id=ds_id)


def _to_1d(arr: np.ndarray, channel: int = 0) -> np.ndarray:
    arr = np.squeeze(arr).astype(float)
    if arr.ndim == 2:
        N, C = arr.shape
        if C == 0:
            raise ValueError(f"Array has 0 columns after numeric coercion (shape {arr.shape})")
        if C > N:                       # (C, N) layout — transpose
            arr = arr.T
            N, C = arr.shape
        channel = min(channel, C - 1)
        return arr[:, channel]
    return arr.flatten()


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to shape (n_samples, n_channels)."""
    arr = np.squeeze(arr).astype(float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        n, c = arr.shape
        if c == 0:
            raise ValueError(f"Array has 0 columns after numeric coercion (shape {arr.shape})")
        if c > n:
            arr = arr.T
        return arr
    # Flatten higher dims into channels while preserving sample axis.
    arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(float)


def _select_signal_matrix(arr: np.ndarray, ds_id: int = 0) -> np.ndarray:
    """Drop time/index axes and preserve all remaining signal channels."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        return _to_2d(arr)

    candidates = np.arange(arr.shape[1])
    not_mono = np.array([not _is_monotonic(arr[:, c]) for c in candidates])
    if not_mono.any():
        candidates = candidates[not_mono]

    is_ac = np.array([arr[:, c].min() < 0 or arr[:, c].max() <= 10 for c in candidates])
    if is_ac.any():
        candidates = candidates[is_ac]

    if len(candidates) == 0:
        candidates = np.arange(arr.shape[1])

    sig = arr[:, candidates].astype(float)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    return sig


# Time-domain statistics

def compute_stats(signal: np.ndarray, fs: int, label: str = "") -> dict:
    """Compute standard vibration fault-diagnosis features (RMS, kurtosis, crest factor, SNR, etc.)."""
    # Accept 1-D or 2-D (take first channel of 2-D)
    x = np.asarray(signal, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    x = x.flatten()
    # Drop NaN / Inf values (can occur from bad CSV rows)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        raise ValueError("Signal is empty or all-NaN/Inf")

    rms        = float(np.sqrt(np.mean(x ** 2)))
    peak       = float(np.max(np.abs(x)))
    crest      = peak / (rms + 1e-12)
    kurt       = float(sp_kurtosis(x, fisher=True))   # excess kurtosis
    skewness   = float(sp_skew(x))

    # Spectral entropy: lower = more structured (fault-like), higher = noise-like
    try:
        x_spec = x[:50_000] if len(x) > 50_000 else x
        _, psd = sp_signal.welch(x_spec, fs=fs, nperseg=min(1024, len(x_spec)))
        psd_norm = psd / (psd.sum() + 1e-12)
        snr = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))  # spectral entropy
    except Exception:
        snr = float("nan")

    return {
        "label":        label,
        "n_samples":    n,
        "duration_s":   round(n / fs, 4),
        "mean":         round(float(np.mean(x)), 6),
        "std":          round(float(np.std(x)), 6),
        "rms":          round(rms, 6),
        "peak":         round(peak, 6),
        "peak_to_peak": round(float(np.max(x) - np.min(x)), 6),
        "crest_factor": round(crest, 4),
        "kurtosis":     round(kurt, 4),
        "skewness":     round(skewness, 4),
        "spectral_entropy": round(snr, 4),
    }


# Frequency-domain

def compute_fft(signal: np.ndarray, fs: int,
                n_fft: Optional[int] = None,
                window: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-sided magnitude spectrum."""
    x = signal.astype(float)
    N = n_fft or len(x)
    if window:
        w = np.hanning(len(x))
        x = x * w
    spec   = np.fft.rfft(x, n=N)
    mags   = np.abs(spec) / (N / 2)
    freqs  = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, mags


def compute_psd(signal: np.ndarray, fs: int,
                nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density estimate."""
    freqs, psd = sp_signal.welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    return freqs, psd


def segment_signal(signal: np.ndarray,
                   win: int = 2048, hop: int = 1024) -> np.ndarray:
    """Slice signal into overlapping windows. Returns (M, win) array."""
    n = len(signal)
    starts = range(0, n - win + 1, hop)
    segs = np.stack([signal[s:s + win] for s in starts])
    return segs


# Plotting helpers

def _color(fault_code: str) -> str:
    return PALETTE.get(fault_code, DEFAULT_COLOR)


def _annotation_box() -> dict:
    return dict(boxstyle="round,pad=0.22", fc=BOX_FACE, ec=BOX_EDGE, lw=0.6)


def _style_axis(ax, tick_size: float = MIN_FONT_SIZE) -> None:
    ax.tick_params(axis="both", labelsize=tick_size, colors=TEXT_DARK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#cbd2d9")
    ax.spines["bottom"].set_color("#cbd2d9")
    ax.xaxis.label.set_color(TEXT_DARK)
    ax.yaxis.label.set_color(TEXT_DARK)
    ax.title.set_color(TEXT_DARK)


def _style_heatmap_axis(ax, tick_size: float = MIN_FONT_SIZE) -> None:
    ax.tick_params(axis="both", labelsize=tick_size, colors=TEXT_DARK)
    ax.title.set_color(TEXT_DARK)
    for spine in ax.spines.values():
        spine.set_color("#cbd2d9")


def plot_waveforms(signals: dict[str, np.ndarray],
                  fs: int,
                  title: str = "Waveforms",
                  max_samples: int = 4096,
                  figsize: tuple = (14, None)) -> plt.Figure:
    """Plot one waveform per fault class in a vertical stack."""
    n_cls  = len(signals)
    height = max(8, 3.2 * n_cls)
    fig, axes = plt.subplots(n_cls, 1, figsize=(max(figsize[0], 18), height),
                             sharex=False, sharey=False)
    if n_cls == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=MIN_FONT_SIZE + 2, fontweight="bold", y=1.01, color=TEXT_DARK)

    for ax, (code, sig) in zip(axes, signals.items()):
        seg   = sig[:max_samples]
        t     = np.arange(len(seg)) / fs * 1000   # ms
        color = _color(code)
        ax.plot(t, seg, color=color, linewidth=0.6, alpha=0.85)
        ax.set_ylabel(code, fontsize=MIN_FONT_SIZE, rotation=0, labelpad=70,
                      ha="right", color=color, fontweight="bold")
        ax.yaxis.set_label_position("left")
        _style_axis(ax, tick_size=MIN_FONT_SIZE)
        # Annotate RMS & kurtosis
        rms  = float(np.sqrt(np.mean(seg ** 2)))
        kurt = float(sp_kurtosis(seg, fisher=True))
        ax.text(0.99, 0.85,
                f"RMS={rms:.4f}  K={kurt:.2f}",
                transform=ax.transAxes, fontsize=MIN_FONT_SIZE,
                ha="right", va="top", color=TEXT_DARK, bbox=_annotation_box())

    axes[-1].set_xlabel("Time (ms)", fontsize=MIN_FONT_SIZE, color=TEXT_DARK)
    fig.tight_layout()
    return fig


def plot_fft(signals: dict[str, np.ndarray],
             fs: int,
             title: str = "FFT Spectra",
             f_max: Optional[float] = None,
             figsize: tuple = (14, None)) -> plt.Figure:
    """
    Plot magnitude spectrum for each fault class.
    Uses Welch PSD for smoother estimates.
    """
    n_cls  = len(signals)
    height = max(8, 3.2 * n_cls)
    fig, axes = plt.subplots(n_cls, 1, figsize=(max(figsize[0], 18), height),
                             sharex=True, sharey=False)
    if n_cls == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=MIN_FONT_SIZE + 2, fontweight="bold", y=1.01, color=TEXT_DARK)
    f_lim = f_max or (fs / 2)

    for ax, (code, sig) in zip(axes, signals.items()):
        freqs, psd = compute_psd(sig, fs)
        mask  = freqs <= f_lim
        color = _color(code)
        ax.semilogy(freqs[mask], psd[mask], color=color, linewidth=0.7, alpha=0.9)
        ax.set_ylabel(code, fontsize=MIN_FONT_SIZE, rotation=0, labelpad=70,
                      ha="right", color=color, fontweight="bold")
        _style_axis(ax, tick_size=MIN_FONT_SIZE)
        # Mark dominant frequency
        dom_idx = np.argmax(psd[mask][1:]) + 1   # skip DC
        dom_f   = freqs[mask][dom_idx]
        ax.axvline(dom_f, color=color, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(dom_f + f_lim * 0.01, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-12,
                f"{dom_f:.0f}Hz", fontsize=MIN_FONT_SIZE, color=TEXT_DARK, va="bottom",
                bbox=_annotation_box())

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=MIN_FONT_SIZE, color=TEXT_DARK)
    fig.tight_layout()
    return fig


def plot_class_grid(signals: dict[str, np.ndarray],
                    fs: int,
                    dataset_name: str,
                    max_samples: int = 4096,
                    f_max: Optional[float] = None) -> plt.Figure:
    """
    Combined figure: waveform + FFT side-by-side for each class.
    One row per fault class.
    """
    codes  = list(signals.keys())
    n_cls  = len(codes)
    f_lim  = f_max or min(fs / 2, 5000)

    # Figure is drawn natively at its intended print size, with row height/spacing tuned so the
    # waveform trace stays visible alongside its labels.
    font = MIN_FONT_SIZE
    fig_w = GRID_PRINT_WIDTH_IN
    header_in = 0.85
    footer_in = 0.32
    # Cap total printed height so the figure plus its caption still fits one portrait page.
    MAX_FIG_H_IN = 8.2
    row_h = min(1.15, (MAX_FIG_H_IN - header_in - footer_in) / max(n_cls, 1))
    fig_h = row_h * n_cls + header_in + footer_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"{dataset_name} — Waveform & Spectrum per Class  (fs={fs:,} Hz)",
                 fontsize=font + 2, fontweight="bold", y=1.0 - 0.22 * header_in / fig_h,
                 color=TEXT_DARK)

    # Margins sized for the rotated class-code label and the waveform/PSD column gutter.
    left_in, right_in = 0.88, 0.14
    outer = gridspec.GridSpec(n_cls, 2, figure=fig,
                              wspace=0.52, hspace=0.55,
                              left=left_in / fig_w, right=1.0 - right_in / fig_w,
                              top=1.0 - header_in / fig_h, bottom=footer_in / fig_h)

    for row_i, code in enumerate(codes):
        sig    = signals[code]
        color  = _color(code)
        seg    = sig[:max_samples]
        t      = np.arange(len(seg)) / fs * 1000

        # ── Waveform ─────────────────────────────────────────────────────────
        ax_t = fig.add_subplot(outer[row_i, 0])
        ax_t.plot(t, seg, color=color, linewidth=0.55, alpha=0.85)
        ax_t.set_ylabel(code, fontsize=font, rotation=0, labelpad=34,
                        ha="right", color=color, fontweight="bold")
        _style_axis(ax_t, tick_size=font)
        if row_i == 0:
            ax_t.set_title("Time Domain", fontsize=font, pad=6, color=TEXT_DARK)
        if row_i == n_cls - 1:
            ax_t.set_xlabel("Time (ms)", fontsize=font, color=TEXT_DARK)

        rms  = float(np.sqrt(np.mean(seg ** 2)))
        kurt = float(sp_kurtosis(seg, fisher=True))
        # Annotation box placed top-left to avoid colliding with the neighbouring PSD subplot.
        ax_t.text(0.03, 0.94, f"RMS={rms:.4f}\nK={kurt:.2f}",
                  transform=ax_t.transAxes, fontsize=font,
                  ha="left", va="top", color=TEXT_DARK,
                  bbox=_annotation_box())

        # ── FFT / PSD ─────────────────────────────────────────────────────────
        ax_f = fig.add_subplot(outer[row_i, 1])
        freqs, psd = compute_psd(sig, fs)
        mask = freqs <= f_lim
        ax_f.semilogy(freqs[mask], psd[mask],
                      color=color, linewidth=0.55, alpha=0.85)
        _style_axis(ax_f, tick_size=font)
        if row_i == 0:
            ax_f.set_title("Power Spectral Density (Welch)", fontsize=font, pad=6, color=TEXT_DARK)
        if row_i == n_cls - 1:
            ax_f.set_xlabel("Frequency (Hz)", fontsize=font, color=TEXT_DARK)

    return fig


def plot_class_balance(catalog: pd.DataFrame,
                       ds_id: int,
                       ds_name: str) -> plt.Figure:
    """
    Bar chart of file count per fault class for one dataset.
    """
    sub = catalog[catalog["dataset_id"] == ds_id].copy()
    path_col = "file_path" if "file_path" in sub.columns else "original_path"
    counts = (sub.groupby("fault_code")[path_col]
                 .count()
                 .reset_index(name="files")
                 .sort_values("files", ascending=False))

    colors = [_color(c) for c in counts["fault_code"]]
    n = len(counts)
    # For many classes, draw natively at print width and use a steeper label rotation to avoid
    # tick-label collisions.
    fig_w = BALANCE_PRINT_WIDTH_IN
    font = MIN_FONT_SIZE
    rotation = 90 if n >= 10 else 35
    # Height scales with class count but stays wide-and-short rather than tall.
    fig_h = 2.9 + (0.55 if n >= 10 else 0.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bars = ax.bar(counts["fault_code"], counts["files"], color=colors, edgecolor="white")
    # pad and ylim headroom reserve room for above-bar value labels at print-scaled font sizes.
    ax.set_title(f"{ds_name} — Class Distribution", fontsize=font + 2, fontweight="bold",
                 pad=8 + 0.5 * font, color=TEXT_DARK)
    ax.set_xlabel("Fault Code", fontsize=font, color=TEXT_DARK)
    ax.set_ylabel("File Count", fontsize=font, color=TEXT_DARK)
    _style_axis(ax, tick_size=font)
    ax.tick_params(axis="x", rotation=rotation)
    ax.set_ylim(0, float(counts["files"].max()) * 1.22)
    for bar, val in zip(bars, counts["files"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=font, color=TEXT_DARK,
                bbox=_annotation_box())
    fig.tight_layout()
    return fig


def plot_cross_dataset_stats(stats_df: pd.DataFrame) -> plt.Figure:
    """
    Heatmap-style comparison of mean kurtosis / crest factor across datasets.
    stats_df must have columns: dataset_id, fault_code, kurtosis, crest_factor, rms
    """
    metrics = ["rms", "kurtosis", "crest_factor"]
    # Figure height is driven by row count (fault codes), independent of width-driven font scaling,
    # to avoid tick-label collisions.
    n_rows = stats_df["fault_code"].nunique()
    fig_w_total = 10 * len(metrics)
    font = _font_for_print(fig_w_total, CROSS_DATASET_PRINT_WIDTH_IN)
    fig_h = max(9.0, 0.42 * font / 18.0 * n_rows + 1.5)
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(fig_w_total, fig_h))
    fig.suptitle("Cross-Dataset Signal Statistics Summary",
                 fontsize=font + 2, fontweight="bold", color=TEXT_DARK)

    for ax, metric in zip(axes, metrics):
        pivot = (stats_df
                 .groupby(["dataset_id", "fault_code"])[metric]
                 .mean()
                 .unstack(level=0))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"D{c}" for c in pivot.columns], fontsize=font)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=font)
        ax.set_title(metric.replace("_", " ").title(), fontsize=font, color=TEXT_DARK)
        _style_heatmap_axis(ax, tick_size=font)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=font, colors=TEXT_DARK)
        cbar.outline.set_edgecolor("#cbd2d9")

    fig.tight_layout()
    return fig


def compute_class_distance_matrix(
    stats_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute pairwise class distances from z-scored per-class summary statistics."""
    if feature_cols is None:
        feature_cols = ["rms", "kurtosis", "crest_factor", "skewness", "spectral_entropy"]

    keep_cols = ["fault_code", *feature_cols]
    df = stats_df[keep_cols].dropna().drop_duplicates("fault_code").copy()
    if df.empty:
        return pd.DataFrame()

    feats = df[list(feature_cols)].astype(float)
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0).replace(0.0, 1.0)
    z = (feats - mu) / sigma

    labels = df["fault_code"].astype(str).tolist()
    arr = z.to_numpy(dtype=float)
    dist = np.sqrt(((arr[:, None, :] - arr[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(dist, index=labels, columns=labels)


def summarize_class_separability(
    stats_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return pairwise class separability rows sorted from hardest to easiest.
    Smaller distance means the classes are more similar under the chosen stats.
    """
    dist_df = compute_class_distance_matrix(stats_df, feature_cols=feature_cols)
    if dist_df.empty or len(dist_df) < 2:
        return pd.DataFrame(columns=["fault_a", "fault_b", "distance"])

    rows = []
    labels = list(dist_df.index)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append(
                {
                    "fault_a": labels[i],
                    "fault_b": labels[j],
                    "distance": float(dist_df.iloc[i, j]),
                }
            )
    return pd.DataFrame(rows).sort_values("distance", ascending=True).reset_index(drop=True)


def plot_class_separability(
    stats_df: pd.DataFrame,
    dataset_name: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Heatmap of pairwise class separability distances for one dataset."""
    dist_df = compute_class_distance_matrix(stats_df, feature_cols=feature_cols)
    if dist_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No separability data", ha="center", va="center", color=TEXT_DARK, bbox=_annotation_box())
        ax.axis("off")
        return fig

    labels = dist_df.index.tolist()
    vals = dist_df.to_numpy()
    n = len(labels)

    fig_w = max(11, 1.6 * n)
    font = _font_for_print(fig_w, SEPARABILITY_PRINT_WIDTH_IN)
    fig, ax = plt.subplots(figsize=(fig_w, max(9, 1.3 * n)))
    im = ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=font)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=font)
    ax.set_title(f"{dataset_name}\nClass Separability (z-scored stats distance)", fontsize=font + 2, pad=14, color=TEXT_DARK)
    _style_heatmap_axis(ax, tick_size=font)

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    vrng = max(vmax - vmin, 1e-12)
    # Per-cell numeric text is dropped for large class counts, relying on color + colorbar instead.
    cell_width_in = SEPARABILITY_PRINT_WIDTH_IN / n
    show_cell_text = cell_width_in >= (MIN_FONT_SIZE * 5 * 0.55 / 72.0)
    if show_cell_text:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                norm_val = (float(vals[i, j]) - vmin) / vrng
                txt_color = "white" if norm_val < 0.55 else "#111111"
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=font, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Distance", rotation=90, fontsize=font)
    cbar.ax.tick_params(labelsize=font, colors=TEXT_DARK)
    cbar.ax.yaxis.label.set_color(TEXT_DARK)
    cbar.outline.set_edgecolor("#cbd2d9")
    fig.tight_layout()
    return fig


# Convenience: load one representative signal per class from catalog

def load_class_representatives(
        catalog: pd.DataFrame,
        ds_id: int,
        project_root: Path,
        split_filter: Optional[str] = "train",
        channel: int = 0) -> dict[str, np.ndarray]:
    """Load one representative signal per fault class for a given dataset."""
    sub = catalog[catalog["dataset_id"] == ds_id].copy()
    if "split" in sub.columns and split_filter:
        # prefer requested split, fall back to any file
        sub_split = sub[sub["split"] == split_filter]
        if sub_split.empty:
            sub_split = sub
    else:
        sub_split = sub

    result: dict[str, np.ndarray] = {}
    for code, grp in sub_split.groupby("fault_code"):
        row  = grp.iloc[0]
        path_col = "file_path" if "file_path" in row.index else "original_path"
        path = project_root / row[path_col]
        if not path.exists():
            continue
        try:
            sig = load_signal(path, ds_id=ds_id, channel=channel)
            result[code] = sig
        except Exception as exc:
            print(f"  [WARN] D{ds_id}/{code}: {exc}")

    # Sort by fault code for consistent plotting order
    return dict(sorted(result.items()))
