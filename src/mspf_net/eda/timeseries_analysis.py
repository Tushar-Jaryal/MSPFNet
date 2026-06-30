from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import yaml

from mspf_net.constants import ACTIVE_THESIS_DATASETS, get_dataset_display
from mspf_net.eda.eda_utils import FS_MAP, load_signal_multichannel
from mspf_net.eda.phase2_eda import DS_INFO, load_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIN_RECOMMENDED_WINDOWS = {
    1: 512,
    2: 8192,
    4: 2048,
    5: 256,
    8: 8192,
}
MAX_RECOMMENDED_WINDOW = 8192


def _finite_1d(x: np.ndarray, max_len: int = 50_000) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) > max_len:
        arr = arr[:max_len]
    return arr


def dominant_frequency_psd(x: np.ndarray, fs: int) -> tuple[float, float]:
    arr = _finite_1d(x)
    if len(arr) < 16:
        return float("nan"), float("nan")
    freqs, psd = sp_signal.welch(arr, fs=fs, nperseg=min(4096, len(arr)))
    mask = freqs > 0
    if not np.any(mask):
        return float("nan"), float("nan")
    dom_idx = int(np.argmax(psd[mask]))
    dom_freq = float(freqs[mask][dom_idx])
    if dom_freq <= 0:
        return float("nan"), float("nan")
    return dom_freq, float(fs / dom_freq)


def dominant_period_autocorr(x: np.ndarray, fs: int, max_lag: int | None = None) -> tuple[float, float]:
    arr = _finite_1d(x)
    if len(arr) < 32:
        return float("nan"), float("nan")
    arr = arr - arr.mean()
    std = float(arr.std())
    if std < 1e-12:
        return float("nan"), float("nan")

    if max_lag is None:
        max_lag = min(len(arr) // 2, max(256, fs // 2))

    ac = sp_signal.correlate(arr, arr, mode="full", method="fft")
    ac = ac[len(arr) - 1 : len(arr) + max_lag]
    if len(ac) <= 2 or ac[0] == 0:
        return float("nan"), float("nan")
    ac = ac / ac[0]

    peaks, _ = sp_signal.find_peaks(ac[1:], prominence=0.05)
    if len(peaks) == 0:
        lag = int(np.argmax(ac[2:]) + 2) if len(ac) > 2 else 1
    else:
        peak_vals = ac[1:][peaks]
        lag = int(peaks[int(np.argmax(peak_vals))] + 1)
    if lag <= 0:
        return float("nan"), float("nan")
    return float(lag), float(lag / fs * 1000.0)


def trend_metrics(x: np.ndarray) -> tuple[float, float]:
    arr = _finite_1d(x)
    if len(arr) < 32:
        return float("nan"), float("nan")
    z = (arr - arr.mean()) / (arr.std() + 1e-12)
    t = np.arange(len(z), dtype=float)
    lin_slope = float(np.polyfit(t, z, 1)[0])

    n_bins = min(64, max(8, len(arr) // 512))
    bins = np.array_split(arr, n_bins)
    rms = np.asarray([np.sqrt(np.mean(np.square(b))) for b in bins if len(b) > 0], dtype=float)
    if len(rms) < 2:
        return lin_slope, float("nan")
    tr = np.arange(len(rms), dtype=float)
    rms_slope = float(np.polyfit(tr, rms, 1)[0] / (np.mean(rms) + 1e-12))
    return lin_slope, rms_slope


def _pick_representative_rows(
    catalog: pd.DataFrame,
    ds_id: int,
    split_filter: str = "train",
    max_files_per_class: int = 1,
) -> pd.DataFrame:
    sub = catalog[catalog["dataset_id"] == ds_id].copy()
    if "split" in sub.columns and split_filter:
        preferred = sub[sub["split"] == split_filter]
        if not preferred.empty:
            sub = preferred
    picked = []
    for _, grp in sub.groupby("fault_code"):
        picked.append(grp.head(max_files_per_class))
    if not picked:
        return pd.DataFrame(columns=sub.columns)
    return pd.concat(picked, ignore_index=True)


def recommend_window(periods_samples: np.ndarray, ds_id: int | None = None) -> int:
    vals = np.asarray(periods_samples, dtype=float)
    vals = vals[np.isfinite(vals) & (vals >= 4)]
    if len(vals) == 0:
        win = 2048
    else:
        median_period = float(np.median(vals))
        p75_period = float(np.quantile(vals, 0.75))
        target = max(256.0, 4.0 * median_period, 2.0 * p75_period)
        win = int(2 ** math.ceil(math.log2(target)))
    min_window = MIN_RECOMMENDED_WINDOWS.get(int(ds_id or -1), 256)
    return int(min(max(win, min_window), MAX_RECOMMENDED_WINDOW))


def _update_window_config(rec_df: pd.DataFrame, config_path: Path) -> None:
    if rec_df.empty:
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    windowing = cfg.setdefault("windowing", {})
    per_dataset = windowing.setdefault("per_dataset", {})
    for _, row in rec_df.iterrows():
        ds_id = int(row["dataset_id"])
        existing = per_dataset.get(str(ds_id), {}) or per_dataset.get(ds_id, {}) or {}
        existing_win = int(existing.get("window_size", 0) or 0)
        existing_hop = existing.get("hop_size")
        floor = max(MIN_RECOMMENDED_WINDOWS.get(ds_id, 256), existing_win)
        win = int(min(max(int(row["recommended_window_samples"]), floor), MAX_RECOMMENDED_WINDOW))
        if existing_hop is not None and existing_win == win:
            hop = int(existing_hop)
        else:
            hop = max(1, win // 2)
        per_dataset[str(ds_id)] = {
            "window_size": win,
            "hop_size": hop,
        }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _plot_matrix(
    ax,
    df: pd.DataFrame,
    title: str,
    cmap: str,
    value_fmt: str = ".1f",
    center: float | None = None,
) -> None:
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    values = df.to_numpy(dtype=float)
    norm = None
    if center is not None and np.isfinite(values).any():
        finite = values[np.isfinite(values)]
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin < center < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    im = ax.imshow(values, cmap=cmap, aspect="auto", norm=norm)
    ax.set_title(title)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([f"ch{int(c) + 1}" for c in df.columns], rotation=0)

    finite = values[np.isfinite(values)]
    threshold = float(np.nanmedian(finite)) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                text = "NA"
                color = "#111111"
            else:
                text = format(val, value_fmt)
                color = "white" if val >= threshold else "#111111"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_dataset_timeseries(
    ds_id: int,
    ds_name: str,
    fs: int,
    representative_signal: np.ndarray,
    ds_df: pd.DataFrame,
    out_path: Path,
    recommended_window: int,
) -> None:
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], width_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    snippet_len = min(representative_signal.shape[0], max(fs // 2, 4000))
    t = np.arange(snippet_len) / fs
    for ch in range(representative_signal.shape[1]):
        sig = representative_signal[:snippet_len, ch]
        sig = (sig - sig.mean()) / (sig.std() + 1e-12)
        ax0.plot(t, sig + ch * 4.0, linewidth=0.8, label=f"ch{ch + 1}")
    ax0.set_title(f"{get_dataset_display(ds_id)} — {ds_name}: channel-wise raw trend snippet")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Standardized amplitude (offset per channel)")
    ax0.legend(ncol=min(6, representative_signal.shape[1]), fontsize=8, loc="upper right")

    ax1 = fig.add_subplot(gs[1, 0])
    period_heat = ds_df.pivot_table(
        index="fault_code",
        columns="channel_idx",
        values="chosen_period_ms",
        aggfunc="median",
    ).sort_index()
    _plot_matrix(ax1, period_heat, "Median dominant period (ms)", cmap="viridis")

    ax2 = fig.add_subplot(gs[1, 1])
    trend_heat = ds_df.pivot_table(
        index="fault_code",
        columns="channel_idx",
        values="rolling_rms_slope",
        aggfunc="median",
    ).sort_index()
    _plot_matrix(ax2, trend_heat, "Rolling-RMS trend slope", cmap="coolwarm", value_fmt=".3f", center=0.0)

    ax3 = fig.add_subplot(gs[2, 0])
    dom_df = ds_df[np.isfinite(ds_df["dominant_freq_hz"])].copy()
    if dom_df.empty:
        ax3.text(0.5, 0.5, "No frequency estimates", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()
    else:
        fault_codes = sorted(dom_df["fault_code"].unique().tolist())
        fault_to_x = {fault: i for i, fault in enumerate(fault_codes)}
        x = dom_df["fault_code"].map(fault_to_x).to_numpy(dtype=float)
        jitter = (dom_df["channel_idx"].to_numpy(dtype=float) - dom_df["channel_idx"].mean()) * 0.05
        scatter = ax3.scatter(
            x + jitter,
            dom_df["dominant_freq_hz"],
            c=dom_df["channel_idx"],
            cmap="tab10",
            s=30,
            alpha=0.85,
            edgecolors="none",
        )
        ax3.set_yscale("log")
        ax3.set_xticks(range(len(fault_codes)))
        ax3.set_xticklabels(fault_codes, rotation=35, ha="right")
        ax3.set_ylabel("Dominant frequency (Hz, log scale)")
        ax3.set_title("FFT dominant frequency by fault and channel")
        fig.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04, label="Channel index")

    ax4 = fig.add_subplot(gs[2, 1])
    agreement = ds_df[
        np.isfinite(ds_df["fft_period_samples"]) & np.isfinite(ds_df["autocorr_period_samples"])
    ].copy()
    if agreement.empty:
        ax4.text(0.5, 0.5, "No period agreement data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_axis_off()
    else:
        scatter = ax4.scatter(
            agreement["fft_period_samples"],
            agreement["autocorr_period_samples"],
            c=agreement["channel_idx"],
            cmap="tab10",
            s=32,
            alpha=0.85,
            edgecolors="none",
        )
        lo = min(float(agreement["fft_period_samples"].min()), float(agreement["autocorr_period_samples"].min()))
        hi = max(float(agreement["fft_period_samples"].max()), float(agreement["autocorr_period_samples"].max()))
        lo = max(lo, 1.0)
        hi = max(hi, lo * 1.1)
        ax4.plot([lo, hi], [lo, hi], linestyle="--", color="#111111", linewidth=1.0)
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_xlabel("FFT period estimate (samples)")
        ax4.set_ylabel("Autocorr period estimate (samples)")
        ax4.set_title("Period-estimation agreement")
        fig.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.04, label="Channel index")

    fig.suptitle(
        f"{get_dataset_display(ds_id)} time-series analysis | "
        f"recommended window = {recommended_window} samples ({recommended_window / fs * 1000:.1f} ms)",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_global_summary(channel_df: pd.DataFrame, rec_df: pd.DataFrame, fig_dir: Path) -> None:
    if channel_df.empty or rec_df.empty:
        return

    period_pivot = (
        channel_df.groupby(["dataset_label", "channel_idx"])["chosen_period_ms"]
        .median()
        .reset_index()
        .pivot(index="dataset_label", columns="channel_idx", values="chosen_period_ms")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _plot_matrix(ax, period_pivot, "Median dominant period by dataset/channel (ms)", cmap="viridis")
    fig.tight_layout()
    fig.savefig(fig_dir / "channel_period_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rec_df = rec_df.sort_values("dataset_label")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].bar(rec_df["dataset_label"], rec_df["recommended_window_samples"], color="#2563eb")
    axes[0].set_title("Recommended window size by dataset")
    axes[0].set_ylabel("Samples")
    axes[0].set_xlabel("Dataset")
    for i, row in rec_df.reset_index(drop=True).iterrows():
        axes[0].text(
            i,
            row["recommended_window_samples"] + 20,
            str(int(row["recommended_window_samples"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    trend_df = (
        channel_df.groupby("dataset_label")
        .agg(
            median_period_ms=("chosen_period_ms", "median"),
            median_abs_rms_slope=("rolling_rms_slope", lambda s: float(np.nanmedian(np.abs(s)))),
        )
        .reset_index()
        .merge(
            rec_df[["dataset_label", "recommended_window_samples"]],
            on="dataset_label",
            how="left",
        )
    )
    axes[1].scatter(
        trend_df["median_period_ms"],
        trend_df["median_abs_rms_slope"],
        s=trend_df["recommended_window_samples"] / 6.0,
        c=trend_df["recommended_window_samples"],
        cmap="plasma",
        alpha=0.9,
        edgecolors="#111111",
        linewidth=0.4,
    )
    for _, row in trend_df.iterrows():
        axes[1].text(
            row["median_period_ms"],
            row["median_abs_rms_slope"],
            row["dataset_label"],
            fontsize=8,
            ha="left",
            va="bottom",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Median chosen period (ms, log scale)")
    axes[1].set_ylabel("Median |rolling RMS slope| (log scale)")
    axes[1].set_title("Temporal complexity map")

    fig.tight_layout()
    fig.savefig(fig_dir / "window_recommendations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    agreement = channel_df[
        np.isfinite(channel_df["fft_period_samples"]) & np.isfinite(channel_df["autocorr_period_samples"])
    ].copy()
    if not agreement.empty:
        labels = sorted(agreement["dataset_label"].unique().tolist())
        cmap = plt.get_cmap("tab10", len(labels))
        color_map = {label: cmap(i) for i, label in enumerate(labels)}

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        for label in labels:
            sub = agreement[agreement["dataset_label"] == label]
            ax.scatter(
                sub["fft_period_samples"],
                sub["autocorr_period_samples"],
                s=28,
                alpha=0.8,
                color=color_map[label],
                label=label,
                edgecolors="none",
            )
        lo = max(
            1.0,
            min(float(agreement["fft_period_samples"].min()), float(agreement["autocorr_period_samples"].min())),
        )
        hi = max(float(agreement["fft_period_samples"].max()), float(agreement["autocorr_period_samples"].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#111111", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FFT period estimate (samples)")
        ax.set_ylabel("Autocorr period estimate (samples)")
        ax.set_title("FFT vs autocorr period estimates across datasets")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "period_method_agreement.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Thesis-facing summary: a single map from temporal difficulty to suggested window size.
    difficulty_df = (
        channel_df.groupby("dataset_label")
        .agg(
            median_period_ms=("chosen_period_ms", "median"),
            period_iqr_ms=("chosen_period_ms", lambda s: float(np.nanquantile(s, 0.75) - np.nanquantile(s, 0.25))),
            median_abs_trend=("rolling_rms_slope", lambda s: float(np.nanmedian(np.abs(s)))),
            median_freq_hz=("dominant_freq_hz", "median"),
        )
        .reset_index()
        .merge(
            rec_df[["dataset_label", "recommended_window_samples", "recommended_window_ms", "n_channels"]],
            on="dataset_label",
            how="left",
        )
    )
    if not difficulty_df.empty:
        # Heuristic discussion metric:
        # larger period spread + stronger trend + lower dominant frequency => harder temporal setting
        difficulty_df["difficulty_score"] = (
            np.log10(np.maximum(difficulty_df["period_iqr_ms"], 1e-3) + 1.0)
            + 0.35 * np.log10(np.maximum(difficulty_df["median_abs_trend"], 1e-6) / 1e-6 + 1.0)
            + 0.25 * np.log10(np.maximum(difficulty_df["recommended_window_ms"], 1.0))
            - 0.20 * np.log10(np.maximum(difficulty_df["median_freq_hz"], 1e-3))
        )

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        scatter = ax.scatter(
            difficulty_df["difficulty_score"],
            difficulty_df["recommended_window_samples"],
            s=np.maximum(difficulty_df["n_channels"], 1) * 55.0,
            c=difficulty_df["median_period_ms"],
            cmap="viridis",
            alpha=0.9,
            edgecolors="#111111",
            linewidth=0.5,
        )
        for _, row in difficulty_df.iterrows():
            ax.text(
                row["difficulty_score"],
                row["recommended_window_samples"],
                row["dataset_label"],
                fontsize=9,
                ha="left",
                va="bottom",
            )
        ax.set_title("Dataset difficulty vs recommended window")
        ax.set_xlabel("Temporal difficulty score (higher = harder / less uniform)")
        ax.set_ylabel("Recommended window (samples)")
        ax.set_yscale("log", base=2)
        ax.set_yticks([256, 512, 1024, 2048, 4096, 8192])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Median dominant period (ms)")

        note = (
            "Bubble size = channel count\n"
            "Difficulty combines period spread, trend magnitude,\n"
            "recommended window scale, and inverse median frequency"
        )
        ax.text(
            0.98,
            0.02,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "dataset_difficulty_vs_window.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="MSPF-Net time-series / period analysis")
    parser.add_argument("--dataset", nargs="+", type=int, default=ACTIVE_THESIS_DATASETS)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-files-per-class", type=int, default=1)
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Write recommended windows into configs/config.yaml without lowering existing per-dataset window/hop settings.",
    )
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "results" / "figures" / "eda_timeseries"
    tbl_dir = PROJECT_ROOT / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(PROJECT_ROOT)
    channel_rows: list[dict] = []
    recommendation_rows: list[dict] = []

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     MSPF-Net  —  Time-Series / Period Analysis                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Project    : {PROJECT_ROOT}")
    print(f"  Datasets   : {args.dataset}")
    print(f"  Split pref : {args.split}")
    print(f"  Max files  : {args.max_files_per_class} per class")

    for ds_id in args.dataset:
        info = DS_INFO.get(ds_id, {"name": f"D{ds_id}", "component": "Unknown"})
        fs = FS_MAP[ds_id]
        picked = _pick_representative_rows(catalog, ds_id, split_filter=args.split, max_files_per_class=args.max_files_per_class)
        if picked.empty:
            print(f"\n  [SKIP] {get_dataset_display(ds_id)} — no representative files found")
            continue

        print(f"\n  {get_dataset_display(ds_id)} — {info['name']}  fs={fs:,} Hz")
        representative_signal = None

        for _, row in picked.iterrows():
            path = PROJECT_ROOT / str(row["file_path"])
            if not path.exists():
                continue
            try:
                sig = load_signal_multichannel(path, ds_id)
            except Exception as exc:
                print(f"    [WARN] {path.name}: {exc}")
                continue
            if representative_signal is None:
                representative_signal = sig

            for ch in range(sig.shape[1]):
                x = sig[:, ch]
                dom_freq_hz, fft_period_samples = dominant_frequency_psd(x, fs)
                ac_period_samples, ac_period_ms = dominant_period_autocorr(x, fs)
                lin_slope, rms_slope = trend_metrics(x)
                chosen_period_samples = ac_period_samples if np.isfinite(ac_period_samples) else fft_period_samples
                chosen_period_ms = (
                    float(chosen_period_samples / fs * 1000.0)
                    if np.isfinite(chosen_period_samples) and chosen_period_samples > 0
                    else float("nan")
                )
                channel_rows.append(
                    {
                        "dataset_id": ds_id,
                        "dataset_label": get_dataset_display(ds_id),
                        "dataset_name": info["name"],
                        "fault_code": row["fault_code"],
                        "source_file": row["file_path"],
                        "channel_idx": ch,
                        "dominant_freq_hz": dom_freq_hz,
                        "fft_period_samples": fft_period_samples,
                        "autocorr_period_samples": ac_period_samples,
                        "autocorr_period_ms": ac_period_ms,
                        "chosen_period_samples": chosen_period_samples,
                        "chosen_period_ms": chosen_period_ms,
                        "linear_trend_slope": lin_slope,
                        "rolling_rms_slope": rms_slope,
                        "n_samples_analyzed": int(min(len(np.asarray(x).reshape(-1)), 50_000)),
                    }
                )

        ds_df = pd.DataFrame([r for r in channel_rows if r["dataset_id"] == ds_id])
        if ds_df.empty:
            print("    [WARN] no channel analysis rows produced")
            continue

        periods = ds_df["chosen_period_samples"].to_numpy(dtype=float)
        recommended_window = recommend_window(periods, ds_id=ds_id)
        rec_row = {
            "dataset_id": ds_id,
            "dataset_label": get_dataset_display(ds_id),
            "dataset_name": info["name"],
            "n_channels": int(ds_df["channel_idx"].max() + 1),
            "n_rows_analyzed": int(len(ds_df)),
            "median_period_samples": float(np.nanmedian(periods)),
            "p75_period_samples": float(np.nanquantile(periods[np.isfinite(periods)], 0.75)) if np.isfinite(periods).any() else float("nan"),
            "max_period_samples": float(np.nanmax(periods)) if np.isfinite(periods).any() else float("nan"),
            "recommended_window_samples": int(recommended_window),
            "recommended_window_ms": float(recommended_window / fs * 1000.0),
            "heuristic": "next_pow2(max(4*median_period, 2*p75_period, 256))",
        }
        recommendation_rows.append(rec_row)
        print(
            f"    Recommended window: {recommended_window} samples "
            f"({recommended_window / fs * 1000.0:.1f} ms)"
        )

        if representative_signal is not None:
            out_path = fig_dir / f"d{ds_id}_{info['name'].lower()}_timeseries.png"
            _plot_dataset_timeseries(
                ds_id=ds_id,
                ds_name=info["name"],
                fs=fs,
                representative_signal=representative_signal,
                ds_df=ds_df,
                out_path=out_path,
                recommended_window=recommended_window,
            )

    channel_df = pd.DataFrame(channel_rows)
    rec_df = pd.DataFrame(recommendation_rows).sort_values("dataset_id")

    channel_path = tbl_dir / "timeseries_channel_analysis.csv"
    rec_path = tbl_dir / "timeseries_window_recommendations.csv"
    channel_df.to_csv(channel_path, index=False)
    rec_df.to_csv(rec_path, index=False)
    if args.update_config:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        _update_window_config(rec_df, config_path)
        print("  Config updated    →", config_path.relative_to(PROJECT_ROOT))
    _plot_global_summary(channel_df, rec_df, fig_dir)

    print("\n  Channel analysis  →", channel_path.relative_to(PROJECT_ROOT))
    print("  Recommendations   →", rec_path.relative_to(PROJECT_ROOT))
    print("  Figures           →", fig_dir.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
