from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import yaml

from mspf_net.constants import ACTIVE_THESIS_DATASETS, get_dataset_display
from mspf_net.eda.eda_utils import FS_MAP, load_signal_multichannel
from mspf_net.eda.phase2_eda import DS_INFO, load_catalog

# Every figure in this module is drawn at roughly its printed size so MIN_FONT_SIZE stays at the
# 18pt floor with no inflation multiplier.
MIN_FONT_SIZE = 9
TEXTWIDTH_IN = 5.771653543307086  # A4 minus 3.17cm L/R margins (thesis/main.tex geometry)
TEXTHEIGHT_IN = 9.692913385826772  # A4 minus 2.54cm T/B margins (thesis/main.tex geometry)
PRINT_WIDTH_IN = 0.88 * TEXTWIDTH_IN
# The 6-panel per-dataset dashboard is placed as a sidewaysfigure to trade page height for printed width.
DASHBOARD_PRINT_WIDTH_IN = 0.92 * TEXTHEIGHT_IN
plt.rcParams.update(
    {
        "font.size": MIN_FONT_SIZE,
        "axes.titlesize": MIN_FONT_SIZE + 2,
        "axes.labelsize": MIN_FONT_SIZE,
        "xtick.labelsize": MIN_FONT_SIZE,
        "ytick.labelsize": MIN_FONT_SIZE,
        "legend.fontsize": MIN_FONT_SIZE,
        "figure.titlesize": MIN_FONT_SIZE + 2,
    }
)


def _save_dpi(fig, print_width_in: float, target_print_dpi: float = 400.0) -> float:
    """Save dpi giving ~target_print_dpi at the printed size (see eda_utils)."""
    canvas_w = float(fig.get_size_inches()[0])
    if canvas_w <= 0:
        return target_print_dpi
    return max(150.0, min(600.0, target_print_dpi * print_width_in / canvas_w))

def _compact_number(x: float) -> str:
    """Short tick label for log-scale axes; avoids scientific notation to save horizontal margin."""
    if x >= 1000:
        return f"{x / 1000:.3g}k"
    if x == int(x):
        return f"{int(x)}"
    return f"{x:.3g}"


def _set_compact_log_ticks(axis) -> None:
    axis.set_major_formatter(FuncFormatter(lambda x, pos: _compact_number(x)))
    # Minor ticks are disabled on log-scale axes spanning under one decade to avoid a second,
    # overlapping tick formatter.
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def _set_plain_colorbar_ticks(cbar) -> None:
    """Disable matplotlib's scientific-notation offset text on colorbars to save space."""
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    cbar.formatter = fmt
    cbar.update_ticks()


def _set_integer_colorbar_ticks(cbar) -> None:
    """Force whole-number tick labels on the channel-index colorbar for consistent width."""
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()


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
    cax=None,
    max_text_cols: int = 2,
) -> None:
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        if cax is not None:
            cax.set_axis_off()
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
    # Row stride is computed from the axes' rendered height so fault-code labels never collide,
    # even with many rows.
    axes_height_in = ax.get_position().height * ax.figure.get_size_inches()[1]
    row_slot_in = axes_height_in / max(len(df.index), 1)
    row_stride = max(1, int(np.ceil(0.24 / row_slot_in)))
    row_labels = [str(v) if i % row_stride == 0 else "" for i, v in enumerate(df.index)]
    ax.set_yticklabels(row_labels)
    ax.set_xticks(range(len(df.columns)))
    # Column labels are spelled out up to max_text_cols, then just numbered, since axes width
    # doesn't grow with channel count.
    if len(df.columns) <= max_text_cols:
        ax.set_xticklabels([f"ch{int(c) + 1}" for c in df.columns], rotation=0)
    else:
        # Tick label stride is computed from the axes' actual rendered width so labels stay readable
        # regardless of column count.
        axes_width_in = ax.get_position().width * ax.figure.get_size_inches()[0]
        slot_in = axes_width_in / len(df.columns)
        col_stride = max(1, int(np.ceil(0.26 / slot_in)))
        labels = [str(int(c) + 1) if i % col_stride == 0 else "" for i, c in enumerate(df.columns)]
        ax.set_xticklabels(labels, rotation=90, ha="center")

    if len(df.columns) <= max_text_cols:
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
                ax.text(j, i, text, ha="center", va="center", fontsize=MIN_FONT_SIZE, color=color)

    if cax is not None:
        # Colorbar axes are pre-allocated via a dedicated GridSpec column rather than shrinking the
        # parent axes after layout.
        cbar = ax.figure.colorbar(im, cax=cax)
    else:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _set_plain_colorbar_ticks(cbar)


def _plot_dataset_timeseries(
    ds_id: int,
    ds_name: str,
    fs: int,
    representative_signal: np.ndarray,
    ds_df: pd.DataFrame,
    out_path: Path,
    recommended_window: int,
) -> None:
    # Every axes is placed at an explicit absolute-inch rectangle (no layout engine), with margins
    # sized from measured rendered bounding boxes.
    fig_w = DASHBOARD_PRINT_WIDTH_IN
    # fig_h is bounded by the sidewaysfigure geometry so the rotated figure fits the page width.
    fig_h = TEXTWIDTH_IN - 0.05

    n_channels = representative_signal.shape[1]
    # The per-point channel colorbar is dropped for single-channel datasets, where it would be uninformative.
    show_channel_cbar = n_channels > 1
    # ax3's width must scale with fault-code count, since each rotated tick label has a fixed footprint.
    n_fault_codes = ds_df["fault_code"].nunique() if not ds_df.empty else 1
    # Above 5 fault codes, ax3's width is capped and only tick labels are thinned, not the plotted data.
    ax3_label_stride = max(1, -(-n_fault_codes // 5))  # ceil division

    def rect(left_in: float, bottom_in: float, width_in: float, height_in: float) -> list[float]:
        return [left_in / fig_w, bottom_in / fig_h, width_in / fig_w, height_in / fig_h]

    top_margin_in = 0.68  # suptitle + ax0's own title
    mid_gap_in = 0.70  # row0's xlabel + tick numbers (measured ~0.63in needed)
    # Bottom margin sized for the longest rotated fault-code label seen (D8's multi-character codes).
    bottom_margin_in = 1.20
    row0_height_in = 1.15
    row1_bottom_in = bottom_margin_in
    row1_height_in = fig_h - top_margin_in - row0_height_in - mid_gap_in - bottom_margin_in
    row0_bottom_in = row1_bottom_in + row1_height_in + mid_gap_in

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Row 0 (snippet plot) has generous margins since it has no competition for horizontal room.
    ax0_left_in = 0.80
    ax0 = fig.add_axes(rect(ax0_left_in, row0_bottom_in, fig_w - ax0_left_in - 0.15, row0_height_in))

    # Row 1 column cursor advances left to right, leaving a gap after each colorbar for its own tick text.
    cursor = 0.50  # left margin: ax1's fault-code y-tick labels (measured ~1.13in needed)
    ax1_left_gap = 1.15
    cursor += ax1_left_gap
    # ax1/ax2 width depends on whether per-cell numeric text is drawn (narrow) or dropped for color
    # swatches only (wide).
    ax1_w = 0.55 if n_channels > 2 else 0.70
    ax1 = fig.add_axes(rect(cursor, row1_bottom_in, ax1_w, row1_height_in))
    cursor += ax1_w + 0.05
    cax1_w = 0.15
    cax1 = fig.add_axes(rect(cursor, row1_bottom_in, cax1_w, row1_height_in))
    cursor += cax1_w + 0.65  # cax1's own tick-number text (measured overflow_right=0.573in) + ax2's own tiny left overflow
    ax2_w = ax1_w
    ax2 = fig.add_axes(rect(cursor, row1_bottom_in, ax2_w, row1_height_in), sharey=ax1)
    cursor += ax2_w + 0.05
    cax2_w = 0.15
    cax2 = fig.add_axes(rect(cursor, row1_bottom_in, cax2_w, row1_height_in))
    # Gap before ax3 must cover both cax2's tick overflow and ax3's own y-tick/ylabel overflow.
    cursor += cax2_w + 0.85 + 1.00
    ax3_w = max(0.8, 0.26 * min(n_fault_codes, 5))
    ax3 = fig.add_axes(rect(cursor, row1_bottom_in, ax3_w, row1_height_in))
    cursor += ax3_w + 0.10  # small breathing room to the right of ax3's plot
    # ax3 and ax4 share one channel-index colorbar (placed after ax4) instead of duplicating it,
    # freeing width for ax4.
    cax3 = None
    cursor += 1.00
    # ax4's width is capped by its own centered xlabel clipping the canvas edge; must stay in sync
    # with cax4's cursor advance.
    ax4_breathing_in = 0.15  # small breathing room to the right of ax4's plot
    cax4_trailing_gap = 0.35  # cax4's own small-integer tick numbers (measured overflow_right~0.257in), clipped at 0.05 and 0.20 (confirmed by rendering)
    trailing_w = (0.15 + cax4_trailing_gap) if show_channel_cbar else 0.0
    ax4_w = max(0.20, min(0.75, fig_w - ax4_breathing_in - trailing_w - cursor))
    ax4 = fig.add_axes(rect(cursor, row1_bottom_in, ax4_w, row1_height_in))
    cursor += ax4_w + ax4_breathing_in
    if show_channel_cbar:
        cax4_w = 0.15
        cax4 = fig.add_axes(rect(cursor, row1_bottom_in, cax4_w, row1_height_in))
        cursor += cax4_w + cax4_trailing_gap
    else:
        cax4 = None
    assert cursor <= fig_w + 0.02, f"timeseries dashboard layout overflows canvas: {cursor:.2f}in > {fig_w:.2f}in"

    snippet_len = min(representative_signal.shape[0], max(fs // 2, 4000))
    t = np.arange(snippet_len) / fs
    for ch in range(n_channels):
        sig = representative_signal[:snippet_len, ch]
        sig = (sig - sig.mean()) / (sig.std() + 1e-12)
        ax0.plot(t, sig + ch * 4.0, linewidth=0.8, label=f"ch{ch + 1}")
    ax0.set_title("Signal snippet", fontsize=MIN_FONT_SIZE, pad=6)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Std. amplitude")
    if n_channels > 1:
        ax0.legend(ncol=min(6, n_channels), fontsize=MIN_FONT_SIZE - 4, loc="upper right")

    period_heat = ds_df.pivot_table(
        index="fault_code",
        columns="channel_idx",
        values="chosen_period_ms",
        aggfunc="median",
    ).sort_index()
    # Per-panel titles are omitted since they'd overflow into neighbouring panels; full description
    # is in the figure caption.
    _plot_matrix(ax1, period_heat, "", cmap="viridis", cax=cax1)

    trend_heat = ds_df.pivot_table(
        index="fault_code",
        columns="channel_idx",
        values="rolling_rms_slope",
        aggfunc="median",
    ).sort_index()
    _plot_matrix(ax2, trend_heat, "", cmap="coolwarm", value_fmt=".3f", center=0.0, cax=cax2)
    ax2.set_ylabel("")
    plt.setp(ax2.get_yticklabels(), visible=False)  # duplicate of ax1's fault-code labels

    dom_df = ds_df[np.isfinite(ds_df["dominant_freq_hz"])].copy()
    if dom_df.empty:
        ax3.text(0.5, 0.5, "No frequency\nestimates", ha="center", va="center", transform=ax3.transAxes)
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
        _set_compact_log_ticks(ax3.yaxis)
        ax3.set_xticks(range(len(fault_codes)))
        # Fault-code labels are rotated 90 (not 45) degrees to fit ax3's narrow width; above 5 categories
        # only every Nth is drawn.
        tick_labels = [
            code if i % ax3_label_stride == 0 else ""
            for i, code in enumerate(fault_codes)
        ]
        ax3.set_xticklabels(tick_labels, rotation=90, ha="center")
        ax3.set_ylabel("Freq (Hz)", labelpad=10)
        # No colorbar here; ax3/ax4 share the single channel-index colorbar placed after ax4.

    agreement = ds_df[
        np.isfinite(ds_df["fft_period_samples"]) & np.isfinite(ds_df["autocorr_period_samples"])
    ].copy()
    if agreement.empty:
        ax4.text(0.5, 0.5, "No period\nagreement data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_axis_off()
        if cax4 is not None:
            cax4.set_axis_off()
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
        _set_compact_log_ticks(ax4.xaxis)
        _set_compact_log_ticks(ax4.yaxis)
        # ax4's log-scale ticks are fixed to just the two decade endpoints, since the default locator
        # packs too many labels into a narrow panel.
        x_lo_decade = 10 ** np.floor(np.log10(lo))
        x_hi_decade = 10 ** np.ceil(np.log10(hi))
        ax4.xaxis.set_major_locator(FixedLocator([x_lo_decade, x_hi_decade]))
        # xlim is set to exactly the two decade endpoints so both fixed ticks are always visible.
        ax4.set_xlim(x_lo_decade, x_hi_decade)
        # ax4's centered xlabel is shortened since its width is capped by the canvas edge; ylabels
        # don't have this problem.
        ax4.set_xlabel("FFT")
        ax4.set_ylabel("Autocorr (samples)", labelpad=10)
        if cax4 is not None:
            _set_integer_colorbar_ticks(fig.colorbar(scatter, cax=cax4))

    fig.suptitle(
        f"{get_dataset_display(ds_id)} time-series analysis | "
        f"window = {recommended_window} samples ({recommended_window / fs * 1000:.1f} ms)",
        fontsize=MIN_FONT_SIZE, fontweight="bold", y=0.99,
    )
    import os
    if os.environ.get("TS_DEBUG_LAYOUT"):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for name, axis in [
            ("ax1", ax1), ("cax1", cax1), ("ax2", ax2), ("cax2", cax2),
            ("ax3", ax3), ("cax3", cax3), ("ax4", ax4), ("cax4", cax4),
        ]:
            if axis is None:
                print(f"{name}: None")
                continue
            bbox = axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            own_pos = axis.get_position()
            own_left_in = own_pos.x0 * fig_w
            own_right_in = own_pos.x1 * fig_w
            own_bottom_in = own_pos.y0 * fig_h
            own_top_in = own_pos.y1 * fig_h
            print(
                f"{name}: own=[{own_left_in:.3f},{own_right_in:.3f}] "
                f"tight=[{bbox.x0:.3f},{bbox.x1:.3f}] "
                f"overflow_left={own_left_in - bbox.x0:.3f} overflow_right={bbox.x1 - own_right_in:.3f} "
                f"overflow_bottom={own_bottom_in - bbox.y0:.3f} overflow_top={bbox.y1 - own_top_in:.3f}"
            )

    # No layout engine or bbox_inches="tight": every axes is already placed at an explicit rectangle.
    fig.savefig(out_path, dpi=_save_dpi(fig, DASHBOARD_PRINT_WIDTH_IN))
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

    # Drawn natively at print size; no inflation needed since the source canvas already matches it.
    fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, 4.2))
    _plot_matrix(ax, period_pivot, "Median dominant period by dataset/channel (ms)", cmap="viridis")
    fig.tight_layout()
    fig.savefig(fig_dir / "channel_period_heatmap.png", dpi=_save_dpi(fig, PRINT_WIDTH_IN), bbox_inches="tight")
    plt.close(fig)

    rec_df = rec_df.sort_values("dataset_label")
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_WIDTH_IN, 4.6))
    axes[0].bar(rec_df["dataset_label"], rec_df["recommended_window_samples"], color="#2563eb")
    # pad + ylim headroom: same fix as plot_class_balance in eda_utils.py --
    # without it the title collides with the above-bar value labels.
    axes[0].set_title("Recommended window size by dataset", fontsize=MIN_FONT_SIZE, pad=8 + MIN_FONT_SIZE)
    axes[0].set_ylabel("Samples")
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylim(0, float(rec_df["recommended_window_samples"].max()) * 1.22)
    axes[0].tick_params(axis="x", rotation=45)
    for i, row in rec_df.reset_index(drop=True).iterrows():
        axes[0].text(
            i,
            row["recommended_window_samples"] + 20,
            str(int(row["recommended_window_samples"])),
            ha="center",
            va="bottom",
            fontsize=MIN_FONT_SIZE,
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
            fontsize=MIN_FONT_SIZE,
            ha="left",
            va="bottom",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Median chosen period (ms, log scale)")
    axes[1].set_ylabel("Median |rolling RMS slope| (log scale)")
    axes[1].set_title("Temporal complexity map")

    fig.tight_layout()
    fig.savefig(fig_dir / "window_recommendations.png", dpi=_save_dpi(fig, PRINT_WIDTH_IN), bbox_inches="tight")
    plt.close(fig)

    agreement = channel_df[
        np.isfinite(channel_df["fft_period_samples"]) & np.isfinite(channel_df["autocorr_period_samples"])
    ].copy()
    if not agreement.empty:
        labels = sorted(agreement["dataset_label"].unique().tolist())
        cmap = plt.get_cmap("tab10", len(labels))
        color_map = {label: cmap(i) for i, label in enumerate(labels)}

        fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, PRINT_WIDTH_IN * 0.85))
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
        ax.legend(ncol=2, fontsize=MIN_FONT_SIZE)
        fig.tight_layout()
        fig.savefig(fig_dir / "period_method_agreement.png", dpi=_save_dpi(fig, PRINT_WIDTH_IN), bbox_inches="tight")
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

        fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, PRINT_WIDTH_IN * 0.85))
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
                fontsize=MIN_FONT_SIZE,
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
            fontsize=MIN_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "dataset_difficulty_vs_window.png", dpi=_save_dpi(fig, PRINT_WIDTH_IN), bbox_inches="tight")
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
