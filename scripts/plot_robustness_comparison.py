#!/usr/bin/env python3
"""Plot D5/D8 variable-speed robustness and noise-SNR comparison figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.config_utils import get_config_path
from mspf_net.utils.aggregate_robustness import (
    ROBUSTNESS_TARGETS,
    get_noise_snr_levels,
    noise_snr_column_names,
)

# Minimum printed text size (pt); this 2-column figure stays portrait at full textwidth rather
# than needing a sidewaysfigure.
TEXTWIDTH_IN = 5.771653543307086
PRINT_WIDTH_IN = TEXTWIDTH_IN
BASE_FONT_SIZE = 9


def _font_for_print(fig_width_in: float, target_pt: float = BASE_FONT_SIZE) -> float:
    return max(target_pt, target_pt * fig_width_in / PRINT_WIDTH_IN)


# Canonical run per model (softmax MSPF-Net, recommended_fine baselines) keeps ablation variants
# out of the cross-model heatmaps.
CANONICAL_MSPF_TAG = "softmax"
BASELINE_TAG = "recommended_fine"


def canonical_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a cross-model table to one canonical run per model (softmax / recommended_fine)."""
    if df.empty or "experiment_tag" not in df.columns:
        return df
    is_mspf = df["model"] == "mspf_net"
    keep = (is_mspf & (df["experiment_tag"] == CANONICAL_MSPF_TAG)) | (
        ~is_mspf & (df["experiment_tag"] == BASELINE_TAG)
    )
    filtered = df[keep].copy()
    dropped = sorted(set(df["model"]) - set(filtered["model"]))
    if dropped:
        print(f"  [canonical_runs] no canonical run for: {', '.join(dropped)} (dropped)")
    return filtered


def _plot_heatmap(
    df: pd.DataFrame,
    plot_path: Path,
    *,
    title: str,
    value_col: str,
    row_col: str,
    col_order: list[str] | None = None,
) -> None:
    if df.empty or value_col not in df.columns:
        return
    # Deterministic dedup: keep the most recent run per (row, target), never select by metric value.
    plot_df = df.dropna(subset=[value_col]).copy()
    if "mtime" in plot_df.columns:
        plot_df = plot_df.sort_values("mtime", ascending=False)
    plot_df = plot_df.drop_duplicates([row_col, "target"], keep="first")
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index=row_col, columns="target", values=value_col)
    if col_order:
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    pivot = pivot.sort_index()
    fig_w = max(11, 1.8 * len(pivot.columns))
    fig_h = max(6, 0.85 * len(pivot.index) + 2)
    font = _font_for_print(fig_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, fontsize=font)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=font)
    ax.set_title(title, fontsize=font + 2, fontweight="bold", pad=14)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                text, color = "NA", "white"
            else:
                text = f"{val:.1f}"
                color = "white" if val < 60 else "#111111"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=font)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro-F1 (%)")
    cbar.ax.tick_params(labelsize=font)
    cbar.set_label("Macro-F1 (%)", fontsize=font)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=max(150.0, min(600.0, 400.0 * PRINT_WIDTH_IN / fig.get_size_inches()[0])), bbox_inches="tight")
    plt.close(fig)


def _plot_noise_curves(df: pd.DataFrame, plot_path: Path, *, row_col: str, title: str) -> None:
    snr_cols = noise_snr_column_names()
    if df.empty or not any(c in df.columns for c in snr_cols):
        return
    levels = get_noise_snr_levels()
    x = levels
    fig, ax = plt.subplots(figsize=(12, 8))
    for label, group in df.groupby(row_col):
        ys = []
        for snr_db, col in zip(levels, snr_cols):
            if col not in group.columns:
                ys.append(np.nan)
                continue
            ys.append(float(group[col].mean()))
        ax.plot(x, ys, marker="o", label=str(label), linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)", fontsize=BASE_FONT_SIZE)
    ax.set_ylabel("Window Macro-F1 (%)", fontsize=BASE_FONT_SIZE)
    ax.set_title(title, fontsize=BASE_FONT_SIZE + 2, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=BASE_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=BASE_FONT_SIZE, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=max(150.0, min(600.0, 400.0 * PRINT_WIDTH_IN / fig.get_size_inches()[0])), bbox_inches="tight")
    plt.close(fig)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot robustness comparison figures from aggregated CSVs.")
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=str(PROJECT_ROOT / get_config_path("paths", "tables", default="results/tables")),
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    phase4_csv = tables_dir / "phase4_robustness_comparison.csv"
    mspf_csv = tables_dir / "mspf_net_robustness_comparison.csv"
    figs_root = PROJECT_ROOT / get_config_path("paths", "figures", default="results/figures")
    phase4_figs = figs_root / "phase4"
    mspf_figs = figs_root / "mspf_net"
    phase4_figs.mkdir(parents=True, exist_ok=True)
    mspf_figs.mkdir(parents=True, exist_ok=True)

    target_order = sorted(ROBUSTNESS_TARGETS)

    # Cross-model figures use only the canonical run per model; the MSPF-variant figures below use
    # the separate, unfiltered CSV.
    phase4_df = canonical_runs(_load_csv(phase4_csv))
    if not phase4_df.empty:
        _plot_heatmap(
            phase4_df,
            phase4_figs / "robustness_comparison_window.png",
            title="Variable-speed robustness — Window Macro-F1 (D5/D8)",
            value_col="robustness_window_macro_f1",
            row_col="model",
            col_order=target_order,
        )
        _plot_heatmap(
            phase4_df.dropna(subset=["robustness_file_macro_f1"]),
            phase4_figs / "robustness_comparison_file.png",
            title="Variable-speed robustness — File Macro-F1 (D5/D8)",
            value_col="robustness_file_macro_f1",
            row_col="model",
            col_order=target_order,
        )
        _plot_noise_curves(
            phase4_df,
            phase4_figs / "noise_robustness_curves.png",
            row_col="model",
            title="AWGN noise robustness on test split (Phase 4 baselines)",
        )
        print(f"  Phase 4 robustness plots → {phase4_figs}")

    mspf_df = _load_csv(mspf_csv)
    if not mspf_df.empty:
        row_col = "variant_label" if "variant_label" in mspf_df.columns else "variant"
        _plot_heatmap(
            mspf_df,
            mspf_figs / "robustness_comparison_window.png",
            title="MSPF-Net variable-speed robustness — Window Macro-F1 (D5/D8)",
            value_col="robustness_window_macro_f1",
            row_col=row_col,
            col_order=target_order,
        )
        _plot_noise_curves(
            mspf_df,
            mspf_figs / "noise_robustness_curves.png",
            row_col=row_col,
            title="MSPF-Net AWGN noise robustness on test split",
        )
        print(f"  MSPF robustness plots      → {mspf_figs}")

    if phase4_df.empty and mspf_df.empty:
        print("  No robustness comparison CSVs found. Run aggregate_results.py / aggregate_mspf_results.py first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
