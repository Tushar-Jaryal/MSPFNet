#!/usr/bin/env python3
"""Plot Figure 4.8: accuracy vs. mean on-device (Raspberry Pi 5) latency per model."""
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

# Minimum printed text size (pt); uses a below-plot legend instead of inline labels since points
# are too tightly clustered.
TEXTWIDTH_IN = 5.771653543307086
# Printed at full textwidth with a squarer canvas instead of a wide-short one, roughly doubling
# the plotted area.
PRINT_WIDTH_IN = TEXTWIDTH_IN
BASE_FONT_SIZE = 9
FIG_WIDTH_IN = 10.0
FIG_HEIGHT_IN = 8.0
MIN_FONT_SIZE = max(BASE_FONT_SIZE, BASE_FONT_SIZE * FIG_WIDTH_IN / PRINT_WIDTH_IN)

# Real-time budget (70ms/window) referenced in the thesis text and table caption.
LATENCY_BUDGET_MS = 70.0

# Canonical display names reused from plot_classwise_f1.py; RF Features has no Pi 5 measurement
# so it's absent.
DISP = {
    "timesnet": "TimesNet", "transformer1d": "Transformer1D", "informer": "Informer",
    "mixnet": "MixNet", "wdcnn": "WDCNN", "resnet1d": "ResNet1D",
    "inception_time": "InceptionTime", "convnext1d": "ConvNeXt1D", "lstnet": "LSTNet",
    "se_cnn1d": "SE-CNN1D", "mspf_net": "MSPF-Net",
}

# Consistent per-model colors so MSPF-Net stands out against baselines.
COLORS = {
    "mspf_net": "#d62728",
    "timesnet": "#7f7f7f",
    "transformer1d": "#9467bd",
    "informer": "#8c564b",
    "mixnet": "#1f77b4",
    "wdcnn": "#2ca02c",
    "resnet1d": "#17becf",
    "inception_time": "#bcbd22",
    "convnext1d": "#e377c2",
    "lstnet": "#ff7f0e",
    "se_cnn1d": "#98df8a",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Figure 4.8: Pi 5 accuracy vs. cost.")
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Override the Pi 5 benchmark CSV to read "
            "(default: results_final/tables/pi5_benchmark_latest.csv)."
        ),
    )
    p.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Override the output figures directory (default: results/figures/phase4).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo = Path.cwd()

    csv_path = (
        Path(args.csv) if args.csv
        else repo / "results_final" / "tables" / "pi5_benchmark_latest.csv"
    )
    if not csv_path.is_file():
        print(f"  [ERROR] Pi 5 benchmark CSV not found: {csv_path}")
        return 1

    figs_dir = (
        Path(args.figures_dir) if args.figures_dir
        else repo / get_config_path("paths", "figures", default="results/figures") / "phase4"
    )
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df[df["model"] != "rf_features"].copy()

    agg = (
        df.groupby("model")
        .agg(
            avg_latency_ms=("mean_ms", "mean"),
            avg_file_macro_f1=("file_macro_f1", "mean"),
            n_datasets=("target", "nunique"),
        )
        .reset_index()
        .sort_values("avg_latency_ms")
    )
    agg["display_name"] = agg["model"].map(lambda m: DISP.get(m, m))

    print("\n" + "═" * 60)
    print("  Pi 5 cost/accuracy summary (averaged over datasets)")
    print("═" * 60)
    with pd.option_context("display.width", 120):
        print(agg[["display_name", "avg_latency_ms", "avg_file_macro_f1", "n_datasets"]]
              .to_string(index=False))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    _scale = MIN_FONT_SIZE / BASE_FONT_SIZE
    handles = []
    for _, row in agg.iterrows():
        color = COLORS.get(row["model"], "#333333")
        is_mspf = row["model"] == "mspf_net"
        marker = "*" if is_mspf else "o"
        # Marker size tuned so nearly-overlapping models stay visually distinct.
        size = (820 if is_mspf else 430) * _scale
        h = ax.scatter(
            row["avg_latency_ms"], row["avg_file_macro_f1"],
            s=size, color=color, marker=marker,
            edgecolors="#111111", linewidth=1.2, zorder=3,
            label=row["display_name"],
        )
        handles.append(h)

    ax.axvline(
        LATENCY_BUDGET_MS, color="#888888", linestyle="--", linewidth=2.0, zorder=1,
        label=f"{LATENCY_BUDGET_MS:.0f} ms/window budget",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Mean on-device latency (ms/window, log scale)", fontsize=MIN_FONT_SIZE)
    ax.set_ylabel("Average file-level\nMacro-F1 (%)", fontsize=MIN_FONT_SIZE)
    ax.set_title(
        "Accuracy vs. mean on-device latency\n(Raspberry Pi 5), avg. over D1/D2/D4/D5/D8",
        fontsize=MIN_FONT_SIZE + 2, fontweight="bold", pad=18,
    )
    ax.tick_params(axis="both", labelsize=MIN_FONT_SIZE)
    ax.set_ylim(0, 108)
    ax.grid(True, which="both", alpha=0.3)

    # Legend placed below the plot rather than as floating labels, which would collide at this
    # figure's required font size.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
        fontsize=MIN_FONT_SIZE, frameon=False, handletextpad=0.5,
        columnspacing=1.0, labelspacing=0.7,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    out_path = figs_dir / "pi5_cost_accuracy.png"
    fig.savefig(out_path, dpi=max(150.0, min(600.0, 400.0 * PRINT_WIDTH_IN / FIG_WIDTH_IN)), bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Figure saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
