from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.config_utils import get_config_path, get_config_value
from mspf_net.utils.aggregate_filters import is_primary_scratch_target, is_processed_fine_record


def _plot_heatmap(df: pd.DataFrame, plot_path: Path, title: str, value_col: str) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    plot_df = (
        df.sort_values([value_col, "accuracy_mean"], ascending=[False, False])
        .drop_duplicates(["model", "target"], keep="first")
    )
    pivot = plot_df.pivot(index="model", columns="target", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(pivot.columns)), max(4, 0.5 * len(pivot.index) + 1)))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = "NA"
                color = "white"
            else:
                text = f"{val:.1f}"
                color = "white" if val < 60 else "#111111"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _metric_block(summary: dict | None, split_key: str, prefix: str = "") -> dict[str, float]:
    if not summary or split_key not in summary or summary[split_key] is None:
        return {}
    block = summary[split_key]
    out = {}
    for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
        stats = block.get(metric)
        if not stats:
            continue
        out[f"{prefix}{metric}_mean"] = float(stats["mean"])
        out[f"{prefix}{metric}_std"] = float(stats["std"])
        out[f"{prefix}{metric}_min"] = float(stats["min"])
        out[f"{prefix}{metric}_max"] = float(stats["max"])
    file_block = block.get("file_level")
    if file_block:
        for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
            stats = file_block.get(metric)
            if not stats:
                continue
            out[f"file_{prefix}{metric}_mean"] = float(stats["mean"])
            out[f"file_{prefix}{metric}_std"] = float(stats["std"])
            out[f"file_{prefix}{metric}_min"] = float(stats["min"])
            out[f"file_{prefix}{metric}_max"] = float(stats["max"])
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate grouped cross-validation summaries from *_cv*_summary.json artifacts. "
            "Per-fold *_results.json files are not scanned; re-run with grouped CV training to regenerate summaries."
        )
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo = Path.cwd()
    results_dir = repo / str(get_config_value("phase4", "results_dir", default="results/baselines"))
    tables_dir = repo / get_config_path("paths", "tables", default="results/tables")
    figs_dir = repo / get_config_path("paths", "figures", default="results/figures") / "phase4_cv"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Scanning {results_dir} for CV summaries ...")
    files = sorted(results_dir.rglob("*_cv*_summary.json"))
    print(f"  Found {len(files)} CV summary files")

    rows = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        evaluation_mode = data.get("evaluation_mode")
        if evaluation_mode != "crossval_summary":
            continue
        if not is_processed_fine_record(data):
            continue
        if not is_primary_scratch_target(str(data.get("target_label", ""))):
            continue
        summary = data.get("summary", {})
        row = {
            "file": str(p),
            "mtime": p.stat().st_mtime,
            "model": data["model"],
            "target": data["target_label"],
            "data_mode": data.get("data_mode", "processed"),
            "group_id": data.get("group_id"),
            "cv_folds": int(data.get("cv_folds", 0)),
            "cv_val_ratio": float(data.get("cv_val_ratio", 0.0)),
            "label_space": data.get("label_space", "fine"),
            "unified_sampling": data.get("unified_sampling"),
            "experiment_tag": data.get("experiment_tag"),
            "evaluation_mode": evaluation_mode,
            "window_size": data.get("window_size"),
            "window_shape": data.get("window_shape"),
            "in_channels": data.get("in_channels"),
        }
        row.update(_metric_block(summary, "test"))
        row.update(_metric_block(summary, "val", prefix="val_"))
        rows.append(row)

    if not rows:
        print("  No CV summaries found.")
        return 0

    df = pd.DataFrame(rows).sort_values("mtime").drop_duplicates(
        [
            "model",
            "target",
            "data_mode",
            "cv_folds",
            "label_space",
            "unified_sampling",
            "experiment_tag",
            "evaluation_mode",
        ],
        keep="last",
    )

    csv_path = tables_dir / "phase4_cv_comparison.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "═" * 72)
    print("  Cross-Validation Comparison")
    print("═" * 72)
    display_cols = [
        "model",
        "target",
        "data_mode",
        "evaluation_mode",
        "cv_folds",
        "window_size",
        "accuracy_mean",
        "macro_f1_mean",
        "macro_f1_std",
        "file_accuracy_mean",
        "file_macro_f1_mean",
        "file_macro_f1_std",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].sort_values(
        ["macro_f1_mean", "accuracy_mean"], ascending=[False, False]
    )
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(display_df.to_string(index=False))

    window_plot = figs_dir / "cv_comparison_window.png"
    _plot_heatmap(df, window_plot, "Cross-validation comparison by Window Macro-F1 mean (%)", "macro_f1_mean")

    file_df = df.dropna(subset=["file_macro_f1_mean"]) if "file_macro_f1_mean" in df.columns else pd.DataFrame()
    if not file_df.empty:
        file_plot = figs_dir / "cv_comparison_file.png"
        _plot_heatmap(file_df, file_plot, "Cross-validation comparison by File Macro-F1 mean (%)", "file_macro_f1_mean")
    else:
        file_plot = None

    best_window = df.sort_values(["macro_f1_mean", "accuracy_mean"], ascending=False).iloc[0]
    print(f"\n  CV CSV saved     → {csv_path}")
    print(f"  Window plot saved → {window_plot}")
    if file_plot is not None:
        print(f"  File plot saved   → {file_plot}")
    print(
        f"\n  Best CV window-level model : {best_window['model']} on {best_window['target']}  "
        f"Acc={best_window['accuracy_mean']:.1f}%  F1={best_window['macro_f1_mean']:.1f}%"
    )

    if "file_macro_f1_mean" in df.columns and df["file_macro_f1_mean"].notna().any():
        best_file = df.dropna(subset=["file_macro_f1_mean"]).sort_values(
            ["file_macro_f1_mean", "file_accuracy_mean"], ascending=False
        ).iloc[0]
        print(
            f"  Best CV file-level model   : {best_file['model']} on {best_file['target']}  "
            f"Acc={best_file['file_accuracy_mean']:.1f}%  F1={best_file['file_macro_f1_mean']:.1f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
