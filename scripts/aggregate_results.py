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
from mspf_net.utils.aggregate_filters import passes_scratch_filters
from mspf_net.utils.aggregate_robustness import ROBUSTNESS_TARGETS, extract_robustness_columns


def _plot_heatmap(df: pd.DataFrame, plot_path: Path, title: str, value_col: str) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    plot_df = (
        df.sort_values([value_col, "window_accuracy"], ascending=[False, False])
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate Phase 4 baseline comparison tables.")
    p.add_argument(
        "--include-fold-results",
        action="store_true",
        help="Include per-fold CV JSON artifacts (default: primary held-out test only).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo = Path.cwd()
    results_dir = repo / str(get_config_value("phase4", "results_dir", default="results/baselines"))
    tables_dir = repo / get_config_path("paths", "tables", default="results/tables")
    figs_dir = repo / get_config_path("paths", "figures", default="results/figures") / "phase4"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Scanning {results_dir} ...")
    files = sorted(results_dir.rglob("*_results.json"))
    print(f"  Found {len(files)} result files")
    rows = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not passes_scratch_filters(data, p, include_folds=args.include_fold_results):
            continue
        evaluation_mode = data.get("evaluation_mode", "primary_test")
        if evaluation_mode != "primary_test":
            continue
        split_key = "test"
        metrics = data.get(split_key)
        if metrics is None:
            continue
        file_metrics = metrics.get("file_level") or {}
        training_cfg = data.get("training_cfg", {})
        memory = data.get("memory") or {}
        rows.append(
            {
                "file": str(p),
                "mtime": p.stat().st_mtime,
                "model": data["model"],
                "target": data["target_label"],
                "data_mode": data.get("data_mode", "processed"),
                "label_space": data.get("label_space", "fine"),
                "experiment_tag": data.get("experiment_tag"),
                "unified_sampling": data.get("unified_sampling"),
                "evaluation_mode": evaluation_mode,
                "primary_comparable": bool(data.get("primary_comparable", True)),
                "metric_split": split_key,
                "window_accuracy": metrics["accuracy"],
                "window_macro_f1": metrics["macro_f1"],
                "window_macro_prec": metrics["macro_precision"],
                "window_macro_rec": metrics["macro_recall"],
                "file_accuracy": file_metrics.get("accuracy", float("nan")),
                "file_macro_f1": file_metrics.get("macro_f1", float("nan")),
                "file_macro_prec": file_metrics.get("macro_precision", float("nan")),
                "file_macro_rec": file_metrics.get("macro_recall", float("nan")),
                "n_files": file_metrics.get("n_files", float("nan")),
                "inf_ms_win": data.get("inference_ms_per_window", float("nan")),
                "batch_size": data.get("batch_size", training_cfg.get("batch_size")),
                "effective_batch_size": data.get(
                    "effective_batch_size", training_cfg.get("effective_batch_size")
                ),
                "peak_train_gpu_mb": memory.get("peak_train_gpu_mb"),
                "peak_inference_gpu_mb": memory.get("peak_inference_gpu_mb"),
                "peak_rss_mb": memory.get("peak_rss_mb"),
                **extract_robustness_columns(data),
            }
        )
    if not rows:
        print("  No result files found.")
        return 0
    df = pd.DataFrame(rows).sort_values("mtime").drop_duplicates(
        ["model", "target", "data_mode", "label_space", "experiment_tag", "unified_sampling", "evaluation_mode"], keep="last"
    )
    primary_df = df.copy()

    primary_csv = tables_dir / "phase4_baseline_comparison.csv"
    primary_df.to_csv(primary_csv, index=False)

    rob_df = pd.DataFrame()
    if not primary_df.empty and "robustness_window_macro_f1" in primary_df.columns:
        rob_df = primary_df[
            primary_df["target"].isin(ROBUSTNESS_TARGETS)
            & primary_df["robustness_window_macro_f1"].notna()
        ].copy()
    rob_csv = tables_dir / "phase4_robustness_comparison.csv"
    if not rob_df.empty:
        rob_df.to_csv(rob_csv, index=False)

    if not primary_df.empty:
        print("\n" + "═" * 72)
        print("  Baseline Comparison — Primary Benchmark")
        print("═" * 72)
        display = primary_df[
            [
                "model",
                "target",
                "label_space",
                "experiment_tag",
                "window_accuracy",
                "window_macro_f1",
                "file_accuracy",
                "file_macro_f1",
                "n_files",
                "inf_ms_win",
                "batch_size",
                "peak_train_gpu_mb",
                "robustness_window_macro_f1",
                "robustness_file_macro_f1",
            ]
        ].sort_values(
            ["window_accuracy", "window_macro_f1"], ascending=[False, False]
        )
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(display.to_string(index=False))
        primary_plot = figs_dir / "baseline_comparison_window.png"
        _plot_heatmap(primary_df, primary_plot, "Primary benchmark comparison by Window Macro-F1 (%)", "window_macro_f1")
        file_plot = figs_dir / "baseline_comparison_file.png"
        file_df = primary_df.dropna(subset=["file_macro_f1"])
        _plot_heatmap(file_df, file_plot, "Primary benchmark comparison by File Macro-F1 (%)", "file_macro_f1")
        best_window = primary_df.sort_values(["window_macro_f1", "window_accuracy"], ascending=False).iloc[0]
        best_file = file_df.sort_values(["file_macro_f1", "file_accuracy"], ascending=False).iloc[0] if not file_df.empty else None
        print(f"\n  Primary CSV saved → {primary_csv}")
        if not rob_df.empty:
            print(f"  Robustness CSV    → {rob_csv}")
        print(f"  Window plot saved → {primary_plot}")
        if not file_df.empty:
            print(f"  File plot saved   → {file_plot}")
        print(
            f"\n  Best window-level model : {best_window['model']}  "
            f"Acc={best_window['window_accuracy']:.1f}%  F1={best_window['window_macro_f1']:.1f}%"
        )
        if best_file is not None:
            print(
                f"  Best file-level model   : {best_file['model']}  "
                f"Acc={best_file['file_accuracy']:.1f}%  F1={best_file['file_macro_f1']:.1f}%"
            )
    else:
        print("\n  No primary benchmark runs found.")

    if not rob_df.empty:
        import subprocess

        plot_script = PROJECT_ROOT / "scripts" / "plot_robustness_comparison.py"
        if plot_script.is_file():
            subprocess.run(
                [sys.executable, str(plot_script), "--tables-dir", str(tables_dir)],
                cwd=PROJECT_ROOT,
                check=False,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
