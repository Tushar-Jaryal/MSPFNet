from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.config_utils import get_config_path, get_config_value, get_primary_scratch_dataset_ids
from mspf_net.constants import get_dataset_display
from mspf_net.utils.aggregate_robustness import ROBUSTNESS_TARGETS, extract_robustness_columns


VARIANT_LABELS = {
    "softmax": "MSPF-Net (Softmax)",
    "rf": "MSPF-Net (RF)",
    "moe": "MSPF-Net MoE (CWT + non-periodic)",
    # Ablation labels use chapter3/appendix Stage 1-4 naming; leading minus = removal, arrow = replacement.
    "no_periodic_path": "−TF path (Stage 3)",
    "no_nonstationary_path": "−NS path (Stage 3)",
    "no_se": "−SE block (TF path)",
    "equal_path_fusion": "Stage 4 gate → equal",
    "simple_channel_mixer": "Stage 1 stem → linear",
    "mean_channel_pooling": "Chan. attn → mean pool",
}

FULL_VARIANTS = {"softmax", "rf", "moe"}
ABLATION_VARIANTS = {
    "no_periodic_path",
    "no_nonstationary_path",
    "no_se",
    "mean_channel_pooling",
    "equal_path_fusion",
    "simple_channel_mixer",
}


def _scratch_targets() -> set[str]:
    return {get_dataset_display(int(ds)).upper() for ds in get_primary_scratch_dataset_ids()}


def _should_include_file(path: Path, include_folds: bool) -> bool:
    from mspf_net.utils.aggregate_filters import is_fold_result_path

    if not include_folds and is_fold_result_path(path):
        return False
    return True


def _base_model_key(row: dict) -> str:
    tag = row.get("experiment_tag")
    architecture = str(row.get("architecture", "slim")).lower()
    classifier = str(row.get("classifier_head", "softmax")).lower()
    if tag == "moe" or architecture == "moe":
        return "moe"
    if tag == "rf" or classifier == "rf":
        return "rf"
    return "softmax"


def _variant_key(row: dict) -> str:
    tag = row.get("experiment_tag")
    if tag in FULL_VARIANTS:
        return str(tag)
    base = _base_model_key(row)
    if tag in ABLATION_VARIANTS:
        if base == "softmax":
            return str(tag)
        return f"{base}__{tag}"
    classifier = str(row.get("classifier_head", "softmax")).lower()
    if tag in {"recommended_fine", "fixed_fine"} or (
        isinstance(tag, str) and tag.endswith("_fine") and tag not in VARIANT_LABELS
    ):
        return "rf" if classifier == "rf" else "softmax"
    if tag:
        return str(tag)
    return "rf" if classifier == "rf" else "softmax"


def _variant_label(row: dict) -> str:
    variant = _variant_key(row)
    if variant in VARIANT_LABELS:
        return VARIANT_LABELS[variant]
    if "__" in variant:
        base, tag = variant.split("__", 1)
        base_name = {"rf": "RF", "moe": "MoE"}.get(base, base)
        ablation = VARIANT_LABELS.get(tag, tag)
        return f"MSPF-Net ({base_name}) {ablation}"
    return VARIANT_LABELS.get(variant, variant)


# Minimum printed text size (pt); these heatmaps are placed as a sidewaysfigure at ~8.92in printed width.
TEXTWIDTH_IN = 5.771653543307086
TEXTHEIGHT_IN = 9.692913385826772
# Printed as a normal portrait float, not a sidewaysfigure, so it shares a page with surrounding text.
PRINT_WIDTH_COMPARISON_IN = TEXTWIDTH_IN
PRINT_WIDTH_ABLATION_IN = TEXTWIDTH_IN
BASE_FONT_SIZE = 9


def _font_for_print(fig_width_in: float, print_width_in: float, target_pt: float = BASE_FONT_SIZE) -> float:
    return max(target_pt, target_pt * fig_width_in / print_width_in)


def _plot_heatmap(
    df: pd.DataFrame, plot_path: Path, title: str, value_col: str, print_width_in: float = PRINT_WIDTH_COMPARISON_IN
) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    plot_df = (
        df.sort_values([value_col, "window_macro_f1"], ascending=[False, False], na_position="last")
        .drop_duplicates(["variant_label", "target"], keep="first")
    )
    if plot_df.empty:
        return
    pivot = plot_df.pivot_table(
        index="variant_label", columns="target", values=value_col, aggfunc="first"
    ).sort_index()
    heatmap_w = 1.8 * len(pivot.columns)
    fig_h = max(6, 0.95 * len(pivot.index) + 2)
    # Reserve explicit width for the longest ytick label so tight_layout doesn't shrink heatmap cells
    # and collide adjacent text.
    max_label_chars = max((len(str(v)) for v in pivot.index), default=0)
    fig_w = max(11, heatmap_w)
    for _ in range(3):
        font = _font_for_print(fig_w, print_width_in)
        label_w_in = max_label_chars * font * 0.52 / 72.0
        fig_w = max(11, heatmap_w + label_w_in + 1.6)
    font = _font_for_print(fig_w, print_width_in)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=font)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=font)
    ax.set_title(title, fontsize=font + 2, fontweight="bold", pad=14)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                text = "NA"
                color = "white"
            else:
                text = f"{val:.1f}"
                color = "white" if val < 60 else "#111111"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=font)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=max(150.0, min(600.0, 400.0 * print_width_in / fig_w)), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate MSPF-Net baseline result JSONs.")
    parser.add_argument(
        "--include-fold-results",
        action="store_true",
        help="Include per-fold cross-validation JSON artifacts (default: primary_test only).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Override the results directory to scan, without the trailing 'mspf_net' "
            "(default: config paths.phase4.results_dir, e.g. results/baselines)."
        ),
    )
    args = parser.parse_args()

    repo = Path.cwd()
    base_results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else repo / str(get_config_value("phase4", "results_dir", default="results/baselines"))
    )
    results_dir = base_results_dir / "mspf_net"
    tables_dir = repo / get_config_path("paths", "tables", default="results/tables")
    figs_dir = repo / get_config_path("paths", "figures", default="results/figures") / "mspf_net"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    scratch_targets = _scratch_targets()
    files = sorted(results_dir.rglob("*_results.json"))
    print(f"\n  Scanning {results_dir} ...")
    print(f"  Found {len(files)} MSPF-Net result files")

    rows = []
    for p in files:
        if not _should_include_file(p, include_folds=args.include_fold_results):
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("model") != "mspf_net":
            continue
        evaluation_mode = data.get("evaluation_mode", "primary_test")
        if evaluation_mode != "primary_test":
            continue
        if data.get("data_mode", "processed") != "processed":
            continue
        if data.get("label_space", "fine") != "fine":
            continue
        target = str(data.get("target_label", "")).upper()
        if target not in scratch_targets:
            continue
        metrics = data.get("test")
        if metrics is None:
            continue
        file_metrics = metrics.get("file_level") or {}
        variant = _variant_key(data)
        training_cfg = data.get("training_cfg", {})
        memory = data.get("memory") or {}
        rows.append(
            {
                "file": str(p),
                "mtime": p.stat().st_mtime,
                "target": data["target_label"],
                "variant": variant,
                "variant_label": _variant_label(data),
                "experiment_tag": data.get("experiment_tag"),
                "run_suffix": data.get("run_suffix"),
                "data_mode": data.get("data_mode", "processed"),
                "label_space": data.get("label_space", "fine"),
                "evaluation_mode": evaluation_mode,
                "classifier_head": data.get("classifier_head", "softmax"),
                "architecture": data.get("architecture", "slim"),
                "epochs": training_cfg.get("epochs"),
                "epochs_ran": data.get("epochs_ran"),
                "batch_size": data.get("batch_size", training_cfg.get("batch_size")),
                "effective_batch_size": data.get(
                    "effective_batch_size", training_cfg.get("effective_batch_size")
                ),
                "peak_train_gpu_mb": memory.get("peak_train_gpu_mb"),
                "peak_inference_gpu_mb": memory.get("peak_inference_gpu_mb"),
                "peak_rss_mb": memory.get("peak_rss_mb"),
                "best_epoch": data.get("best_epoch"),
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
                "params": data.get("params", float("nan")),
                "primary_comparable": bool(data.get("primary_comparable", True)),
                **extract_robustness_columns(data),
            }
        )

    if not rows:
        print("  No MSPF-Net primary-test scratch results found.")
        return 0

    df = pd.DataFrame(rows).sort_values("mtime").drop_duplicates(
        [
            "target",
            "variant",
            "classifier_head",
            "architecture",
            "data_mode",
            "label_space",
            "evaluation_mode",
            "run_suffix",
        ],
        keep="last",
    )

    full_df = df[df["variant"].isin(FULL_VARIANTS)].copy()
    ablation_df = df[
        df["experiment_tag"].isin(ABLATION_VARIANTS)
        | df["variant"].str.contains("__", regex=False, na=False)
    ].copy()

    all_csv = tables_dir / "mspf_net_all_results.csv"
    full_csv = tables_dir / "mspf_net_comparison.csv"
    ablation_csv = tables_dir / "mspf_net_ablation.csv"
    df.to_csv(all_csv, index=False)
    full_df.to_csv(full_csv, index=False)
    ablation_df.to_csv(ablation_csv, index=False)

    rob_df = pd.DataFrame()
    if not df.empty and "robustness_window_macro_f1" in df.columns:
        rob_df = df[
            df["target"].isin(ROBUSTNESS_TARGETS)
            & df["robustness_window_macro_f1"].notna()
        ].copy()
    rob_csv = tables_dir / "mspf_net_robustness_comparison.csv"
    if not rob_df.empty:
        rob_df.to_csv(rob_csv, index=False)

    full_plot = None
    full_file_plot = None
    if not full_df.empty:
        full_plot = figs_dir / "mspf_net_comparison_window.png"
        _plot_heatmap(full_df, full_plot, "MSPF-Net comparison by Window Macro-F1 (%)", "window_macro_f1")
        file_df = full_df.dropna(subset=["file_macro_f1"])
        if not file_df.empty:
            full_file_plot = figs_dir / "mspf_net_comparison_file.png"
            _plot_heatmap(file_df, full_file_plot, "MSPF-Net comparison by File Macro-F1 (%)", "file_macro_f1")

    ablation_plot = None
    ablation_file_plot = None
    if not ablation_df.empty:
        ablation_plot = figs_dir / "mspf_net_ablation_window.png"
        _plot_heatmap(
            ablation_df, ablation_plot, "MSPF-Net ablation by Window Macro-F1 (%)", "window_macro_f1",
            print_width_in=PRINT_WIDTH_ABLATION_IN,
        )
        ablation_file_df = ablation_df.dropna(subset=["file_macro_f1"])
        if not ablation_file_df.empty:
            ablation_file_plot = figs_dir / "mspf_net_ablation_file.png"
            _plot_heatmap(
                ablation_file_df, ablation_file_plot, "MSPF-Net ablation by File Macro-F1 (%)", "file_macro_f1",
                print_width_in=PRINT_WIDTH_ABLATION_IN,
            )

    print("\n" + "═" * 72)
    print("  MSPF-Net Aggregation (primary scratch, processed fine)")
    print("═" * 72)
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(
            df[
                [
                    "variant_label",
                    "target",
                    "evaluation_mode",
                    "run_suffix",
                    "epochs",
                    "batch_size",
                    "effective_batch_size",
                    "peak_train_gpu_mb",
                    "window_macro_f1",
                    "file_macro_f1",
                    "robustness_window_macro_f1",
                    "robustness_file_macro_f1",
                    "inf_ms_win",
                ]
            ]
            .sort_values(["variant_label", "target"])
            .to_string(index=False)
        )
    print(f"\n  All results CSV        → {all_csv}")
    print(f"  Full model CSV         → {full_csv}")
    print(f"  Ablation CSV           → {ablation_csv}")
    if not rob_df.empty:
        print(f"  Robustness CSV         → {rob_csv}")
    if full_plot is not None:
        print(f"  Full plot              → {full_plot}")
    if full_file_plot is not None:
        print(f"  Full file plot         → {full_file_plot}")
    if ablation_plot is not None:
        print(f"  Ablation plot          → {ablation_plot}")
    if ablation_file_plot is not None:
        print(f"  Ablation file plot     → {ablation_file_plot}")

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
