"""Helpers for variable-speed and noise-SNR robustness aggregation."""

from __future__ import annotations

from typing import Any

from mspf_net.config_utils import get_config_list

ROBUSTNESS_TARGETS = frozenset({"D5", "D8"})


def get_noise_snr_levels() -> list[float]:
    levels = get_config_list("evaluation", "noise_snr_levels", default=[10, 5, 0])
    return [float(v) for v in levels]


def _snr_column(snr_db: float) -> str:
    label = str(int(snr_db)) if snr_db == int(snr_db) else str(snr_db).replace(".", "p").replace("-", "m")
    return f"noise_macro_f1_snr{label}"


def compact_eval_metrics(block: dict | None) -> dict[str, Any] | None:
    """Drop heavy arrays; keep scalar metrics and optional file_level summary."""
    if not block:
        return None
    out = {
        "accuracy": block.get("accuracy"),
        "macro_f1": block.get("macro_f1"),
        "macro_precision": block.get("macro_precision"),
        "macro_recall": block.get("macro_recall"),
        "loss": block.get("loss"),
    }
    file_level = block.get("file_level")
    if file_level:
        out["file_level"] = {
            "n_files": file_level.get("n_files"),
            "accuracy": file_level.get("accuracy"),
            "macro_f1": file_level.get("macro_f1"),
            "macro_precision": file_level.get("macro_precision"),
            "macro_recall": file_level.get("macro_recall"),
        }
    return out


def extract_robustness_columns(data: dict) -> dict[str, float]:
    """Extract flat CSV columns for variable-speed and noise-SNR robustness."""
    target = str(data.get("target_label", "")).upper()
    out: dict[str, float] = {}

    rob = data.get("robustness")
    if rob and target in ROBUSTNESS_TARGETS:
        out["robustness_window_macro_f1"] = float(rob.get("macro_f1", float("nan")))
        out["robustness_window_accuracy"] = float(rob.get("accuracy", float("nan")))
        file_level = rob.get("file_level") or {}
        out["robustness_file_macro_f1"] = float(file_level.get("macro_f1", float("nan")))
        out["robustness_file_accuracy"] = float(file_level.get("accuracy", float("nan")))
        out["robustness_n_files"] = float(file_level.get("n_files", float("nan")))
    else:
        out["robustness_window_macro_f1"] = float("nan")
        out["robustness_window_accuracy"] = float("nan")
        out["robustness_file_macro_f1"] = float("nan")
        out["robustness_file_accuracy"] = float("nan")
        out["robustness_n_files"] = float("nan")

    noise = data.get("noise_robustness") or {}
    by_snr = noise.get("by_snr") or {}
    for snr_db in get_noise_snr_levels():
        key = str(int(snr_db)) if snr_db == int(snr_db) else str(snr_db)
        block = by_snr.get(key) or by_snr.get(str(snr_db))
        col = _snr_column(snr_db)
        out[col] = float(block.get("macro_f1", float("nan"))) if block else float("nan")

    return out


def noise_snr_column_names() -> list[str]:
    return [_snr_column(s) for s in get_noise_snr_levels()]
