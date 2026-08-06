"""Shared filters for Phase 4 / MSPF result aggregation."""

from __future__ import annotations

from pathlib import Path

from mspf_net.constants import PRIMARY_SCRATCH_DATASETS, get_dataset_display

PRIMARY_SCRATCH_TARGETS = {get_dataset_display(ds_id) for ds_id in PRIMARY_SCRATCH_DATASETS}


def is_primary_scratch_target(target: str) -> bool:
    return str(target).strip() in PRIMARY_SCRATCH_TARGETS


def is_processed_fine_record(data: dict) -> bool:
    return data.get("data_mode", "processed") == "processed" and data.get("label_space", "fine") == "fine"


def is_fold_result_path(path: Path) -> bool:
    return any(part.startswith("fold_") for part in path.parts) or "_fold" in path.stem


def passes_scratch_filters(data: dict, path: Path, *, include_folds: bool = False) -> bool:
    if not include_folds and is_fold_result_path(path):
        return False
    if not is_processed_fine_record(data):
        return False
    return is_primary_scratch_target(str(data.get("target_label", "")))
