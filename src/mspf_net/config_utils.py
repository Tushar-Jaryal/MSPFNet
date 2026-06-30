from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_global_config() -> dict[str, Any]:
    if not GLOBAL_CONFIG_PATH.exists():
        return {}
    with open(GLOBAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config_path(*keys: str, default: str) -> str:
    value: Any = load_global_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return str(value)


def get_config_list(*keys: str, default: list[Any]) -> list[Any]:
    value: Any = load_global_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return list(default)
        value = value[key]
    return list(value) if isinstance(value, list) else list(default)


def get_config_value(*keys: str, default: Any) -> Any:
    value: Any = load_global_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def get_active_dataset_ids() -> list[int]:
    return [int(v) for v in get_config_list("datasets", "active_thesis", default=[1, 2, 4, 5, 8])]


def get_primary_scratch_dataset_ids() -> list[int]:
    return [int(v) for v in get_config_list("datasets", "primary_scratch", default=get_active_dataset_ids())]


def get_unified_group_ids() -> list[str]:
    return [str(v) for v in get_config_list("unified", "groups", default=["bearing_md", "gearbox_md"])]


def get_processed_dir_for_policy(policy: str) -> Path:
    """Return processed-data root for a window policy (recommended vs fixed)."""
    normalized = str(policy or "recommended").strip().lower()
    if normalized in {"recommended", "default", "rec"}:
        rel = get_config_path("paths", "data_processed", default="data/processed")
    elif normalized in {"fixed", "fix"}:
        rel = get_config_path("paths", "data_processed_fixed", default="data/processed_fixed")
    else:
        raise ValueError(f"Unknown window policy {policy!r}; expected 'recommended' or 'fixed'")
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_phase4_epochs() -> int:
    return int(get_config_value("phase4", "epochs", default=50))


def get_fs_target_hz() -> int:
    return int(get_config_value("preprocessing", "fs_target", default=10_000))


def get_dataset_window_shape(dataset_id: int, in_channels: int) -> list[int] | None:
    """Return ``[channels, window_size]`` from global windowing config."""
    per_ds = get_config_value("windowing", "per_dataset", default={}) or {}
    entry = per_ds.get(str(dataset_id)) or per_ds.get(int(dataset_id))
    if isinstance(entry, dict) and entry.get("window_size") is not None:
        return [int(in_channels), int(entry["window_size"])]
    default_size = get_config_value("windowing", "default_window_size", default=None)
    if default_size is not None:
        return [int(in_channels), int(default_size)]
    return None


def _fill_missing_training_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill training keys from phase4 defaults without overwriting model YAML."""
    out = dict(cfg)
    out.setdefault("training", {})
    phase4_cfg = load_global_config().get("phase4", {})
    if not isinstance(phase4_cfg, dict):
        phase4_cfg = {}
    for key, default in (
        ("epochs", get_phase4_epochs()),
        ("patience", int(get_config_value("phase4", "patience", default=10))),
        ("batch_size", int(get_config_value("phase4", "batch_size", default=128))),
        ("lr", float(get_config_value("phase4", "lr", default=1e-3))),
        ("weight_decay", float(get_config_value("phase4", "weight_decay", default=1e-4))),
        ("scheduler", str(get_config_value("phase4", "scheduler", default="cosine"))),
        ("num_workers", int(get_config_value("phase4", "num_workers", default=4))),
        ("unified_sampling", str(get_config_value("phase4", "unified_sampling", default="dataset_balanced"))),
        ("use_class_weights", bool(get_config_value("phase4", "use_class_weights", default=True))),
        ("device", str(get_config_value("phase4", "device", default="auto"))),
    ):
        if key not in out["training"]:
            out["training"][key] = phase4_cfg.get(key, default)
    return out
