"""Export and load deployment bundles for embedded (Pi 5) inference."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from mspf_embedded._paths import ensure_repo_src, repo_root

ensure_repo_src()

from mspf_net.constants import get_dataset_display
from mspf_net.config_utils import get_dataset_window_shape
from mspf_net.data.dataset import FaultDataset
from mspf_net.models.baselines.factory import create_baseline
from mspf_net.training.baseline_runner import _apply_dataset_overrides, _prepare_mspf_model_cfg, load_config
from mspf_net.training.inference_check import MODEL_CONFIG_PATHS


def load_embedded_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("embedded", data))


def _deep_merge_cfg(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_cfg(out[k], v)
        else:
            out[k] = v
    return out


def _infer_window_shape(processed_dir: str, dataset_id: int, label_space: str = "fine") -> list[int]:
    for split in ("test", "train", "val"):
        try:
            ds = FaultDataset(
                processed_dir,
                [dataset_id],
                split=split,
                label_space=label_space,
                exclude_ineligible=True,
            )
        except FileNotFoundError:
            break
        if len(ds) > 0:
            sample = ds[0][0]
            return [int(sample.shape[0]), int(sample.shape[1])]
    raise RuntimeError(f"No windows found for D{dataset_id} under {processed_dir}")


def _window_shape_from_results(results: dict, in_channels: int) -> list[int] | None:
    shape = results.get("window_shape")
    if shape and len(shape) >= 2:
        return [int(shape[0]), int(shape[1])]
    window_size = results.get("window_size")
    if window_size is not None:
        return [int(in_channels), int(window_size)]
    return None


def _resolve_window_shape(
    *,
    processed_dir: str | None,
    dataset_id: int,
    in_channels: int,
    results: dict | None = None,
) -> list[int]:
    if results is not None:
        from_results = _window_shape_from_results(results, in_channels)
        if from_results is not None:
            return from_results

    if processed_dir:
        p = Path(processed_dir)
        if p.is_dir():
            try:
                return _infer_window_shape(str(p), dataset_id)
            except RuntimeError:
                pass

    from_config = get_dataset_window_shape(dataset_id, in_channels)
    if from_config is not None:
        return from_config

    raise RuntimeError(
        f"Could not infer window shape for D{dataset_id}. "
        "Pass --processed-dir or ensure configs/config.yaml windowing.per_dataset is set."
    )


def _model_cfg_for_export(model_name: str, model_cfg: dict, results: dict | None) -> dict:
    cfg = dict(model_cfg)
    if model_name == "mspf_net":
        cfg = _prepare_mspf_model_cfg(model_name, cfg)
        if results:
            cfg.setdefault("architecture", results.get("architecture", "slim"))
            cfg.setdefault("classifier_head", results.get("classifier_head", "softmax"))
    return cfg


def _model_cfg_for_load(model_name: str, model_cfg: dict, manifest: dict) -> dict:
    cfg = dict(model_cfg)
    if model_name == "mspf_net":
        cfg = _prepare_mspf_model_cfg(model_name, cfg)
        cfg.setdefault("architecture", manifest.get("architecture", "slim"))
        cfg.setdefault("classifier_head", manifest.get("classifier_head", "softmax"))
    return cfg


def export_baseline_bundle(
    *,
    model_name: str | None = None,
    out_dir: str | Path,
    results_json: str | Path | None = None,
    checkpoint: str | Path | None = None,
    config_path: str | Path | None = None,
    dataset_id: int | None = None,
    processed_dir: str | Path | None = None,
    base_config: str | Path | None = None,
) -> Path:
    """Write checkpoint.pt, model.yaml, and manifest.json into ``out_dir``."""
    root = repo_root()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict | None = None

    if results_json is not None:
        with open(results_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        model_name = model_name or str(results.get("model", "mspf_net"))
        ckpt_src = Path(results["best_checkpoint"])
        if not ckpt_src.is_absolute():
            ckpt_src = root / ckpt_src
        dataset_id = int(results["datasets"][0] if dataset_id is None else dataset_id)
        class_names = list(results["class_names"])
        in_channels = int(results["in_channels"])
        num_classes = int(results["num_classes"])
        params = int(results.get("params", 0))
        target_label = str(results.get("target_label", get_dataset_display(dataset_id)))
        experiment_tag = str(results.get("experiment_tag", "recommended_fine"))
        architecture = str(results.get("architecture", "slim"))
        classifier_head = str(results.get("classifier_head", "softmax"))
    else:
        if checkpoint is None or dataset_id is None or model_name is None:
            raise ValueError("Provide --results-json or (--model, --checkpoint, --dataset)")
        ckpt_src = Path(checkpoint)
        if not ckpt_src.is_absolute():
            ckpt_src = root / ckpt_src
        base_cfg_path = base_config or MODEL_CONFIG_PATHS.get(
            model_name, f"configs/baselines/{model_name}.yaml"
        )
        cfg_peek = load_config(str(root / "configs/baselines/default.yaml"), str(root / base_cfg_path))
        if config_path is not None:
            override = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
            cfg_peek = _deep_merge_cfg(cfg_peek, override)
        peek = FaultDataset(
            str(processed_dir or cfg_peek["paths"]["processed_dir"]),
            [dataset_id],
            split="train",
            label_space="fine",
        )
        class_names = list(peek.class_names)
        in_channels = int(peek.input_channels)
        num_classes = int(peek.num_classes)
        params = 0
        target_label = get_dataset_display(dataset_id)
        experiment_tag = "softmax" if model_name == "mspf_net" else "recommended_fine"
        architecture = "slim"
        classifier_head = "softmax"

    model_name = str(model_name)
    if model_name == "rf_features":
        raise ValueError("rf_features has no torch checkpoint; skip export for sklearn RF baseline.")

    base_cfg_path = base_config or MODEL_CONFIG_PATHS.get(
        model_name, f"configs/baselines/{model_name}.yaml"
    )
    cfg = load_config(
        str(root / "configs/baselines/default.yaml"),
        str(root / base_cfg_path if not Path(base_cfg_path).is_absolute() else base_cfg_path),
    )

    if not ckpt_src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_src}")

    cfg, _ = _apply_dataset_overrides(cfg, model_name, "processed", [int(dataset_id)], None)
    pdir = str(processed_dir) if processed_dir else str(cfg["paths"]["processed_dir"])
    window_shape = _resolve_window_shape(
        processed_dir=pdir if Path(pdir).is_dir() else None,
        dataset_id=int(dataset_id),
        in_channels=in_channels,
        results=results,
    )

    export_model_cfg = _model_cfg_for_export(model_name, cfg.get("model", {}), results)
    if params <= 0:
        tmp = create_baseline(model_name, in_channels, num_classes, model_cfg=export_model_cfg)
        params = int(sum(p.numel() for p in tmp.parameters()))

    shutil.copy2(ckpt_src, out / "checkpoint.pt")

    export_cfg = deepcopy(cfg)
    export_cfg["model"] = export_model_cfg
    with open(out / "model.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(export_cfg, f, sort_keys=False)

    manifest = {
        "model": model_name,
        "variant": experiment_tag,
        "dataset_id": int(dataset_id),
        "target_label": target_label,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "class_names": class_names,
        "window_shape": window_shape,
        "params": params,
        "checkpoint_file": "checkpoint.pt",
        "config_file": "model.yaml",
        "label_space": "fine",
    }
    if model_name == "mspf_net":
        manifest["architecture"] = architecture
        manifest["classifier_head"] = classifier_head

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out


def export_bundle(**kwargs) -> Path:
    """Backward-compatible alias (defaults to MSPF-Net when model omitted)."""
    return export_baseline_bundle(**kwargs)


def load_bundle(bundle_dir: str | Path, device: torch.device | None = None):
    """Load manifest, rebuild model, and load checkpoint weights."""
    from mspf_net.utils.device import get_device

    bundle = Path(bundle_dir)
    with open(bundle / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(bundle / "model.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    model_name = str(manifest.get("model", "mspf_net"))
    model_cfg = _model_cfg_for_load(model_name, dict(cfg.get("model", {})), manifest)
    model = create_baseline(
        model_name,
        in_channels=int(manifest["in_channels"]),
        num_classes=int(manifest["num_classes"]),
        model_cfg=model_cfg,
    )

    if device is None:
        device = get_device("cpu")
    ckpt_path = bundle / manifest.get("checkpoint_file", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return model, manifest, cfg
