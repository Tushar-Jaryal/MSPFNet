"""Load checkpoints and measure inference for Phase 4 baseline models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mspf_net.config_utils import PROJECT_ROOT, get_dataset_window_shape
from mspf_net.models.baselines.factory import create_baseline
from mspf_net.training.baseline_runner import (
    _apply_dataset_overrides,
    _prepare_mspf_model_cfg,
    load_config,
)
from mspf_net.utils.device import benchmark_latency, device_info, get_device, profile_flops

PHASE4_MODELS = [
    "timesnet",
    "transformer1d",
    "informer",
    "mixnet",
    "wdcnn",
    "resnet1d",
    "inception_time",
    "convnext1d",
    "lstnet",
    "se_cnn1d",
    "rf_features",
    "mspf_net",
]

MODEL_CONFIG_PATHS = {
    "timesnet": "configs/baselines/timesnet.yaml",
    "transformer1d": "configs/baselines/transformer1d.yaml",
    "informer": "configs/baselines/informer.yaml",
    "mixnet": "configs/baselines/mixnet.yaml",
    "wdcnn": "configs/baselines/wdcnn.yaml",
    "resnet1d": "configs/baselines/resnet1d.yaml",
    "inception_time": "configs/baselines/inception_time.yaml",
    "convnext1d": "configs/baselines/convnext1d.yaml",
    "lstnet": "configs/baselines/lstnet.yaml",
    "se_cnn1d": "configs/baselines/se_cnn1d.yaml",
    "rf_features": "configs/baselines/rf_features.yaml",
    "mspf_net": "configs/baselines/mspf_net.yaml",
}

DEFAULT_EXPERIMENT_TAGS = {
    "mspf_net": "softmax",
}


def _resolve_window_shape(dataset_id: int, in_channels: int, results: dict | None) -> list[int]:
    if results:
        shape = results.get("window_shape")
        if shape and len(shape) >= 2:
            return [int(shape[0]), int(shape[1])]
        window_size = results.get("window_size")
        if window_size is not None:
            return [int(in_channels), int(window_size)]
    from_config = get_dataset_window_shape(dataset_id, in_channels)
    if from_config is not None:
        return from_config
    raise RuntimeError(f"Could not resolve window shape for D{dataset_id}")


def find_results_json(
    results_dir: Path,
    model_name: str,
    dataset_id: int,
    *,
    experiment_tag: str | None = None,
) -> tuple[Path, dict] | None:
    """Pick the newest primary-test fine results JSON for a model/dataset."""
    search_dir = results_dir / model_name / "processed" / f"d{dataset_id}"
    if not search_dir.is_dir():
        return None

    preferred_tag = experiment_tag or DEFAULT_EXPERIMENT_TAGS.get(model_name, "recommended_fine")
    candidates: list[tuple[float, Path, dict]] = []
    for path in search_dir.glob("*_results.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("evaluation_mode", "primary_test") != "primary_test":
            continue
        if data.get("label_space", "fine") != "fine":
            continue
        tag = str(data.get("experiment_tag") or "")
        priority = 0
        if tag == preferred_tag:
            priority = 2
        elif tag == "recommended_fine":
            priority = 1
        candidates.append((priority + path.stat().st_mtime / 1e20, path, data))

    if not candidates:
        return None
    _, path, data = max(candidates, key=lambda item: item[0])
    return path, data


def _build_model_from_results(
    model_name: str,
    dataset_id: int,
    results: dict,
    device: torch.device,
) -> nn.Module:
    cfg_path = MODEL_CONFIG_PATHS[model_name]
    cfg = load_config("configs/baselines/default.yaml", str(PROJECT_ROOT / cfg_path))
    cfg, _ = _apply_dataset_overrides(cfg, model_name, "processed", [dataset_id], None)
    in_channels = int(results["in_channels"])
    num_classes = int(results["num_classes"])
    model_cfg = _prepare_mspf_model_cfg(model_name, cfg["model"])
    if model_name == "mspf_net":
        model_cfg.setdefault("architecture", results.get("architecture", "slim"))
        model_cfg.setdefault("classifier_head", results.get("classifier_head", "softmax"))
    model = create_baseline(model_name, in_channels, num_classes, model_cfg=model_cfg)
    ckpt_path = Path(results["best_checkpoint"])
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return model.eval().to(device)


def check_model_inference(
    model_name: str,
    dataset_id: int,
    *,
    results_dir: Path,
    device: torch.device | None = None,
    experiment_tag: str | None = None,
    n_runs: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    """Run a forward-pass smoke test and latency/FLOPs profile for one model."""
    device = device or get_device("auto")
    found = find_results_json(results_dir, model_name, dataset_id, experiment_tag=experiment_tag)
    if found is None:
        return {
            "model": model_name,
            "dataset_id": dataset_id,
            "status": "missing_results",
            "error": f"No primary-test results under {results_dir / model_name / 'processed' / f'd{dataset_id}'}",
        }

    results_path, results = found
    if model_name == "rf_features":
        return {
            "model": model_name,
            "dataset_id": dataset_id,
            "target": results.get("target_label"),
            "experiment_tag": results.get("experiment_tag"),
            "status": "sklearn_rf",
            "results_json": str(results_path),
            "inference_ms_per_window": results.get("inference_ms_per_window"),
            "note": "sklearn RF — no torch checkpoint; see training JSON inference_ms_per_window",
        }

    try:
        model = _build_model_from_results(model_name, dataset_id, results, device)
        in_channels = int(results["in_channels"])
        window_shape = _resolve_window_shape(dataset_id, in_channels, results)
        input_shape = (1, int(window_shape[0]), int(window_shape[1]))

        with torch.no_grad():
            dummy = torch.randn(*input_shape, device=device, dtype=torch.float32)
            logits = model(dummy)
        expected_classes = int(results["num_classes"])
        if tuple(logits.shape) != (1, expected_classes):
            raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)} != (1, {expected_classes})")

        latency = benchmark_latency(model, input_shape, device=device, n_runs=n_runs, warmup=warmup)
        flops = profile_flops(model, input_shape)
        params = int(sum(p.numel() for p in model.parameters()))

        return {
            "model": model_name,
            "dataset_id": dataset_id,
            "target": results.get("target_label"),
            "experiment_tag": results.get("experiment_tag"),
            "status": "ok",
            "results_json": str(results_path),
            "device": str(device),
            "device_info": device_info(device),
            "window_shape": window_shape,
            "input_shape": list(input_shape),
            "logits_shape": list(logits.shape),
            "params": params,
            "latency": latency,
            "flops": flops,
            "training_inference_ms": results.get("inference_ms_per_window"),
            "mean_ms": latency.get("mean_ms"),
            "target_ok": latency.get("target_ok"),
        }
    except Exception as exc:
        if device.type == "mps" and model_name == "mspf_net":
            cpu = torch.device("cpu")
            retry = check_model_inference(
                model_name,
                dataset_id,
                results_dir=results_dir,
                device=cpu,
                experiment_tag=experiment_tag,
                n_runs=n_runs,
                warmup=warmup,
            )
            if retry.get("status") == "ok":
                retry["note"] = f"MPS fallback: {exc}"
                return retry
        return {
            "model": model_name,
            "dataset_id": dataset_id,
            "target": results.get("target_label"),
            "experiment_tag": results.get("experiment_tag"),
            "status": "error",
            "results_json": str(results_path),
            "error": str(exc),
        }
