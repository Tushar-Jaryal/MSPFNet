"""Run embedded latency, FLOPs, memory, and accuracy benchmarks."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from mspf_embedded._paths import ensure_repo_src, repo_root
from mspf_embedded.bundle import load_bundle, load_embedded_config

ensure_repo_src()

from mspf_net.data.dataset import build_dataloaders
from mspf_net.utils.device import device_info, get_device, profile_flops


def _peak_rss_during_eval(model, loader, device, class_names):
    try:
        import psutil

        proc = psutil.Process()
        peak = proc.memory_info().rss
    except ImportError:
        proc = None
        peak = 0

    import numpy as np
    import torch

    from mspf_net.training.metrics import classification_metrics, metrics_from_confusion_matrix

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    file_logit_sums: dict[str, np.ndarray] = {}
    file_label_counts: dict[str, np.ndarray] = {}
    file_order: list[str] = []

    dataset = getattr(loader, "dataset", None)
    meta = getattr(dataset, "meta", None)
    has_files = meta is not None and "source_file" in meta.columns
    offset = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            logits = model(x)
            if proc is not None:
                peak = max(peak, proc.memory_info().rss)

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for t, p in zip(y_np.astype(int), preds.astype(int)):
                cm[t, p] += 1

            if has_files:
                batch_size = int(len(y_np))
                sources = meta.iloc[offset : offset + batch_size]["source_file"].astype(str).to_numpy()
                logits_np = logits.detach().cpu().numpy()
                for yt, lg, src in zip(y_np.astype(int), logits_np, sources):
                    if src not in file_logit_sums:
                        file_order.append(src)
                        file_logit_sums[src] = np.asarray(lg, dtype=np.float64)
                        counts = np.zeros(num_classes, dtype=np.int64)
                        counts[yt] = 1
                        file_label_counts[src] = counts
                    else:
                        file_logit_sums[src] += np.asarray(lg, dtype=np.float64)
                        file_label_counts[src][yt] += 1
                offset += batch_size

    metrics = metrics_from_confusion_matrix(cm)
    result = metrics.to_percent_dict()
    result["confusion_matrix"] = metrics.confusion_matrix

    if file_order:
        file_true, file_pred = [], []
        for src in file_order:
            counts = file_label_counts[src]
            file_true.append(int(counts.argmax()))
            file_pred.append(int(file_logit_sums[src].argmax()))
        file_true_np = np.asarray(file_true, dtype=np.int64)
        file_pred_np = np.asarray(file_pred, dtype=np.int64)
        file_metrics = classification_metrics(file_true_np, file_pred_np, num_classes)
        result["file_level"] = {
            "n_files": int(len(file_true_np)),
            **file_metrics.to_percent_dict(),
        }

    peak_mb = round(peak / (1024 * 1024), 3) if proc is not None else float("nan")
    return result, peak_mb


def run_embedded_inference_check(
    *,
    bundle_dir: str | Path,
    embedded_cfg: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    device_name: str = "cpu",
) -> dict:
    """Latency + FLOPs smoke test (no processed test split required)."""
    root = repo_root()
    if config_path is not None:
        embedded_cfg = load_embedded_config(config_path)
    embedded_cfg = embedded_cfg or {}

    device = get_device(device_name)
    threads = int(embedded_cfg.get("torch_num_threads", 4))
    torch.set_num_threads(threads)

    model, manifest, _ = load_bundle(bundle_dir, device=device)
    model_name = str(manifest.get("model", "mspf_net"))
    dataset_id = int(manifest["dataset_id"])
    in_ch = int(manifest["in_channels"])
    seq_len = int(manifest["window_shape"][1])
    input_shape = (1, in_ch, seq_len)

    latency_runs = int(embedded_cfg.get("latency_runs", 1000))
    latency_warmup = int(embedded_cfg.get("latency_warmup", 50))

    with torch.no_grad():
        logits = model(torch.randn(*input_shape, device=device, dtype=torch.float32))
    if tuple(logits.shape) != (1, int(manifest["num_classes"])):
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")

    flops_info = profile_flops(model, input_shape)
    latency_stats = _benchmark_with_warmup(
        model, input_shape, device, n_runs=latency_runs, warmup=latency_warmup
    )

    try:
        import psutil

        peak_rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    except ImportError:
        peak_rss_mb = float("nan")

    payload = {
        "platform": "pi5",
        "mode": "inference_check",
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "device": str(device),
        "device_info": device_info(device),
        "manifest": manifest,
        "model": model_name,
        "flops": flops_info,
        "latency": latency_stats,
        "peak_rss_mb": peak_rss_mb,
        "torch_num_threads": threads,
        "logits_shape": list(logits.shape),
    }

    results_dir = Path(embedded_cfg.get("results_dir", root / "embedded/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    bundle_tag = Path(bundle_dir).name
    json_path = results_dir / f"{bundle_tag}_inference.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "pi5_inference_check.csv"
    row = {
        "model": model_name,
        "target": manifest.get("target_label"),
        "dataset_id": dataset_id,
        "variant": manifest.get("variant"),
        "bundle": bundle_tag,
        "device": str(device),
        "mean_ms": latency_stats.get("mean_ms"),
        "std_ms": latency_stats.get("std_ms"),
        "min_ms": latency_stats.get("min_ms"),
        "max_ms": latency_stats.get("max_ms"),
        "flops_g": flops_info.get("flops_g"),
        "params": flops_info.get("params", manifest.get("params")),
        "peak_rss_mb": peak_rss_mb,
        "latency_target_ok": latency_stats.get("target_ok"),
    }
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

    payload["results_json"] = str(json_path)
    payload["results_csv"] = str(csv_path)
    return payload


def run_embedded_benchmark(
    *,
    bundle_dir: str | Path,
    processed_dir: str | Path,
    embedded_cfg: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    device_name: str = "cpu",
) -> dict:
    """Run full embedded benchmark and write JSON + CSV artifacts."""
    root = repo_root()
    if config_path is not None:
        embedded_cfg = load_embedded_config(config_path)
    embedded_cfg = embedded_cfg or {}

    device = get_device(device_name)
    threads = int(embedded_cfg.get("torch_num_threads", 4))
    torch.set_num_threads(threads)

    model, manifest, _ = load_bundle(bundle_dir, device=device)
    dataset_id = int(manifest["dataset_id"])
    in_ch = int(manifest["in_channels"])
    seq_len = int(manifest["window_shape"][1])
    input_shape = (1, in_ch, seq_len)

    latency_runs = int(
        embedded_cfg.get("benchmark_latency_runs", embedded_cfg.get("latency_runs", 1000))
    )
    latency_warmup = int(embedded_cfg.get("latency_warmup", 50))
    batch_size = int(embedded_cfg.get("batch_size", 1))

    flops_info = profile_flops(model, input_shape)
    latency_stats = _benchmark_with_warmup(
        model, input_shape, device, n_runs=latency_runs, warmup=latency_warmup
    )

    loaders = build_dataloaders(
        processed_dir=str(processed_dir),
        dataset_ids=[dataset_id],
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        exclude_ineligible=True,
        aug_cfg=None,
        label_space=str(manifest.get("label_space", "fine")),
        require_test=True,
    )
    test_loader = loaders["test"]
    eval_out, peak_rss_mb = _peak_rss_during_eval(
        model,
        test_loader,
        device,
        list(manifest["class_names"]),
    )

    file_level = eval_out.get("file_level") or {}
    payload = {
        "platform": "pi5",
        "hostname": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "processed_dir": str(processed_dir),
        "device": str(device),
        "device_info": device_info(device),
        "manifest": manifest,
        "flops": flops_info,
        "latency": latency_stats,
        "peak_rss_mb": peak_rss_mb,
        "torch_num_threads": threads,
        "test_windows": int(len(test_loader.dataset)),
        "window_accuracy": eval_out.get("accuracy"),
        "window_macro_f1": eval_out.get("macro_f1"),
        "file_macro_f1": file_level.get("macro_f1"),
        "file_accuracy": file_level.get("accuracy"),
    }

    results_dir = Path(embedded_cfg.get("results_dir", root / "embedded/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(manifest.get("model", "mspf_net"))
    bundle_tag = Path(bundle_dir).name
    json_path = results_dir / f"{bundle_tag}_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "pi5_benchmark.csv"
    row = {
        "model": model_name,
        "target": manifest.get("target_label"),
        "dataset_id": dataset_id,
        "variant": manifest.get("variant", "softmax"),
        "bundle": bundle_tag,
        "device": str(device),
        "mean_ms": latency_stats.get("mean_ms"),
        "std_ms": latency_stats.get("std_ms"),
        "flops_g": flops_info.get("flops_g"),
        "params": flops_info.get("params", manifest.get("params")),
        "peak_rss_mb": peak_rss_mb,
        "window_macro_f1": eval_out.get("macro_f1"),
        "file_macro_f1": file_level.get("macro_f1"),
        "test_windows": payload["test_windows"],
        "latency_target_ok": latency_stats.get("target_ok"),
    }
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

    payload["results_json"] = str(json_path)
    payload["results_csv"] = str(csv_path)
    return payload


def _benchmark_with_warmup(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    n_runs: int,
    warmup: int,
) -> dict:
    import statistics
    import time

    from mspf_net.utils.device import _sync

    model = model.eval().to(device)
    dummy = torch.randn(*input_shape).float().to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            _sync(device)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(times)
    return {
        "device": str(device),
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "n_runs": n_runs,
        "warmup": warmup,
        "target_ok": mean_ms < 70.0,
    }
