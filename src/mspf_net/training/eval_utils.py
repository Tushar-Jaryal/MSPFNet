"""Shared evaluation helpers for training and embedded inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from mspf_net.training.metrics import aggregate_file_logits, classification_metrics


def add_awgn_at_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN so signal RMS matches ``snr_db`` (deterministic scale, fresh noise)."""
    rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True).clamp_min(1e-12))
    noise_scale = rms / (10.0 ** (float(snr_db) / 20.0))
    return x + torch.randn_like(x) * noise_scale


def evaluate_classifier(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    criterion: nn.Module | None = None,
    use_amp: bool = False,
    snr_db: float | None = None,
) -> dict:
    """Run inference on a dataloader and return window + optional file-level metrics."""
    model.eval()
    all_true, all_pred, all_logits = [], [], []
    total_loss = 0.0
    total_count = 0
    use_amp = bool(use_amp) and device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError(f"Expected batch of (x, y), got {type(batch)!r}")

            x = x.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")
            y = y.to(device, dtype=torch.long, non_blocking=device.type == "cuda")
            if snr_db is not None:
                x = add_awgn_at_snr(x, snr_db)

            with autocast(device_type=amp_device, enabled=use_amp):
                logits = model(x)
                if criterion is not None:
                    total_loss += float(criterion(logits, y).item()) * len(x)
            total_count += int(len(x))
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    total_count = max(total_count, 1)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    logits_np = (
        np.concatenate(all_logits)
        if all_logits
        else np.empty((0, len(class_names)), dtype=np.float32)
    )
    metrics = classification_metrics(y_true, y_pred, len(class_names))
    out = {
        "loss": total_loss / total_count if criterion is not None else float("nan"),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits_np,
    }
    out.update(metrics.to_percent_dict())
    out["confusion_matrix"] = metrics.confusion_matrix

    dataset = getattr(loader, "dataset", None)
    meta = getattr(dataset, "meta", None)
    if meta is not None and len(meta) == len(y_true) and "source_file" in meta.columns:
        source_files = meta["source_file"].astype(str).to_numpy()
        file_true, file_pred = aggregate_file_logits(y_true, logits_np, source_files)
        file_metrics = classification_metrics(file_true, file_pred, len(class_names))
        out["file_level"] = {
            "n_files": int(len(file_true)),
            **file_metrics.to_percent_dict(),
            "confusion_matrix": file_metrics.confusion_matrix,
            "y_true": file_true,
            "y_pred": file_pred,
        }
    return out


def evaluate_noise_robustness(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    snr_levels: list[float] | None = None,
    use_amp: bool = False,
) -> dict | None:
    """Evaluate test loader at fixed AWGN SNR levels from global config."""
    if loader is None:
        return None
    from mspf_net.utils.aggregate_robustness import compact_eval_metrics, get_noise_snr_levels

    levels = snr_levels if snr_levels is not None else get_noise_snr_levels()
    if not levels:
        return None

    by_snr: dict[str, dict] = {}
    for snr_db in levels:
        raw = evaluate_classifier(
            model,
            loader,
            device,
            class_names,
            criterion=None,
            use_amp=use_amp,
            snr_db=snr_db,
        )
        key = str(int(snr_db)) if snr_db == int(snr_db) else str(snr_db)
        compact = compact_eval_metrics(raw)
        if compact is not None:
            by_snr[key] = compact

    return {"snr_db_levels": levels, "by_snr": by_snr}
