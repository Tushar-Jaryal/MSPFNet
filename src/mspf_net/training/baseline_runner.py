from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier

from mspf_net.constants import ACTIVE_THESIS_DATASETS, get_dataset_display
from mspf_net.config_utils import (
    _fill_missing_training_defaults,
    get_config_path,
    get_config_value,
    get_processed_dir_for_policy,
    load_global_config,
)
from mspf_net.data.dataset import (
    build_crossval_dataloaders,
    build_dataloaders,
    build_unified_dataloaders,
    inspect_processed_splits,
    inspect_unified_splits,
)
from mspf_net.models.baselines.factory import create_baseline
from mspf_net.models.baselines.rf_features import extract_rf_features
from mspf_net.training.metrics import aggregate_file_logits, classification_metrics
from mspf_net.training.trainer import Trainer, _to_jsonable
from mspf_net.utils.device import get_device


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_with_extends(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    extends = data.pop("extends", None)
    if not extends:
        return data
    parent_path = Path(cfg_path).parent / extends
    parent_cfg = _load_yaml_with_extends(str(parent_path))
    return _deep_merge(parent_cfg, data)


def load_config(default_cfg_path: str, model_cfg_path: str) -> dict:
    with open(default_cfg_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    base = _apply_global_defaults(base)
    override = _load_yaml_with_extends(model_cfg_path)
    return _force_global_runtime(_deep_merge(base, override))


def _force_global_runtime(cfg: dict) -> dict:
    return _fill_missing_training_defaults(deepcopy(cfg))


def _apply_global_defaults(base: dict) -> dict:
    cfg = deepcopy(base)
    cfg.setdefault("paths", {})
    cfg.setdefault("training", {})

    window_policy = _resolve_window_policy(cfg, cfg.get("training", {}).get("experiment_tag"))
    cfg["paths"]["processed_dir"] = str(get_processed_dir_for_policy(window_policy))
    cfg["paths"]["unified_dir"] = get_config_path(
        "paths",
        "data_unified",
        default=str(cfg["paths"].get("unified_dir", "data/unified")),
    )
    cfg["paths"]["results_dir"] = get_config_value(
        "phase4",
        "results_dir",
        default=str(cfg["paths"].get("results_dir", "results/baselines")),
    )

    global_cfg = load_global_config()
    project_cfg = global_cfg.get("project", {}) if isinstance(global_cfg.get("project"), dict) else {}
    phase4_cfg = global_cfg.get("phase4", {}) if isinstance(global_cfg.get("phase4"), dict) else {}
    if "seed" in project_cfg:
        cfg["training"]["seed"] = project_cfg["seed"]
    for key in (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "patience",
        "scheduler",
        "num_workers",
        "unified_sampling",
        "use_class_weights",
        "device",
    ):
        if key in phase4_cfg and key not in cfg["training"]:
            cfg["training"][key] = phase4_cfg[key]
    return cfg


def _resolve_window_policy(cfg: dict, experiment_tag: str | None) -> str:
    data_policy = cfg.get("data", {}).get("window_policy")
    if data_policy:
        return str(data_policy).lower()
    tag = str(experiment_tag or cfg.get("training", {}).get("experiment_tag") or "").lower()
    if tag.startswith("fixed"):
        return "fixed"
    return "recommended"


def _prepare_mspf_model_cfg(model_name: str, model_cfg: dict) -> dict:
    if model_name != "mspf_net":
        return model_cfg
    from mspf_net.models.mspf.mspf_core import flatten_mspf_kwargs

    return flatten_mspf_kwargs(dict(model_cfg))


def _apply_dataset_overrides(
    cfg: dict,
    model_name: str,
    data_mode: str,
    dataset_ids: list[int] | None,
    group_id: str | None,
) -> tuple[dict, str | None]:
    overrides = cfg.get("dataset_overrides") or {}
    if not overrides:
        return cfg, None
    if data_mode != "processed" or not dataset_ids or len(dataset_ids) != 1:
        return cfg, None

    ds_id = int(dataset_ids[0])
    display = get_dataset_display(ds_id)
    candidate_keys = [display, display.upper(), str(ds_id), f"D{ds_id}"]
    for key in candidate_keys:
        override = overrides.get(key)
        if override:
            merged = _deep_merge(cfg, override)
            return merged, str(key)
    return cfg, None


def _resolve_augmentation_cfg(cfg: dict) -> dict | None:
    training_aug = cfg.get("training", {}).get("augmentation")
    if training_aug is not None:
        return training_aug if training_aug.get("enabled", True) else None

    project_cfg = load_global_config()
    aug_cfg = project_cfg.get("preprocessing", {}).get("augmentation")
    if not aug_cfg or not aug_cfg.get("enabled", True):
        return None
    return aug_cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one Phase 4 baseline model.")
    p.add_argument("--datasets", type=int, nargs="+", default=None)
    p.add_argument("--data-mode", choices=["processed", "unified"], default=None)
    p.add_argument("--group-id", type=str, default=None)
    p.add_argument("--label-space", choices=["fine", "coarse"], default=None)
    p.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--unified-sampling", choices=["dataset_balanced", "none"], default=None)
    p.add_argument("--cv-folds", type=int, default=None)
    p.add_argument("--cv-val-ratio", type=float, default=None)
    p.add_argument("--experiment-tag", type=str, default=None)
    p.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Append a sanitized suffix to artifact filenames without changing the experiment variant.",
    )
    p.add_argument("--exclude-ineligible", action="store_true")
    p.add_argument("--include-ineligible", action="store_true")
    p.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Override processed data directory (e.g. data/processed or data/processed_fixed).",
    )
    return p


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_run_tag(dataset_ids: list[int] | None, group_id: str | None, data_mode: str) -> str:
    def _sanitize(token: str) -> str:
        return token.replace("\\", "__").replace("/", "__").replace(",", "_").replace(" ", "_")

    if data_mode == "unified":
        return _sanitize(group_id or "unified")
    assert dataset_ids is not None
    if len(dataset_ids) == 1:
        return get_dataset_display(dataset_ids[0]).lower()
    return _sanitize("multi_" + "_".join(get_dataset_display(ds).lower() for ds in dataset_ids))


def _variant_suffix(
    data_mode: str,
    label_space: str,
    unified_sampling: str | None,
    experiment_tag: str | None,
) -> str:
    parts: list[str] = []
    if label_space != "fine":
        parts.append(label_space)
    if data_mode == "unified" and unified_sampling and unified_sampling != "dataset_balanced":
        parts.append(f"us_{unified_sampling}")
    if experiment_tag:
        parts.append(str(experiment_tag).replace("/", "__").replace(" ", "_"))
    return "" if not parts else "_" + "_".join(parts)


def _sanitize_suffix(suffix: str | None) -> str:
    if not suffix:
        return ""
    cleaned = str(suffix).strip().replace("\\", "__").replace("/", "__").replace(",", "_").replace(" ", "_")
    return "" if not cleaned else f"_{cleaned.lstrip('_')}"


def _class_weight_tensor(train_ds, cfg: dict, device: torch.device) -> torch.Tensor | None:
    if not cfg["training"].get("use_class_weights", True):
        return None
    weights = train_ds.class_weights().to(device)
    power = float(cfg["training"].get("class_weight_power", 1.0))
    if power != 1.0:
        weights = torch.pow(weights.clamp_min(1e-8), power)
        weights = weights / weights.sum() * len(weights)
    max_weight = cfg["training"].get("max_class_weight")
    if max_weight is not None:
        weights = weights.clamp_max(float(max_weight))
        weights = weights / weights.sum() * len(weights)
    return weights


def _build_criterion(train_ds, cfg: dict, device: torch.device) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(
        weight=_class_weight_tensor(train_ds, cfg, device),
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, epochs: int):
    scheduler_name = str(cfg["training"].get("scheduler", "cosine")).lower()
    if scheduler_name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if scheduler_name == "cosine_warmup":
        warmup_epochs = max(0, int(cfg["training"].get("warmup_epochs", 0)))

        def lr_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)
            denom = max(1, int(epochs) - warmup_epochs)
            progress = min(1.0, max(0.0, (epoch_idx - warmup_epochs + 1) / denom))
            min_lr_ratio = float(cfg["training"].get("min_lr_ratio", 0.05))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def _training_cfg_payload(
    cfg: dict,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    num_workers: int | None = None,
) -> dict:
    accum_steps = int(cfg["training"].get("accum_steps", 1))
    return {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "effective_batch_size": int(batch_size * accum_steps),
        "accum_steps": accum_steps,
        "num_workers": num_workers,
        "lr": lr,
        "patience": patience,
        "scheduler": cfg["training"].get("scheduler", "cosine"),
        "warmup_epochs": cfg["training"].get("warmup_epochs"),
        "label_smoothing": cfg["training"].get("label_smoothing", 0.0),
        "grad_clip_norm": cfg["training"].get("grad_clip_norm"),
        "use_amp": cfg["training"].get("use_amp", False),
        "selection_metric": cfg["training"].get("selection_metric", "val_macro_f1"),
        "class_weight_power": cfg["training"].get("class_weight_power", 1.0),
        "max_class_weight": cfg["training"].get("max_class_weight"),
    }


def _dataset_window_shape(dataset) -> dict:
    windows = getattr(dataset, "windows", None)
    if windows is None:
        return {}
    arr = np.asarray(windows)
    out = {"window_shape": list(arr.shape[1:])}
    if arr.ndim >= 3:
        out["in_channels"] = int(arr.shape[1])
        out["window_size"] = int(arr.shape[-1])
    elif arr.ndim == 2:
        out["in_channels"] = 1
        out["window_size"] = int(arr.shape[-1])
    return out


def _plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sums == 0, 1.0, row_sums))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            val = norm[i, j]
            color = "white" if val > 0.55 else "#111111"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summarize_cv_results(results: list[dict]) -> dict:
    metric_keys = [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
    ]
    split_keys = ["test", "val"]
    summary: dict[str, dict] = {}
    for split_key in split_keys:
        split_rows = [r.get(split_key) for r in results if r.get(split_key) is not None]
        if not split_rows:
            summary[split_key] = None
            continue
        split_summary = {}
        for metric in metric_keys:
            values = [float(row[metric]) for row in split_rows if row.get(metric) is not None]
            if not values:
                continue
            split_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        file_rows = [row.get("file_level") for row in split_rows if row.get("file_level") is not None]
        if file_rows:
            file_summary = {}
            for metric in metric_keys:
                values = [float(row[metric]) for row in file_rows if row.get(metric) is not None]
                if not values:
                    continue
                file_summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            split_summary["file_level"] = file_summary
        summary[split_key] = split_summary
    return summary


def _evaluate_rf(model: RandomForestClassifier, loader, class_names: list[str]) -> dict:
    dataset = loader.dataset
    feats = extract_rf_features(dataset.windows)
    y_true = np.asarray(dataset.labels, dtype=np.int64)
    y_pred = model.predict(feats)
    probs = model.predict_proba(feats)
    metrics = classification_metrics(y_true, y_pred, len(class_names))
    out = {
        "loss": float("nan"),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": probs,
        **metrics.to_percent_dict(),
        "confusion_matrix": metrics.confusion_matrix,
    }
    meta = getattr(dataset, "meta", None)
    if meta is not None and len(meta) == len(y_true) and "source_file" in meta.columns:
        source_files = meta["source_file"].astype(str).to_numpy()
        file_true, file_pred = aggregate_file_logits(y_true, probs, source_files)
        file_metrics = classification_metrics(file_true, file_pred, len(class_names))
        out["file_level"] = {
            "n_files": int(len(file_true)),
            **file_metrics.to_percent_dict(),
            "confusion_matrix": file_metrics.confusion_matrix,
            "y_true": file_true,
            "y_pred": file_pred,
        }
    return out


@torch.no_grad()
def _extract_deep_features(model, dataset, device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    windows = np.asarray(dataset.windows)
    if windows.ndim == 2:
        windows = windows[:, None, :]
    labels = np.asarray(dataset.labels, dtype=np.int64)
    meta = getattr(dataset, "meta", None)

    model.eval()
    feats = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start:start + batch_size]).float().to(device)
        feat = model.extract_features(batch)
        feats.append(feat.detach().cpu().numpy())

    if feats:
        features = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    else:
        feat_dim = int(getattr(model, "feat_dim", 0) or 0)
        features = np.empty((0, feat_dim), dtype=np.float32)
    return features, labels, meta


def _evaluate_deep_rf(model, rf_model: RandomForestClassifier, dataset, class_names: list[str], device: torch.device) -> dict:
    features, y_true, meta = _extract_deep_features(model, dataset, device=device)
    y_pred = rf_model.predict(features)
    probs = rf_model.predict_proba(features)
    metrics = classification_metrics(y_true, y_pred, len(class_names))
    out = {
        "loss": float("nan"),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": probs,
        **metrics.to_percent_dict(),
        "confusion_matrix": metrics.confusion_matrix,
    }
    if meta is not None and len(meta) == len(y_true) and "source_file" in meta.columns:
        source_files = meta["source_file"].astype(str).to_numpy()
        file_true, file_pred = aggregate_file_logits(y_true, probs, source_files)
        file_metrics = classification_metrics(file_true, file_pred, len(class_names))
        out["file_level"] = {
            "n_files": int(len(file_true)),
            **file_metrics.to_percent_dict(),
            "confusion_matrix": file_metrics.confusion_matrix,
            "y_true": file_true,
            "y_pred": file_pred,
        }
    return out


def _maybe_apply_mspf_rf_head(
    model_name: str,
    cfg: dict,
    model,
    train_ds,
    result: dict,
    loaders: dict,
    device: torch.device,
    seed: int,
) -> dict:
    if model_name != "mspf_net":
        return result
    if str(cfg["model"].get("classifier_head", "softmax")).lower() != "rf":
        return result

    train_features, y_train, _ = _extract_deep_features(model, train_ds, device=device)
    if hasattr(model, "fit_random_forest"):
        model.fit_random_forest(train_features, y_train, random_state=seed)
        rf_model = model.rf_classifier
    else:
        rf_model = RandomForestClassifier(
            n_estimators=int(cfg["model"].get("rf_n_estimators", 300)),
            max_depth=cfg["model"].get("rf_max_depth"),
            max_features=cfg["model"].get("rf_max_features", "sqrt"),
            min_samples_split=int(cfg["model"].get("rf_min_samples_split", 2)),
            n_jobs=int(cfg["model"].get("rf_n_jobs", -1)),
            random_state=int(seed),
        )
        rf_model.fit(train_features, y_train)

    softmax_eval = {
        "val": _to_jsonable(result.get("val")),
        "test": _to_jsonable(result.get("test")),
        "robustness": _to_jsonable(result.get("robustness")),
    }
    result["val"] = _to_jsonable(_evaluate_deep_rf(model, rf_model, loaders["val"].dataset, train_ds.class_names, device))
    result["test"] = _to_jsonable(_evaluate_deep_rf(model, rf_model, loaders["test"].dataset, train_ds.class_names, device))
    if loaders.get("robustness") is not None:
        result["robustness"] = _to_jsonable(
            _evaluate_deep_rf(model, rf_model, loaders["robustness"].dataset, train_ds.class_names, device)
        )
    else:
        result["robustness"] = None
    result["softmax_eval"] = softmax_eval
    result["classifier_head"] = "rf"
    result["rf_cfg"] = {
        "n_estimators": int(cfg["model"].get("rf_n_estimators", 300)),
        "max_depth": cfg["model"].get("rf_max_depth"),
        "max_features": cfg["model"].get("rf_max_features", "sqrt"),
        "min_samples_split": int(cfg["model"].get("rf_min_samples_split", 2)),
        "n_jobs": int(cfg["model"].get("rf_n_jobs", -1)),
    }
    return result


def _create_baseline_model(model_name: str, cfg: dict, train_ds, device: torch.device):
    model_cfg = _prepare_mspf_model_cfg(model_name, cfg["model"])
    return create_baseline(
        model_name,
        in_channels=train_ds.input_channels,
        num_classes=train_ds.num_classes,
        model_cfg=model_cfg,
    ).to(device)


def run_baseline(model_name: str, model_cfg_path: str, argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    cfg = load_config("configs/baselines/default.yaml", model_cfg_path)

    data_mode = args.data_mode or cfg["data"]["mode"]
    dataset_ids = args.datasets or cfg["data"]["dataset_ids"]
    group_id = args.group_id or cfg["data"].get("group_id")
    cfg, dataset_override_key = _apply_dataset_overrides(cfg, model_name, data_mode, dataset_ids, group_id)
    cfg = _force_global_runtime(cfg)

    label_space = args.label_space or cfg["data"].get("label_space", "fine")
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    epochs = args.epochs or cfg["training"]["epochs"]
    lr = args.lr or cfg["training"]["lr"]
    patience = args.patience or cfg["training"]["patience"]
    num_workers = cfg["training"]["num_workers"] if args.num_workers is None else args.num_workers
    device = get_device(args.device or cfg["training"]["device"])
    seed = args.seed or cfg["training"]["seed"]
    unified_sampling = args.unified_sampling or cfg["training"].get("unified_sampling", "dataset_balanced")
    cv_folds = args.cv_folds or int(cfg["training"].get("cv_folds", 1) or 1)
    cv_val_ratio = args.cv_val_ratio if args.cv_val_ratio is not None else float(cfg["training"].get("cv_val_ratio", 0.2))
    experiment_tag = args.experiment_tag or cfg["training"].get("experiment_tag")
    run_suffix = _sanitize_suffix(args.run_suffix)
    if args.processed_dir:
        cfg["paths"]["processed_dir"] = args.processed_dir
    elif data_mode == "processed":
        cfg["paths"]["processed_dir"] = str(
            get_processed_dir_for_policy(_resolve_window_policy(cfg, experiment_tag))
        )
    exclude_ineligible = True
    if args.include_ineligible:
        exclude_ineligible = False
    elif args.exclude_ineligible:
        exclude_ineligible = True
    else:
        exclude_ineligible = cfg["data"].get("exclude_ineligible", True)
    aug_cfg = _resolve_augmentation_cfg(cfg)

    _set_seed(seed)

    ds_label = group_id if data_mode == "unified" else ",".join(get_dataset_display(ds) for ds in dataset_ids)

    run_tag = _safe_run_tag(dataset_ids, group_id, data_mode)
    run_name = f"{model_name}_{run_tag}{_variant_suffix(data_mode, label_space, unified_sampling, experiment_tag)}{run_suffix}"
    results_root = Path(cfg["paths"]["results_dir"])
    if data_mode == "unified":
        output_dir = results_root / model_name / "unified" / Path(group_id or "unified")
    else:
        output_dir = results_root / model_name / "processed" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if cv_folds > 1:
        folds = build_crossval_dataloaders(
            data_mode=data_mode,
            label_space=label_space,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            exclude_ineligible=exclude_ineligible,
            aug_cfg=aug_cfg,
            n_splits=cv_folds,
            val_ratio=cv_val_ratio,
            seed=seed,
            processed_dir=cfg["paths"]["processed_dir"],
            dataset_ids=dataset_ids,
            unified_dir=cfg["paths"]["unified_dir"],
            group_id=group_id,
            unified_train_sampling=unified_sampling,
        )

        print("\n" + "═" * 64)
        print(f"  MSPF-Net — Baseline: {model_name.upper()} (5-fold CV)" if cv_folds == 5 else f"  MSPF-Net — Baseline: {model_name.upper()} ({cv_folds}-fold CV)")
        print(f"  Data mode : {data_mode}")
        print(f"  Target    : {ds_label}")
        print(f"  Device    : {device}")
        print("═" * 64)

        fold_payloads = []
        for fold in folds:
            fold_idx = int(fold["fold_idx"]) + 1
            train_ds = fold["train"].dataset
            fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_name = f"{run_name}_fold{fold_idx}"
            print(f"\n  Fold {fold_idx}/{cv_folds}")
            if model_name == "rf_features":
                rf_model = RandomForestClassifier(
                    n_estimators=int(cfg["model"].get("n_estimators", 300)),
                    max_depth=cfg["model"].get("max_depth"),
                    max_features=cfg["model"].get("max_features", "sqrt"),
                    min_samples_split=int(cfg["model"].get("min_samples_split", 2)),
                    n_jobs=int(cfg["model"].get("n_jobs", -1)),
                    random_state=int(seed + fold_idx),
                )
                rf_model.fit(extract_rf_features(train_ds.windows), np.asarray(train_ds.labels, dtype=np.int64))
                result = {
                    "run_name": fold_name,
                    "best_epoch": None,
                    "epochs_ran": 0,
                    "elapsed_s": float("nan"),
                    "best_checkpoint": None,
                    "history_csv": None,
                    "curves_png": None,
                    "val": _to_jsonable(_evaluate_rf(rf_model, fold["val"], train_ds.class_names)),
                    "test": _to_jsonable(_evaluate_rf(rf_model, fold["test"], train_ds.class_names)),
                    "robustness": None,
                    "inference_ms_per_window": float("nan"),
                }
                param_count = 0
            else:
                model = _create_baseline_model(model_name, cfg, train_ds, device)
                criterion = _build_criterion(train_ds, cfg, device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
                scheduler = _build_scheduler(optimizer, cfg, epochs)

                trainer = Trainer(
                    model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=fold_dir,
                    run_name=fold_name,
                    class_names=train_ds.class_names,
                    max_epochs=epochs,
                    patience=patience,
                    grad_clip_norm=cfg["training"].get("grad_clip_norm"),
                    use_amp=bool(cfg["training"].get("use_amp", False)),
                    accum_steps=int(cfg["training"].get("accum_steps", 1)),
                    selection_metric=str(cfg["training"].get("selection_metric", "val_macro_f1")),
                    batch_size=batch_size,
                )
                result = trainer.fit(
                    fold["train"],
                    fold["val"],
                    fold["test"],
                    fold.get("robustness"),
                    evaluation_split="test",
                )
                result = _maybe_apply_mspf_rf_head(
                    model_name=model_name,
                    cfg=cfg,
                    model=model,
                    train_ds=train_ds,
                    result=result,
                    loaders=fold,
                    device=device,
                    seed=seed + fold_idx,
                )
                param_count = int(sum(p.numel() for p in model.parameters()))
            payload = {
                "model": model_name,
                "data_mode": data_mode,
                "datasets": dataset_ids if data_mode == "processed" else None,
                "group_id": group_id if data_mode == "unified" else None,
                "target_label": ds_label,
                "label_space": label_space,
                "unified_sampling": unified_sampling if data_mode == "unified" else None,
                "experiment_tag": experiment_tag,
                "run_suffix": run_suffix.lstrip("_") or None,
                "device": str(device),
                "evaluation_mode": "crossval_fold",
                "primary_comparable": False,
                "cv_folds": cv_folds,
                "cv_fold_idx": fold_idx,
                "cv_val_ratio": cv_val_ratio,
                "num_classes": train_ds.num_classes,
                "class_names": train_ds.class_names,
                "in_channels": train_ds.input_channels,
                **_dataset_window_shape(train_ds),
                "params": param_count,
                "batch_size": batch_size,
                "effective_batch_size": int(batch_size * int(cfg["training"].get("accum_steps", 1))),
                "num_workers": num_workers,
                "training_cfg": _training_cfg_payload(cfg, seed, epochs, batch_size, lr, patience, num_workers),
                "dataset_override_key": dataset_override_key,
                "mspf_profile": cfg.get("profile"),
                "classifier_head": str(cfg["model"].get("classifier_head", "softmax")).lower(),
                "architecture": str(cfg["model"].get("architecture", "slim")).lower(),
                **result,
            }
            result_path = fold_dir / f"{fold_name}_results.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(payload), f, indent=2)
            fold_payloads.append(payload)

        summary = _summarize_cv_results(fold_payloads)
        summary_payload = {
            "model": model_name,
            "data_mode": data_mode,
            "datasets": dataset_ids if data_mode == "processed" else None,
            "group_id": group_id if data_mode == "unified" else None,
            "target_label": ds_label,
            "label_space": label_space,
            "unified_sampling": unified_sampling if data_mode == "unified" else None,
            "experiment_tag": experiment_tag,
            "run_suffix": run_suffix.lstrip("_") or None,
            "device": str(device),
            "evaluation_mode": "crossval_summary",
            "primary_comparable": False,
            "cv_folds": cv_folds,
            "cv_val_ratio": cv_val_ratio,
            "window_size": fold_payloads[0].get("window_size") if fold_payloads else None,
            "window_shape": fold_payloads[0].get("window_shape") if fold_payloads else None,
            "in_channels": fold_payloads[0].get("in_channels") if fold_payloads else None,
            "fold_results": [_to_jsonable(p) for p in fold_payloads],
            "summary": _to_jsonable(summary),
        }
        summary_path = output_dir / f"{run_name}_cv{cv_folds}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        print("\n  Cross-validation summary")
        if summary.get("test"):
            test_summary = summary["test"]
            print(
                f"  TEST mean: Acc={test_summary['accuracy']['mean']:.2f}%  "
                f"F1={test_summary['macro_f1']['mean']:.2f}%  "
                f"Prec={test_summary['macro_precision']['mean']:.2f}%  "
                f"Rec={test_summary['macro_recall']['mean']:.2f}%"
            )
        print(f"  Summary → {summary_path}")
        return summary_payload

    if data_mode == "processed":
        availability = inspect_processed_splits(
            processed_dir=cfg["paths"]["processed_dir"],
            dataset_ids=dataset_ids,
        )
    else:
        if not group_id:
            raise ValueError("--group-id is required for unified mode")
        availability = inspect_unified_splits(
            unified_dir=cfg["paths"]["unified_dir"],
            group_id=group_id,
        )

    target_label = group_id if data_mode == "unified" else ",".join(get_dataset_display(ds) for ds in dataset_ids)
    if not availability["test"]:
        print("\n" + "═" * 64)
        print(f"  MSPF-Net — Baseline: {model_name.upper()}")
        print(f"  Data mode : {data_mode}")
        print(f"  Target    : {target_label}")
        print("═" * 64)
        print("  SKIP: no held-out test split exists for this target under the current split policy")
        print("  Reason: datasets without a real file-level test split are excluded from Phase 4 benchmark runs")
        return {
            "status": "skipped",
            "reason": "no_test_split",
            "target_label": target_label,
            "has_test_split": False,
            "has_val_split": bool(availability["val"]),
            "primary_comparable": False,
            "evaluation_mode": "skipped_no_test",
        }
    evaluation_mode = "primary_test"
    primary_comparable = True

    if data_mode == "processed":
        loaders = build_dataloaders(
            processed_dir=cfg["paths"]["processed_dir"],
            dataset_ids=dataset_ids,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            exclude_ineligible=exclude_ineligible,
            aug_cfg=aug_cfg,
            label_space=label_space,
            require_test=True,
        )
        train_ds = loaders["train"].dataset
    else:
        if not group_id:
            raise ValueError("--group-id is required for unified mode")
        loaders = build_unified_dataloaders(
            unified_dir=cfg["paths"]["unified_dir"],
            group_id=group_id,
            label_space=label_space,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            exclude_ineligible=exclude_ineligible,
            aug_cfg=aug_cfg,
            train_sampling=unified_sampling,
            require_test=True,
        )
        train_ds = loaders["train"].dataset

    if model_name == "rf_features":
        rf_model = RandomForestClassifier(
            n_estimators=int(cfg["model"].get("n_estimators", 300)),
            max_depth=cfg["model"].get("max_depth"),
            max_features=cfg["model"].get("max_features", "sqrt"),
            min_samples_split=int(cfg["model"].get("min_samples_split", 2)),
            n_jobs=int(cfg["model"].get("n_jobs", -1)),
            random_state=int(seed),
        )
        X_train = extract_rf_features(train_ds.windows)
        y_train = np.asarray(train_ds.labels, dtype=np.int64)
        rf_model.fit(X_train, y_train)

        val_metrics = _evaluate_rf(rf_model, loaders["val"], train_ds.class_names)
        test_metrics = _evaluate_rf(rf_model, loaders["test"], train_ds.class_names)
        rob_metrics = None if loaders.get("robustness") is None else _evaluate_rf(rf_model, loaders["robustness"], train_ds.class_names)

        payload = {
            "model": model_name,
            "data_mode": data_mode,
            "datasets": dataset_ids if data_mode == "processed" else None,
            "group_id": group_id if data_mode == "unified" else None,
            "target_label": ds_label,
            "label_space": label_space,
            "unified_sampling": unified_sampling if data_mode == "unified" else None,
            "experiment_tag": experiment_tag,
            "run_suffix": run_suffix.lstrip("_") or None,
            "device": str(device),
            "evaluation_mode": evaluation_mode,
            "has_test_split": bool(availability["test"]),
            "has_val_split": bool(availability["val"]),
            "primary_comparable": primary_comparable,
            "num_classes": train_ds.num_classes,
            "class_names": train_ds.class_names,
            "in_channels": train_ds.input_channels,
            "params": 0,
            "run_name": run_name,
            "best_epoch": None,
            "epochs_ran": 0,
            "elapsed_s": float("nan"),
            "best_checkpoint": None,
            "history_csv": None,
            "curves_png": None,
            "val": _to_jsonable(val_metrics),
            "test": _to_jsonable(test_metrics),
            "robustness": None if rob_metrics is None else _to_jsonable(rob_metrics),
            "inference_ms_per_window": float("nan"),
            "batch_size": batch_size,
            "effective_batch_size": int(batch_size * int(cfg["training"].get("accum_steps", 1))),
            "num_workers": num_workers,
            "training_cfg": _training_cfg_payload(cfg, seed, epochs, batch_size, lr, patience, num_workers),
        }
        result_path = output_dir / f"{run_name}_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  RF baseline done → {result_path}")
        return payload

    model = _create_baseline_model(model_name, cfg, train_ds, device)

    criterion = _build_criterion(train_ds, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = _build_scheduler(optimizer, cfg, epochs)

    print("\n" + "═" * 64)
    print(f"  MSPF-Net — Baseline: {model_name.upper()}")
    print(f"  Data mode : {data_mode}")
    print(f"  Target    : {ds_label}")
    print(f"  Device    : {device}")
    print("═" * 64)
    print(f"  Classes  : {train_ds.num_classes}  ({train_ds.class_names})")
    test_count = 0 if loaders["test"] is None else len(loaders["test"].dataset)
    print(
        f"  Windows  : train={len(loaders['train'].dataset):,}  "
        f"val={len(loaders['val'].dataset):,}  test={test_count:,}"
    )
    print(f"  Channels : {train_ds.input_channels}")
    accum_steps = int(cfg["training"].get("accum_steps", 1))
    print(f"  Batch    : {batch_size}  (effective={batch_size * accum_steps}, workers={num_workers})")
    print("  Using class weights" if cfg["training"].get("use_class_weights", True) else "  No class weights")
    print()

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        run_name=run_name,
        class_names=train_ds.class_names,
        max_epochs=epochs,
        patience=patience,
        grad_clip_norm=cfg["training"].get("grad_clip_norm"),
        use_amp=bool(cfg["training"].get("use_amp", False)),
        accum_steps=int(cfg["training"].get("accum_steps", 1)),
        selection_metric=str(cfg["training"].get("selection_metric", "val_macro_f1")),
        batch_size=batch_size,
    )
    eval_loader = loaders["test"]
    eval_split = "test"
    result = trainer.fit(
        loaders["train"],
        loaders["val"],
        eval_loader,
        loaders.get("robustness"),
        evaluation_split=eval_split,
    )
    result = _maybe_apply_mspf_rf_head(
        model_name=model_name,
        cfg=cfg,
        model=model,
        train_ds=train_ds,
        result=result,
        loaders=loaders,
        device=device,
        seed=seed,
    )

    test_cm_path = output_dir / f"{run_name}_confusion.png"
    _plot_confusion(
        np.asarray(result[eval_split]["confusion_matrix"]),
        train_ds.class_names,
        test_cm_path,
        title=f"{model_name} — {ds_label} test confusion",
    )
    file_cm_path = None
    if result[eval_split].get("file_level") is not None:
        file_cm_path = output_dir / f"{run_name}_file_confusion.png"
        _plot_confusion(
            np.asarray(result[eval_split]["file_level"]["confusion_matrix"]),
            train_ds.class_names,
            file_cm_path,
            title=f"{model_name} — {ds_label} test file-level confusion",
        )

    param_count = int(sum(p.numel() for p in model.parameters()))
    payload = {
        "model": model_name,
        "data_mode": data_mode,
        "datasets": dataset_ids if data_mode == "processed" else None,
        "group_id": group_id if data_mode == "unified" else None,
        "target_label": ds_label,
        "label_space": label_space,
        "unified_sampling": unified_sampling if data_mode == "unified" else None,
        "experiment_tag": experiment_tag,
        "run_suffix": run_suffix.lstrip("_") or None,
        "device": str(device),
        "evaluation_mode": evaluation_mode,
        "has_test_split": bool(availability["test"]),
        "has_val_split": bool(availability["val"]),
        "primary_comparable": primary_comparable,
        "num_classes": train_ds.num_classes,
        "class_names": train_ds.class_names,
        "in_channels": train_ds.input_channels,
        "params": param_count,
        "batch_size": batch_size,
        "effective_batch_size": int(batch_size * int(cfg["training"].get("accum_steps", 1))),
        "num_workers": num_workers,
        "training_cfg": _training_cfg_payload(cfg, seed, epochs, batch_size, lr, patience, num_workers),
        "dataset_override_key": dataset_override_key,
        "mspf_profile": cfg.get("profile"),
        "classifier_head": str(cfg["model"].get("classifier_head", "softmax")).lower(),
        "architecture": str(cfg["model"].get("architecture", "slim")).lower(),
        **result,
        "confusion_png": str(test_cm_path),
        "file_confusion_png": None if file_cm_path is None else str(file_cm_path),
    }
    result_path = output_dir / f"{run_name}_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    print(f"  Training done in {payload['elapsed_s']:.1f}s")
    print()
    metric_block = payload["test"]
    metric_label = "TEST"
    print(
        f"  {metric_label}: Acc={metric_block['accuracy']:.2f}%  "
        f"F1={metric_block['macro_f1']:.2f}%  "
        f"Prec={metric_block['macro_precision']:.2f}%  "
        f"Rec={metric_block['macro_recall']:.2f}%"
    )
    file_metric_block = metric_block.get("file_level")
    if file_metric_block is not None:
        print(
            f"  FILE {metric_label}: Acc={file_metric_block['accuracy']:.2f}%  "
            f"F1={file_metric_block['macro_f1']:.2f}%  "
            f"Prec={file_metric_block['macro_precision']:.2f}%  "
            f"Rec={file_metric_block['macro_recall']:.2f}%  "
            f"(n_files={file_metric_block['n_files']})"
        )
    print(f"        Inference: {payload['inference_ms_per_window']:.3f} ms/window   Params: {param_count:,}")
    memory = payload.get("memory") or {}
    train_peak = memory.get("peak_train_gpu_mb")
    if train_peak is not None:
        infer_peak = memory.get("peak_inference_gpu_mb")
        rss_peak = memory.get("peak_rss_mb")
        infer_txt = f"{infer_peak:.0f}" if infer_peak is not None else "n/a"
        rss_txt = f"{rss_peak:.0f}" if rss_peak == rss_peak else "n/a"
        print(
            f"        Memory  : train GPU peak={train_peak:.0f} MB  "
            f"inference GPU peak={infer_txt} MB  RSS={rss_txt} MB"
        )
    print(f"  Results → {result_path}")
    print(f"  History → {payload['history_csv']}")
    print(f"  Plot    → {test_cm_path}")
    if file_cm_path is not None:
        print(f"  FilePlot → {file_cm_path}")
    return payload
