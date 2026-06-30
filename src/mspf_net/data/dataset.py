from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class LabelEncoder:
    """Maps fault_code strings to integer class indices and back."""

    def __init__(self, classes: list[str]):
        self.classes_ = sorted(classes)
        self._c2i = {c: i for i, c in enumerate(self.classes_)}
        self._i2c = {i: c for c, i in self._c2i.items()}

    def __len__(self) -> int:
        return len(self.classes_)

    def encode(self, label: str) -> int:
        return self._c2i[label]

    def decode(self, idx: int) -> str:
        return self._i2c[idx]

    def encode_array(self, labels) -> np.ndarray:
        unknown = sorted({l for l in labels if l not in self._c2i})
        if unknown:
            raise ValueError(
                f"fault_code(s) not seen during training: {unknown}. "
                "Ensure all splits share the same label taxonomy."
            )
        return np.array([self._c2i[l] for l in labels], dtype=np.int64)


def inspect_processed_splits(
    processed_dir: Union[str, Path],
    dataset_ids: Union[int, list[int]],
    processed_group: Optional[str] = None,
) -> dict[str, bool]:
    """
    Check which processed splits exist for one or more datasets.
    """
    root = Path(processed_dir)
    ids = [dataset_ids] if isinstance(dataset_ids, int) else list(dataset_ids)
    found = {split: False for split in ("train", "val", "test", "robustness")}
    for ds_id in ids:
        for split in found:
            if processed_group is None:
                patterns = [f"d{ds_id}_{split}_windows.npy", f"d{ds_id}_*_{split}_windows.npy"]
            else:
                patterns = [f"d{ds_id}_{processed_group}_{split}_windows.npy"]
            for pattern in patterns:
                if any(root.glob(pattern)):
                    found[split] = True
                    break
    return found


def inspect_unified_splits(
    unified_dir: Union[str, Path],
    group_id: str,
) -> dict[str, bool]:
    """
    Check which unified splits exist for one channel-compatible group.
    """
    group_dir = Path(unified_dir) / group_id
    return {
        split: (group_dir / f"{split}_windows.npy").exists() and (group_dir / f"{split}_meta.csv").exists()
        for split in ("train", "val", "test", "robustness")
    }


def _apply_augmentation(x: torch.Tensor, aug_cfg: Optional[dict]) -> torch.Tensor:
    if aug_cfg is None:
        return x
    if not aug_cfg.get("enabled", True):
        return x

    noise_std = aug_cfg.get("noise_std", 0.0)
    noise_snr_db = aug_cfg.get("noise_snr_db", None)
    if noise_snr_db:
        choices = noise_snr_db if isinstance(noise_snr_db, (list, tuple)) else [noise_snr_db]
        snr_db = float(choices[int(torch.randint(0, len(choices), (1,)).item())])
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True).clamp_min(1e-12))
        noise_scale = rms / (10.0 ** (snr_db / 20.0))
        x = x + torch.randn_like(x) * noise_scale
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    scale_range = aug_cfg.get("scale_range", None) or aug_cfg.get("amplitude_scale", None)
    if scale_range:
        lo, hi = scale_range
        scale = lo + torch.rand(1).item() * (hi - lo)
        x = x * scale

    shift_max = int(aug_cfg.get("shift_max", 0) or 0)
    jitter_pct = aug_cfg.get("jitter_pct", None)
    if jitter_pct is not None and shift_max <= 0:
        shift_max = max(0, int(round(float(jitter_pct) * x.shape[-1])))
    if shift_max > 0:
        shift = int(torch.randint(0, shift_max + 1, (1,)).item())
        if shift > 0:
            x = torch.roll(x, shift, dims=-1)

    return x


def _build_dataset_balanced_sampler(dataset: Dataset) -> WeightedRandomSampler | None:
    """
    Build a train sampler that equalizes dataset contribution inside one unified group.

    This is intentionally based on `dataset_id` only. Label imbalance is already handled
    downstream by class-weighted loss, while this sampler protects unified training from
    being dominated by the largest source dataset in the group.
    """
    meta = getattr(dataset, "meta", None)
    if meta is None or "dataset_id" not in meta.columns:
        return None

    dataset_ids = meta["dataset_id"].to_numpy()
    if len(dataset_ids) == 0:
        return None

    unique_ids, counts = np.unique(dataset_ids, return_counts=True)
    if len(unique_ids) <= 1:
        return None

    id_to_weight = {ds_id: 1.0 / count for ds_id, count in zip(unique_ids.tolist(), counts.tolist())}
    weights = np.asarray([id_to_weight[ds_id] for ds_id in dataset_ids], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def _domain_key_from_row(row: pd.Series, source: str) -> str:
    if source == "sub_dataset" and "sub_dataset" in row.index:
        return str(row["sub_dataset"])
    if "dataset_id" in row.index:
        return str(row["dataset_id"])
    return "0"


def build_domain_id_map(meta: pd.DataFrame, source: str = "sub_dataset") -> dict[str, int]:
    if source == "sub_dataset" and "sub_dataset" in meta.columns:
        labels = sorted(meta["sub_dataset"].astype(str).unique().tolist())
    elif "dataset_id" in meta.columns:
        labels = sorted(meta["dataset_id"].astype(str).unique().tolist())
    else:
        labels = ["0"]
    return {label: idx for idx, label in enumerate(labels)}


class FaultDataset(Dataset):
    """
    Read split-specific Phase 3 windows for one or more datasets.
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        dataset_ids: Union[int, list[int]],
        split: str = "train",
        label_space: str = "fine",
        encoder: Optional[LabelEncoder] = None,
        exclude_ineligible: bool = True,
        aug_cfg: Optional[dict] = None,
        processed_group: Optional[str] = None,
        channel_align: Optional[Union[int, str]] = None,
        return_domain_id: bool = False,
        domain_id_map: Optional[dict[str, int]] = None,
        domain_label_source: str = "sub_dataset",
    ):
        self.processed_dir = Path(processed_dir)
        self.dataset_ids = [dataset_ids] if isinstance(dataset_ids, int) else list(dataset_ids)
        self.split = split
        self.label_space = label_space
        self.processed_group = processed_group

        windows_list, meta_list = [], []

        for ds_id in self.dataset_ids:
            splits_to_load = ["train", "val", "test", "robustness"] if split == "all" else [split]
            for sp in splits_to_load:
                pairs = []
                if processed_group is None:
                    patterns = [
                        f"d{ds_id}_{sp}_windows.npy",
                        f"d{ds_id}_*_{sp}_windows.npy",
                    ]
                else:
                    patterns = [f"d{ds_id}_{processed_group}_{sp}_windows.npy"]
                seen = set()
                for pattern in patterns:
                    for npy_path in sorted(self.processed_dir.glob(pattern)):
                        csv_path = npy_path.with_name(npy_path.name.replace("_windows.npy", "_meta.csv"))
                        if not csv_path.exists():
                            continue
                        key = (str(npy_path), str(csv_path))
                        if key not in seen:
                            seen.add(key)
                            pairs.append((npy_path, csv_path))

                for npy_path, csv_path in pairs:
                    w = np.load(str(npy_path))
                    m = pd.read_csv(csv_path)
                    if len(w) != len(m):
                        raise ValueError(f"Shape mismatch between {npy_path.name} and {csv_path.name}")
                    windows_list.append(w)
                    meta_list.append(m)

        if not windows_list:
            raise FileNotFoundError(
                f"No windows found for datasets {self.dataset_ids} split='{split}' in {self.processed_dir}"
            )

        channel_shapes = sorted({tuple(np.asarray(w).shape[1:]) for w in windows_list})
        if len(channel_shapes) > 1:
            if channel_align is None:
                raise ValueError(
                    "Cannot combine datasets with different window shapes/channels: "
                    f"{channel_shapes}. Pass channel_align='min'/'max' or an int to align automatically."
                )
            all_ch = [w.shape[1] for w in windows_list]
            if channel_align == "min":
                target_ch = min(all_ch)
            elif channel_align == "max":
                target_ch = max(all_ch)
            elif isinstance(channel_align, int):
                target_ch = channel_align
            else:
                raise ValueError(f"channel_align must be 'min', 'max', or an int; got {channel_align!r}")
            aligned = []
            for w in windows_list:
                c = w.shape[1]
                if c == target_ch:
                    aligned.append(w)
                elif c > target_ch:
                    aligned.append(w[:, :target_ch, :])
                else:
                    # Repeat channels cyclically to reach target_ch
                    reps = (target_ch + c - 1) // c
                    padded = np.concatenate([w] * reps, axis=1)[:, :target_ch, :]
                    aligned.append(padded)
            windows_list = aligned

        self.windows = np.vstack(windows_list)
        self.meta = pd.concat(meta_list, ignore_index=True)

        if exclude_ineligible and split == "test" and "test_eligible" in self.meta.columns:
            mask = self.meta["test_eligible"].astype(bool)
            self.windows = self.windows[mask.values]
            self.meta = self.meta[mask].reset_index(drop=True)

        label_col = "fault_code" if label_space == "fine" else "coarse_fault_code"
        if label_col not in self.meta.columns:
            raise ValueError(f"Missing '{label_col}' in processed metadata")
        all_classes = sorted(self.meta[label_col].astype(str).unique().tolist())
        self.encoder = encoder if encoder is not None else LabelEncoder(all_classes)
        self.labels = self.encoder.encode_array(self.meta[label_col].astype(str).tolist())

        # Augmentation is only active for the training split
        self._aug = aug_cfg if split == "train" else None
        self.return_domain_id = bool(return_domain_id)
        self.domain_id_map = dict(domain_id_map or {})
        self.domain_label_source = str(domain_label_source)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx]).float()
        x = _apply_augmentation(x, self._aug)
        if self.return_domain_id:
            key = _domain_key_from_row(self.meta.iloc[idx], self.domain_label_source)
            return x, int(self.labels[idx]), int(self.domain_id_map.get(key, 0))
        return x, int(self.labels[idx])

    @property
    def input_channels(self) -> int:
        if self.windows.ndim == 3:
            return int(self.windows.shape[1])
        return 1

    @property
    def num_classes(self) -> int:
        return len(self.encoder)

    @property
    def class_names(self) -> list[str]:
        return self.encoder.classes_

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        w = w / w.sum() * self.num_classes
        return torch.tensor(w, dtype=torch.float32)

    def summary(self) -> pd.DataFrame:
        df = self.meta.copy()
        df["label_idx"] = self.labels
        label_col = "fault_code" if self.label_space == "fine" else "coarse_fault_code"
        return (
            df.groupby([label_col, "label_idx", "dataset_id"])
            .size()
            .reset_index(name="n_windows")
            .sort_values(["dataset_id", "label_idx"])
        )


class UnifiedFaultDataset(Dataset):
    """
    Read one component-focused unified group from data/unified/{group_id}/.
    """

    def __init__(
        self,
        unified_dir: Union[str, Path],
        group_id: str,
        split: str = "train",
        label_space: str = "fine",
        encoder: Optional[LabelEncoder] = None,
        exclude_ineligible: bool = True,
        aug_cfg: Optional[dict] = None,
    ):
        self.unified_dir = Path(unified_dir)
        self.group_id = group_id
        self.group_dir = self.unified_dir / group_id
        self.split = split
        self.label_space = label_space

        npy_path = self.group_dir / f"{split}_windows.npy"
        csv_path = self.group_dir / f"{split}_meta.csv"
        if not npy_path.exists() or not csv_path.exists():
            raise FileNotFoundError(
                f"No unified windows found for group='{group_id}' split='{split}' in {self.group_dir}"
            )

        self.windows = np.load(str(npy_path))
        self.meta = pd.read_csv(csv_path)
        if len(self.windows) != len(self.meta):
            raise ValueError(f"Shape mismatch between {npy_path.name} and {csv_path.name}")

        if exclude_ineligible and split == "test" and "test_eligible" in self.meta.columns:
            mask = self.meta["test_eligible"].astype(bool)
            self.windows = self.windows[mask.values]
            self.meta = self.meta[mask].reset_index(drop=True)

        if "label_space" in self.meta.columns:
            found = set(self.meta["label_space"].astype(str).unique().tolist())
            if found != {label_space}:
                raise ValueError(
                    f"Unified metadata label_space mismatch for {group_id}/{split}: "
                    f"found {sorted(found)}, expected '{label_space}'"
                )

        label_col = "label_code"
        if label_col not in self.meta.columns:
            fallback_col = "fault_code" if label_space == "fine" else "coarse_fault_code"
            if fallback_col not in self.meta.columns:
                raise ValueError(f"Missing '{label_col}' and '{fallback_col}' in unified metadata")
            label_col = fallback_col

        all_classes = sorted(self.meta[label_col].astype(str).unique().tolist())
        self.encoder = encoder if encoder is not None else LabelEncoder(all_classes)
        self.labels = self.encoder.encode_array(self.meta[label_col].astype(str).tolist())

        self._aug = aug_cfg if split == "train" else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx]).float()
        x = _apply_augmentation(x, self._aug)
        return x, int(self.labels[idx])

    @property
    def input_channels(self) -> int:
        if self.windows.ndim == 3:
            return int(self.windows.shape[1])
        return 1

    @property
    def num_classes(self) -> int:
        return len(self.encoder)

    @property
    def class_names(self) -> list[str]:
        return self.encoder.classes_

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        w = w / w.sum() * self.num_classes
        return torch.tensor(w, dtype=torch.float32)

    def summary(self) -> pd.DataFrame:
        df = self.meta.copy()
        df["label_idx"] = self.labels
        group_cols = ["label_idx"]
        if "label_code" in df.columns:
            group_cols.insert(0, "label_code")
        if "dataset_id" in df.columns:
            group_cols.append("dataset_id")
        return (
            df.groupby(group_cols)
            .size()
            .reset_index(name="n_windows")
            .sort_values(group_cols)
        )


class DatasetView(Dataset):
    """
    Lightweight dataset view over an existing loaded dataset.

    Keeps file-level grouping metadata intact so downstream evaluation can still
    aggregate windows back to source files.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: np.ndarray,
        class_names: list[str],
        aug_cfg: Optional[dict] = None,
        label_space: str = "fine",
        return_domain_id: bool = False,
        domain_id_map: Optional[dict[str, int]] = None,
        domain_label_source: str = "sub_dataset",
    ):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.windows = np.asarray(base_dataset.windows)[self.indices]
        self.meta = base_dataset.meta.iloc[self.indices].reset_index(drop=True).copy()
        self.labels = np.asarray(base_dataset.labels)[self.indices]
        self._class_names = list(class_names)
        self._aug = aug_cfg
        self.label_space = label_space
        self.return_domain_id = bool(return_domain_id)
        self.domain_id_map = dict(domain_id_map or {})
        self.domain_label_source = str(domain_label_source)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx]).float()
        x = _apply_augmentation(x, self._aug)
        if self.return_domain_id:
            key = _domain_key_from_row(self.meta.iloc[idx], self.domain_label_source)
            return x, int(self.labels[idx]), int(self.domain_id_map.get(key, 0))
        return x, int(self.labels[idx])

    @property
    def input_channels(self) -> int:
        if self.windows.ndim == 3:
            return int(self.windows.shape[1])
        return 1

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        w = w / w.sum() * self.num_classes
        return torch.tensor(w, dtype=torch.float32)

    def summary(self) -> pd.DataFrame:
        df = self.meta.copy()
        df["label_idx"] = self.labels
        label_col = "fault_code" if self.label_space == "fine" else "coarse_fault_code"
        if label_col not in df.columns and "label_code" in df.columns:
            label_col = "label_code"
        return (
            df.groupby([label_col, "label_idx"])
            .size()
            .reset_index(name="n_windows")
            .sort_values(["label_idx"])
        )


def _label_column(meta: pd.DataFrame, label_space: str) -> str:
    preferred = "fault_code" if label_space == "fine" else "coarse_fault_code"
    if preferred in meta.columns:
        return preferred
    if "label_code" in meta.columns:
        return "label_code"
    raise ValueError(f"Could not infer label column for label_space='{label_space}'")


def _pool_processed_dataset(
    processed_dir: Union[str, Path],
    dataset_ids: Union[int, list[int]],
    label_space: str = "fine",
    exclude_ineligible: bool = True,
    processed_group: Optional[str] = None,
    channel_align: Optional[Union[int, str]] = None,
) -> FaultDataset:
    pooled = FaultDataset(
        processed_dir=processed_dir,
        dataset_ids=dataset_ids,
        split="all",
        label_space=label_space,
        encoder=None,
        exclude_ineligible=False,
        aug_cfg=None,
        processed_group=processed_group,
        channel_align=channel_align,
    )
    if "split" in pooled.meta.columns:
        valid_mask = pooled.meta["split"].astype(str).isin(["train", "val", "test"]).to_numpy()
        if exclude_ineligible and "test_eligible" in pooled.meta.columns:
            valid_mask &= pooled.meta["test_eligible"].astype(bool).to_numpy()
        keep_idx = np.flatnonzero(valid_mask)
        pooled = DatasetView(
            base_dataset=pooled,
            indices=keep_idx,
            class_names=pooled.class_names,
            aug_cfg=None,
            label_space=label_space,
        )
    return pooled


def _pool_unified_dataset(
    unified_dir: Union[str, Path],
    group_id: str,
    label_space: str = "fine",
    exclude_ineligible: bool = True,
) -> DatasetView:
    splits = []
    for split in ("train", "val", "test"):
        try:
            ds = UnifiedFaultDataset(
                unified_dir=unified_dir,
                group_id=group_id,
                split=split,
                label_space=label_space,
                encoder=None,
                exclude_ineligible=False,
                aug_cfg=None,
            )
            splits.append(ds)
        except FileNotFoundError:
            continue
    if not splits:
        raise FileNotFoundError(
            f"No unified train/val/test windows found for group='{group_id}' in {Path(unified_dir) / group_id}"
        )

    windows = np.vstack([np.asarray(ds.windows) for ds in splits])
    meta = pd.concat([ds.meta for ds in splits], ignore_index=True)
    label_col = _label_column(meta, label_space)
    classes = sorted(meta[label_col].astype(str).unique().tolist())
    pooled = DatasetView.__new__(DatasetView)
    pooled.base_dataset = None
    pooled.indices = np.arange(len(meta), dtype=np.int64)
    pooled.windows = windows
    pooled.meta = meta
    encoder = LabelEncoder(classes)
    pooled.labels = encoder.encode_array(meta[label_col].astype(str).tolist())
    pooled._class_names = classes
    pooled._aug = None
    pooled.label_space = label_space

    if exclude_ineligible and "test_eligible" in pooled.meta.columns:
        keep_idx = np.flatnonzero(pooled.meta["test_eligible"].astype(bool).to_numpy())
        pooled = DatasetView(
            base_dataset=pooled,
            indices=keep_idx,
            class_names=classes,
            aug_cfg=None,
            label_space=label_space,
        )
    return pooled


def _build_group_folds(
    meta: pd.DataFrame,
    label_space: str,
    n_splits: int,
    seed: int,
) -> list[set[str]]:
    if "source_file" not in meta.columns:
        raise ValueError("Cross-validation requires 'source_file' in metadata")

    label_col = _label_column(meta, label_space)
    file_labels = (
        meta[["source_file", label_col]]
        .astype({label_col: str})
        .drop_duplicates("source_file")
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for _, grp in file_labels.groupby(label_col):
        files = grp["source_file"].astype(str).tolist()
        rng.shuffle(files)
        for i, src in enumerate(files):
            candidate = i % n_splits
            min_size = min(fold_sizes)
            tied = [j for j, size in enumerate(fold_sizes) if size == min_size]
            if candidate not in tied:
                candidate = tied[0]
            folds[candidate].append(src)
            fold_sizes[candidate] += 1

    return [set(fold) for fold in folds]


def _split_train_val_groups(
    meta: pd.DataFrame,
    candidate_groups: set[str],
    label_space: str,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    label_col = _label_column(meta, label_space)
    file_labels = (
        meta.loc[meta["source_file"].astype(str).isin(candidate_groups), ["source_file", label_col]]
        .astype({label_col: str})
        .drop_duplicates("source_file")
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    train_groups: set[str] = set()
    val_groups: set[str] = set()

    for _, grp in file_labels.groupby(label_col):
        files = grp["source_file"].astype(str).tolist()
        rng.shuffle(files)
        if len(files) <= 1:
            train_groups.update(files)
            continue
        n_val = int(round(len(files) * val_ratio))
        n_val = max(1, n_val)
        if n_val >= len(files):
            n_val = len(files) - 1
        val_groups.update(files[:n_val])
        train_groups.update(files[n_val:])

    leftovers = set(candidate_groups) - train_groups - val_groups
    train_groups.update(leftovers)
    return train_groups, val_groups


def build_crossval_dataloaders(
    data_mode: str,
    label_space: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    exclude_ineligible: bool = True,
    aug_cfg: Optional[dict] = None,
    n_splits: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
    processed_dir: Optional[Union[str, Path]] = None,
    dataset_ids: Optional[Union[int, list[int]]] = None,
    unified_dir: Optional[Union[str, Path]] = None,
    group_id: Optional[str] = None,
    processed_group: Optional[str] = None,
    channel_align: Optional[Union[int, str]] = None,
    unified_train_sampling: str = "dataset_balanced",
    return_domain_id: bool = False,
    domain_label_source: str = "sub_dataset",
) -> list[dict[str, Optional[DataLoader]]]:
    """
    Build leakage-safe cross-validation folds using source files as the grouping unit.

    The pooled dataset is assembled from the existing non-robustness windows, then
    split by `source_file` so no windows from one raw recording can leak across folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if data_mode == "processed":
        pooled = _pool_processed_dataset(
            processed_dir=processed_dir,
            dataset_ids=dataset_ids,
            label_space=label_space,
            exclude_ineligible=exclude_ineligible,
            processed_group=processed_group,
            channel_align=channel_align,
        )
    elif data_mode == "unified":
        pooled = _pool_unified_dataset(
            unified_dir=unified_dir,
            group_id=group_id,
            label_space=label_space,
            exclude_ineligible=exclude_ineligible,
        )
    else:
        raise ValueError(f"Unsupported data_mode={data_mode!r}")

    folds = _build_group_folds(pooled.meta, label_space=label_space, n_splits=n_splits, seed=seed)
    class_names = pooled.class_names
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    results: list[dict[str, Optional[DataLoader]]] = []

    all_groups = set(pooled.meta["source_file"].astype(str).unique().tolist())

    for fold_idx in range(n_splits):
        test_groups = folds[fold_idx]
        train_candidate_groups = all_groups - test_groups
        train_groups, val_groups = _split_train_val_groups(
            pooled.meta,
            candidate_groups=train_candidate_groups,
            label_space=label_space,
            val_ratio=val_ratio,
            seed=seed + fold_idx + 1,
        )

        meta_source = pooled.meta["source_file"].astype(str)
        train_idx = np.flatnonzero(meta_source.isin(train_groups).to_numpy())
        val_idx = np.flatnonzero(meta_source.isin(val_groups).to_numpy())
        test_idx = np.flatnonzero(meta_source.isin(test_groups).to_numpy())

        train_meta = pooled.meta.iloc[train_idx]
        domain_id_map = build_domain_id_map(train_meta, domain_label_source) if return_domain_id else {}

        train_ds = DatasetView(
            pooled,
            train_idx,
            class_names=class_names,
            aug_cfg=aug_cfg,
            label_space=label_space,
            return_domain_id=return_domain_id,
            domain_id_map=domain_id_map,
            domain_label_source=domain_label_source,
        )
        val_ds = DatasetView(
            pooled,
            val_idx,
            class_names=class_names,
            aug_cfg=None,
            label_space=label_space,
            return_domain_id=return_domain_id,
            domain_id_map=domain_id_map,
            domain_label_source=domain_label_source,
        )
        test_ds = DatasetView(
            pooled,
            test_idx,
            class_names=class_names,
            aug_cfg=None,
            label_space=label_space,
            return_domain_id=return_domain_id,
            domain_id_map=domain_id_map,
            domain_label_source=domain_label_source,
        )

        train_loader_kwargs = dict(loader_kwargs)
        sampler = None
        if data_mode == "unified" and unified_train_sampling == "dataset_balanced":
            sampler = _build_dataset_balanced_sampler(train_ds)
        elif data_mode == "unified" and unified_train_sampling != "none":
            raise ValueError(
                f"Unsupported unified_train_sampling='{unified_train_sampling}'. "
                "Expected 'dataset_balanced' or 'none'."
            )
        if sampler is not None:
            train_loader_kwargs["sampler"] = sampler
            train_loader_kwargs["shuffle"] = False
        else:
            train_loader_kwargs["shuffle"] = True

        results.append(
            {
                "fold_idx": fold_idx,
                "train": DataLoader(train_ds, drop_last=True, **train_loader_kwargs),
                "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
                "test": DataLoader(test_ds, shuffle=False, **loader_kwargs),
                "robustness": None,
            }
        )

    return results


def build_dataloaders(
    processed_dir: Union[str, Path],
    dataset_ids: Union[int, list[int]],
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    exclude_ineligible: bool = True,
    aug_cfg: Optional[dict] = None,
    processed_group: Optional[str] = None,
    channel_align: Optional[Union[int, str]] = None,
    label_space: str = "fine",
    require_val: bool = True,
    require_test: bool = True,
    return_domain_id: bool = False,
    domain_id_map: Optional[dict[str, int]] = None,
    domain_label_source: str = "sub_dataset",
) -> dict[str, Optional[DataLoader]]:
    """
    Build train/val/test/robustness DataLoaders sharing one train-fitted label
    encoder. The robustness key is None when no robustness windows exist.

    channel_align: 'min', 'max', or an int — aligns channel counts when combining
                   datasets with mismatched channels (e.g. D6=1ch + D7=8ch).
                   'min' truncates to the smallest channel count; 'max' pads by
                   repeating channels cyclically; an int takes the first N channels.
    """
    availability = inspect_processed_splits(processed_dir, dataset_ids, processed_group=processed_group)
    if not availability["train"]:
        raise FileNotFoundError(
            f"No train windows found for datasets {dataset_ids} in {processed_dir}"
        )
    if require_val and not availability["val"]:
        raise FileNotFoundError(
            f"No val windows found for datasets {dataset_ids} in {processed_dir}"
        )
    if require_test and not availability["test"]:
        raise FileNotFoundError(
            f"No test windows found for datasets {dataset_ids} in {processed_dir}"
        )

    _ds_kwargs = dict(processed_group=processed_group, channel_align=channel_align, label_space=label_space)

    train_ds = FaultDataset(
        processed_dir, dataset_ids, split="train", encoder=None,
        exclude_ineligible=False, aug_cfg=aug_cfg, **_ds_kwargs,
        return_domain_id=return_domain_id,
        domain_id_map=domain_id_map,
        domain_label_source=domain_label_source,
    )
    if return_domain_id and not domain_id_map:
        domain_id_map = build_domain_id_map(train_ds.meta, domain_label_source)
        train_ds.domain_id_map = domain_id_map
    encoder = train_ds.encoder

    val_ds = FaultDataset(
        processed_dir, dataset_ids, split="val", encoder=encoder,
        exclude_ineligible=False, **_ds_kwargs,
        return_domain_id=return_domain_id,
        domain_id_map=domain_id_map,
        domain_label_source=domain_label_source,
    )
    test_ds = None
    if availability["test"]:
        test_ds = FaultDataset(
            processed_dir, dataset_ids, split="test", encoder=encoder,
            exclude_ineligible=exclude_ineligible, **_ds_kwargs,
            return_domain_id=return_domain_id,
            domain_id_map=domain_id_map,
            domain_label_source=domain_label_source,
        )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    try:
        rob_ds = FaultDataset(
            processed_dir, dataset_ids, split="robustness", encoder=encoder,
            exclude_ineligible=False, **_ds_kwargs,
            return_domain_id=return_domain_id,
            domain_id_map=domain_id_map,
            domain_label_source=domain_label_source,
        )
        rob_loader: Optional[DataLoader] = DataLoader(rob_ds, shuffle=False, **loader_kwargs)
    except FileNotFoundError:
        rob_loader = None

    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs),
        "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        "test": None if test_ds is None else DataLoader(test_ds, shuffle=False, **loader_kwargs),
        "robustness": rob_loader,
        "availability": availability,
    }


def build_unified_dataloaders(
    unified_dir: Union[str, Path],
    group_id: str,
    label_space: str = "fine",
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    exclude_ineligible: bool = True,
    aug_cfg: Optional[dict] = None,
    train_sampling: str = "dataset_balanced",
    require_val: bool = True,
    require_test: bool = True,
) -> dict[str, Optional[DataLoader]]:
    """
    Build train/val/test/robustness DataLoaders from one unified group.
    """
    availability = inspect_unified_splits(unified_dir, group_id)
    if not availability["train"]:
        raise FileNotFoundError(
            f"No unified train windows found for group='{group_id}' in {Path(unified_dir) / group_id}"
        )
    if require_val and not availability["val"]:
        raise FileNotFoundError(
            f"No unified val windows found for group='{group_id}' in {Path(unified_dir) / group_id}"
        )
    if require_test and not availability["test"]:
        raise FileNotFoundError(
            f"No unified test windows found for group='{group_id}' in {Path(unified_dir) / group_id}"
        )

    train_ds = UnifiedFaultDataset(
        unified_dir,
        group_id,
        split="train",
        label_space=label_space,
        encoder=None,
        exclude_ineligible=False,
        aug_cfg=aug_cfg,
    )
    encoder = train_ds.encoder

    val_ds = UnifiedFaultDataset(
        unified_dir,
        group_id,
        split="val",
        label_space=label_space,
        encoder=encoder,
        exclude_ineligible=False,
    )
    test_ds = None
    if availability["test"]:
        test_ds = UnifiedFaultDataset(
            unified_dir,
            group_id,
            split="test",
            label_space=label_space,
            encoder=encoder,
            exclude_ineligible=exclude_ineligible,
        )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_kwargs = dict(loader_kwargs)
    train_sampler = None
    if train_sampling == "dataset_balanced":
        train_sampler = _build_dataset_balanced_sampler(train_ds)
    elif train_sampling != "none":
        raise ValueError(
            f"Unsupported unified train_sampling='{train_sampling}'. "
            "Expected 'dataset_balanced' or 'none'."
        )

    if train_sampler is not None:
        train_loader_kwargs["sampler"] = train_sampler
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True

    try:
        rob_ds = UnifiedFaultDataset(
            unified_dir,
            group_id,
            split="robustness",
            label_space=label_space,
            encoder=encoder,
            exclude_ineligible=False,
        )
        rob_loader: Optional[DataLoader] = DataLoader(rob_ds, shuffle=False, **loader_kwargs)
    except FileNotFoundError:
        rob_loader = None

    return {
        "train": DataLoader(train_ds, drop_last=True, **train_loader_kwargs),
        "val": DataLoader(val_ds, shuffle=False, **loader_kwargs),
        "test": None if test_ds is None else DataLoader(test_ds, shuffle=False, **loader_kwargs),
        "robustness": rob_loader,
        "availability": availability,
    }
