from __future__ import annotations

import json
import warnings
from math import gcd
from numpy.lib.format import open_memmap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from mspf_net.constants import FS_NATIVE, get_fs_native

FS_TARGET = 10_000
SPLIT_ORDER = ["train", "val", "test", "robustness"]
LEGACY_CODES = {"GBX_SW", "GBX_TB"}
LEGACY_OUTPUT_PREFIXES: dict[int, list[str]] = {
    31: ["d3_c1"],
    39: ["d3_c9"],
}


def resample_signal(signal: np.ndarray, fs_in: int, fs_out: int = FS_TARGET) -> np.ndarray:
    """Polyphase rational resampling. Returns float64."""
    if fs_in == fs_out:
        return signal.astype(np.float64)
    g = gcd(int(fs_in), int(fs_out))
    up = int(fs_out) // g
    down = int(fs_in) // g
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axis = 0 if np.ndim(signal) > 1 else -1
        return resample_poly(signal.astype(np.float64), up, down, axis=axis)


def make_windows(signal: np.ndarray, win_size: int = 2048, hop: int = 1024) -> np.ndarray:
    """Sliding-window segmentation. Returns windows as (M, C, L)."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    if n < win_size:
        return np.empty((0, x.shape[1], win_size), dtype=np.float64)
    starts = np.arange(0, n - win_size + 1, hop)
    return np.stack([x[s : s + win_size].T for s in starts], axis=0)


def align_signal_channels(
    signal: np.ndarray,
    expected_channels: int | float | None,
) -> np.ndarray:
    """
    Force each loaded signal to the catalog-declared channel count.

    This prevents non-signal columns or loader-specific matrix layouts from
    creating mixed channel counts inside one dataset.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if expected_channels is None or pd.isna(expected_channels):
        return x

    expected = int(expected_channels)
    if expected <= 0:
        return x

    actual = int(x.shape[1])
    if actual == expected:
        return x
    if actual > expected:
        variances = np.var(x, axis=0)
        keep = np.argsort(variances)[::-1][:expected]
        keep = np.sort(keep)
        return x[:, keep]
    raise ValueError(f"Loaded only {actual} channel(s), expected {expected}")


def fit_normalizer(windows: np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Fit a train-only normalizer on Phase 3 windows.

    For multichannel data, use one mean/std pair per channel so a high-amplitude
    sensor cannot suppress lower-amplitude channels during z-score scaling.
    """
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected windows with shape (N, C, L); got {x.shape}")
    mean = x.mean(axis=(0, 2))
    std = x.std(axis=(0, 2))
    std = np.where(std < 1e-12, 1.0, std)
    if mean.size == 1:
        return float(mean[0]), float(std[0])
    return mean.astype(np.float64), std.astype(np.float64)


def apply_normalizer(
    windows: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    if mean_arr.ndim == 0:
        return ((x - float(mean_arr)) / float(std_arr)).astype(np.float32)
    return ((x - mean_arr.reshape(1, -1, 1)) / std_arr.reshape(1, -1, 1)).astype(np.float32)


def load_standardized_catalog(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "interim" / "catalog_standardized.csv"
    if not path.exists():
        raise FileNotFoundError(
            "data/interim/catalog_standardized.csv not found. "
            "Run: python scripts/run_standardize_catalog.py"
        )
    df = pd.read_csv(path)
    missing = [c for c in ["file_path", "fault_code", "coarse_fault_code", "split"] if c not in df.columns]
    if missing:
        raise ValueError(f"catalog_standardized.csv is missing required columns: {missing}")
    legacy = sorted(set(df["fault_code"]).intersection(LEGACY_CODES))
    if legacy:
        raise ValueError(f"Legacy fault codes remain in catalog_standardized.csv: {legacy}")
    return df


def validate_file_level_splits(catalog: pd.DataFrame) -> None:
    dup = (
        catalog.groupby("file_path")["split"]
        .nunique()
        .reset_index(name="n_splits")
        .query("n_splits > 1")
    )
    if not dup.empty:
        raise ValueError(
            "A raw file appears in multiple splits: "
            + ", ".join(dup["file_path"].astype(str).tolist()[:5])
        )


def cleanup_dataset_outputs(out_dir: Path, ds_id: int, verbose: bool = True) -> None:
    """
    Remove all prior outputs for one dataset before rewriting them.

    This prevents stale split files from surviving reruns when the current
    preprocessing pass produces a different set of splits or window shapes.
    """
    prefixes = [f"d{ds_id}", *LEGACY_OUTPUT_PREFIXES.get(int(ds_id), [])]
    patterns = []
    for prefix in prefixes:
        patterns.extend(
            [
                f"{prefix}_*_windows.npy",
                f"{prefix}_*_meta.csv",
                f"{prefix}_norm.json",
            ]
        )
    removed = 0
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    if verbose and removed:
        print(f"    Cleared {removed} stale output file(s)")


def _output_prefix(ds_id: int, partition_name: str | None = None) -> str:
    if not partition_name:
        return f"d{ds_id}"
    return f"d{ds_id}_{partition_name}"


def _partition_dataset_rows(subset: pd.DataFrame) -> list[tuple[str | None, pd.DataFrame]]:
    """
    Split one dataset into homogeneous preprocessing partitions.

    Most datasets stay as a single partition and keep the historical
    `d{ds_id}_<split>` output naming. Datasets that genuinely mix channel
    counts are emitted as separate channel groups so all raw channels can be
    preserved without forcing lossy alignment.
    """
    channel_counts = sorted(subset["n_channels"].dropna().astype(int).unique().tolist())
    if len(channel_counts) <= 1:
        return [(None, subset.copy())]

    partitions = []
    for n_channels, part in subset.groupby("n_channels", dropna=False):
        label = f"c{int(n_channels)}"
        partitions.append((label, part.copy()))
    return partitions


def _count_windows_for_row(row: pd.Series, project_root: Path, win_size: int, hop: int, fs_target: int) -> int:
    from mspf_net.eda.eda_utils import load_signal_multichannel

    abs_path = project_root / str(row["file_path"])
    if not abs_path.exists():
        return 0
    sig = load_signal_multichannel(abs_path, int(row["dataset_id"]))
    sig = align_signal_channels(sig, row.get("n_channels"))
    fs_in = get_fs_native(int(row["dataset_id"]))
    if fs_in != fs_target:
        sig = resample_signal(sig, fs_in, fs_target)
    return int(make_windows(sig, win_size, hop).shape[0])


def estimate_dataset_windows(
    catalog: pd.DataFrame,
    ds_id: int,
    project_root: Path,
    win_size: int = 2048,
    hop: int = 1024,
    fs_target: int = FS_TARGET,
) -> dict:
    subset = catalog[catalog["dataset_id"] == ds_id].copy()
    counts = {split: 0 for split in SPLIT_ORDER}
    file_counts = {split: 0 for split in SPLIT_ORDER}

    for _, row in subset.iterrows():
        split = str(row["split"])
        if split not in counts:
            continue
        n_wins = _count_windows_for_row(row, project_root, win_size, hop, fs_target)
        if n_wins == 0:
            continue
        counts[split] += n_wins
        file_counts[split] += 1

    return {
        "n_windows": sum(counts.values()),
        **counts,
        **{f"{split}_files": file_counts[split] for split in SPLIT_ORDER},
    }


def preprocess_dataset(
    ds_id: int,
    catalog: pd.DataFrame,
    project_root: Path,
    out_dir: Path,
    win_size: int = 2048,
    hop: int = 1024,
    fs_target: int = FS_TARGET,
    verbose: bool = True,
) -> dict:
    """
    Preprocess one dataset using the file-level split already stored in the
    standardized catalog. Every generated window inherits its parent-file split.
    """
    from mspf_net.eda.eda_utils import load_signal_multichannel

    validate_file_level_splits(catalog)
    subset = catalog[catalog["dataset_id"] == ds_id].copy()
    if subset.empty:
        raise RuntimeError(f"No catalog rows found for dataset {ds_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_dataset_outputs(out_dir, ds_id, verbose=verbose)

    if verbose:
        split_file_counts = subset["split"].value_counts().to_dict()
        print(f"    Files to process: {len(subset)}")
        for split in SPLIT_ORDER:
            n_files = split_file_counts.get(split, 0)
            if n_files:
                print(f"    {split:10s} {n_files:>4} files")

    def load_file_windows(row: pd.Series) -> tuple[str, str, np.ndarray] | None:
        split = str(row["split"])
        if split not in SPLIT_ORDER:
            return None

        rel_path = str(row["file_path"])
        abs_path = project_root / rel_path
        if not abs_path.exists():
            if verbose:
                print(f"    [SKIP] not found: {abs_path.name}")
            return None

        try:
            sig = load_signal_multichannel(abs_path, ds_id)
            sig = align_signal_channels(sig, row.get("n_channels"))
        except Exception as e:
            if verbose:
                print(f"    [ERR]  {abs_path.name}: {e}")
            return None

        fs_in = get_fs_native(ds_id)
        if fs_in != fs_target:
            sig = resample_signal(sig, fs_in, fs_target)

        wins = make_windows(sig, win_size, hop)
        if wins.shape[0] == 0:
            if verbose:
                print(f"    [SKIP] too short: {abs_path.name}")
            return None
        return split, rel_path, wins

    partitions = _partition_dataset_rows(subset)
    if verbose and len(partitions) > 1:
        labels = ", ".join(f"{name}" for name, _ in partitions)
        print(f"    Mixed channel dataset: writing partitions {labels}")

    aggregate_split_counts: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
    aggregate_file_counts: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
    partition_summaries: list[dict] = []

    for partition_name, part_df in partitions:
        prefix = _output_prefix(ds_id, partition_name)
        split_counts: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
        file_counts: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
        meta_rows_by_split: dict[str, list[dict]] = {split: [] for split in SPLIT_ORDER}
        source_file_split: dict[str, str] = {}
        preserved_channels: int | None = None

        train_windows_for_norm: list[np.ndarray] = []

        for _, row in part_df.iterrows():
            loaded = load_file_windows(row)
            if loaded is None:
                continue
            split, rel_path, wins = loaded

            prev_split = source_file_split.setdefault(rel_path, split)
            if prev_split != split:
                raise ValueError(f"Source file assigned to multiple splits: {rel_path}")

            if preserved_channels is None:
                preserved_channels = int(wins.shape[1])
            elif int(wins.shape[1]) != preserved_channels:
                raise ValueError(
                    f"Dataset {ds_id} partition {prefix} produced mixed channel counts "
                    f"({preserved_channels} vs {wins.shape[1]}). Check loader heuristics and catalog n_channels."
                )

            split_counts[split] += int(wins.shape[0])
            file_counts[split] += 1

            if split == "train":
                train_windows_for_norm.append(wins)

        if split_counts["train"] == 0 or not train_windows_for_norm:
            raise RuntimeError(f"No train windows produced for dataset {ds_id} partition {prefix}")

        train_windows = np.vstack(train_windows_for_norm)
        mean, std = fit_normalizer(train_windows)

        memmaps: dict[str, np.memmap] = {}
        write_offsets: dict[str, int] = {split: 0 for split in SPLIT_ORDER}

        for split in SPLIT_ORDER:
            n_split = split_counts.get(split, 0)
            if n_split <= 0:
                continue
            memmaps[split] = open_memmap(
                out_dir / f"{prefix}_{split}_windows.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_split, int(preserved_channels), win_size),
            )

        for _, row in part_df.iterrows():
            loaded = load_file_windows(row)
            if loaded is None:
                continue
            split, rel_path, wins = loaded
            n_wins = int(wins.shape[0])
            start = write_offsets[split]
            end = start + n_wins
            memmaps[split][start:end] = apply_normalizer(wins, mean, std)
            write_offsets[split] = end

            meta_rows_by_split[split].extend(
                {
                    "dataset_id": ds_id,
                    "fault_code": row["fault_code"],
                    "coarse_fault_code": row.get("coarse_fault_code"),
                    "split": split,
                    "test_eligible": bool(row.get("test_eligible", True)),
                    "source_file": rel_path,
                    "std_name": row.get("std_name"),
                    "sub_dataset": row.get("sub_dataset"),
                    "processed_group": partition_name or "default",
                    "n_channels": int(preserved_channels),
                    "start_sample": i * hop,
                }
                for i in range(n_wins)
            )

        for split in SPLIT_ORDER:
            if split_counts.get(split, 0) <= 0:
                continue

            memmaps[split].flush()
            split_meta = pd.DataFrame(meta_rows_by_split[split])
            split_meta.insert(0, "window_idx", np.arange(len(split_meta)))

            bad_sources = (
                split_meta.groupby("source_file")["split"]
                .nunique()
                .reset_index(name="n_splits")
                .query("n_splits > 1")
            )
            if not bad_sources.empty:
                raise ValueError(f"Window split leakage detected in dataset {ds_id}")

            split_meta.to_csv(out_dir / f"{prefix}_{split}_meta.csv", index=False)

        with open(out_dir / f"{prefix}_norm.json", "w") as fh:
            json.dump(
                {
                    "mean": np.asarray(mean).tolist() if np.ndim(mean) else float(mean),
                    "std": np.asarray(std).tolist() if np.ndim(std) else float(std),
                    "normalization": "per_channel_zscore" if int(preserved_channels) > 1 else "scalar_zscore",
                    "fit_on": "train",
                    "split_strategy": "file_level_catalog",
                    "win_size": win_size,
                    "hop": hop,
                    "fs_native": get_fs_native(ds_id),
                    "fs_target": fs_target,
                    "window_counts": split_counts,
                    "file_counts": file_counts,
                    "channels_preserved": True,
                    "processed_group": partition_name or "default",
                    "n_channels": int(preserved_channels),
                },
                fh,
                indent=2,
            )

        if verbose:
            total = sum(split_counts.values())
            label = f" [{partition_name}]" if partition_name else ""
            for split in SPLIT_ORDER:
                n = split_counts.get(split, 0)
                if n:
                    print(f"    {split:10s} {n:>8,} windows{label}")
            print(f"    total       {total:>8,} windows{label}")

        for split in SPLIT_ORDER:
            aggregate_split_counts[split] += split_counts.get(split, 0)
            aggregate_file_counts[split] += file_counts.get(split, 0)

        partition_summaries.append(
            {
                "prefix": prefix,
                "partition": partition_name or "default",
                "n_channels": int(preserved_channels),
                "norm_mean": (
                    round(float(mean), 6)
                    if np.ndim(mean) == 0
                    else [round(float(v), 6) for v in np.asarray(mean).tolist()]
                ),
                "norm_std": (
                    round(float(std), 6)
                    if np.ndim(std) == 0
                    else [round(float(v), 6) for v in np.asarray(std).tolist()]
                ),
                "split_counts": split_counts,
                "file_counts": file_counts,
                "n_windows": sum(split_counts.values()),
            }
        )

    channel_summary = (
        partition_summaries[0]["n_channels"]
        if len(partition_summaries) == 1
        else ",".join(str(p["n_channels"]) for p in partition_summaries)
    )

    return {
        "n_windows": sum(aggregate_split_counts.values()),
        "n_train": aggregate_split_counts.get("train", 0),
        "n_val": aggregate_split_counts.get("val", 0),
        "n_test": aggregate_split_counts.get("test", 0),
        "n_robustness": aggregate_split_counts.get("robustness", 0),
        "norm_mean": partition_summaries[0]["norm_mean"] if len(partition_summaries) == 1 else np.nan,
        "norm_std": partition_summaries[0]["norm_std"] if len(partition_summaries) == 1 else np.nan,
        "n_channels": channel_summary,
        "file_counts": aggregate_file_counts,
        "split_counts": aggregate_split_counts,
        "partitions": partition_summaries,
    }
