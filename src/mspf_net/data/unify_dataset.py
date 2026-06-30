from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mspf_net.config_utils import get_config_path, get_config_value, get_unified_group_ids
from mspf_net.constants import ACTIVE_THESIS_DATASETS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / get_config_path("paths", "data_processed", default="data/processed")
STD_CATALOG = PROJECT_ROOT / "data" / "interim" / "catalog_standardized.csv"
UNIFIED_DIR = PROJECT_ROOT / get_config_path("paths", "data_unified", default="data/unified")
SPLITS = ["train", "val", "test", "robustness"]
COMPONENT_GROUPS = tuple(get_unified_group_ids())
N_SEGMENTS = int(get_config_value("unified", "n_segments", default=64))
FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "peak_abs",
    "peak_to_peak",
    "crest_factor",
    "skewness",
    "kurtosis",
    "energy",
    "spectral_entropy",
]
AGGREGATIONS = ["channel_mean"]
EVIDENCE_DIM_NAMES = [f"{agg}_{name}" for name in FEATURE_NAMES for agg in AGGREGATIONS]
FEATURE_CHUNK_SIZE = 32768


def load_standardized_catalog() -> pd.DataFrame:
    if not STD_CATALOG.exists():
        raise FileNotFoundError(
            "data/interim/catalog_standardized.csv not found. "
            "Run: python scripts/run_standardize_catalog.py"
        )
    df = pd.read_csv(STD_CATALOG)
    required = ["dataset_id", "file_path", "fault_code", "coarse_fault_code", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"catalog_standardized.csv is missing required columns: {missing}")
    return df


def _iter_split_exports(ds_id: int, split: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    patterns = [
        f"d{ds_id}_{split}_windows.npy",
        f"d{ds_id}_*_{split}_windows.npy",
    ]
    for pattern in patterns:
        for npy_path in sorted(PROCESSED_DIR.glob(pattern)):
            csv_path = npy_path.with_name(npy_path.name.replace("_windows.npy", "_meta.csv"))
            if not csv_path.exists():
                continue
            key = (str(npy_path), str(csv_path))
            if key not in seen:
                seen.add(key)
                pairs.append((npy_path, csv_path))
    return pairs


def _read_split_exports(
    ds_id: int,
    split: str,
    load_windows: bool = True,
) -> list[tuple[np.ndarray | None, pd.DataFrame, Path, Path]]:
    loaded = []
    for npy_path, csv_path in _iter_split_exports(ds_id, split):
        meta = pd.read_csv(csv_path)
        windows = None
        if load_windows:
            windows = np.load(str(npy_path))
            if len(windows) != len(meta):
                raise ValueError(f"Shape mismatch between {npy_path.name} and {csv_path.name}")
        loaded.append((windows, meta, npy_path, csv_path))
    return loaded


def _prepare_meta(meta: pd.DataFrame, label_space: str, component_group: str) -> pd.DataFrame:
    meta = meta.copy()
    label_col = "fault_code" if label_space == "fine" else "coarse_fault_code"
    if label_col not in meta.columns:
        raise ValueError(f"Missing label column '{label_col}' in processed metadata")
    if meta[label_col].isna().any():
        missing = sorted(meta.loc[meta[label_col].isna(), "source_file"].astype(str).unique().tolist()[:5])
        raise ValueError(f"Missing {label_col} values in processed metadata, examples: {missing}")

    meta["label_space"] = label_space
    meta["label_code"] = meta[label_col].astype(str)
    meta["unified_view"] = "component_md"
    meta["component_group"] = component_group
    meta["source_n_channels"] = meta.get("n_channels", pd.NA)
    meta["evidence_n_dims"] = len(EVIDENCE_DIM_NAMES)
    meta["evidence_n_segments"] = N_SEGMENTS
    return meta


def _component_mask(meta: pd.DataFrame, component_group: str) -> pd.Series:
    fault = meta["fault_code"].astype(str)
    sub_dataset = meta.get("sub_dataset", pd.Series("", index=meta.index)).astype(str).str.lower()

    if component_group == "bearing_md":
        component_like = sub_dataset.str.contains("bearing|cwru", regex=True)
        label_like = fault.str.startswith("BRG_") | fault.eq("NOR")
        return component_like & label_like

    if component_group == "gearbox_md":
        component_like = sub_dataset.str.contains("gearbox|gearset", regex=True)
        label_like = fault.str.startswith("GBX_") | fault.eq("NOR")
        return component_like & label_like

    raise ValueError(f"Unsupported component group: {component_group}")


def _as_3d(windows: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported window array rank: {arr.ndim}")


def _safe_std(x: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.maximum(np.var(x, axis=axis), 1e-12))


def _skewness(x: np.ndarray, axis: int) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = _safe_std(x, axis=axis)
    centered = x - mean
    third = np.mean(centered ** 3, axis=axis)
    return third / np.maximum(std ** 3, 1e-12)


def _kurtosis(x: np.ndarray, axis: int) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = _safe_std(x, axis=axis)
    centered = x - mean
    fourth = np.mean(centered ** 4, axis=axis)
    return fourth / np.maximum(std ** 4, 1e-12)


def _spectral_entropy(x: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(x, axis=2)) ** 2
    probs = spectrum / np.maximum(np.sum(spectrum, axis=2, keepdims=True), 1e-12)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=2)
    return entropy / np.maximum(np.log(max(spectrum.shape[2], 2)), 1e-12)


def _segment_features(segment: np.ndarray) -> np.ndarray:
    mean = np.mean(segment, axis=2)
    std = _safe_std(segment, axis=2)
    rms = np.sqrt(np.mean(np.square(segment), axis=2))
    peak = np.max(np.abs(segment), axis=2)
    p2p = np.ptp(segment, axis=2)
    crest = peak / np.maximum(rms, 1e-12)
    skew = _skewness(segment, axis=2)
    kurt = _kurtosis(segment, axis=2)
    energy = np.mean(np.square(segment), axis=2)
    entropy = _spectral_entropy(segment)

    per_channel = [mean, std, rms, peak, p2p, crest, skew, kurt, energy, entropy]
    blocks = []
    for feat in per_channel:
        blocks.append(np.mean(feat, axis=1))
    return np.stack(blocks, axis=1).astype(np.float32)


def _extract_evidence_tensor_block(windows: np.ndarray, n_segments: int = N_SEGMENTS) -> np.ndarray:
    arr = _as_3d(windows)
    n_windows, _, seq_len = arr.shape
    edges = np.linspace(0, seq_len, n_segments + 1, dtype=int)
    evidence = np.empty((n_windows, len(EVIDENCE_DIM_NAMES), n_segments), dtype=np.float32)
    for idx in range(n_segments):
        start = int(edges[idx])
        stop = int(edges[idx + 1])
        if stop <= start:
            stop = min(seq_len, start + 1)
        evidence[:, :, idx] = _segment_features(arr[:, :, start:stop])
    return np.nan_to_num(evidence, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_evidence_tensor(windows: np.ndarray, chunk_size: int = FEATURE_CHUNK_SIZE) -> np.ndarray:
    chunks = []
    for start in range(0, len(windows), chunk_size):
        block = _extract_evidence_tensor_block(windows[start:start + chunk_size])
        chunks.append(block.astype(np.float16))
    if not chunks:
        return np.empty((0, len(EVIDENCE_DIM_NAMES), N_SEGMENTS), dtype=np.float16)
    return np.vstack(chunks)


def _fit_normalizer(train_windows: np.ndarray) -> dict[str, list[float]]:
    train_windows = np.asarray(train_windows, dtype=np.float32)
    mean = np.mean(train_windows, axis=(0, 2), keepdims=True)
    std = np.std(train_windows, axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return {
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
    }


def _apply_normalizer(windows: np.ndarray, norm: dict[str, list[float]]) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None]
    return ((windows - mean) / std).astype(np.float16)


def build_unified_dataset(
    dataset_ids: list[int],
    label_space: str = "fine",
    out_dir: Path = UNIFIED_DIR,
    dry_run: bool = False,
) -> dict:
    catalog = load_standardized_catalog()
    catalog = catalog[catalog["dataset_id"].isin(dataset_ids)].copy()
    if catalog.empty:
        raise ValueError(f"No standardized catalog rows found for datasets: {dataset_ids}")

    groups = {
        group_id: {
            "n_channels": len(EVIDENCE_DIM_NAMES),
            "seq_len": N_SEGMENTS,
            "splits": {sp: {"windows": [], "meta": []} for sp in SPLITS},
        }
        for group_id in COMPONENT_GROUPS
    }
    manifest_rows: list[dict] = []

    for ds_id in dataset_ids:
        for split in SPLITS:
            loaded_exports = _read_split_exports(ds_id, split, load_windows=not dry_run)
            if not loaded_exports:
                continue
            for windows, meta, npy_path, csv_path in loaded_exports:
                for group_id in COMPONENT_GROUPS:
                    mask = _component_mask(meta, group_id)
                    if not mask.any():
                        continue
                    group_meta = _prepare_meta(meta.loc[mask].reset_index(drop=True), label_space, group_id)
                    group_windows = None
                    if not dry_run:
                        assert windows is not None
                        group_windows = _extract_evidence_tensor(windows[mask.to_numpy()])

                    if group_windows is not None:
                        groups[group_id]["splits"][split]["windows"].append(group_windows)
                    groups[group_id]["splits"][split]["meta"].append(group_meta)
                    manifest_rows.append(
                        {
                            "dataset_id": ds_id,
                            "group_id": group_id,
                            "split": split,
                            "label_space": label_space,
                            "view": "component_md",
                            "component_group": group_id,
                            "n_channels": len(EVIDENCE_DIM_NAMES),
                            "seq_len": N_SEGMENTS,
                            "n_windows": int(len(group_meta)),
                            "window_file": str(npy_path.relative_to(PROJECT_ROOT)),
                            "meta_file": str(csv_path.relative_to(PROJECT_ROOT)),
                        }
                    )

    groups = {
        gid: group
        for gid, group in groups.items()
        if any(group["splits"][sp]["meta"] for sp in SPLITS)
    }
    if not groups:
        raise RuntimeError("No component-focused processed splits found. Run Phase 3 preprocessing first.")

    label_vocab = {
        gid: sorted(
            {
                label
                for split in SPLITS
                for meta in group["splits"][split]["meta"]
                for label in meta["label_code"].astype(str).unique().tolist()
            }
        )
        for gid, group in groups.items()
    }

    if dry_run:
        return {
            "dataset_ids": dataset_ids,
            "label_space": label_space,
            "view": "component_md",
            "groups": {
                gid: {
                    "n_channels": g["n_channels"],
                    "seq_len": g["seq_len"],
                    "split_counts": {
                        sp: int(sum(len(m) for m in g["splits"][sp]["meta"]))
                        for sp in SPLITS
                    },
                }
                for gid, g in groups.items()
            },
            "label_vocab": label_vocab,
            "manifest_rows": manifest_rows,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_rows = []
    normalizers = {}

    for group_id, group in sorted(groups.items()):
        group_dir = out_dir / group_id
        group_dir.mkdir(parents=True, exist_ok=True)

        train_blocks = group["splits"]["train"]["windows"]
        if not train_blocks:
            raise RuntimeError(f"No train split found for unified group '{group_id}'")
        train_windows = np.vstack(train_blocks)
        norm = _fit_normalizer(train_windows)
        normalizers[group_id] = norm

        for split in SPLITS:
            split_windows = group["splits"][split]["windows"]
            split_meta = group["splits"][split]["meta"]
            if not split_windows:
                continue

            windows = _apply_normalizer(np.vstack(split_windows), norm)
            meta = pd.concat(split_meta, ignore_index=True)
            if "window_idx" in meta.columns:
                meta = meta.drop(columns=["window_idx"])
            meta.insert(0, "window_idx", np.arange(len(meta)))

            np.save(group_dir / f"{split}_windows.npy", windows)
            meta.to_csv(group_dir / f"{split}_meta.csv", index=False)

            saved_rows.append(
                {
                    "group_id": group_id,
                    "split": split,
                    "label_space": label_space,
                    "view": "component_md",
                    "component_group": group_id,
                    "n_channels": group["n_channels"],
                    "seq_len": group["seq_len"],
                    "n_windows": int(len(meta)),
                    "window_file": str((group_dir / f"{split}_windows.npy").relative_to(PROJECT_ROOT)),
                    "meta_file": str((group_dir / f"{split}_meta.csv").relative_to(PROJECT_ROOT)),
                }
            )

        with open(group_dir / "feature_spec.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "view": "component_md",
                    "component_group": group_id,
                    "feature_names": FEATURE_NAMES,
                    "aggregations": AGGREGATIONS,
                    "evidence_dim_names": EVIDENCE_DIM_NAMES,
                    "n_segments": N_SEGMENTS,
                    "normalization": "train_split_channel_standardization",
                    "normalizer": norm,
                },
                fh,
                indent=2,
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(["group_id", "dataset_id", "split"]).reset_index(drop=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    export_manifest = pd.DataFrame(saved_rows).sort_values(["group_id", "split"]).reset_index(drop=True)
    export_manifest.to_csv(out_dir / "exports.csv", index=False)

    spec = {
        "dataset_ids": dataset_ids,
        "label_space": label_space,
        "view": "component_md",
        "groups": {
            gid: {
                "n_channels": g["n_channels"],
                "seq_len": g["seq_len"],
                "splits": {
                    sp: int(sum(len(m) for m in g["splits"][sp]["meta"]))
                    for sp in SPLITS
                },
                "label_vocab": label_vocab[gid],
            }
            for gid, g in sorted(groups.items())
        },
        "feature_spec": {
            "feature_names": FEATURE_NAMES,
            "aggregations": AGGREGATIONS,
            "evidence_dim_names": EVIDENCE_DIM_NAMES,
            "n_segments": N_SEGMENTS,
        },
        "grouping_rule": (
            "component-focused multidimensional evidence tensors: bearing labels in bearing_md, "
            "gearbox labels in gearbox_md; mixed labels are excluded from primary unified groups"
        ),
        "source_manifest": str(STD_CATALOG.relative_to(PROJECT_ROOT)),
        "processed_dir": str(PROCESSED_DIR.relative_to(PROJECT_ROOT)),
        "normalizers": normalizers,
    }
    with open(out_dir / "spec.json", "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)

    return {
        "dataset_ids": dataset_ids,
        "label_space": label_space,
        "view": "component_md",
        "groups": spec["groups"],
        "label_vocab": label_vocab,
        "manifest_rows": manifest_rows,
        "export_rows": saved_rows,
    }


def _header() -> None:
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         MSPF-Net  —  Component Unified Dataset Builder         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Project    : {PROJECT_ROOT}")
    print(f"  Processed  : {PROCESSED_DIR}")
    print(f"  Output     : {UNIFIED_DIR}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=int, default=ACTIVE_THESIS_DATASETS)
    parser.add_argument("--label-space", choices=["fine", "coarse"], default="fine")
    parser.add_argument("--view", choices=["component_md"], default="component_md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _header()
    print(f"  Datasets    : {args.datasets}")
    print(f"  Label space : {args.label_space}")
    print(f"  View        : {args.view}")
    print(f"  Mode        : {'DRY RUN - no files written' if args.dry_run else 'FULL RUN'}")

    summary = build_unified_dataset(
        dataset_ids=args.datasets,
        label_space=args.label_space,
        out_dir=UNIFIED_DIR,
        dry_run=args.dry_run,
    )

    print(f"\n{'-'*68}")
    print("  Unified Groups - component_md")
    print(f"{'-'*68}")
    for group_id, info in sorted(summary["groups"].items()):
        splits = info.get("split_counts") or info["splits"]
        print(
            f"  {group_id:>10}  dims={info['n_channels']}  segments={info['seq_len']}  "
            f"train={splits['train']}  val={splits['val']}  "
            f"test={splits['test']}  robustness={splits['robustness']}"
        )

    print(f"\n  Labels:")
    for group_id, labels in sorted(summary["label_vocab"].items()):
        print(f"  - {group_id}: {', '.join(labels)}")

    if args.dry_run:
        print("  Dry run complete. No unified files were written.")
        return

    print(f"  Saved       : {UNIFIED_DIR / 'manifest.csv'}")
    print(f"  Saved       : {UNIFIED_DIR / 'exports.csv'}")
    print(f"  Saved       : {UNIFIED_DIR / 'spec.json'}")


if __name__ == "__main__":
    main()
