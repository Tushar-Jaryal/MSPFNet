import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from mspf_net.constants import ACTIVE_THESIS_DATASETS, get_dataset_display

from mspf_net.preprocessing.preprocess_utils import (
    FS_NATIVE,
    FS_TARGET,
    estimate_dataset_windows,
    load_standardized_catalog,
    preprocess_dataset,
)

ALL_DATASETS = ACTIVE_THESIS_DATASETS
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def _load_window_config() -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    default_win = 2048
    default_hop = 1024
    if not cfg_path.exists():
        return {
            "default_window_size": default_win,
            "default_hop_size": default_hop,
            "overlap_ratio": 0.5,
            "per_dataset": {},
        }

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    windowing_cfg = cfg.get("windowing", {}) or {}
    return {
        "default_window_size": int(
            windowing_cfg.get(
                "default_window_size",
                preprocessing_cfg.get("window_size", default_win),
            )
        ),
        "default_hop_size": windowing_cfg.get("default_hop_size"),
        "overlap_ratio": float(preprocessing_cfg.get("overlap_ratio", 0.5)),
        "per_dataset": windowing_cfg.get("per_dataset", {}) or {},
    }


def _resolve_dataset_window(cfg: dict, ds_id: int) -> tuple[int, int]:
    per_dataset = cfg.get("per_dataset", {}) or {}
    ds_cfg = per_dataset.get(str(ds_id), {}) or per_dataset.get(int(ds_id), {}) or {}

    win = int(ds_cfg.get("window_size", cfg["default_window_size"]))
    hop_cfg = ds_cfg.get("hop_size", cfg.get("default_hop_size"))
    if hop_cfg is not None:
        hop = int(hop_cfg)
    else:
        overlap_ratio = float(cfg.get("overlap_ratio", 0.5))
        hop = max(1, int(round(win * (1.0 - overlap_ratio))))
    return win, hop


def _header():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       MSPF-Net  —  Phase 3: Signal Preprocessing               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    from datetime import datetime

    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project    : {PROJECT_ROOT}")
    print(f"  Target fs  : {FS_TARGET:,} Hz")
    print()


def _bar(label: str):
    print(f"\n{'─'*68}")
    print(f"  {label}")
    print(f"{'─'*68}")


def _format_norm_stat(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(f"{float(v):.4f}" for v in value) + "]"
    return f"{float(value):.4f}"


def _csv_norm_stat(value):
    if isinstance(value, (list, tuple)):
        return json.dumps([float(v) for v in value])
    return float(value)


def main():
    window_cfg = _load_window_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", type=int, default=ALL_DATASETS, help="Dataset IDs to process (default: all)")
    parser.add_argument("--win", type=int, default=None, help="Override window size for all selected datasets")
    parser.add_argument("--hop", type=int, default=None, help="Override hop size for all selected datasets")
    parser.add_argument("--dry-run", action="store_true", help="Estimate window counts only, do not write files")
    args = parser.parse_args()

    _header()

    catalog = load_standardized_catalog(PROJECT_ROOT)
    print(f"  Catalog     : {len(catalog)} files across {catalog.dataset_id.nunique()} datasets")
    if args.win is not None or args.hop is not None:
        win = args.win if args.win is not None else window_cfg["default_window_size"]
        hop = args.hop if args.hop is not None else _resolve_dataset_window(window_cfg, args.dataset[0])[1]
        print(f"  Window      : override → {win} samples @ {FS_TARGET:,} Hz = {win / FS_TARGET * 1000:.1f} ms")
        print(f"  Hop         : override → {hop} samples")
    else:
        print("  Window      : per-dataset from configs/config.yaml -> windowing.per_dataset")
        print(
            f"  Default     : win={window_cfg['default_window_size']}  "
            f"hop={'auto' if window_cfg['default_hop_size'] is None else window_cfg['default_hop_size']}"
        )
    print("  Split rule  : windows inherit the parent raw-file split from catalog_standardized.csv")
    print(f"  Mode        : {'DRY RUN — no files written' if args.dry_run else 'FULL RUN'}")

    ds_names = {
        1: "D1_BearingTV    [200kHz→10kHz]  ★ PRIMARY",
        2: "D2_PlanetaryGB  [48kHz→10kHz]",
        31: "D3C1_CWRU_Bearing [12kHz→10kHz]",
        39: "D3C9_Gearbox [12kHz→10kHz]",
        4: "D4_MultiModeGB  [10kHz — no resample]",
        5: "D5_MixedBG      [10kHz — no resample]  ★ PRIMARY",
        8: "D8_HUSTGearbox  [25.6kHz→10kHz]",
    }

    report_rows = []

    for ds_id in args.dataset:
        _bar(ds_names.get(ds_id, f"Dataset {ds_id}"))
        t0 = time.time()
        win_size, hop_size = (
            (args.win, args.hop if args.hop is not None else max(1, args.win // 2))
            if args.win is not None
            else _resolve_dataset_window(window_cfg, ds_id)
        )
        if args.win is None and args.hop is not None:
            win_size = _resolve_dataset_window(window_cfg, ds_id)[0]
            hop_size = args.hop
        print(
            f"  Window cfg  : win={win_size}  hop={hop_size}  "
            f"({win_size / FS_TARGET * 1000:.1f} ms window)"
        )

        if args.dry_run:
            print("  Estimating window counts...")
            est = estimate_dataset_windows(catalog, ds_id, PROJECT_ROOT, win_size, hop_size)
            print(f"  Estimated windows : {est['n_windows']:,}")
            print(f"    train : {est['train']:,}")
            print(f"    val   : {est['val']:,}")
            print(f"    test  : {est['test']:,}")
            if est.get("robustness", 0):
                print(f"    robustness : {est['robustness']:,}")
            report_rows.append({"dataset_id": ds_id, "status": "dry-run", "win_size": win_size, "hop": hop_size, **est})
            continue

        try:
            summary = preprocess_dataset(
                ds_id=ds_id,
                catalog=catalog,
                project_root=PROJECT_ROOT,
                out_dir=OUT_DIR,
                win_size=win_size,
                hop=hop_size,
                fs_target=FS_TARGET,
                verbose=True,
            )
            elapsed = time.time() - t0
            total_windows = summary["n_windows"]
            size_mb = 0.0
            for path in OUT_DIR.glob(f"d{ds_id}_*_windows.npy"):
                size_mb += path.stat().st_size / 1024**2

            print(f"\n  ✅ Done in {elapsed:.1f}s")
            print(f"  Windows      : {total_windows:,} total")
            print(f"  Channels     : {summary['n_channels']} preserved")
            print(f"  Size on disk : {size_mb:.1f} MB  (float32)")
            print(
                f"  Split        : train={summary['n_train']:,}  val={summary['n_val']:,}  "
                f"test={summary['n_test']:,}  robustness={summary['n_robustness']:,}"
            )
            if summary.get("partitions") and len(summary["partitions"]) > 1:
                print("  Norm         : per-partition normalizers fitted on train")
                saved = ", ".join(f"{p['prefix']}_<split>_windows.npy" for p in summary["partitions"])
                print(f"  Saved        → data/processed/{saved}  (C,L windows)")
            else:
                print(
                    f"  Norm         : mean={_format_norm_stat(summary['norm_mean'])}  "
                    f"std={_format_norm_stat(summary['norm_std'])}  (fitted on train)"
                )
                print(f"  Saved        → data/processed/d{ds_id}_<split>_windows.npy  (C,L windows)")

            report_rows.append(
                {
                    "dataset_id": ds_id,
                    "dataset_label": get_dataset_display(ds_id),
                    "status": "ok",
                    "win_size": win_size,
                    "hop": hop_size,
                    "n_windows": summary["n_windows"],
                    "n_train": summary["n_train"],
                    "n_val": summary["n_val"],
                    "n_test": summary["n_test"],
                    "n_robustness": summary["n_robustness"],
                    "n_channels": summary["n_channels"],
                    "norm_mean": _csv_norm_stat(summary["norm_mean"]),
                    "norm_std": _csv_norm_stat(summary["norm_std"]),
                    "size_mb": round(size_mb, 1),
                    "elapsed_s": round(elapsed, 1),
                }
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ❌ FAILED in {elapsed:.1f}s: {e}")
            import traceback

            traceback.print_exc()
            report_rows.append({"dataset_id": ds_id, "status": f"FAILED: {e}"})

    print(f"\n{'═'*68}")
    print("  Phase 3 Preprocessing Summary")
    print(f"{'═'*68}")
    report = pd.DataFrame(report_rows)
    print(report.to_string(index=False))

    if not args.dry_run and len(report_rows) > 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        report.to_csv(OUT_DIR / "preprocessing_report.csv", index=False)
        print("\n  Report saved → data/processed/preprocessing_report.csv")

    window_rows = report[report["status"].isin(["ok", "dry-run"])] if "status" in report.columns else report
    if "n_windows" in window_rows.columns:
        total = window_rows["n_windows"].sum()
        print(f"\n  Total windows across processed datasets : {total:,}")


if __name__ == "__main__":
    main()
