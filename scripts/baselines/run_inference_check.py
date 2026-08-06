#!/usr/bin/env python3
"""Check inference (smoke + latency) for all Phase 4 baselines and MSPF-Net."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.config_utils import get_config_path, get_config_value, get_primary_scratch_dataset_ids
from mspf_net.constants import get_dataset_display
from mspf_net.training.inference_check import PHASE4_MODELS, check_model_inference
from mspf_net.utils.device import get_device


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test and benchmark inference for trained baseline checkpoints."
    )
    parser.add_argument("--models", nargs="+", default=PHASE4_MODELS)
    parser.add_argument("--datasets", nargs="+", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--mspf-tag", type=str, default="softmax", help="experiment_tag for MSPF-Net results")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir or get_config_value("phase4", "results_dir", default="results/baselines"))
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    tables_dir = PROJECT_ROOT / get_config_path("paths", "tables", default="results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else tables_dir / "inference_check.json"
    out_csv = Path(args.out_csv) if args.out_csv else tables_dir / "inference_check.csv"

    dataset_ids = args.datasets or get_primary_scratch_dataset_ids()
    device = get_device(args.device)

    print(f"\n  Device      : {device}")
    print(f"  Results dir : {results_dir}")
    print(f"  Datasets    : {dataset_ids}")
    print(f"  Models      : {args.models}\n")

    rows: list[dict] = []
    failures = 0
    for model_name in args.models:
        for dataset_id in dataset_ids:
            tag = args.mspf_tag if model_name == "mspf_net" else None
            payload = check_model_inference(
                model_name,
                int(dataset_id),
                results_dir=results_dir,
                device=device,
                experiment_tag=tag,
                n_runs=int(args.latency_runs),
                warmup=int(args.latency_warmup),
            )
            rows.append(payload)
            status = payload.get("status")
            target = payload.get("target") or get_dataset_display(dataset_id)
            if status == "ok":
                print(
                    f"  OK   {model_name:16s} {target:4s}  "
                    f"{payload['mean_ms']:.3f} ms/window  "
                    f"params={payload['params']:,}  "
                    f"shape={payload['window_shape']}"
                )
            elif status == "sklearn_rf":
                print(
                    f"  RF   {model_name:16s} {target:4s}  "
                    f"training_ms={payload.get('inference_ms_per_window')}"
                )
            else:
                failures += 1
                print(f"  FAIL {model_name:16s} {target:4s}  {payload.get('error', status)}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    flat_rows = []
    for r in rows:
        latency = r.get("latency") or {}
        flops = r.get("flops") or {}
        flat_rows.append(
            {
                "model": r.get("model"),
                "dataset_id": r.get("dataset_id"),
                "target": r.get("target"),
                "experiment_tag": r.get("experiment_tag"),
                "status": r.get("status"),
                "mean_ms": latency.get("mean_ms", r.get("mean_ms")),
                "std_ms": latency.get("std_ms"),
                "target_ok": latency.get("target_ok", r.get("target_ok")),
                "params": r.get("params", flops.get("params")),
                "flops_g": flops.get("flops_g"),
                "window_shape": r.get("window_shape"),
                "training_inference_ms": r.get("training_inference_ms", r.get("inference_ms_per_window")),
                "device": r.get("device"),
                "results_json": r.get("results_json"),
                "error": r.get("error"),
            }
        )
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print(f"\n  JSON → {out_json}")
    print(f"  CSV  → {out_csv}")
    print(f"  Done ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
