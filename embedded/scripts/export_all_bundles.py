#!/usr/bin/env python3
"""Export embedded bundles for all Phase 4 torch baselines + MSPF-Net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EMBEDDED = Path(__file__).resolve().parents[1]
_REPO = _EMBEDDED.parent
sys.path.insert(0, str(_EMBEDDED))

from mspf_embedded._paths import ensure_repo_src, repo_root
from mspf_embedded.bundle import export_baseline_bundle
from mspf_net.config_utils import get_primary_scratch_dataset_ids
from mspf_net.training.inference_check import DEFAULT_EXPERIMENT_TAGS, PHASE4_MODELS, find_results_json

ensure_repo_src()

TORCH_MODELS = [m for m in PHASE4_MODELS if m != "rf_features"]


def _bundle_name(model: str, dataset_id: int, experiment_tag: str) -> str:
    if model == "mspf_net":
        return f"{model}_d{dataset_id}_{experiment_tag}"
    return f"{model}_d{dataset_id}"


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Export Pi deployment bundles for all torch baselines.")
    parser.add_argument("--models", nargs="+", default=TORCH_MODELS)
    parser.add_argument("--datasets", nargs="+", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default="results/baselines")
    parser.add_argument("--bundles-root", type=str, default=str(root / "embedded/bundles"))
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--mspf-tag", type=str, default="softmax")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    bundles_root = Path(args.bundles_root)
    bundles_root.mkdir(parents=True, exist_ok=True)
    dataset_ids = args.datasets or get_primary_scratch_dataset_ids()

    failures = 0
    exported = 0
    for model in args.models:
        if model == "rf_features":
            print(f"SKIP {model} (sklearn, no torch checkpoint)")
            continue
        for dataset_id in dataset_ids:
            tag = args.mspf_tag if model == "mspf_net" else None
            found = find_results_json(results_dir, model, int(dataset_id), experiment_tag=tag)
            if found is None:
                print(f"SKIP {model} D{dataset_id} — no results JSON")
                continue
            results_path, data = found
            if not data.get("best_checkpoint"):
                print(f"SKIP {model} D{dataset_id} — no checkpoint in {results_path.name}")
                continue
            out_dir = bundles_root / _bundle_name(model, int(dataset_id), str(data.get("experiment_tag", "")))
            try:
                export_baseline_bundle(
                    model_name=model,
                    results_json=results_path,
                    out_dir=out_dir,
                    processed_dir=args.processed_dir,
                )
                exported += 1
                print(f"OK   {out_dir.name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {model} D{dataset_id}: {exc}")
                if args.stop_on_failure:
                    return 1

    print(f"\nExported {exported} bundles to {bundles_root} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
