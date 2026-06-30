#!/usr/bin/env python3
"""Run inference smoke + latency for every bundle on Pi 5 (no test split required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EMBEDDED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EMBEDDED))

from mspf_embedded._paths import ensure_repo_src, repo_root
from mspf_embedded.benchmark import run_embedded_inference_check
from mspf_embedded.bundle import load_embedded_config

ensure_repo_src()


def main() -> int:
    root = repo_root()
    default_cfg = root / "embedded/configs/pi5_softmax.yaml"

    parser = argparse.ArgumentParser(
        description="Pi 5 inference check for all exported bundles (latency + forward pass)."
    )
    parser.add_argument("--bundles-root", type=str, default=str(root / "embedded/bundles"))
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--latency-runs", type=int, default=None)
    parser.add_argument("--latency-warmup", type=int, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    cfg = load_embedded_config(args.config)
    if args.latency_runs is not None:
        cfg["latency_runs"] = int(args.latency_runs)
    if args.latency_warmup is not None:
        cfg["latency_warmup"] = int(args.latency_warmup)

    bundles_root = Path(args.bundles_root)
    if not bundles_root.is_dir():
        print(f"No bundles directory: {bundles_root}")
        return 1

    failures = 0
    for bundle_dir in sorted(bundles_root.iterdir()):
        if not bundle_dir.is_dir() or not (bundle_dir / "manifest.json").is_file():
            continue
        print(f"\n=== Inference {bundle_dir.name} ===")
        try:
            payload = run_embedded_inference_check(
                bundle_dir=bundle_dir,
                embedded_cfg=cfg,
                device_name=args.device,
            )
            lat = payload["latency"]
            print(
                f"  {payload['manifest'].get('model')} {payload['manifest'].get('target_label')}  "
                f"{lat['mean_ms']:.3f} ms/window  params={payload['manifest'].get('params')}"
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED: {exc}")
            if args.stop_on_failure:
                return 1

    print(f"\nDone ({failures} failures). See embedded/results/tables/pi5_inference_check.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
