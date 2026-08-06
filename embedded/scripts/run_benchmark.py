#!/usr/bin/env python3
"""Run embedded benchmark (latency, FLOPs, memory, test accuracy) on this machine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EMBEDDED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EMBEDDED))

from mspf_embedded._paths import ensure_repo_src, repo_root
from mspf_embedded.benchmark import run_embedded_benchmark

ensure_repo_src()


def main() -> int:
    root = repo_root()
    default_cfg = root / "embedded/configs/pi5_softmax.yaml"

    parser = argparse.ArgumentParser(description="Run MSPF-Net embedded benchmark (Softmax).")
    parser.add_argument("--bundle-dir", type=str, required=True)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    from mspf_embedded.bundle import load_embedded_config

    cfg = load_embedded_config(args.config)
    processed_dir = args.processed_dir or cfg.get("processed_dir", "data/processed")
    if not Path(processed_dir).is_absolute():
        processed_dir = str(root / processed_dir)

    payload = run_embedded_benchmark(
        bundle_dir=args.bundle_dir,
        processed_dir=processed_dir,
        embedded_cfg=cfg,
        device_name=args.device,
    )

    print(json.dumps(payload, indent=2))
    print(f"\nResults: {payload['results_json']}")
    print(f"Table:   {payload['results_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
