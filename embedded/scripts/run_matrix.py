#!/usr/bin/env python3
"""Run embedded benchmarks for all bundles under embedded/bundles/."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_EMBEDDED = Path(__file__).resolve().parents[1]
_REPO = _EMBEDDED.parent
sys.path.insert(0, str(_EMBEDDED))

from mspf_embedded._paths import ensure_repo_src, repo_root
from mspf_embedded.benchmark import run_embedded_benchmark
from mspf_embedded.bundle import load_embedded_config

ensure_repo_src()


def main() -> int:
    root = repo_root()
    default_cfg = root / "embedded/configs/pi5_softmax.yaml"

    parser = argparse.ArgumentParser(description="Run embedded benchmarks for all exported bundles.")
    parser.add_argument("--bundles-root", type=str, default=str(root / "embedded/bundles"))
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    cfg = load_embedded_config(args.config)
    processed_dir = args.processed_dir or cfg.get("processed_dir", "data/processed")
    if not Path(processed_dir).is_absolute():
        processed_dir = str(root / processed_dir)

    bundles_root = Path(args.bundles_root)
    if not bundles_root.is_dir():
        print(f"No bundles directory: {bundles_root}")
        return 1

    failures = 0
    for bundle_dir in sorted(bundles_root.iterdir()):
        if not bundle_dir.is_dir():
            continue
        manifest = bundle_dir / "manifest.json"
        if not manifest.is_file():
            continue
        print(f"\n=== Benchmark {bundle_dir.name} ===")
        try:
            payload = run_embedded_benchmark(
                bundle_dir=bundle_dir,
                processed_dir=processed_dir,
                embedded_cfg=cfg,
                device_name=args.device,
            )
            print(
                f"  OK  window_f1={payload.get('window_macro_f1'):.2f}%  "
                f"peak_rss={payload.get('peak_rss_mb')} MB"
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED: {exc}")
            if args.stop_on_failure:
                return 1
        finally:
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
