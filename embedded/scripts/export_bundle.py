#!/usr/bin/env python3
"""Export a Softmax MSPF-Net bundle for embedded (Pi 5) deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EMBEDDED = Path(__file__).resolve().parents[1]
_REPO = _EMBEDDED.parent
sys.path.insert(0, str(_EMBEDDED))

from mspf_embedded.bundle import export_bundle
from mspf_embedded._paths import ensure_repo_src

ensure_repo_src()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MSPF-Net Softmax bundle for embedded inference.")
    parser.add_argument("--results-json", type=str, default=None, help="Training results JSON with best_checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=int, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory, e.g. embedded/bundles/d1_softmax",
    )
    args = parser.parse_args()

    out = export_bundle(
        out_dir=args.out_dir,
        results_json=args.results_json,
        checkpoint=args.checkpoint,
        config_path=args.config,
        dataset_id=args.dataset,
        processed_dir=args.processed_dir,
    )
    print(f"Exported bundle to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
