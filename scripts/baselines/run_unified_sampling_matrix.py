from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.config_utils import get_config_value

DEFAULT_EPOCHS = int(get_config_value("phase4", "epochs", default=50))
DEFAULT_DEVICE = str(get_config_value("phase4", "device", default="auto"))


def _run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Run unified-mode baseline matrix across sampling strategies.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--label-space", choices=["fine", "coarse"], default="fine")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--stop-on-failure", action="store_true")
    args = p.parse_args()

    py = sys.executable
    samplings = ["dataset_balanced", "none"]
    for sampling in samplings:
        tag = f"unified_{args.label_space}_{sampling}"
        cmd = [
            py,
            "scripts/baselines/run_all_baselines.py",
            "--data-mode",
            "unified",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--label-space",
            args.label_space,
            "--unified-sampling",
            sampling,
            "--experiment-tag",
            tag,
        ]
        if args.models:
            cmd += ["--models", *args.models]
        if args.stop_on_failure:
            cmd.append("--stop-on-failure")
        _run(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
