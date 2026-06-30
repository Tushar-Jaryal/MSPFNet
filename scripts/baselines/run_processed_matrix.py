from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.constants import ACTIVE_THESIS_DATASETS
from mspf_net.config_utils import get_config_value

DEFAULT_EPOCHS = int(get_config_value("phase4", "epochs", default=50))
DEFAULT_DEVICE = str(get_config_value("phase4", "device", default="auto"))
DEFAULT_FIXED_WIN = int(get_config_value("phase4", "fixed_window_size", default=2048))
DEFAULT_FIXED_HOP = int(get_config_value("phase4", "fixed_hop_size", default=1024))


def _run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Run processed-mode baseline matrix across label-space and window policy.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--fixed-win", type=int, default=DEFAULT_FIXED_WIN)
    p.add_argument("--fixed-hop", type=int, default=DEFAULT_FIXED_HOP)
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument(
        "--window-policies",
        nargs="+",
        choices=["recommended", "fixed"],
        default=None,
        help="Window policies to run. Defaults to both, or recommended only when --skip-preprocess is used.",
    )
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", type=int, nargs="+", default=None)
    p.add_argument("--stop-on-failure", action="store_true")
    args = p.parse_args()

    py = sys.executable
    label_spaces = ["fine", "coarse"]
    selected_policies = args.window_policies or (["recommended"] if args.skip_preprocess else ["recommended", "fixed"])
    if args.skip_preprocess and "fixed" in selected_policies:
        p.error("--skip-preprocess cannot be combined with --window-policies fixed because fixed windows require regeneration.")
    window_policies = []
    if "recommended" in selected_policies:
        window_policies.append(("recommended", None, None))
    if "fixed" in selected_policies:
        window_policies.append(("fixed", args.fixed_win, args.fixed_hop))

    for policy_name, win, hop in window_policies:
        if not args.skip_preprocess:
            cmd = [py, "scripts/run_phase3_preprocess.py"]
            if win is not None:
                cmd += ["--win", str(win), "--hop", str(hop)]
            _run(cmd)

        for label_space in label_spaces:
            tag = f"{policy_name}_{label_space}"
            cmd = [
                py,
                "scripts/baselines/run_all_baselines.py",
                "--data-mode",
                "processed",
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--label-space",
                label_space,
                "--experiment-tag",
                tag,
            ]
            datasets = args.datasets if args.datasets is not None else ACTIVE_THESIS_DATASETS
            cmd += ["--datasets", *[str(ds_id) for ds_id in datasets]]
            if args.models:
                cmd += ["--models", *args.models]
            if args.stop_on_failure:
                cmd.append("--stop-on-failure")
            _run(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
