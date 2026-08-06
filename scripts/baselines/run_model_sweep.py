from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.constants import ACTIVE_THESIS_DATASETS
from mspf_net.config_utils import get_config_path, get_config_value

DEFAULT_EPOCHS = int(get_config_value("phase4", "epochs", default=50))
DEFAULT_DEVICE = str(get_config_value("phase4", "device", default="auto"))
DEFAULT_RESULTS_DIR = str(get_config_value("phase4", "results_dir", default="results/baselines"))


def _discover_unified_groups(repo: Path) -> list[str]:
    unified_dir = repo / get_config_path("paths", "data_unified", default="data/unified")
    spec_path = unified_dir / "spec.json"
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        groups = sorted((spec.get("groups") or {}).keys())
        if groups:
            return groups

    if not unified_dir.exists():
        raise FileNotFoundError(
            "Unified directory not found at data/unified. "
            "Run the unified export first or pass --group-id explicitly."
        )

    groups = sorted(
        p.name for p in unified_dir.iterdir()
        if p.is_dir() and (p / "train_windows.npy").exists()
    )
    if groups:
        return groups

    raise ValueError(
        "No unified groups were discovered under data/unified. "
        "Run the unified export first or pass --group-id explicitly."
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Run one baseline across multiple datasets.")
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--data-mode", choices=["processed", "unified"], default="processed")
    p.add_argument("--group-id", type=str, default=None)
    p.add_argument("--label-space", choices=["fine", "coarse"], default="fine")
    p.add_argument("--cv-folds", type=int, default=None)
    p.add_argument("--cv-val-ratio", type=float, default=None)
    p.add_argument("--unified-sampling", choices=["dataset_balanced", "none"], default=None)
    p.add_argument("--experiment-tag", type=str, default=None)
    args = p.parse_args()

    repo = Path.cwd()
    def _safe_label(value: str) -> str:
        return value.replace("\\", "__").replace("/", "__").replace(",", "_").replace(" ", "_")

    log_dir = repo / DEFAULT_RESULTS_DIR / args.model / args.data_mode / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.data_mode == "unified":
        if args.group_id is not None:
            targets = [args.group_id]
        else:
            targets = _discover_unified_groups(repo)
            print(f"Auto-discovered unified groups: {', '.join(targets)}")
    else:
        if args.datasets is not None:
            targets = args.datasets
        else:
            targets = ACTIVE_THESIS_DATASETS

    total = len(targets)
    for i, target in enumerate(targets, start=1):
        label = str(target)
        log_path = log_dir / f"{args.model}_{_safe_label(label)}.log"
        print("\n" + "═" * 72)
        print(f"[{i}/{total}] Running {args.model} on {label}")
        print(f"Log: {log_path}")
        print("═" * 72 + "\n")

        cmd = [sys.executable, f"scripts/baselines/run_{args.model}.py", "--epochs", str(args.epochs), "--device", args.device]
        if args.batch_size is not None:
            cmd += ["--batch_size", str(args.batch_size)]
        if args.cv_folds is not None:
            cmd += ["--cv-folds", str(args.cv_folds)]
        if args.cv_val_ratio is not None:
            cmd += ["--cv-val-ratio", str(args.cv_val_ratio)]
        if args.unified_sampling is not None:
            cmd += ["--unified-sampling", args.unified_sampling]
        if args.experiment_tag is not None:
            cmd += ["--experiment-tag", args.experiment_tag]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if args.data_mode == "unified":
            cmd += ["--data-mode", "unified", "--group-id", str(target), "--label-space", args.label_space]
        else:
            cmd += ["--datasets", str(target), "--label-space", args.label_space]
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=repo, env=env)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                logf.write(line)
            rc = proc.wait()
        if rc != 0:
            print(f"  {args.model} {label}: FAILED (exit {rc})")
        else:
            print(f"  {label}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
