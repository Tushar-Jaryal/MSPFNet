from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.constants import PRIMARY_SCRATCH_DATASETS
from mspf_net.config_utils import get_config_list, get_config_value

DEFAULT_ABLATION_DATASETS = [int(v) for v in get_config_list("datasets", "mspf_ablation", default=[2, 4, 5, 8])]
DEFAULT_EPOCHS = int(get_config_value("phase4", "epochs", default=50))
DEFAULT_DEVICE = str(get_config_value("phase4", "device", default="auto"))


def _run(cmd: list[str]) -> None:
    print("\n" + "═" * 80)
    print("RUN:", " ".join(cmd))
    print("═" * 80)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the MSPF-Net-only evaluation workflow: Softmax, RF, MoE, "
            "ablation studies, aggregation, and audit."
        )
    )
    p.add_argument("--datasets", type=int, nargs="+", default=None)
    p.add_argument("--ablation-datasets", type=int, nargs="+", default=DEFAULT_ABLATION_DATASETS)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--processed-dir", type=str, default=None)
    p.add_argument("--run-suffix", type=str, default=None, help="Optional artifact suffix for side-by-side runs.")
    p.add_argument("--skip-softmax", action="store_true")
    p.add_argument("--skip-rf", action="store_true")
    p.add_argument("--skip-moe", action="store_true")
    p.add_argument(
        "--moe-only",
        action="store_true",
        help="Run only the two-expert MoE model (+ aggregate/audit unless skipped).",
    )
    p.add_argument("--skip-ablations", action="store_true")
    p.add_argument(
        "--ablation-base-models",
        nargs="+",
        choices=["softmax", "rf", "moe"],
        default=["softmax", "rf", "moe"],
        help="Which full-model bases to use for ablation runs.",
    )
    p.add_argument("--skip-aggregate", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    return p


def _run_per_dataset(
    script: str,
    datasets: list[int],
    epochs: int,
    device: str,
    batch_size: int | None,
    processed_dir: str | None,
    run_suffix: str | None,
    stop_on_failure: bool,
) -> None:
    py = sys.executable
    for dataset_id in datasets:
        cmd = [
            py,
            script,
            "--datasets",
            str(dataset_id),
            "--epochs",
            str(epochs),
            "--device",
            device,
        ]
        if batch_size is not None:
            cmd += ["--batch_size", str(batch_size)]
        if processed_dir:
            cmd += ["--processed-dir", processed_dir]
        if run_suffix:
            cmd += ["--run-suffix", run_suffix]
        try:
            _run(cmd)
        except subprocess.CalledProcessError:
            if stop_on_failure:
                raise
            print(f"\n  WARNING: failed on dataset D{dataset_id} for {script}; continuing.")


def main() -> int:
    args = build_parser().parse_args()
    if args.moe_only:
        args.skip_softmax = True
        args.skip_rf = True
        args.skip_ablations = True
    run_suffix = args.run_suffix
    scratch_datasets = list(args.datasets) if args.datasets is not None else list(PRIMARY_SCRATCH_DATASETS)
    py = sys.executable

    if not args.skip_softmax:
        _run_per_dataset(
            script="scripts/baselines/run_mspf_net.py",
            datasets=scratch_datasets,
            epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size,
            processed_dir=args.processed_dir,
            run_suffix=run_suffix,
            stop_on_failure=args.stop_on_failure,
        )

    if not args.skip_rf:
        _run_per_dataset(
            script="scripts/baselines/run_mspf_net_rf.py",
            datasets=scratch_datasets,
            epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size,
            processed_dir=args.processed_dir,
            run_suffix=run_suffix,
            stop_on_failure=args.stop_on_failure,
        )

    if not args.skip_moe:
        _run_per_dataset(
            script="scripts/baselines/run_mspf_net_moe.py",
            datasets=scratch_datasets,
            epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size,
            processed_dir=args.processed_dir,
            run_suffix=run_suffix,
            stop_on_failure=args.stop_on_failure,
        )

    if not args.skip_ablations:
        cmd = [
            py,
            "scripts/baselines/run_mspf_ablations.py",
            "--base-models",
            *args.ablation_base_models,
            "--datasets",
            *[str(ds) for ds in args.ablation_datasets],
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
        if args.batch_size is not None:
            cmd += ["--batch_size", str(args.batch_size)]
        if args.processed_dir:
            cmd += ["--processed-dir", args.processed_dir]
        if run_suffix:
            cmd += ["--run-suffix", run_suffix]
        if args.stop_on_failure:
            cmd += ["--stop-on-failure"]
        _run(cmd)

    if not args.skip_aggregate:
        _run([py, "scripts/aggregate_mspf_results.py"])

    if not args.skip_audit:
        _run([py, "scripts/audit_baseline_matrix.py"])

    print("\nMSPF-Net evaluation workflow completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
