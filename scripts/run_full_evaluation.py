from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.config_utils import get_config_value

DEFAULT_EPOCHS = int(get_config_value("phase4", "epochs", default=50))
DEFAULT_DEVICE = str(get_config_value("phase4", "device", default="auto"))
DEFAULT_FIXED_WIN = int(get_config_value("phase4", "fixed_window_size", default=2048))
DEFAULT_FIXED_HOP = int(get_config_value("phase4", "fixed_hop_size", default=1024))


def _run(cmd: list[str]) -> None:
    print("\n" + "═" * 80)
    print("RUN:", " ".join(cmd))
    print("═" * 80)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the full end-to-end evaluation stack: verification, catalog, "
            "standardization, EDA, time-series analysis, preprocessing, unified export, "
            "processed/unified baselines, grouped 5-fold CV, aggregation, and audit."
        )
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--datasets", type=int, nargs="+", default=None, help="Dataset ids for MSPF suite forwarding.")
    p.add_argument("--processed-dir", type=str, default=None, help="Processed data root forwarded to MSPF suite.")
    p.add_argument("--run-suffix", type=str, default=None, help="Artifact suffix forwarded to MSPF suite.")
    p.add_argument("--fixed-win", type=int, default=DEFAULT_FIXED_WIN)
    p.add_argument("--fixed-hop", type=int, default=DEFAULT_FIXED_HOP)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument(
        "--include-fixed-processed",
        action="store_true",
        help=(
            "Also run the fixed-window processed matrix. This regenerates processed data "
            "for fixed windows, then restores recommended preprocessing before unified export."
        ),
    )
    p.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue matrix jobs after an individual model/target failure instead of failing fast.",
    )
    p.add_argument("--run-mspf-suite", action="store_true", help="Also run the MSPF-Net-only evaluation pipeline.")
    p.add_argument("--skip-ablations", action="store_true", help="Forward to MSPF suite: skip ablation runs.")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-build-catalog", action="store_true")
    p.add_argument("--skip-standardize", action="store_true")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-timeseries", action="store_true")
    p.add_argument("--skip-phase3", action="store_true")
    p.add_argument("--skip-unify", action="store_true")
    p.add_argument("--skip-processed-matrix", action="store_true")
    p.add_argument("--skip-unified-matrix", action="store_true")
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    py = sys.executable
    ran_mspf_suite = False

    if not args.skip_verify:
        _run([py, "scripts/run_verify_datasets.py"])

    if not args.skip_build_catalog:
        _run([py, "scripts/run_build_catalog.py"])

    if not args.skip_standardize:
        _run([py, "scripts/run_standardize_catalog.py"])

    if not args.skip_eda:
        _run([py, "scripts/run_phase2_eda.py"])

    if not args.skip_timeseries:
        _run([py, "scripts/run_timeseries_analysis.py"])

    if not args.skip_phase3:
        _run([py, "scripts/run_phase3_preprocess.py"])

    if not args.skip_processed_matrix:
        cmd = [
            py,
            "scripts/baselines/run_processed_matrix.py",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--fixed-win",
            str(args.fixed_win),
            "--fixed-hop",
            str(args.fixed_hop),
        ]
        if args.include_fixed_processed:
            cmd += ["--window-policies", "recommended", "fixed"]
        else:
            cmd.append("--skip-preprocess")
            cmd += ["--window-policies", "recommended"]
        if args.models:
            cmd += ["--models", *args.models]
        if not args.continue_on_failure:
            cmd.append("--stop-on-failure")
        _run(cmd)

    if args.include_fixed_processed and not args.skip_processed_matrix and not args.skip_unify:
        _run([py, "scripts/run_phase3_preprocess.py"])

    if not args.skip_unify:
        _run([py, "scripts/run_unify_dataset.py"])

    if not args.skip_unified_matrix:
        cmd = [
            py,
            "scripts/baselines/run_unified_sampling_matrix.py",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--label-space",
            "fine",
        ]
        if args.models:
            cmd += ["--models", *args.models]
        if not args.continue_on_failure:
            cmd.append("--stop-on-failure")
        _run(cmd)

    if not args.skip_cv:
        cmd = [
            py,
            "scripts/baselines/run_all_baselines.py",
            "--cv-folds",
            "5",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
        if args.models:
            cmd += ["--models", *args.models]
        if not args.continue_on_failure:
            cmd.append("--stop-on-failure")
        _run(cmd)

    if args.run_mspf_suite:
        mspf_cmd = [
            py,
            "scripts/run_mspf_full_evaluation.py",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
        if args.datasets:
            mspf_cmd += ["--datasets", *[str(ds) for ds in args.datasets]]
        if args.batch_size is not None:
            mspf_cmd += ["--batch_size", str(args.batch_size)]
        if args.processed_dir:
            mspf_cmd += ["--processed-dir", args.processed_dir]
        if args.run_suffix:
            mspf_cmd += ["--run-suffix", args.run_suffix]
        if args.skip_ablations:
            mspf_cmd.append("--skip-ablations")
        if args.continue_on_failure:
            mspf_cmd.append("--stop-on-failure")
        _run(mspf_cmd)
        ran_mspf_suite = True

    if not args.skip_aggregate:
        _run([py, "scripts/aggregate_results.py"])
        _run([py, "scripts/aggregate_cv_results.py"])
        if ran_mspf_suite:
            _run([py, "scripts/aggregate_mspf_results.py"])

    if not args.skip_audit:
        _run([py, "scripts/audit_baseline_matrix.py"])

    print("\nFull evaluation workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
