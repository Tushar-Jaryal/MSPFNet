from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.config_utils import get_config_list, get_config_value
from mspf_net.training.baseline_runner import _deep_merge, _load_yaml_with_extends

DEFAULT_ABLATION_DATASETS = [int(v) for v in get_config_list("datasets", "mspf_ablation", default=[2, 4, 5, 8])]
DEFAULT_EPOCHS = int(get_config_value("phase4", "epochs", default=50))
DEFAULT_DEVICE = str(get_config_value("phase4", "device", default="auto"))
CONFIG_RUNNER = "scripts/baselines/run_mspf_net_config.py"

BASE_CONFIGS = {
    "softmax": "configs/baselines/mspf_net.yaml",
    "rf": "configs/baselines/mspf_net_rf.yaml",
    "moe": "configs/baselines/mspf_net_moe.yaml",
}

SLIM_ABLATION_CONFIGS = {
    "no_periodic_path": "configs/baselines/mspf_net_no_periodic_path.yaml",
    "no_nonstationary_path": "configs/baselines/mspf_net_no_nonstationary_path.yaml",
    "equal_path_fusion": "configs/baselines/mspf_net_equal_path_fusion.yaml",
    "simple_channel_mixer": "configs/baselines/mspf_net_simple_channel_mixer.yaml",
    "no_se": "configs/baselines/mspf_net_no_se.yaml",
    "mean_channel_pooling": "configs/baselines/mspf_net_mean_channel.yaml",
}

DEFAULT_ABLATIONS = list(SLIM_ABLATION_CONFIGS.keys())

MOE_ABLATION_OVERRIDES: dict[str, dict] = {
    "no_periodic_path": {"model": {"moe_disabled_experts": ["cwt"]}},
    "no_nonstationary_path": {"model": {"moe_disabled_experts": ["non_periodic"]}},
    "equal_path_fusion": {"model": {"moe_router_mode": "equal"}},
}


def _build_merged_config(base_model: str, ablation_name: str, tmp_dir: Path) -> Path:
    if base_model not in BASE_CONFIGS:
        raise ValueError(f"Unknown base model: {base_model}")
    if ablation_name not in SLIM_ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    ablation_path = PROJECT_ROOT / SLIM_ABLATION_CONFIGS[ablation_name]
    cfg = _load_yaml_with_extends(str(ablation_path))

    if base_model != "softmax":
        base_path = PROJECT_ROOT / BASE_CONFIGS[base_model]
        base_cfg = _load_yaml_with_extends(str(base_path))
        base_training = deepcopy(base_cfg.get("training", {}))
        base_training.pop("experiment_tag", None)
        cfg = _deep_merge(cfg, {"training": base_training, "model": base_cfg.get("model", {})})
        if base_model == "moe" and ablation_name in MOE_ABLATION_OVERRIDES:
            cfg = _deep_merge(cfg, MOE_ABLATION_OVERRIDES[ablation_name])

    out_path = tmp_dir / f"mspf_{base_model}_{ablation_name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MSPF-Net slim ablations for Softmax, RF, and/or MoE.")
    parser.add_argument(
        "--base-models",
        nargs="+",
        choices=list(BASE_CONFIGS.keys()),
        default=list(BASE_CONFIGS.keys()),
        help="Which full-model bases to ablate (default: softmax rf moe).",
    )
    parser.add_argument("--variants", nargs="+", choices=list(SLIM_ABLATION_CONFIGS.keys()), default=None)
    parser.add_argument("--datasets", nargs="+", type=int, default=DEFAULT_ABLATION_DATASETS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    variants = args.variants if args.variants is not None else DEFAULT_ABLATIONS
    py = sys.executable
    failures = 0

    with tempfile.TemporaryDirectory(prefix="mspf_ablation_") as tmp:
        tmp_dir = Path(tmp)
        for base_model in args.base_models:
            for variant in variants:
                config_path = _build_merged_config(base_model, variant, tmp_dir)
                for dataset_id in args.datasets:
                    cmd = [
                        py,
                        CONFIG_RUNNER,
                        str(config_path),
                        "--datasets",
                        str(dataset_id),
                        "--epochs",
                        str(args.epochs),
                        "--device",
                        args.device,
                    ]
                    if args.batch_size is not None:
                        cmd += ["--batch_size", str(args.batch_size)]
                    if args.run_suffix:
                        cmd += ["--run-suffix", args.run_suffix]
                    if args.processed_dir:
                        cmd += ["--processed-dir", args.processed_dir]
                    print(f"\n=== MSPF-Net {base_model} / {variant} on D{dataset_id} ===")
                    rc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode
                    if rc != 0:
                        failures += 1
                        print(f"  FAILED (exit {rc})")
                        if args.stop_on_failure:
                            return rc
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
