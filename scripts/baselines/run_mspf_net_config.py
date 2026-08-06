from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.training.baseline_runner import run_baseline


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_mspf_net_config.py <config_yaml> [--datasets ...] [--epochs ...]")
    config_path = sys.argv[1]
    run_baseline("mspf_net", config_path, argv=sys.argv[2:])
