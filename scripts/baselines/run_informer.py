from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.training.baseline_runner import run_baseline


if __name__ == "__main__":
    run_baseline("informer", "configs/baselines/informer.yaml")
