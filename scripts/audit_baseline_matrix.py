from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mspf_net.constants import PRIMARY_SCRATCH_DATASETS, get_dataset_display
from mspf_net.config_utils import get_config_list, get_config_value


ACTIVE_TARGETS = [get_dataset_display(ds_id) for ds_id in PRIMARY_SCRATCH_DATASETS]
DEFAULT_ABLATION_DATASETS = [get_dataset_display(int(v)) for v in get_config_list("datasets", "mspf_ablation", default=[2, 4, 5, 8])]

FULL_MSPF_TAGS = {"softmax", "rf", "moe", None}
ABLATION_TAGS = {
    "no_periodic_path",
    "no_nonstationary_path",
    "no_se",
    "mean_channel_pooling",
    "equal_path_fusion",
    "simple_channel_mixer",
}


def _load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(results_dir.rglob("*_results.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("evaluation_mode") != "primary_test":
            continue
        rows.append(
            {
                "model": data["model"],
                "target": data["target_label"],
                "data_mode": data.get("data_mode", "processed"),
                "label_space": data.get("label_space", "fine"),
                "experiment_tag": data.get("experiment_tag"),
                "classifier_head": data.get("classifier_head", "softmax"),
                "architecture": data.get("architecture", "slim"),
                "seed": data.get("training_cfg", {}).get("seed"),
                "epochs": data.get("training_cfg", {}).get("epochs"),
                "patience": data.get("training_cfg", {}).get("patience"),
                "epochs_ran": data.get("epochs_ran"),
            }
        )
    return rows


def _scratch_rows(rows: list[dict]) -> list[dict]:
    return [
        r
        for r in rows
        if r["data_mode"] == "processed"
        and r["label_space"] == "fine"
        and r["target"] in ACTIVE_TARGETS
    ]


def _coverage_status(by_target: set[str], expected: list[str]) -> str:
    missing = [t for t in expected if t not in by_target]
    return "COMPLETE" if not missing else f"MISSING {missing}"


def _print_mspf_coverage(title: str, rows: list[dict], expected: list[str]) -> None:
    by_tag: dict[str | None, set[str]] = defaultdict(set)
    for row in rows:
        if row["model"] != "mspf_net":
            continue
        by_tag[row.get("experiment_tag")].add(row["target"])

    print(title)
    if not by_tag:
        print("  (no results)")
        return
    for tag in sorted(by_tag, key=lambda t: (t is None, str(t))):
        label = tag or "default"
        print(f"  {label:24s} {_coverage_status(by_tag[tag], expected)}")


def main() -> int:
    repo = Path.cwd()
    results_dir = repo / str(get_config_value("phase4", "results_dir", default="results/baselines"))
    rows = _load_rows(results_dir)
    scratch = _scratch_rows(rows)

    full_rows = [r for r in scratch if r["model"] == "mspf_net" and r.get("experiment_tag") in FULL_MSPF_TAGS]
    ablation_rows = [r for r in scratch if r["model"] == "mspf_net" and r.get("experiment_tag") in ABLATION_TAGS]

    _print_mspf_coverage("Processed fine-label MSPF-Net full-model coverage (softmax, rf, moe)", full_rows, ACTIVE_TARGETS)
    _print_mspf_coverage(
        "MSPF-Net ablation coverage (ablation datasets)",
        ablation_rows,
        DEFAULT_ABLATION_DATASETS,
    )

    other_models = sorted({r["model"] for r in scratch if r["model"] != "mspf_net"})
    if other_models:
        print("\nOther baseline models (full runs, no experiment tag)")
        for model in other_models:
            targets = {r["target"] for r in scratch if r["model"] == model and not r.get("experiment_tag")}
            print(f"  {model:16s} {_coverage_status(targets, ACTIVE_TARGETS)}")

    combos = defaultdict(set)
    for r in rows:
        key = (r["model"], r["data_mode"], r["label_space"], r["experiment_tag"])
        combos[key].add((r["seed"], r["epochs"], r["patience"]))
    print("\nTraining-config consistency by result family")
    for key, cfgs in sorted(combos.items()):
        status = "CONSISTENT" if len(cfgs) == 1 else f"INCONSISTENT {sorted(cfgs)}"
        print(f"  {key}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
