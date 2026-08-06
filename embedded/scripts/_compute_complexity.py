#!/usr/bin/env python3
from __future__ import annotations
import csv, json, sys
from pathlib import Path

_EMB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EMB)); sys.path.insert(0, str(_EMB.parent / "src"))
import torch
from thop import profile
from mspf_embedded._paths import repo_root
from mspf_embedded.bundle import load_bundle

ROOT = repo_root()
B = ROOT / "embedded/bundles"
OUT = ROOT / "embedded/results/tables/pi5_complexity_canonical.csv"

# FLOPs fallback for architectures thop cannot hook (from existing Pi benchmark CSV)
bench_flops = {}
bench_csv = ROOT / "embedded/results/tables/pi5_benchmark_latest.csv"
if bench_csv.exists():
    for r in csv.DictReader(open(bench_csv)):
        bench_flops[(r["model"], r["target"].lower())] = float(r["flops_g"])

THOP_UNRELIABLE = {"transformer1d"}  # attention not hooked by installed thop

rows = []
for d in sorted(p for p in B.iterdir() if p.is_dir() and (p / "manifest.json").exists()):
    m = json.load(open(d / "manifest.json"))
    model, _, _ = load_bundle(d)
    model.eval()
    c, L = m["window_shape"]
    x = torch.randn(1, int(c), int(L))
    params_full = int(sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        macs, params_thop = profile(model, inputs=(x,), verbose=False)
    tgt = f"D{m['dataset_id']}"
    if m["model"] in THOP_UNRELIABLE:
        flops_g = bench_flops.get((m["model"], tgt.lower()), float("nan"))
        flops_source = "pi_benchmark_csv"
    else:
        flops_g = round(2 * macs / 1e9, 4)
        flops_source = "thop"
    rows.append({
        "bundle": d.name, "model": m["model"], "target": tgt,
        "dataset_id": m["dataset_id"], "in_channels": m["in_channels"],
        "window_len": int(L), "num_classes": m["num_classes"],
        "params_full": params_full, "params_thop": int(params_thop),
        "flops_g": flops_g, "flops_source": flops_source,
    })
    print(f"{d.name:26} full={params_full:>9} thop={int(params_thop):>9} "
          f"flops_g={flops_g} ({flops_source})", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {OUT} ({len(rows)} rows)", flush=True)
