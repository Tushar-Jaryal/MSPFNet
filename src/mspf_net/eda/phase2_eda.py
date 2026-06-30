from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

# ─── Project root resolution ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

from mspf_net.eda.eda_utils import (
    FS_MAP,
    compute_stats,
    summarize_class_separability,
    load_class_representatives,
    plot_class_balance,
    plot_class_grid,
    plot_class_separability,
    plot_cross_dataset_stats,
)
from mspf_net.constants import ACTIVE_THESIS_DATASETS, get_dataset_display

# ─── Dataset metadata ─────────────────────────────────────────────────────────
DS_INFO = {
    1: {"name": "D1_BearingTV",    "component": "Bearing",         "primary": True},
    2: {"name": "D2_PlanetaryGB",  "component": "Gearbox",         "primary": False},
    31: {"name": "D3C1_CWRU_Bearing",          "component": "Bearing",         "primary": False},
    39: {"name": "D3C9_Gearbox",               "component": "Gearbox (9-ch)",  "primary": False},
    4: {"name": "D4_MultiModeGB",  "component": "Gearbox",         "primary": False},
    5: {"name": "D5_MixedBG",      "component": "Mixed (3-ch)",    "primary": True},
    8: {"name": "D8_HUSTGearbox",  "component": "Gearbox",         "primary": False},
}

# Frequency ceiling for PSD plots (Hz) — avoid plotting up to Nyquist for high-fs datasets
F_MAX_PLOT = {1: 5_000, 2: 4_000, 31: 3_000, 39: 3_000, 4: 2_000, 5: 2_000, 8: 4_000}


# ═══════════════════════════════════════════════════════════════════════════════

def load_catalog(project_root: Path) -> pd.DataFrame:
    """
    Load catalog.csv (which has file_path) and left-join split/test_eligible
    from catalog_standardized.csv. Falls back to catalog.csv alone if missing.
    """
    base_p = project_root / "data" / "interim" / "catalog.csv"
    std_p  = project_root / "data" / "interim" / "catalog_standardized.csv"
    if not base_p.exists():
        raise FileNotFoundError(
            "data/interim/catalog.csv not found. Run: python scripts/run_build_catalog.py"
        )
    base = pd.read_csv(base_p)
    print(f"  Base catalog : data/interim/catalog.csv  ({len(base)} rows)")
    if std_p.exists():
        std = pd.read_csv(std_p)
        merge_cols = [c for c in ["dataset_id", "sub_dataset", "raw_filename", "fault_code"]
                      if c in base.columns and c in std.columns]
        add_cols   = [c for c in ["split", "test_eligible", "std_name", "coarse_fault_code"]
                      if c in std.columns and c not in base.columns]
        if merge_cols and add_cols:
            base = base.merge(std[merge_cols + add_cols], on=merge_cols, how="left")
            print(f"  + split info from data/interim/catalog_standardized.csv  (added: {add_cols})")
    else:
        print("  [WARN] data/interim/catalog_standardized.csv not found")
    return base


def run_eda_dataset(ds_id: int, catalog: pd.DataFrame,
                    project_root: Path,
                    out_dir: Path,
                    peek_only: bool = False) -> tuple[list[dict], pd.DataFrame]:
    """
    Run full EDA for one dataset.
    Returns list of per-class stat dicts for the summary CSV.
    """
    info = DS_INFO[ds_id]
    name = info["name"]
    fs   = FS_MAP[ds_id]
    pri  = "★ PRIMARY" if info["primary"] else ""
    f_max = F_MAX_PLOT[ds_id]

    ds_label = name.split("_", 1)[0]
    print(f"\n{'─'*68}")
    print(f"  {ds_label} — {name}  [{info['component']}]  fs={fs:,} Hz  {pri}")
    print(f"{'─'*68}")

    # ── Load one signal per class ─────────────────────────────────────────────
    t0 = time.time()
    signals = load_class_representatives(catalog, ds_id, project_root)
    elapsed = time.time() - t0

    if not signals:
        print(f"  [SKIP] No readable files found (check paths under {project_root})")
        return [], pd.DataFrame()

    n_loaded = len(signals)
    print(f"  Loaded {n_loaded} class representative(s) in {elapsed:.1f}s")
    for code, sig in signals.items():
        print(f"    {code:12s}  shape={sig.shape}  dtype={sig.dtype}")

    # ── Compute stats ─────────────────────────────────────────────────────────
    coarse_lookup = (
        catalog[catalog["dataset_id"] == ds_id][["fault_code", "coarse_fault_code"]]
        .dropna()
        .drop_duplicates("fault_code")
        .set_index("fault_code")["coarse_fault_code"]
        .to_dict()
    )
    stats_rows = []
    for code, sig in signals.items():
        st = compute_stats(sig, fs=fs, label=code)
        st["dataset_id"]   = ds_id
        st["dataset_name"] = name
        st["fault_code"]   = code
        st["coarse_fault_code"] = coarse_lookup.get(code)
        stats_rows.append(st)

    # Print summary table
    cols = ["fault_code", "n_samples", "duration_s", "rms",
            "kurtosis", "crest_factor", "skewness", "spectral_entropy"]
    df_stats = pd.DataFrame(stats_rows)[cols]
    print()
    print(df_stats.to_string(index=False))

    sep_df = summarize_class_separability(pd.DataFrame(stats_rows))
    if not sep_df.empty:
        print("\n  Closest class pairs (hardest to distinguish by summary stats):")
        for _, row in sep_df.head(3).iterrows():
            print(f"    {row['fault_a']:12s} vs {row['fault_b']:12s}  distance={row['distance']:.2f}")

        if "NOR" in set(sep_df["fault_a"]).union(set(sep_df["fault_b"])):
            nor_pairs = sep_df[(sep_df["fault_a"] == "NOR") | (sep_df["fault_b"] == "NOR")].head(3)
            if not nor_pairs.empty:
                print("\n  Classes closest to NOR (potential confusion risk):")
                for _, row in nor_pairs.iterrows():
                    other = row["fault_b"] if row["fault_a"] == "NOR" else row["fault_a"]
                    print(f"    {other:12s}  distance_to_NOR={row['distance']:.2f}")

    if peek_only:
        return stats_rows, sep_df

    # ── Generate figures ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    # 1. Waveform + PSD grid
    grid_path = out_dir / f"d{ds_id}_{name.lower()}_grid.png"
    print(f"\n  Plotting grid  → {grid_path.name} ...", end=" ", flush=True)
    fig = plot_class_grid(signals, fs=fs,
                          dataset_name=f"D{ds_id} — {name}",
                          f_max=f_max)
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done")

    # 2. Class balance
    bal_path = out_dir / f"d{ds_id}_{name.lower()}_balance.png"
    print(f"  Plotting balance → {bal_path.name} ...", end=" ", flush=True)
    sub_catalog = catalog[catalog["dataset_id"] == ds_id]
    fig2 = plot_class_balance(sub_catalog, ds_id, ds_name=f"D{ds_id} — {name}")
    fig2.savefig(bal_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("done")

    # 3. Class separability heatmap
    sep_path = out_dir / f"d{ds_id}_{name.lower()}_separation.png"
    print(f"  Plotting separation → {sep_path.name} ...", end=" ", flush=True)
    fig3 = plot_class_separability(pd.DataFrame(stats_rows), dataset_name=f"D{ds_id} — {name}")
    fig3.savefig(sep_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("done")

    return stats_rows, sep_df


def build_split_diagnostic(catalog: pd.DataFrame, ds_id: int) -> pd.DataFrame:
    """Summarize file-level split coverage for one dataset and each class."""
    sub = catalog[catalog["dataset_id"] == ds_id].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for fault_code, grp in sub.groupby("fault_code"):
        counts = grp["split"].value_counts() if "split" in grp.columns else {}
        rows.append(
            {
                "dataset_id": ds_id,
                "dataset_label": get_dataset_display(ds_id),
                "dataset_name": DS_INFO[ds_id]["name"],
                "fault_code": fault_code,
                "n_files": int(len(grp)),
                "n_train_files": int(counts.get("train", 0)),
                "n_val_files": int(counts.get("val", 0)),
                "n_test_files": int(counts.get("test", 0)),
                "n_robustness_files": int(counts.get("robustness", 0)),
                "test_eligible": bool(grp["test_eligible"].any()) if "test_eligible" in grp.columns else True,
            }
        )
    return pd.DataFrame(rows).sort_values(["fault_code"]).reset_index(drop=True)


def print_phase2_header():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          MSPF-Net  —  Phase 2: Exploratory Data Analysis        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    import datetime
    print(f"  Timestamp  : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Project    : {PROJECT_ROOT}")
    print()


def print_phase2_footer(all_stats: list[dict], out_dir: Path):
    df = pd.DataFrame(all_stats)
    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  Phase 2 EDA Summary")
    print("══════════════════════════════════════════════════════════════════")
    print(f"  Classes analysed : {len(df)}")

    # Highest kurtosis (most impulsive / easiest to distinguish)
    top_k = df.nlargest(3, "kurtosis")[["dataset_name","fault_code","kurtosis","crest_factor"]]
    print("\n  Top-3 highest kurtosis (most impulsive signals):")
    for _, row in top_k.iterrows():
        ds_label = row["dataset_name"].split("_", 1)[0]
        print(f"    {ds_label}  {row['fault_code']:12s}  "
              f"kurtosis={row['kurtosis']:.2f}  CF={row['crest_factor']:.2f}")

    # Lowest kurtosis (most similar to Normal — hardest to separate)
    bot_k = df.nsmallest(3, "kurtosis")[["dataset_name","fault_code","kurtosis"]]
    print("\n  Bottom-3 lowest kurtosis (closest to Gaussian / Normal-like):")
    for _, row in bot_k.iterrows():
        ds_label = row["dataset_name"].split("_", 1)[0]
        print(f"    {ds_label}  {row['fault_code']:12s}  "
              f"kurtosis={row['kurtosis']:.2f}")

    print()
    print(f"  Results saved → {out_dir}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MSPF-Net Phase 2 EDA")
    parser.add_argument("--dataset", nargs="+", type=int,
                        choices=sorted(DS_INFO.keys()), metavar="N",
                        help="Specific dataset IDs to run (default: all)")
    parser.add_argument("--peek", action="store_true",
                        help="Print stats only, skip figure generation")
    args = parser.parse_args()

    ds_ids    = args.dataset or ACTIVE_THESIS_DATASETS
    peek_only = args.peek

    print_phase2_header()

    # ── Setup output dirs ─────────────────────────────────────────────────────
    fig_dir = PROJECT_ROOT / "results" / "figures" / "eda"
    tbl_dir = PROJECT_ROOT / "results" / "tables"
    if not peek_only:
        fig_dir.mkdir(parents=True, exist_ok=True)
        tbl_dir.mkdir(parents=True, exist_ok=True)

    # ── Load catalog ──────────────────────────────────────────────────────────
    catalog = load_catalog(PROJECT_ROOT)

    # ── Per-dataset EDA ───────────────────────────────────────────────────────
    all_stats: list[dict] = []
    all_sep: list[pd.DataFrame] = []
    all_split_diag: list[pd.DataFrame] = []
    for ds_id in ds_ids:
        try:
            rows, sep_df = run_eda_dataset(ds_id, catalog, PROJECT_ROOT,
                                           out_dir=fig_dir, peek_only=peek_only)
            all_stats.extend(rows)
            split_diag = build_split_diagnostic(catalog, ds_id)
            if not split_diag.empty:
                all_split_diag.append(split_diag)
            if not sep_df.empty:
                sep_df = sep_df.copy()
                sep_df.insert(0, "dataset_id", ds_id)
                sep_df.insert(1, "dataset_name", DS_INFO[ds_id]["name"])
                all_sep.append(sep_df)
        except Exception as exc:
            print(f"\n  [ERROR] D{ds_id}: {exc}")
            import traceback
            traceback.print_exc()

    if not all_stats:
        print("\n  No stats collected — check dataset paths.")
        return

    # ── Cross-dataset figure ──────────────────────────────────────────────────
    if not peek_only and len(all_stats) > 0:
        import matplotlib.pyplot as plt
        df_all = pd.DataFrame(all_stats)
        cross_path = fig_dir / "cross_dataset_stats.png"
        print(f"\n  Plotting cross-dataset heatmap → {cross_path.name} ...",
              end=" ", flush=True)
        fig_c = plot_cross_dataset_stats(df_all)
        fig_c.savefig(cross_path, dpi=150, bbox_inches="tight")
        plt.close(fig_c)
        print("done")

    # ── Save stats CSV ────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_stats)
    col_order = [
        "dataset_id","dataset_name","fault_code","coarse_fault_code","label",
        "n_samples","duration_s","mean","std","rms",
        "peak","peak_to_peak","crest_factor","kurtosis","skewness","spectral_entropy"
    ]
    col_order = [c for c in col_order if c in df_all.columns]
    df_all = df_all[col_order].sort_values(["dataset_id","fault_code"])

    stats_path = tbl_dir / "eda_stats.csv"
    df_all.to_csv(stats_path, index=False)
    print(f"\n  Stats table → {stats_path.relative_to(PROJECT_ROOT)}")

    if all_sep:
        sep_all = pd.concat(all_sep, ignore_index=True)
        sep_path = tbl_dir / "eda_class_separation.csv"
        sep_all.to_csv(sep_path, index=False)
        print(f"  Separability → {sep_path.relative_to(PROJECT_ROOT)}")

    if all_split_diag:
        split_diag_all = pd.concat(all_split_diag, ignore_index=True)
        split_diag_path = tbl_dir / "eda_split_diagnostics.csv"
        split_diag_all.to_csv(split_diag_path, index=False)
        print(f"  Split coverage → {split_diag_path.relative_to(PROJECT_ROOT)}")

    print_phase2_footer(all_stats, fig_dir if not peek_only else tbl_dir)


if __name__ == "__main__":
    main()
