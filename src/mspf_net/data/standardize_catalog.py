import re, sys, argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parents[3]
CATALOG_PATH   = PROJECT_ROOT / "data" / "interim" / "catalog.csv"
STD_CATALOG    = PROJECT_ROOT / "data" / "interim" / "catalog_standardized.csv"
NAMING_REPORT  = PROJECT_ROOT / "results" / "tables" / "naming_convention_report.csv"

# Sub-dataset integer codes (used in std_name)
SUB_CODES = {
    "bearing":          "brg",
    "gearbox":          "gbx",
    "mixed":            "mix",
    "cwru_bearing":     "cwru",
    "gearbox_bearing":  "gbxbrg",
    "gearbox_gearset":  "gbxgear",
    "gearbox_crack":    "gbxcrk",
    "gearbox_hust":     "gbxhust",
}

def clean_condition(cond_str):
    """Normalize condition string for use in a filename."""
    if not isinstance(cond_str, str):
        return "c000"
    # Remove spaces, special chars; keep alphanumeric, dash, underscore
    c = re.sub(r'[^\w\-]', '_', cond_str)
    c = re.sub(r'_+', '_', c).strip('_')
    return c[:30]  # cap length


def dataset_token(ds_id: int) -> str:
    ds_id = int(ds_id)
    if ds_id == 31:
        return "3c1"
    if ds_id == 39:
        return "3c9"
    return str(ds_id)


def dataset_display(ds_id: int) -> str:
    return f"D{dataset_token(ds_id).upper()}"

def make_std_name(row):
    """Build the standardized canonical filename for one catalog row."""
    ds  = dataset_token(int(row["dataset_id"]))
    sub = SUB_CODES.get(str(row["sub_dataset"]), "unk")
    fc  = str(row["fault_code"])
    rep  = int(row.get("rep", 0)) if pd.notna(row.get("rep")) else 0
    ext  = Path(str(row["raw_filename"])).suffix.lower()

    # Include severity if present and not NA
    sev = str(row.get("severity_code", "NA"))
    sev_part = f"_{sev}" if sev not in ("NA", "nan", "") else ""

    # D5 fix: variable-speed sweep files use 'varspd' instead of '200-2400-200'
    # to prevent dash-delimited glob patterns from misinterpreting the token.
    # Detected by speed_rpm being null/None in the catalog (only sweep files).
    if ds == 5 and pd.isna(row.get("speed_rpm")):
        cond = "varspd"
    else:
        cond = clean_condition(str(row.get("condition_str", "c000")))

    return f"d{ds}_{sub}_{fc}{sev_part}_{cond}_{rep:03d}{ext}"

def make_std_path(row):
    """
    Build an informational standardized raw-data path.

    This is a canonical catalog path for the source file naming scheme, not the
    Phase 3 processed-window output location. Phase 3 writes flat split-specific
    artifacts such as `d{ds}_train_windows.npy`.
    """
    ds  = dataset_token(int(row["dataset_id"]))
    sub = SUB_CODES.get(str(row["sub_dataset"]), "unk")
    fc  = str(row["fault_code"])
    std_name = row["std_name"]
    return f"data/standardized/dataset_{ds}/{sub}/{fc}/{std_name}"

def assign_split(df, train=0.60, val=0.20, seed=42):
    """
    File-level split with dataset-specific condition protocols where available.

    Deterministic condition-based splits:
        D1  cond_A/cond_B → train, cond_C → val, cond_D → test
        D2  20–35 Hz      → train, 40–45 Hz → val, 50–55 Hz → test
        D3C1 CWRU load 0/1 → train, load 2 → val, load 3 → test
        D5  1200 RPM      → train, 1800 RPM → val, 2400 RPM → test,
            NaN RPM       → robustness
        D8  20–30 Hz      → train, 35 Hz → val, 40 Hz → test,
            *_VS_*        → robustness

    All remaining files use stratified random 60/20/20 by
    (dataset_id, sub_dataset, fault_code).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    split_map = {}
    small_group_cycle: defaultdict[int, int] = defaultdict(int)

    cond_series = df.get("condition_str", pd.Series(index=df.index, dtype=object)).fillna("").astype(str)
    sub_series = df.get("sub_dataset", pd.Series(index=df.index, dtype=object)).fillna("").astype(str)
    speed_series = pd.to_numeric(df.get("speed_rpm", pd.Series(index=df.index, dtype=float)), errors="coerce")

    def assign_by_index(mask, split):
        for ix in df[mask].index:
            split_map[ix] = split

    def assign_fallback_group(idx: list[int], dataset_id: int) -> None:
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        if n == 0:
            return
        if n == 1:
            split_map[idx[0]] = "train"
            return
        if n == 2:
            split_map[idx[0]] = "train"
            second_split = "val" if small_group_cycle[int(dataset_id)] % 2 == 0 else "test"
            split_map[idx[1]] = second_split
            small_group_cycle[int(dataset_id)] += 1
            return
        if n == 3:
            split_map[idx[0]] = "train"
            split_map[idx[1]] = "val"
            split_map[idx[2]] = "test"
            return

        n_train = max(1, round(n * train))
        n_val = max(1, round(n * val))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_train = min(n_train, n - n_val - 1)
        for i, ix in enumerate(idx):
            if i < n_train:
                split_map[ix] = "train"
            elif i < n_train + n_val:
                split_map[ix] = "val"
            else:
                split_map[ix] = "test"

    # D1: hold out operating conditions C and D as val/test.
    is_d1 = df["dataset_id"] == 1
    assign_by_index(is_d1 & cond_series.isin(["cond_A", "cond_B"]), "train")
    assign_by_index(is_d1 & (cond_series == "cond_C"), "val")
    assign_by_index(is_d1 & (cond_series == "cond_D"), "test")

    # D2: condition split by shaft speed.
    is_d2 = df["dataset_id"] == 2
    assign_by_index(is_d2 & speed_series.isin([1200.0, 1500.0, 1800.0, 2100.0]), "train")
    assign_by_index(is_d2 & speed_series.isin([2400.0, 2700.0]), "val")
    assign_by_index(is_d2 & speed_series.isin([3000.0, 3300.0]), "test")

    # D3: apply the canonical load-based split only to the CWRU bearing subset.
    is_d3c1 = df["dataset_id"] == 31
    assign_by_index(is_d3c1 & cond_series.isin(["load0hp", "load1hp"]), "train")
    assign_by_index(is_d3c1 & (cond_series == "load2hp"), "val")
    assign_by_index(is_d3c1 & (cond_series == "load3hp"), "test")

    # D5: deterministic speed-condition split to preserve the intended
    # cross-speed evaluation protocol.
    is_d5 = df["dataset_id"] == 5
    assign_by_index(is_d5 & speed_series.isna(), "robustness")
    assign_by_index(is_d5 & (speed_series == 1200.0), "train")
    assign_by_index(is_d5 & (speed_series == 1800.0), "val")
    assign_by_index(is_d5 & (speed_series == 2400.0), "test")

    # D8: keep variable-speed files held out, and split stationary files by speed.
    is_d8 = df["dataset_id"] == 8
    assign_by_index(is_d8 & cond_series.str.contains("VS"), "robustness")
    assign_by_index(is_d8 & speed_series.isin([1200.0, 1500.0, 1800.0]), "train")
    assign_by_index(is_d8 & (speed_series == 2100.0), "val")
    assign_by_index(is_d8 & (speed_series == 2400.0), "test")

    assigned_idx = pd.Index(split_map.keys())

    # All remaining datasets/files: stratified random 60/20/20.
    steady = df[~df.index.isin(assigned_idx)]
    for (dataset_id, _, _), grp in steady.groupby(["dataset_id", "sub_dataset", "fault_code"]):
        assign_fallback_group(grp.index.tolist(), int(dataset_id))

    return df.index.map(lambda i: split_map.get(i, "train"))


def compute_test_eligible(df):
    """
    Returns a boolean Series: True if the (dataset_id, fault_code) group
    has at least one file assigned to a real held-out `test` split.

    Condition-based splits cover most benchmark datasets directly. Sparse
    groups that only have 1–2 files in a given dataset×sub_dataset×fault cell
    may still produce no proper held-out test file, so those rows stay
    test_eligible=False for downstream evaluation filtering.
    """
    # Build set of (dataset_id, fault_code) pairs that have a real held-out
    # test file. Robustness is intentionally excluded because it is a separate
    # evaluation slice and should not make a class look primary-benchmark-ready.
    eligible_pairs = set(
        df[df["split"] == "test"]
        .groupby(["dataset_id", "fault_code"])
        .groups.keys()
    )
    return df.apply(
        lambda r: (int(r["dataset_id"]), r["fault_code"]) in eligible_pairs,
        axis=1
    )


def validate_taxonomy(df: pd.DataFrame) -> None:
    legacy = {"GBX_SW", "GBX_TB"}
    bad_codes = sorted(set(df["fault_code"]).intersection(legacy))
    if bad_codes:
        raise ValueError(f"Legacy fault codes remain after merge: {bad_codes}")
    bad_coarse = df["coarse_fault_code"].isna()
    if bad_coarse.any():
        missing = sorted(df.loc[bad_coarse, "fault_code"].unique().tolist())
        raise ValueError(f"Missing coarse_fault_code mapping for: {missing}")


def validate_split_integrity(df: pd.DataFrame) -> None:
    bad_split = df["split"].isna()
    if bad_split.any():
        raise ValueError("Some rows are missing split assignments")

    # File-level exclusivity is the leakage guard for the whole pipeline.
    dup = (
        df.groupby("file_path")["split"]
        .nunique()
        .reset_index(name="n_splits")
        .query("n_splits > 1")
    )
    if not dup.empty:
        raise ValueError(
            "A raw file appears in multiple splits: "
            + ", ".join(dup["file_path"].astype(str).tolist()[:5])
        )


def validate_naming(df: pd.DataFrame) -> None:
    legacy_pattern = r"GBX_SW|GBX_TB"
    bad_names = df["std_name"].astype(str).str.contains(legacy_pattern, regex=True, na=False)
    if bad_names.any():
        raise ValueError(
            "Legacy label tokens remain in standardized names: "
            + ", ".join(df.loc[bad_names, "std_name"].astype(str).head(5).tolist())
        )


def print_report(df):
    """Print the naming convention mapping table and split distribution."""
    print(f"\n\033[1m{'='*100}\033[0m")
    print(f"  \033[1mStandardized Naming Convention Report\033[0m — {len(df)} files")
    print(f"{'='*100}")
    print(f"\n\033[1m{'DS':<4} {'Raw Filename':<52} {'Standardized Name':<52}\033[0m")
    print("-"*108)

    current_ds = None
    for _, row in df.iterrows():
        ds = int(row["dataset_id"])
        ds_label = dataset_display(ds)
        if ds != current_ds:
            print(f"\n  \033[34m── {ds_label} ──────────────────────────────────────────────────────────────────────\033[0m")
            current_ds = ds
        raw = str(row["raw_filename"])
        std = str(row["std_name"])
        print(f"{ds_label:<4} {raw:<52} → {std}")

    # Split distribution with test_eligible and robustness columns
    print(f"\n\033[1mSplit Distribution:\033[0m")
    print(f"  {'DS':<4} {'Fault Code':<12} {'Train':>7} {'Val':>7} {'Test':>7} {'Robust':>8} {'Total':>7}  {'Eligible'}")
    print(f"  {'-'*62}")

    for (ds, fc), grp in df.groupby(["dataset_id", "fault_code"]):
        ds_label = dataset_display(int(ds))
        counts = grp["split"].value_counts()
        tr   = counts.get("train", 0)
        vl   = counts.get("val", 0)
        te   = counts.get("test", 0)
        rb   = counts.get("robustness", 0)
        elig = grp["test_eligible"].any()
        flag = "✅" if elig else "⚠️  no held-out test"
        rb_str = f"{rb:>8}" if rb > 0 else "        "
        print(f"  {ds_label:<4} {fc:<12} {tr:>7} {vl:>7} {te:>7}{rb_str} {tr+vl+te+rb:>7}  {flag}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        print(f"\033[91mCatalog not found: {CATALOG_PATH}\033[0m")
        print("Run: python scripts/run_build_catalog.py first.")
        sys.exit(1)

    df = pd.read_csv(CATALOG_PATH)
    print(f"\n\033[1mMSPF-Net — Standardize Catalog\033[0m")
    print(f"  Loaded {len(df)} entries from {CATALOG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Add standardized names
    df["std_name"] = df.apply(make_std_name, axis=1)
    df["std_path"] = df.apply(make_std_path, axis=1)

    # Assign file-level splits using dataset-specific protocols plus fallback random.
    df["split"] = assign_split(df)

    # Add test_eligible flag
    df["test_eligible"] = compute_test_eligible(df)

    # Validate taxonomy, split integrity, and naming consistency before write.
    validate_taxonomy(df)
    validate_split_integrity(df)
    validate_naming(df)

    # Print summary of special assignments
    n_robust = (df["split"] == "robustness").sum()
    n_ineligible = (~df["test_eligible"]).sum()
    if n_robust > 0:
        robust_ds = sorted(df.loc[df["split"] == "robustness", "dataset_id"].unique().tolist())
        robust_labels = ", ".join(dataset_display(ds) for ds in robust_ds)
        print(f"  \033[93m⚠️  {n_robust} variable-speed files from {robust_labels} → split='robustness' (speed-invariance experiment)\033[0m")
    if n_ineligible > 0:
        inelig_classes = (
            df[~df["test_eligible"]]
            .groupby(["dataset_id", "fault_code"])
            .size()
            .index.tolist()
        )
        labels = [f"{dataset_display(ds)}/{fc}" for ds, fc in inelig_classes]
        print(f"  \033[93m⚠️  {n_ineligible} files marked test_eligible=False (no test files for that dataset×fault): {', '.join(labels)}\033[0m")
    print()

    print_report(df)

    if not args.dry_run:
        STD_CATALOG.parent.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
        df.to_csv(STD_CATALOG, index=False)
        # Save naming report (raw → std mapping)
        df[["dataset_id", "dataset_name", "sub_dataset", "raw_filename", "std_name",
            "fault_code", "coarse_fault_code", "fault_label", "condition_str",
            "split", "test_eligible"]].to_csv(
                NAMING_REPORT, index=False)
        print(f"\n  Standardized catalog → {STD_CATALOG.relative_to(PROJECT_ROOT)}")
        print(f"  Naming report        → {NAMING_REPORT.relative_to(PROJECT_ROOT)}")
    else:
        print("\n  \033[93m[DRY RUN] No files written.\033[0m")

if __name__ == "__main__":
    main()
