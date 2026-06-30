import re, sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from mspf_net.constants import ACTIVE_RAW_DATASETS, DATASET_NAMES, get_dataset_display, get_dataset_name

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
LABEL_MAP    = PROJECT_ROOT / "data" / "metadata" / "label_mapping.json"
CATALOG_PATH = PROJECT_ROOT / "data" / "interim" / "catalog.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "interim" / "catalog_summary.csv"

# ─── Load label mapping ────────────────────────────────────────────────────────
with open(LABEL_MAP) as f:
    LM = json.load(f)

FAULT_CODES  = LM["fault_codes"]
DS_CLASS_MAP = LM["dataset_class_maps"]
COARSE_FAULT_MAP = LM.get("coarse_fault_map", {})

FS_TARGET = 10000
N_CHANNELS = {1: 1, 2: 1, 3: 1, 31: 1, 39: 9, 4: 1, 5: 3, 6: 1, 7: 8, 8: 4}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def fault_label(code):
    return FAULT_CODES.get(code, {}).get("label", code)

def fault_component(code):
    return FAULT_CODES.get(code, {}).get("component", "unknown")

def coarse_fault_code(code):
    return COARSE_FAULT_MAP.get(code, "UNKNOWN")

def dataset_name(ds_id):
    return get_dataset_name(int(ds_id))


def display_dataset_id(ds_id: int) -> str:
    return get_dataset_display(int(ds_id))

def is_variable_speed_row(row):
    ds_id = int(row["dataset_id"])
    if ds_id == 1:
        return True
    if ds_id == 5 and pd.isna(row.get("speed_rpm")):
        return True
    if ds_id == 8 and isinstance(row.get("condition_str"), str) and "VS" in row.get("condition_str"):
        return True
    return False


def inspect_n_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect real files to populate authoritative channel counts where needed.

    D2 and D3-family datasets are inspected explicitly because loader heuristics can expose
    more columns than the old hardcoded fallback suggested.
    """
    from mspf_net.eda.eda_utils import load_signal_multichannel

    df = df.copy()
    needs_inspect = df["dataset_id"].isin([2, 31, 39]) | df["n_channels"].isna()
    if not needs_inspect.any():
        return df

    print("  Inspecting real channel counts for D2/D3-family ...")
    for ix, row in df.loc[needs_inspect].iterrows():
        path = PROJECT_ROOT / str(row["file_path"])
        if not path.exists():
            continue
        try:
            sig = load_signal_multichannel(path, int(row["dataset_id"]))
            sig = np.asarray(sig)
            n_channels = 1 if sig.ndim == 1 else int(sig.shape[1])
            df.at[ix, "n_channels"] = n_channels
        except Exception:
            # Keep parser-declared fallback if inspection fails.
            fallback = row["n_channels"] if pd.notna(row["n_channels"]) else N_CHANNELS.get(int(row["dataset_id"]))
            df.at[ix, "n_channels"] = fallback
    return df

def finalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical columns expected downstream and keep a stable schema."""
    df = df.copy()
    # Normalize every parser output to one catalog schema so downstream phases
    # can treat all datasets uniformly, even when some fields are missing.
    df["dataset_name"] = df["dataset_id"].map(dataset_name)
    df["coarse_fault_code"] = df["fault_code"].map(coarse_fault_code)
    df["fault_label"] = df["fault_code"].map(fault_label)
    df["component"] = df["fault_code"].map(fault_component)
    df["severity_code"] = df.get("severity_code", pd.Series(index=df.index, dtype=object)).fillna("NA")
    df["is_variable_speed"] = df.apply(is_variable_speed_row, axis=1)
    df["n_channels"] = df.get("n_channels", pd.Series(index=df.index, dtype=float)).fillna(df["dataset_id"].map(N_CHANNELS))
    df["fs_raw"] = df.get("fs_raw", pd.Series(index=df.index, dtype=float))
    df["fs_target"] = FS_TARGET
    df["std_name"] = pd.NA
    df["split"] = pd.NA
    df["test_eligible"] = pd.NA

    required = [
        "dataset_id", "dataset_name", "sub_dataset", "file_path", "raw_filename",
        "std_name", "raw_label", "fault_code", "coarse_fault_code", "fault_label",
        "component", "split", "test_eligible", "is_variable_speed", "n_channels",
        "fs_raw", "fs_target", "speed_rpm", "load_nm", "severity_code",
        "condition_str", "condition_desc", "rep", "file_format",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    df = inspect_n_channels(df[required])
    return df[required]

def peek_mat(path):
    """Return array key names inside a .mat or .MAT file."""
    try:
        import scipy.io as sio
        m = sio.loadmat(str(path))
        return [k for k in m.keys() if not k.startswith("_")]
    except Exception as e:
        return [f"ERROR: {e}"]

def peek_csv(path, n=3):
    try:
        df = pd.read_csv(path, nrows=n)
        return list(df.columns)
    except Exception as e:
        return [f"ERROR: {e}"]

def peek_txt(path, n=3):
    try:
        with open(path) as f:
            lines = [f.readline().strip() for _ in range(n)]
        return lines
    except Exception as e:
        return [f"ERROR: {e}"]

# ─── Per-dataset parsers ──────────────────────────────────────────────────────

def parse_d1(raw_dir):
    """
    D1 — Bearing Time-Varying Speed (Huang & Baddour 2018)
    File pattern: {fault_prefix}-{condition}-{rep}.mat
      H=Normal, B=Ball, C=Combined, I=Inner_Race, O=Outer_Race
      Conditions: A, B, C, D
      Reps: 1, 2, 3
    """
    rows = []
    dm = DS_CLASS_MAP["dataset_1"]
    for f in sorted(raw_dir.rglob("*.mat")):
        name = f.stem           # e.g. H-A-1
        parts = name.split("-")
        if len(parts) != 3:
            continue
        prefix, cond, rep = parts[0].upper(), parts[1].upper(), parts[2]
        fault_code = dm["fault_prefix_map"].get(prefix)
        if fault_code is None:
            continue
        cond_info = dm["condition_map"].get(cond, {})
        rows.append({
            "dataset_id":   1,
            "sub_dataset":  "bearing",
            "file_path":    str(f.relative_to(PROJECT_ROOT)),
            "raw_filename": f.name,
            "raw_label":    prefix,
            "fault_code":   fault_code,
            "severity_code": "NA",
            "condition_str":f"cond_{cond}",
            "condition_desc": cond_info.get("description", ""),
            "speed_rpm":    None,
            "load_nm":      None,
            "rep":          int(rep),
            "file_format":  "mat",
            "fs_raw":       200000,
            "n_channels":   1,
        })
    return rows

def parse_d2(raw_dir):
    """
    D2 — Gearbox Benchmark (Liu et al. / Nguyen & Diep)
    Structure: {fault_folder}/{sensor_id}/{code}{sensor}_{speed_hz}.MAT
    Folder names: healthy, broken, missing_tooth, root_crack, wear
    """
    rows = []
    dm = DS_CLASS_MAP["dataset_2"]
    folder_map = dm["folder_map"]

    for f in sorted(raw_dir.rglob("*.MAT")):
        # Folder structure: fault_folder / sensor_id / file.MAT
        parts = f.parts
        # find the fault folder
        fault_folder = None
        for p in parts:
            if p.lower() in folder_map:
                fault_folder = p.lower()
                break
        if fault_folder is None:
            continue

        # sensor = parent folder name (1 or 2)
        sensor = f.parent.name

        # filename: e.g. N1_20.MAT → code=N, sensor=1, speed=20
        stem = f.stem  # N1_20
        m = re.match(r'^([A-Za-z]+)(\d+)_(\d+)$', stem)
        if not m:
            continue
        code_letter, sensor_in_name, speed_hz = m.group(1), m.group(2), int(m.group(3))
        speed_rpm = speed_hz * 60  # Hz × 60

        fault_code = folder_map[fault_folder]
        rows.append({
            "dataset_id":   2,
            "sub_dataset":  "gearbox",
            "file_path":    str(f.relative_to(PROJECT_ROOT)),
            "raw_filename": f.name,
            "raw_label":    fault_folder,
            "fault_code":   fault_code,
            "severity_code": "NA",
            "condition_str":f"sensor{sensor}_spd{speed_hz}hz",
            "condition_desc": f"Sensor {sensor}, {speed_hz} Hz shaft speed",
            "speed_rpm":    speed_rpm,
            "load_nm":      None,
            "rep":          int(sensor),
            "file_format":  "mat",
            "fs_raw":       48000,
            "n_channels":   1,
        })
    return rows

def parse_d3(raw_dir):
    """
    D3 — Independent CWRU bearing and gearbox benchmarks
    Three sub-datasets:
      dataset/          → CWRU bearing .mat files
      gearbox/bearingset/ → gearbox bearing .csv
      gearbox/gearset/    → gearbox gear .csv
    """
    rows = []
    dm_cwru = DS_CLASS_MAP["dataset_3"]["sub_datasets"]["bearing_cwru"]
    dm_brg  = DS_CLASS_MAP["dataset_3"]["sub_datasets"]["gearbox_bearing"]
    dm_gear = DS_CLASS_MAP["dataset_3"]["sub_datasets"]["gearbox_gearset"]

    # ── CWRU bearing .mat ──────────────────────────────────────────────────────
    cwru_dir = raw_dir / "dataset"
    if cwru_dir.exists():
        for f in sorted(cwru_dir.rglob("*.mat")):
            stem = f.stem  # e.g. B007_0, IR007_0, OR007@6_0, normal_0
            stem_clean = stem.replace("@6", "")  # normalize OR007@6 → OR007

            # normal_{load}
            m = re.match(r'^normal_(\d)$', stem_clean, re.IGNORECASE)
            if m:
                load_hp = int(m.group(1))
                fault_code = "NOR"
                size_in, position = None, None
            else:
                # {prefix}{size}_{load}
                m2 = re.match(r'^(B|IR|OR)(\d{3})_(\d)$', stem_clean, re.IGNORECASE)
                if not m2:
                    continue
                prefix = m2.group(1).upper()
                size_in = int(m2.group(2))  # e.g. 007 → 7 thou
                load_hp = int(m2.group(3))
                fault_code = dm_cwru["prefix_map"].get(prefix)
                if fault_code is None:
                    continue

            rows.append({
                "dataset_id":   31,
                "sub_dataset":  "cwru_bearing",
                "file_path":    str(f.relative_to(PROJECT_ROOT)),
                "raw_filename": f.name,
                "raw_label":    prefix if fault_code != "NOR" else "normal",
                "fault_code":   fault_code,
                "severity_code": f"{size_in}thou" if size_in else "NA",
                "condition_str":f"load{load_hp}hp",
                "condition_desc": f"{load_hp} HP load",
                "speed_rpm":    1797 - load_hp * 7,  # CWRU nominal speeds
                "load_nm":      None,
                "rep":          load_hp,
                "file_format":  "mat",
                "fs_raw":       12000,
                "n_channels":   1,
            })

    # ── Gearbox bearing .csv ───────────────────────────────────────────────────
    brg_dir = raw_dir / "gearbox" / "bearingset"
    if brg_dir.exists():
        for f in sorted(brg_dir.glob("*.csv")):
            stem = f.stem  # e.g. ball_20_0, inner_30_2, health_20_0
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            prefix  = parts[0].lower()
            speed   = int(parts[1]) if parts[1].isdigit() else None
            cond    = parts[2] if len(parts) > 2 else "0"
            fault_code = dm_brg["prefix_map"].get(prefix)
            if fault_code is None:
                continue
            rows.append({
                "dataset_id":   39,
                "sub_dataset":  "gearbox_bearing",
                "file_path":    str(f.relative_to(PROJECT_ROOT)),
                "raw_filename": f.name,
                "raw_label":    prefix,
                "fault_code":   fault_code,
                "severity_code": "NA",
                "condition_str":f"spd{speed}_cond{cond}",
                "condition_desc": f"Speed {speed} units, condition {cond}",
                "speed_rpm":    speed,
                "load_nm":      None,
                "rep":          int(cond) if str(cond).isdigit() else 0,
                "file_format":  "csv",
                "fs_raw":       None,
                "n_channels":   9,
            })

    # ── Gearbox gearset .csv ───────────────────────────────────────────────────
    gear_dir = raw_dir / "gearbox" / "gearset"
    if gear_dir.exists():
        for f in sorted(gear_dir.glob("*.csv")):
            stem = f.stem  # e.g. Chipped_20_0, Health_20_0
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            prefix  = parts[0]
            speed   = int(parts[1]) if parts[1].isdigit() else None
            cond    = parts[2] if len(parts) > 2 else "0"
            fault_code = dm_gear["prefix_map"].get(prefix)
            if fault_code is None:
                continue
            rows.append({
                "dataset_id":   39,
                "sub_dataset":  "gearbox_gearset",
                "file_path":    str(f.relative_to(PROJECT_ROOT)),
                "raw_filename": f.name,
                "raw_label":    prefix,
                "fault_code":   fault_code,
                "severity_code": "NA",
                "condition_str":f"spd{speed}_cond{cond}",
                "condition_desc": f"Speed {speed} units, condition {cond}",
                "speed_rpm":    speed,
                "load_nm":      None,
                "rep":          int(cond) if str(cond).isdigit() else 0,
                "file_format":  "csv",
                "fs_raw":       None,
                "n_channels":   9,
            })

    return rows

def parse_d4(raw_dir):
    """
    D4 — Multi-mode Gearbox Faults (Chen et al. 2024)
    Pattern: {fault}_{severity}_{mode}_{load}-{rpm}.csv
      OR:    {fault}_{mode}_{load}-{rpm}.csv  (no severity for health/miss_teeth)
    Severity: H, M, L
    Mode: speed_circulation, torque_circulation
    Load: 10Nm, 20Nm
    RPM: 1000, 2000, 3000
    """
    rows = []
    dm = DS_CLASS_MAP["dataset_4"]

    # Ordered by length (longest first) to avoid greedy prefix matching
    FAULT_PREFIXES = sorted(dm["prefix_map"].keys(), key=len, reverse=True)

    for f in sorted(raw_dir.glob("*.csv")):
        stem = f.stem.lower()

        # Match fault prefix
        fault_key = None
        for pf in FAULT_PREFIXES:
            if stem.startswith(pf):
                fault_key = pf
                break
        if fault_key is None:
            continue

        remainder = stem[len(fault_key):].lstrip("_")
        fault_code = dm["prefix_map"][fault_key]

        # Try to extract severity (H/M/L)
        severity = "NA"
        if remainder and remainder[0].upper() in ("H", "M", "L"):
            severity_letter = remainder[0].upper()
            severity = LM["severity_codes"].get(severity_letter, "NA")
            remainder = remainder[1:].lstrip("_")

        # Extract mode (speed_circulation or torque_circulation)
        mode = "unknown"
        if "speed_circulation" in remainder:
            mode = "speed"
            remainder = remainder.replace("speed_circulation_", "").replace("speed_circulation", "")
        elif "torque_circulation" in remainder:
            mode = "torque"
            remainder = remainder.replace("torque_circulation_", "").replace("torque_circulation", "")

        # Extract load (Nm) and rpm
        load_nm, speed_rpm = None, None
        # Pattern: 10nm-1000rpm or 1000rpm_10nm
        m1 = re.search(r'(\d+)nm[_-](\d+)rpm', remainder)
        m2 = re.search(r'(\d+)rpm[_-](\d+)nm', remainder)
        if m1:
            load_nm   = int(m1.group(1))
            speed_rpm = int(m1.group(2))
        elif m2:
            speed_rpm = int(m2.group(1))
            load_nm   = int(m2.group(2))

        rows.append({
            "dataset_id":   4,
            "sub_dataset":  "gearbox",
            "file_path":    str(f.relative_to(PROJECT_ROOT)),
            "raw_filename": f.name,
            "raw_label":    fault_key,
            "fault_code":   fault_code,
            "severity_code": severity,
            "condition_str":f"mode{mode}_load{load_nm}Nm_{speed_rpm}rpm",
            "condition_desc": f"{mode} mode, {load_nm}Nm, {speed_rpm}rpm",
            "speed_rpm":    speed_rpm,
            "load_nm":      load_nm,
            "rep":          0,
            "file_format":  "csv",
            "fs_raw":       10000,
            "n_channels":   1,
        })
    return rows

def parse_d5(raw_dir):
    """
    D5 — Mixed Bearing + Gearbox (Hou et al. 2025)
    Three subdirectories:
      Bearing Dataset/       → {code}_{rpm}.csv
      Parallel Gearbox Dataset/ → {fault name}_{rpm}.csv
      Mixed Fault Dataset/   → {bearing_fault}+{gear_fault}_{rpm}.csv
    Speed tokens: 1200, 1800, 2400, 200-2400-200
    """
    rows = []
    dm = DS_CLASS_MAP["dataset_5"]

    def parse_speed(token):
        """Convert speed token to numeric or string."""
        if "-" in token:
            return None, token   # variable speed e.g. 200-2400-200
        try:
            return int(token), f"{token}rpm"
        except ValueError:
            return None, token

    # ── Bearing Dataset ──────────────────────────────────────────────────────
    brg_dir = raw_dir / "Bearing Dataset"
    if brg_dir.exists():
        # Sort fault keys by length (longest first) to handle "IF+OF" before "IF"
        brg_keys = sorted(dm["bearing_map"].keys(), key=len, reverse=True)
        for f in sorted(brg_dir.glob("*.csv")):
            stem = f.stem  # e.g. IF+OF_1200, Normal_1200
            fault_key = None
            for k in brg_keys:
                if stem.startswith(k):
                    fault_key = k
                    break
            if fault_key is None:
                continue
            speed_token = stem[len(fault_key):].lstrip("_")
            speed_rpm, speed_str = parse_speed(speed_token)
            fault_code = dm["bearing_map"][fault_key]
            rows.append({
                "dataset_id":   5,
                "sub_dataset":  "bearing",
                "file_path":    str(f.relative_to(PROJECT_ROOT)),
                "raw_filename": f.name,
                "raw_label":    fault_key,
                "fault_code":   fault_code,
                "severity_code": "NA",
                "condition_str":speed_str,
                "condition_desc": f"{speed_str} shaft speed",
                "speed_rpm":    speed_rpm,
                "load_nm":      None,
                "rep":          0,
                "file_format":  "csv",
                "fs_raw":       10000,
                "n_channels":   3,
            })

    # ── Parallel Gearbox Dataset ──────────────────────────────────────────────
    gbx_dir = raw_dir / "Parallel Gearbox Dataset"
    if gbx_dir.exists():
        gbx_keys = sorted(dm["gearbox_map"].keys(), key=len, reverse=True)
        for f in sorted(gbx_dir.glob("*.csv")):
            # Raw filename tokens are preserved here because they come directly
            # from the source dataset naming convention.
            stem = f.stem  # e.g. normal_1200, gear surface wear_1200
            fault_key = None
            for k in gbx_keys:
                if stem.startswith(k):
                    fault_key = k
                    break
            if fault_key is None:
                continue
            speed_token = stem[len(fault_key):].lstrip("_")
            speed_rpm, speed_str = parse_speed(speed_token)
            fault_code = dm["gearbox_map"][fault_key]
            rows.append({
                "dataset_id":   5,
                "sub_dataset":  "gearbox",
                "file_path":    str(f.relative_to(PROJECT_ROOT)),
                "raw_filename": f.name,
                "raw_label":    fault_key,
                "fault_code":   fault_code,
                "severity_code": "NA",
                "condition_str":speed_str,
                "condition_desc": f"{speed_str} shaft speed",
                "speed_rpm":    speed_rpm,
                "load_nm":      None,
                "rep":          0,
                "file_format":  "csv",
                "fs_raw":       10000,
                "n_channels":   3,
            })

    # ── Mixed Fault Dataset ───────────────────────────────────────────────────
    mix_dir = raw_dir / "Mixed Fault Dataset"
    if mix_dir.exists():
        mix_keys = sorted(dm["mixed_map"].keys(), key=len, reverse=True)
        for f in sorted(mix_dir.glob("*.csv")):
            stem = f.stem  # e.g. IF+eccentric gear_1200
            fault_key = None
            for k in mix_keys:
                if stem.startswith(k):
                    fault_key = k
                    break
            if fault_key is None:
                continue
            speed_token = stem[len(fault_key):].lstrip("_")
            speed_rpm, speed_str = parse_speed(speed_token)
            fault_code = dm["mixed_map"][fault_key]
            rows.append({
                "dataset_id":   5,
                "sub_dataset":  "mixed",
                "file_path":    str(f.relative_to(PROJECT_ROOT)),
                "raw_filename": f.name,
                "raw_label":    fault_key,
                "fault_code":   fault_code,
                "severity_code": "NA",
                "condition_str":speed_str,
                "condition_desc": f"{speed_str} shaft speed",
                "speed_rpm":    speed_rpm,
                "load_nm":      None,
                "rep":          0,
                "file_format":  "csv",
                "fs_raw":       10000,
                "n_channels":   3,
            })

    return rows

def parse_d6(raw_dir):
    """
    D6 — Two-stage Gearbox Crack (Pandya et al. 2011)
    Pattern: {state}30hz{load_pct}.txt
      h=Healthy, b=BrokenTooth
      30hz = fixed shaft speed at 30 Hz = 1800 RPM
      load_pct: 0, 10, 20, ..., 90
    """
    rows = []
    dm = DS_CLASS_MAP["dataset_6"]

    for f in sorted(raw_dir.rglob("*.txt")):
        name = f.name.lower()
        # pattern: {h|b}30hz{load}.txt
        m = re.match(r'^([hb])30hz(\d+)\.txt$', name)
        if not m:
            continue
        state, load_pct = m.group(1), int(m.group(2))
        fault_code = dm["state_map"].get(state)
        if fault_code is None:
            continue
        rows.append({
            "dataset_id":   6,
            "sub_dataset":  "gearbox_crack",
            "file_path":    str(f.relative_to(PROJECT_ROOT)),
            "raw_filename": f.name,
            "raw_label":    state,
            "fault_code":   fault_code,
            "severity_code": "NA",
            "condition_str":f"1800rpm_load{load_pct}pct",
            "condition_desc": f"1800 RPM (30 Hz), {load_pct}% load",
            "speed_rpm":    1800,
            "load_nm":      None,
            "rep":          0,
            "file_format":  "txt",
            "fs_raw":       10000,
            "n_channels":   1,
        })
    return rows

def parse_d7(raw_dir):
    """
    D7 — Gearbox Fault Benchmark (Nguyen & Diep 2025)
    Pattern: {state}{rep}.mat
      H=Healthy, D=Damaged
      Reps: 1–10
    """
    rows = []
    dm = DS_CLASS_MAP["dataset_7"]
    for f in sorted(raw_dir.rglob("*.mat")):
        name = f.stem  # e.g. H1, D10
        m = re.match(r'^([HD])(\d+)$', name, re.IGNORECASE)
        if not m:
            continue
        state, rep = m.group(1).upper(), int(m.group(2))
        fault_code = dm["state_map"].get(state)
        if fault_code is None:
            continue
        rows.append({
            "dataset_id":   7,
            "sub_dataset":  "gearbox",
            "file_path":    str(f.relative_to(PROJECT_ROOT)),
            "raw_filename": f.name,
            "raw_label":    state,
            "fault_code":   fault_code,
            "severity_code": "NA",
            "condition_str":f"rep{rep:02d}",
            "condition_desc": f"Recording {rep} of 10",
            "speed_rpm":    None,
            "load_nm":      None,
            "rep":          rep,
            "file_format":  "mat",
            "fs_raw":       10000,
            "n_channels":   8,
        })
    return rows

def parse_d8(raw_dir):
    """
    D8 — HUST gearbox TXT dataset
    Patterns:
      steady:        {prefix}_{speed_hz}_{rep}.txt        e.g. H_20_0.txt
      variable-speed:{prefix}_L{load}_VS_0_40_{rep}.txt  e.g. B_L3_VS_0_40_0.txt

    Prefix mapping is inferred from the local raw filenames:
      H -> NOR
      B -> GBX_BT
      M -> GBX_MT
    """
    rows = []
    dm = DS_CLASS_MAP["dataset_8"]
    for f in sorted(raw_dir.glob("*.txt")):
        stem = f.stem

        m_vs = re.match(r'^([HBM])_L(\d+)_VS_(\d+)_(\d+)_(\d+)$', stem, re.IGNORECASE)
        if m_vs:
            prefix, load_level, start_hz, end_hz, rep = m_vs.groups()
            fault_code = dm["prefix_map"].get(prefix.upper())
            if fault_code is None:
                continue
            rows.append({
                "dataset_id":   8,
                "sub_dataset":  "gearbox",
                "file_path":    str(f.relative_to(PROJECT_ROOT)),
                "raw_filename": f.name,
                "raw_label":    prefix.upper(),
                "fault_code":   fault_code,
                "severity_code": "NA",
                "condition_str": f"L{load_level}_VS_{start_hz}_{end_hz}",
                "condition_desc": f"Load level L{load_level}, variable speed {start_hz}-{end_hz} Hz",
                "speed_rpm":    None,
                "load_nm":      None,
                "rep":          int(rep),
                "file_format":  "txt",
                "fs_raw":       25600,
                "n_channels":   4,
            })
            continue

        m = re.match(r'^([HBM])_(\d+)_(\d+)$', stem, re.IGNORECASE)
        if not m:
            continue
        prefix, speed_hz, rep = m.groups()
        fault_code = dm["prefix_map"].get(prefix.upper())
        if fault_code is None:
            continue
        rows.append({
            "dataset_id":   8,
            "sub_dataset":  "gearbox",
            "file_path":    str(f.relative_to(PROJECT_ROOT)),
            "raw_filename": f.name,
            "raw_label":    prefix.upper(),
            "fault_code":   fault_code,
            "severity_code": "NA",
            "condition_str": f"spd{speed_hz}hz",
            "condition_desc": f"Steady speed {speed_hz} Hz",
            "speed_rpm":    int(speed_hz) * 60,
            "load_nm":      None,
            "rep":          int(rep),
            "file_format":  "txt",
            "fs_raw":       25600,
            "n_channels":   4,
        })
    return rows

# ─── Peek mode ────────────────────────────────────────────────────────────────
def peek_all():
    """Load the first file of each dataset and print array keys / column names."""
    for i in ACTIVE_RAW_DATASETS:
        parser = PARSERS[i]
        raw_dir = DATA_RAW / f"dataset_{i}"
        print(f"\n{'='*60}")
        print(f"  D{i} — {raw_dir}")
        rows = parser(raw_dir)
        if not rows:
            print("  ⚠️  No files parsed (check directory)")
            continue
        first = rows[0]
        fp = PROJECT_ROOT / first["file_path"]
        print(f"  First file : {first['raw_filename']}")
        print(f"  Fault code : {first['fault_code']}  →  {fault_label(first['fault_code'])}")
        print(f"  Condition  : {first['condition_desc']}")
        ext = fp.suffix.lower()
        if ext in (".mat",):
            keys = peek_mat(fp)
            print(f"  MAT keys   : {keys}")
        elif ext == ".csv":
            cols = peek_csv(fp)
            print(f"  CSV cols   : {cols[:8]}{'...' if len(cols)>8 else ''}")
        elif ext == ".txt":
            lines = peek_txt(fp)
            print(f"  TXT lines  : {lines}")

# ─── Main ─────────────────────────────────────────────────────────────────────
PARSERS = {1: parse_d1, 2: parse_d2, 3: parse_d3,
           4: parse_d4, 5: parse_d5, 6: parse_d6, 7: parse_d7, 8: parse_d8}

def main():
    parser = argparse.ArgumentParser(description="Build MSPF-Net unified data catalog")
    parser.add_argument("--dataset", type=int, help="Process single dataset only")
    parser.add_argument("--peek", action="store_true", help="Peek into first file of each dataset")
    args = parser.parse_args()

    if args.peek:
        peek_all()
        return

    ds_ids = [args.dataset] if args.dataset else ACTIVE_RAW_DATASETS
    all_rows = []

    print(f"\n\033[1mMSPF-Net — Build Data Catalog\033[0m")
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for ds_id in ds_ids:
        raw_dir = DATA_RAW / f"dataset_{ds_id}"
        if not raw_dir.exists():
            print(f"  D{ds_id} : directory not found — skip")
            continue
        rows = PARSERS[ds_id](raw_dir)
        print(f"  {display_dataset_id(ds_id)} : {len(rows):>4} files parsed", end="")
        if rows:
            codes = sorted(set(r["fault_code"] for r in rows))
            print(f"  |  classes: {', '.join(codes)}")
        else:
            print("  ⚠️  0 files parsed — check naming or directory")
        all_rows.extend(rows)

    if not all_rows:
        print("\n\033[91mNo files parsed. Ensure datasets are in data/raw/dataset_N/\033[0m\n")
        sys.exit(1)

    df = finalize_catalog(pd.DataFrame(all_rows))

    # Save catalog
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG_PATH, index=False)
    print(f"\n  Catalog saved  → {CATALOG_PATH.relative_to(PROJECT_ROOT)}  ({len(df)} rows)")

    # Save per-dataset class distribution summary
    summary = (
        df.groupby(["dataset_id", "dataset_name", "sub_dataset", "fault_code", "coarse_fault_code", "fault_label"])
        .size()
        .reset_index(name="n_files")
        .sort_values(["dataset_id", "fault_code"])
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"  Summary saved  → {SUMMARY_PATH.relative_to(PROJECT_ROOT)}\n")

    # Print summary table
    print(f"\033[1m{'DS':<4} {'Sub-dataset':<20} {'Fault Code':<12} {'Coarse':<18} {'Fault Label':<32} {'Files':>6}\033[0m")
    print("-" * 112)
    for _, row in summary.iterrows():
        ds_label = display_dataset_id(int(row.dataset_id))
        print(
            f"{ds_label:<4} {row.sub_dataset:<20} {row.fault_code:<12} "
            f"{row.coarse_fault_code:<18} {row.fault_label:<32} {int(row.n_files):>6}"
        )
    print()

if __name__ == "__main__":
    main()
