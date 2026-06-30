import os, sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from mspf_net.constants import ACTIVE_THESIS_DATASETS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
SCHEMA_PATH  = PROJECT_ROOT / "data" / "metadata" / "dataset_schema.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "tables"

SUPPORTED_EXT = {".mat", ".npy", ".npz", ".csv", ".txt", ".h5", ".hdf5"}

DOWNLOAD_LINKS = {
    1: ("Mendeley Data",  "https://data.mendeley.com/datasets/v43hmbwxpm/2"),
    2: ("GitHub",         "https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset.git"),
    3: ("IEEE DataPort",  "https://github.com/cathysiyu/Mechanical-datasets.git"),
    4: ("Data Brief",     "https://doi.org/10.1016/j.dib.2024.110453"),
    5: ("Data Brief",     "https://doi.org/10.1016/j.dib.2025.112187"),
    6: ("ResearchGate",   "https://www.researchgate.net/publication/287092196"),
    7: ("PLoS ONE",       "https://doi.org/10.1371/journal.pone.0324905"),
    8: ("Reliability Engineering & System Safety", "https://doi.org/10.1016/j.ress.2024.109964"),
}

# ─── Filename-based label patterns per dataset ────────────────────────────────
# All checked-in datasets encode labels in filenames or folder names.
LABEL_PATTERNS = {
    1: {
        "source":  "filename prefix (H/B/C/I/O)",
        "classes": 5,
        "detect":  lambda f: f.stem[0].upper() in ("H","B","C","I","O"),
    },
    2: {
        "source":  "parent folder (healthy/broken/missing_tooth/root_crack/wear)",
        "classes": 5,
        "detect":  lambda f: any(
            p.lower() in ("healthy","broken","missing_tooth","root_crack","wear")
            for p in f.parts
        ),
    },
    3: {
        "source":  "filename prefix (CWRU: normal/B/IR/OR; gearbox: health/ball/inner/outer/comb/Chipped/Miss/Root/Surface)",
        "classes": 9,
        "detect":  lambda f: True,
    },
    31: {
        "source":  "dataset_3/dataset filename prefix (normal/B/IR/OR)",
        "classes": 4,
        "expected_codes": ["NOR", "BRG_BF", "BRG_IR", "BRG_OR"],
        "detect":  lambda f: f.suffix.lower() == ".mat" and "dataset_3/dataset" in str(f).replace("\\", "/"),
    },
    39: {
        "source":  "dataset_3/gearbox filename prefix (health/ball/inner/outer/comb/Chipped/Miss/Root/Surface)",
        "classes": 9,
        "expected_codes": ["NOR", "BRG_BF", "BRG_CF", "BRG_IR", "BRG_OR", "GBX_BT", "GBX_MT", "GBX_RC", "GBX_WR"],
        "detect":  lambda f: (
            f.suffix.lower() == ".csv"
            and "dataset_3/gearbox" in str(f).replace("\\", "/")
        ),
    },
    4: {
        "source":  "filename prefix (health/gear_pitting/gear_wear/miss_teeth/teeth_break/teeth_crack/teeth_break_and_bearing_*)",
        "classes": 8,
        "detect":  lambda f: True,
    },
    5: {
        "source":  "filename prefix in Bearing/Gearbox/Mixed sub-folders",
        "classes": 15,
        "detect":  lambda f: f.suffix.lower() == ".csv",
    },
    6: {
        "source":  "filename prefix (h=healthy / b=broken)",
        "classes": 2,
        "detect":  lambda f: f.stem[0].lower() in ("h","b"),
    },
    7: {
        "source":  "parent folder (Healthy / Damaged)",
        "classes": 2,
        "detect":  lambda f: any(
            p in ("Healthy","Damaged") for p in f.parts
        ),
    },
    8: {
        "source":  "filename prefix (H/B/M) with steady and *_VS_* variants",
        "classes": 3,
        "detect":  lambda f: f.suffix.lower() == ".txt" and f.stem[:1].upper() in ("H", "B", "M"),
    },
}


# ─── Infer label from file path (used for class distribution table) ──────────

def _infer_label(ds_id: int, f: Path) -> str | None:
    name  = f.stem.lower()
    parts = list(f.parts)

    if ds_id == 1:
        return {"h":"NOR","b":"BRG_BF","c":"BRG_CF","i":"BRG_IR","o":"BRG_OR"}.get(name[0])

    if ds_id == 2:
        for p in parts:
            c = {"healthy":"NOR","broken":"GBX_BT","missing_tooth":"GBX_MT",
                 "root_crack":"GBX_RC","wear":"GBX_WR"}.get(p.lower())
            if c: return c

    if ds_id in (3, 31, 39):
        if name.startswith("normal"):      return "NOR"
        if name[:2] == "ir":               return "BRG_IR"
        if name[:2] == "or":               return "BRG_OR"
        if name[0] == "b" and len(name)>1 and name[1].isdigit(): return "BRG_BF"
        if name.startswith("inner"):       return "BRG_IR"
        if name.startswith("outer"):       return "BRG_OR"
        if name.startswith("ball"):        return "BRG_BF"
        if name.startswith("comb"):        return "BRG_CF"
        if name.startswith("health"):      return "NOR"
        if name[0] == "c":                 return "GBX_BT"
        if name[0] == "m":                 return "GBX_MT"
        if name[0] == "r":                 return "GBX_RC"
        if name[0] == "s":                 return "GBX_WR"

    if ds_id == 4:
        for pf, code in [
            ("teeth_break_and_bearing_inner","MIX_TI"),
            ("teeth_break_and_bearing_outer","MIX_TO"),
            ("gear_pitting","GBX_PT"), ("gear_wear","GBX_WR"),
            ("miss_teeth","GBX_MT"),   ("teeth_break","GBX_BT"),
            ("teeth_crack","GBX_CK"),  ("health","NOR"),
        ]:
            if name.startswith(pf): return code

    if ds_id == 5:
        # These prefixes mirror the raw source filenames. The canonical labels
        # attached to the returned fault codes come from label_mapping.json.
        for pf, code in [
            ("if+of","BRG_IO"),("if+eccentric gear","MIX_II"),
            ("if+gear surface wear","MIX_IS"),("if+gear tooth break","MIX_IB"),
            ("of+eccentric gear","MIX_OI"),("of+gear surface wear","MIX_OS"),
            ("of+gear tooth break","MIX_OB"),
            ("normal","NOR"),
            ("bf","BRG_BF"),("cf","BRG_CF"),("if","BRG_IR"),("of","BRG_OR"),
            ("eccentric gear","GBX_EG"),("gear surface wear","GBX_WR"),
            ("gear tooth break","GBX_BT"),
        ]:
            if name.startswith(pf): return code

    if ds_id == 6:
        return {"h":"NOR","b":"GBX_BT"}.get(name[0])

    if ds_id == 7:
        if any("Healthy" in p for p in parts): return "NOR"
        if any("Damaged" in p for p in parts): return "GBX_CF"

    if ds_id == 8:
        return {"h":"NOR","b":"GBX_BT","m":"GBX_MT"}.get(name[0])

    return None


# ─── File loading: scipy + h5py fallback ─────────────────────────────────────

def try_load(path: Path):
    """
    Load a data file. Returns (arrays_dict, error_str).

    .mat files: tries scipy first; if it raises (MATLAB v7.3 HDF5 format),
    automatically falls back to h5py.
    """
    ext = path.suffix.lower()
    try:
        if ext == ".mat":
            return _load_mat(path)
        if ext == ".npy":
            return {"data": np.load(str(path), allow_pickle=True)}, None
        if ext == ".npz":
            return dict(np.load(str(path), allow_pickle=True)), None
        if ext in (".csv", ".txt"):
            return _load_csv(path)
        if ext in (".h5", ".hdf5"):
            return _load_hdf5(path)
        return None, f"unsupported extension: {ext}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _load_mat(path: Path):
    """
    Try to load a .MAT file using multiple strategies in order:
      1. scipy.io.loadmat      → MATLAB v4 / v5 / v6
      2. h5py                  → MATLAB v7.3 (HDF5)
      3. text / whitespace     → some .MAT files are plain-text columns
      4. raw binary fallback   → try numpy fromfile (last resort)

    On failure, includes hex magic bytes in the error for easy diagnosis.
    """
    # ── Read magic bytes once (used for diagnostics and routing) ─────────────
    try:
        with open(path, "rb") as fh:
            magic = fh.read(16)
    except Exception as e:
        return None, f"Cannot open file: {e}"

    magic_hex = magic.hex(" ")
    magic_str = "".join(chr(b) if 32 <= b < 127 else "." for b in magic)

    # ── Strategy 1: scipy (MATLAB v4/v5/v6) ──────────────────────────────────
    scipy_err = None
    try:
        import scipy.io as sio
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m = sio.loadmat(str(path))
        return {k: v for k, v in m.items() if not k.startswith("_")}, None
    except Exception as e:
        scipy_err = e

    # ── Strategy 2: h5py (MATLAB v7.3 = HDF5) ────────────────────────────────
    # HDF5 magic: first 8 bytes = 89 48 44 46 0d 0a 1a 0a
    HDF5_MAGIC = bytes([0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A])
    if magic[:8] == HDF5_MAGIC:
        try:
            import h5py
        except ImportError:
            return None, (
                "MATLAB v7.3 HDF5 file — scipy cannot read it.\n"
                "    Fix: pip install h5py   (in the mspf_net conda env)"
            )
        try:
            arrays = {}
            with h5py.File(str(path), "r") as hf:
                def _ex(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try: arrays[name] = np.array(obj)
                        except Exception: pass
                hf.visititems(_ex)
            return arrays, None
        except Exception as h5_err:
            return None, f"h5py error: {h5_err}"

    # ── Strategy 3: plain-text columns (some .MAT files are ASCII) ───────────
    # Detect: magic bytes are all printable ASCII (digits, spaces, minus, dot)
    is_text = all(b in range(32, 128) or b in (9, 10, 13) for b in magic)
    if is_text:
        try:
            arr = np.loadtxt(str(path))
            if arr.size >= 100:
                return {"data": arr}, None
        except Exception:
            pass
        # also try pandas CSV with common separators
        for sep in (",", "\t", " ", ";"):
            try:
                df = pd.read_csv(path, sep=sep, header=None,
                                 encoding="utf-8", on_bad_lines="skip")
                numeric_df = df.apply(pd.to_numeric, errors="coerce")
                numeric_df = numeric_df.dropna(how="all", axis=1).dropna(how="all", axis=0)
                if numeric_df.size >= 100:
                    return {"data": numeric_df.values.astype(np.float64)}, None
            except Exception:
                continue

    # ── Strategy 4: raw binary (try float64 and float32) ─────────────────────
    file_size = path.stat().st_size
    for dtype in (np.float64, np.float32, np.int16):
        n_elements = file_size // np.dtype(dtype).itemsize
        if n_elements >= 100:
            try:
                arr = np.fromfile(str(path), dtype=dtype)
                # Sanity check: values shouldn't be all NaN or wildly huge
                if np.isfinite(arr).mean() > 0.5:
                    return {"data": arr}, None
            except Exception:
                continue

    # ── All strategies failed — return diagnostic error ───────────────────────
    return None, (
        f"Cannot read .MAT file (all strategies failed).\n"
        f"    File size  : {path.stat().st_size:,} bytes\n"
        f"    Magic hex  : {magic_hex}\n"
        f"    Magic str  : '{magic_str}'\n"
        f"    scipy error: {scipy_err}\n"
        f"    Hint: if magic starts with 'MATLAB 5.0' → scipy should work;\n"
        f"          if magic is '89 48 44 46 ...'     → HDF5/h5py;\n"
        f"          if magic is all ASCII digits      → text file;\n"
        f"          otherwise contact dataset authors for format spec."
    )


def _load_csv(path: Path):
    """
    Load CSV/TXT with robust fallbacks:
      - Tries multiple encodings: utf-8, gbk (common in Chinese datasets), latin-1
      - Tries multiple separators: comma, tab, whitespace
      - Always coerces to numeric dtype so count_signal_samples works correctly
    """
    if path.suffix.lower() == ".txt":
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    vals = [float(x) for x in parts]
                except Exception:
                    continue
                rows.append(vals)
        if rows:
            return {"data": np.asarray(rows, dtype=np.float64)}, None

    ENCODINGS = ["utf-8", "gbk", "gb2312", "latin-1", "cp1252"]
    SEPARATORS = [",", "\t", None, ";", r"\s+"]   # None = pandas auto-detect

    last_err = None
    for enc in ENCODINGS:
        for sep in SEPARATORS:
            try:
                kwargs = dict(
                    nrows=10000,
                    encoding=enc,
                    on_bad_lines="skip",
                )
                if sep is None:
                    kwargs["sep"] = ","
                    kwargs["low_memory"] = False
                elif sep == r"\s+":
                    kwargs["sep"] = sep
                    kwargs["engine"] = "python"
                else:
                    kwargs["sep"] = sep
                    kwargs["low_memory"] = False

                df = pd.read_csv(path, **kwargs)

                if df.empty or df.shape[1] == 0:
                    continue

                # Force all columns to numeric — non-numeric (headers in data,
                # Chinese labels, units) become NaN, then we drop empty cols
                numeric_df = df.apply(pd.to_numeric, errors="coerce")
                numeric_df = (
                    numeric_df
                    .dropna(how="all", axis=1)   # drop all-NaN columns
                    .dropna(how="all", axis=0)   # drop all-NaN rows
                )

                # Keep the densest numeric block so header text in TXT exports
                # does not block verification of otherwise valid signals.
                if not numeric_df.empty:
                    dense_rows = numeric_df.notna().sum(axis=1) >= max(2, numeric_df.shape[1] // 2)
                    if dense_rows.any():
                        numeric_df = numeric_df.loc[dense_rows]

                if numeric_df.size >= 10:        # at least some numeric data
                    return {"data": numeric_df.values.astype(np.float64)}, None

            except Exception as e:
                last_err = e
                continue

    return None, f"Could not parse CSV/TXT (tried {len(ENCODINGS)} encodings × {len(SEPARATORS)} separators). Last error: {last_err}"


def _load_hdf5(path: Path):
    try:
        import h5py
        arrays = {}
        with h5py.File(str(path), "r") as hf:
            def _ex(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try: arrays[name] = np.array(obj)
                    except Exception: pass
            hf.visititems(_ex)
        return arrays, None
    except Exception as e:
        return None, str(e)


# ─── Sample counting ──────────────────────────────────────────────────────────

def count_signal_samples(arrays: dict) -> int:
    """
    Return element count of the largest numeric array in the dict.
    Handles object dtype (strings that should be numbers) by attempting coercion.
    Skips tiny arrays (< 100 elements) which are likely metadata.
    """
    best = 0
    for v in arrays.values():
        if not isinstance(v, np.ndarray):
            continue
        # Coerce object/string arrays to float (e.g. CSV loaded as strings)
        if v.dtype == object:
            try:
                v = v.astype(np.float64)
            except (ValueError, TypeError):
                try:
                    # Try element-wise coercion
                    v = pd.to_numeric(v.flatten(), errors="coerce")
                    v = v[~np.isnan(v)]
                except Exception:
                    continue
        if np.issubdtype(v.dtype, np.number):
            n = v.shape[0] if v.ndim >= 2 else v.size
            if n > best and n >= 100:
                best = n
    return best


def estimate_windows(raw: int, w: int = 2048, overlap: float = 0.5) -> int:
    step = int(w * (1 - overlap))
    return max(0, (raw - w) // step + 1)


# ─── Label coverage ───────────────────────────────────────────────────────────

def check_labels(ds_id: int, files: list) -> dict:
    pat = LABEL_PATTERNS.get(ds_id, {})
    exp_n = pat.get("classes", 0)
    detect = pat.get("detect", lambda _: True)

    counts = {}
    for f in files:
        try:
            if detect(f):
                lbl = _infer_label(ds_id, f)
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
        except Exception:
            pass

    found_n = len(counts)
    return {
        "classes_found":   found_n,
        "classes_ok":      found_n >= exp_n,
        "label_counts":    counts,
        "exp_classes":     exp_n,
    }


# ─── Per-dataset check ────────────────────────────────────────────────────────

def resolve_dataset_dir(ds_id: int) -> Path:
    if ds_id == 31:
        return DATA_RAW / "dataset_3" / "dataset"
    if ds_id == 39:
        return DATA_RAW / "dataset_3" / "gearbox"
    return DATA_RAW / f"dataset_{ds_id}"

def find_files(ds_dir: Path) -> list:
    return sorted([
        p for p in Path(ds_dir).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXT and p.is_file()
    ])


def detect_read_method(path: Path) -> str:
    ext = path.suffix.lower()
    if ext != ".mat":
        return ext.lstrip(".")
    try:
        with open(path, "rb") as fh:
            magic = fh.read(8)
    except Exception:
        return "unknown"
    HDF5_SIG = bytes([0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A])
    if magic[:8] == HDF5_SIG:
        return "h5py (v7.3)"
    try:
        import scipy.io as sio
        sio.loadmat(str(path))
        return "scipy"
    except Exception:
        pass
    is_text = all(b in range(32, 128) or b in (9, 10, 13) for b in magic)
    return "text" if is_text else "raw binary"


def check_dataset(ds_id: int, meta: dict) -> dict:
    r = {
        "id": ds_id, "name": meta["short_name"],
        "dir_ok": False, "n_files": 0,
        "readable": False, "read_method": None,
        "raw_samples": None, "est_windows": None,
        "classes_found": None, "classes_ok": None,
        "label_counts": {}, "missing_classes": [],
        "issues": [], "status": "FAIL",
    }

    ds_dir = resolve_dataset_dir(ds_id)
    if not ds_dir.exists():
        rel = ds_dir.relative_to(PROJECT_ROOT)
        r["issues"].append(f"Directory missing: {rel}/")
        return r
    r["dir_ok"] = True

    files = find_files(ds_dir)
    r["n_files"] = len(files)
    if not files:
        r["issues"].append("No data files found")
        return r

    # readability
    arr, err = try_load(files[0])
    if err:
        r["issues"].append(f"Cannot read {files[0].name}: {err}")
        return r
    r["readable"]    = True
    r["read_method"] = detect_read_method(files[0])

    # raw sample count
    total_raw, bad = 0, []
    for f in files:
        ad, lerr = try_load(f)
        if lerr:
            bad.append(f.name)
        else:
            total_raw += count_signal_samples(ad)
    r["raw_samples"] = total_raw
    r["est_windows"] = estimate_windows(total_raw)

    if bad:
        r["issues"].append(
            f"{len(bad)}/{len(files)} files unreadable: "
            f"{', '.join(bad[:3])}{'...' if len(bad)>3 else ''}"
        )

    # filename-based label coverage
    lbl = check_labels(ds_id, files)
    r["classes_found"] = lbl["classes_found"]
    r["classes_ok"]    = lbl["classes_ok"]
    r["label_counts"]  = lbl["label_counts"]
    pat = LABEL_PATTERNS.get(ds_id, {})
    exp_n = pat.get("classes", 0)
    if lbl["classes_found"] < exp_n:
        r["missing_classes"] = [
            c for c in (LABEL_PATTERNS[ds_id].get("expected_codes") or [])
            if c not in lbl["label_counts"]
        ]
        r["issues"].append(
            f"Only {lbl['classes_found']}/{exp_n} classes detected "
            f"(labels from: {pat.get('source','filename')})"
        )

    # status
    if not r["readable"] or (bad and len(bad) == len(files)):
        r["status"] = "FAIL"
    elif bad or lbl["classes_found"] < exp_n:
        r["status"] = "WARN"
    else:
        r["status"] = "PASS"

    return r


# ─── Peek ─────────────────────────────────────────────────────────────────────

def peek_dataset(ds_id: int):
    ds_dir = resolve_dataset_dir(ds_id)
    files  = find_files(ds_dir)
    if not files:
        print(f"  No files in {ds_dir}"); return
    f = files[0]
    print(f"\n  D{ds_id} first file: {f}")
    arr, err = try_load(f)
    if err:
        print(f"  ❌ {err}"); return
    print(f"  ✅ Loaded  (method: {detect_read_method(f)})")
    print(f"  {'Key':<30} {'dtype':<12} {'shape'}")
    print(f"  {'─'*60}")
    for k, v in arr.items():
        if isinstance(v, np.ndarray):
            print(f"  {str(k):<30} {str(v.dtype):<12} {v.shape}")
        else:
            print(f"  {str(k):<30} {type(v).__name__}")
    lbl = check_labels(ds_id, files)
    print(f"\n  Label source : {LABEL_PATTERNS.get(ds_id,{}).get('source','?')}")
    print(f"  Classes      : {lbl['classes_found']}/{lbl['exp_classes']} detected")
    for code, n in sorted(lbl["label_counts"].items()):
        print(f"    {code:<14} {n} files")


# ─── Table ────────────────────────────────────────────────────────────────────

def print_table(results: list):
    C = {"G":"\033[92m","Y":"\033[93m","R":"\033[91m","B":"\033[1m","E":"\033[0m","D":"\033[2m"}
    w = [5, 24, 5, 7, 10, 16, 17, 9, 8]
    hdr = ["DS","Name","Dir","Files","Readable","Raw Samples","~Windows (w=2048)","Classes","Status"]
    sep = "+" + "+".join("-"*(c+2) for c in w) + "+"

    def row(*cells):
        return "| " + " | ".join(str(c).ljust(w[i]) for i,c in enumerate(cells)) + " |"

    title = "  Dataset Verification Summary — MSPF-Net Phase 1 (v2)  "
    total_w = sum(w) + 3*len(w) - 1
    print(sep)
    print("| " + C["B"] + title.center(total_w) + C["E"] + " |")
    print(sep)
    print(row(*hdr))
    print(sep)
    for r in results:
        sc  = {"PASS":C["G"],"WARN":C["Y"],"FAIL":C["R"]}.get(r["status"],"")
        raw = f"{r['raw_samples']:,}"  if isinstance(r["raw_samples"], int) else "—"
        win = f"~{r['est_windows']:,}" if isinstance(r["est_windows"],  int) else "—"
        cls = str(r["classes_found"])  if r["classes_found"] is not None    else "—"
        mth = (C["D"] + f" [{r['read_method']}]" + C["E"]) if r.get("read_method") else ""
        print(row(
            f"D{r['id']}", r["name"],
            "✓" if r["dir_ok"] else "✗", str(r["n_files"]),
            ("✓" + mth) if r["readable"] else "✗",
            raw, win, cls,
            sc + r["status"] + C["E"],
        ))
    print(sep)


def print_issues(results: list):
    has = [r for r in results if r["issues"]]
    if not has:
        print("\n\033[92m✅  All datasets verified successfully!\033[0m\n")
        return
    print("\n\033[93m⚠️  Issues Found:\033[0m")
    for r in has:
        print(f"\n  \033[1mD{r['id']} – {r['name']}\033[0m")
        for iss in r["issues"]:
            print(f"    \033[91m•\033[0m {iss}")


def print_class_dist(results: list):
    ds_with_labels = [r for r in results if r.get("label_counts")]
    if not ds_with_labels: return
    print(f"\n\033[1m  File count per class (from filenames):\033[0m")
    print(f"  {'DS':<5} {'Fault Code':<16} {'Files':>6}  {'Label method'}")
    print(f"  {'─'*58}")
    for r in ds_with_labels:
        src = LABEL_PATTERNS.get(r["id"],{}).get("source","filename")
        first = True
        for code, n in sorted(r["label_counts"].items()):
            print(f"  D{r['id']:<4} {code:<16} {n:>6}  {src if first else ''}")
            first = False


# ─── Entry point ──────────────────────────────────────────────────────────────

def load_schema():
    with open(SCHEMA_PATH) as f:
        s = json.load(f)
    return {k: v for k, v in s.items() if not k.startswith("_")}


def raw_location_hint(ds_id: int) -> str:
    if ds_id == 31:
        return "data/raw/dataset_3/dataset/"
    if ds_id == 39:
        return "data/raw/dataset_3/gearbox/"
    return f"data/raw/dataset_{ds_id}/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=int,
                        help="Single dataset id to verify (default: active thesis datasets only)")
    parser.add_argument("--peek",           type=int,  metavar="DS_ID",
                        help="Print array keys/shapes of first file in dataset N")
    parser.add_argument("--download-guide", action="store_true")
    parser.add_argument("--no-dist",        action="store_true")
    args = parser.parse_args()

    schema = load_schema()

    if args.download_guide:
        print(f"\n\033[1m📥  Dataset Download Guide\033[0m\n")
        for i in ACTIVE_THESIS_DATASETS:
            src, url = DOWNLOAD_LINKS.get(i, ("Unknown", ""))
            m = schema.get(f"dataset_{i}",{})
            print(f"  D{i}  {m.get('short_name',''):<24} → {raw_location_hint(i)}")
            print(f"       {src:<16}  {url}\n")
        return

    if args.peek is not None:
        peek_dataset(args.peek)
        return

    ids = [args.dataset] if args.dataset else list(ACTIVE_THESIS_DATASETS)

    print(f"\n\033[1mMSPF-Net Phase 1 — Dataset Verification\033[0m")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []
    for ds_id in ids:
        mk = f"dataset_{ds_id}"
        if mk not in schema:
            print(f"  Schema not found for dataset_{ds_id}"); continue
        print(f"  Checking D{ds_id} — {schema[mk]['short_name']} ...",
              end=" ", flush=True)
        r = check_dataset(ds_id, schema[mk])
        results.append(r)
        sc = {"PASS":"\033[92m","WARN":"\033[93m","FAIL":"\033[91m"}.get(r["status"],"")
        print(f"{sc}{r['status']}\033[0m")

    print()
    print_table(results)
    print_issues(results)
    if not args.no_dist:
        print_class_dist(results)

    n_pass = sum(1 for r in results if r["status"]=="PASS")
    n_warn = sum(1 for r in results if r["status"]=="WARN")
    n_fail = sum(1 for r in results if r["status"]=="FAIL")
    total_raw = sum(r["raw_samples"] or 0 for r in results)
    total_win = sum(r["est_windows"] or 0 for r in results)
    print(f"\n  Result  : \033[92m{n_pass} PASS\033[0m  \033[93m{n_warn} WARN\033[0m  \033[91m{n_fail} FAIL\033[0m")
    print(f"  Total   : {total_raw:,} raw samples  |  ~{total_win:,} windows (w=2048, overlap=50%)\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "dataset_verification.csv"
    pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("issues","label_counts")}
        | {"issues": "; ".join(r["issues"])}
        for r in results
    ]).to_csv(out, index=False)
    print(f"\033[2m  Report saved → {out.relative_to(PROJECT_ROOT)}\033[0m\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
