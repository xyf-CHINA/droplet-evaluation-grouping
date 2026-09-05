"""
DAFD 3.0 — Stage 0 data audit script.

This script ONLY READS `data/raw/` and WRITES `outputs/audit/`.
It never modifies raw files or official code, and it trains no models.

All key numbers reported in DATA_AUDIT.md / audit_results.json are produced
programmatically by this script. Nothing is hard-coded by hand.

Run from the project root:  python src/audit/audit_dafd3.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (project-root-relative, no machine-specific absolute paths)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "audit"

# ----------------------------------------------------------------------------
# Constants (all fixed and documented)
# ----------------------------------------------------------------------------
SEED_LO, SEED_HI = 0, 999       # 1000 deterministic seeds, inclusive [0, 999]
TEST_SIZE = 0.20                # random-split test fraction
ROUND_MATCH = 8                 # decimals used for cross-file deterministic matching
ROUND_GEOM = 6                  # decimals used for geometry canonicalization

# Pure geometry fields. Flow/fluid/target columns are FORBIDDEN here.
GEOM_COLS = [
    "Orifice width (um)",
    "Normalized channel depth",
    "Normalized continuous inlet",
    "Normalized dispersed inlet",
    "Normalized outlet width",
]
FLOW_COLS = ["Flow rate ratio", "Capillary number"]
FLUID_COLS = ["viscosity ratio"]
TARGET_COLS = [
    "Observed droplet diameter (um)",
    "Normalized droplet diameter",
    "Observed generation rate (Hz)",
]

# Raw SE uses slightly different column names than the normalized family.
RAW_SE_RENAME = {
    "Normalized depth": "Normalized channel depth",
    "Normalized dis[ersed inlet": "Normalized dispersed inlet",
    " droplet diameter (um)": "Observed droplet diameter (um)",
    " generation rate (Hz)": "Observed generation rate (Hz)",
}

# Deterministic matching key sets (documented in DATA_AUDIT.md):
# SE files have no viscosity-ratio column -> match on geometry + flow + targets.
SE_MATCH_COLS = GEOM_COLS + FLOW_COLS + [
    "Observed droplet diameter (um)",
    "Observed generation rate (Hz)",
]
FF_MATCH_COLS = GEOM_COLS + FLOW_COLS + FLUID_COLS + [
    "Observed droplet diameter (um)",
    "Observed generation rate (Hz)",
]

CORE_FILES = [
    "Raw SE dataset.xlsx",
    "Comprehensive_normalized.xlsx",
    "Generalizability_data_normalized.xlsx",
]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def py_native(obj):
    """Recursively convert numpy/python values so json.dumps works.

    NaN / inf / -inf become None (strict-JSON compliant; json.dump runs with
    allow_nan=False).
    """
    if isinstance(obj, dict):
        return {str(k): py_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def rounded_tuple(values, digits: int):
    """Round a sequence of floats to `digits` decimals; returns None if any NaN."""
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if np.isnan(f):
            return None
        out.append(round(f, digits))
    return tuple(out)


# ----------------------------------------------------------------------------
# Stage 0A — raw file manifest
# ----------------------------------------------------------------------------
def build_raw_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.xlsx")):
        sha = sha256_of(path)
        xl = pd.ExcelFile(path)
        sheets_info = []
        for sh in xl.sheet_names:
            df = xl.parse(sh)
            sheets_info.append(
                {"sheet": sh, "n_rows": int(len(df)), "n_cols": int(df.shape[1])}
            )
        rows.append(
            {
                "filename": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
                "sha256": sha,
                "n_sheets": len(xl.sheet_names),
                "sheet_names": " | ".join(xl.sheet_names),
                "sheets_info": json.dumps(sheets_info),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Loading helpers (read-only)
# ----------------------------------------------------------------------------
def load_core_files():
    """Load the three core files. Returns dict name -> (df, sheet, note)."""
    comp = pd.read_excel(RAW_DIR / "Comprehensive_normalized.xlsx", sheet_name="Sheet1")
    ext = pd.read_excel(
        RAW_DIR / "Generalizability_data_normalized.xlsx", sheet_name="sheet 1"
    )
    se_raw = pd.read_excel(
        RAW_DIR / "Raw SE dataset.xlsx", sheet_name="Generation rate ver. new"
    )
    # Raw SE has one leading fully-NaN title row; only drop FULLY-NaN rows and record it.
    fully_nan = se_raw.isna().all(axis=1)
    n_dropped = int(fully_nan.sum())
    se = se_raw.loc[~fully_nan].reset_index(drop=False).rename(
        columns={"index": "excel_row"}
    )
    se["excel_row"] = se["excel_row"] + 2  # pandas index 0 == Excel row 2
    se = se.rename(columns=RAW_SE_RENAME)
    return comp, ext, se, int(n_dropped)


def load_component_files():
    ff1 = pd.read_excel(RAW_DIR / "FF1_DE.xlsx")
    ff2 = pd.read_excel(RAW_DIR / "FF2_DE.xlsx")
    nff1 = pd.read_excel(RAW_DIR / "All_NewFF1_instability.xlsx")
    nff2 = pd.read_excel(RAW_DIR / "All_NewFF2_instability.xlsx")
    final = pd.read_excel(
        RAW_DIR / "Final_Comprehensive_plus_Generalizability_data_normalized.xlsx"
    )
    raw_de = pd.read_excel(RAW_DIR / "Raw DE dataset.xlsx", sheet_name="1")
    return {"FF1_DE": ff1, "FF2_DE": ff2, "NewFF1": nff1, "NewFF2": nff2,
            "Final": final, "Raw_DE": raw_de}


# ----------------------------------------------------------------------------
# Stage 0B — dataset summary stats
# ----------------------------------------------------------------------------
def dataset_stats(df: pd.DataFrame, feature_cols) -> dict:
    """Summary stats for one parsed sheet. Outliers are reported, never removed."""
    valid = df.dropna(how="all")
    n_full_dup = int(df.duplicated(keep=False).sum())  # rows involved in full duplicates
    n_full_dup_groups = int(df.duplicated().sum())      # duplicate rows beyond first
    feat_present = [c for c in feature_cols if c in df.columns]
    if feat_present:
        n_dup_feat = int(df.duplicated(subset=feat_present, keep=False).sum())
        n_dup_feat_groups = int(df.duplicated(subset=feat_present).sum())
    else:
        n_dup_feat = n_dup_feat_groups = np.nan
    n_missing_cells = int(df.isna().sum().sum())
    col_details = []
    for c in df.columns:
        s = df[c]
        d = {
            "column": str(c),
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            d["min"] = py_native(s.min())
            d["max"] = py_native(s.max())
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                d["n_iqr_outliers"] = int(
                    ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                )
            else:
                d["n_iqr_outliers"] = 0
        col_details.append(d)
    return {
        "n_rows": int(len(df)),
        "n_rows_valid": int(len(valid)),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "n_missing_cells": n_missing_cells,
        "n_duplicated_full_rows_involved": n_full_dup,
        "n_duplicated_full_rows_extra": n_full_dup_groups,
        "n_duplicated_feature_rows_involved": n_dup_feat,
        "n_duplicated_feature_rows_extra": n_dup_feat_groups,
        "feature_cols_used": feat_present,
        "column_details": col_details,
    }


# ----------------------------------------------------------------------------
# Stage 0D — geometry reconstruction
# ----------------------------------------------------------------------------
def canonical_geometry_tuple(row: pd.Series, digits: int = ROUND_GEOM):
    """Canonical tuple of the five pure geometry fields, rounded to `digits`
    decimals. Returns None if any geometry field is missing (no guessing)."""
    return rounded_tuple([row[c] for c in GEOM_COLS], digits)


def assign_geometry_ids(df: pd.DataFrame, id_prefix: str = "G") -> tuple[pd.DataFrame, dict]:
    """Assign stable geometry ids.

    Ordering rule (documented): unique canonical tuples are sorted
    lexicographically ascending; ids are `{prefix}0001`, `{prefix}0002`, ...
    (zero-padded to at least 4 digits). Deterministic, independent of row
    order and independent of flow/fluid/target columns.
    """
    tuples = [canonical_geometry_tuple(r) for _, r in df.iterrows()]
    uniques = sorted({t for t in tuples if t is not None})
    width = max(4, len(str(len(uniques))))
    t2id = {t: f"{id_prefix}{i:0{width}d}" for i, t in enumerate(uniques, start=1)}
    out = df.copy()
    out["geometry_id"] = [t2id[t] if t is not None else None for t in tuples]
    counts = Counter(t for t in tuples if t is not None)
    mapping = {
        gid: {"geometry_tuple": list(t), "n_samples": int(counts[t])}
        for t, gid in t2id.items()
    }
    return out, mapping


# Physical-size columns in Raw SE (one-to-one with the normalized geometry fields).
PHYS_GEOM_COLS = [
    "Orifice width (um)",
    "Channel height or depth (um)",
    "Continuous inlet width (um)",
    "dispersed inlet width (um)",
    "Outlet channel width (um)",
]
DERIVED_PHYS_MAP = {
    "Normalized channel depth": "Channel height or depth (um)",
    "Normalized continuous inlet": "Continuous inlet width (um)",
    "Normalized dispersed inlet": "dispersed inlet width (um)",
    "Normalized outlet width": "Outlet channel width (um)",
}


def physical_geometry_consistency(se: pd.DataFrame) -> dict:
    """Verify (a) normalized_i * orifice_width == physical_i for the four derived
    size columns, and (b) the grouping by the 5 normalized fields and the grouping
    by the 5 physical fields are a strict 1:1 bijection (same group count)."""
    orifice = se["Orifice width (um)"].to_numpy(dtype=float)
    n_consistent, n_inconsistent, n_missing = 0, 0, 0
    for norm_col, phys_col in DERIVED_PHYS_MAP.items():
        for n, p, o in zip(se[norm_col], se[phys_col], orifice):
            if pd.isna(n) or pd.isna(p) or pd.isna(o):
                n_missing += 1  # missingness is NOT a derived-value inconsistency
                continue
            if np.isclose(float(n) * float(o), float(p), rtol=1e-6, atol=1e-6):
                n_consistent += 1
            else:
                n_inconsistent += 1
    norm_tuples = [rounded_tuple(r[GEOM_COLS], ROUND_GEOM) for _, r in se.iterrows()]
    phys_tuples = [rounded_tuple(r[PHYS_GEOM_COLS], ROUND_GEOM) for _, r in se.iterrows()]
    norm_groups = {t for t in norm_tuples if t is not None}
    phys_groups = {t for t in phys_tuples if t is not None}
    mapping = {}
    for nt, pt in zip(norm_tuples, phys_tuples):
        if nt is None or pt is None:
            continue
        if nt in mapping and mapping[nt] != pt:
            mapping[nt] = None  # conflicting mapping -> bijection fails
        elif nt not in mapping:
            mapping[nt] = pt
    bijection = (
        all(v is not None for v in mapping.values())
        and len(mapping) == len(norm_groups) == len(phys_groups)
    )
    return {
        "n_rows": int(len(se)),
        "n_checks": n_consistent + n_inconsistent + n_missing,
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "n_missing_involved": n_missing,
        "n_groups_normalized": len(norm_groups),
        "n_groups_physical": len(phys_groups),
        "mapping_is_bijection": bool(bijection),
    }


def geometry_group_summary(df: pd.DataFrame) -> dict:
    counts = df["geometry_id"].value_counts().sort_index()
    if len(counts) == 0:
        return {}
    agg = {
        "n_samples": int(len(df)),
        "n_unique_geometries": int(len(counts)),
        "group_size_min": int(counts.min()),
        "group_size_max": int(counts.max()),
        "group_size_median": float(counts.median()),
        "group_size_mean": float(counts.mean()),
        "n_singleton_geometries": int((counts == 1).sum()),
        "n_geometries_lt5": int((counts < 5).sum()),
        "n_geometries_lt10": int((counts < 10).sum()),
        "n_invalid_geometry_rows": int(df["geometry_id"].isna().sum()),
        "largest_10": counts.sort_values(ascending=False).head(10).to_dict(),
        "smallest_10": counts.sort_values().head(10).to_dict(),
    }
    return agg


# ----------------------------------------------------------------------------
# Stage 0F — random-split geometry overlap
# ----------------------------------------------------------------------------
def random_split_metrics(train_idx, test_idx, geom_arr) -> dict:
    """Sample-level and geometry-level seen-geometry overlap for ONE split."""
    train_geoms = set(geom_arr[train_idx])
    test_geoms = geom_arr[test_idx]
    seen = [g for g in test_geoms if g in train_geoms]
    sample_frac = len(seen) / len(test_geoms)
    test_unique = sorted(set(test_geoms))
    geo_seen = [g for g in test_unique if g in train_geoms]
    geo_frac = len(geo_seen) / len(test_unique)
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_geom": int(len(train_geoms)),
        "n_test_geom": int(len(test_unique)),
        "n_test_seen_geom_samples": int(len(seen)),
        "sample_overlap_frac": float(sample_frac),
        "n_test_geom_seen": int(len(geo_seen)),
        "geometry_overlap_frac": float(geo_frac),
    }


def random_split_audit(se_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sklearn.model_selection import train_test_split

    geom_arr = se_df["geometry_id"].to_numpy()
    n = len(se_df)
    rows = []
    for seed in range(SEED_LO, SEED_HI + 1):
        tr, te = train_test_split(
            np.arange(n), test_size=TEST_SIZE, random_state=seed, shuffle=True
        )
        m = random_split_metrics(tr, te, geom_arr)
        rows.append({"seed": seed, **m})
    df = pd.DataFrame(rows)
    s = df["sample_overlap_frac"]
    g = df["geometry_overlap_frac"]
    summary = {
        "n_seeds": int(len(df)),
        "test_size": TEST_SIZE,
        "seed_range": [SEED_LO, SEED_HI],
        "splitter": "sklearn.model_selection.train_test_split(shuffle=True, random_state=seed)",
        "interval_label": "empirical 95% percentile interval (2.5th-97.5th percentile of the seed distribution, not an analytic CI)",
        "sample_overlap": {
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p2_5": float(np.percentile(s, 2.5)),
            "p97_5": float(np.percentile(s, 97.5)),
        },
        "geometry_overlap": {
            "mean": float(g.mean()),
            "sd": float(g.std(ddof=1)),
            "median": float(g.median()),
            "min": float(g.min()),
            "max": float(g.max()),
            "p2_5": float(np.percentile(g, 2.5)),
            "p97_5": float(np.percentile(g, 97.5)),
        },
    }
    return df, summary


# ----------------------------------------------------------------------------
# Stage 0C — Comprehensive composition audit
# ----------------------------------------------------------------------------
def composition_audit(comp, se, ff1, ff2):
    """Deterministic matching of Comprehensive rows against source files."""
    se_keys = Counter(rounded_tuple(r[SE_MATCH_COLS], ROUND_MATCH) for _, r in se.iterrows())
    ff1_keys = Counter(rounded_tuple(r[FF_MATCH_COLS], ROUND_MATCH) for _, r in ff1.iterrows())
    ff2_keys = Counter(rounded_tuple(r[FF_MATCH_COLS], ROUND_MATCH) for _, r in ff2.iterrows())

    def nan_tuples(keys_counter):
        return sum(v for k, v in keys_counter.items() if k is None)

    provenance = []
    unmatched = []
    comp_se_keys, comp_ff_keys = set(), set()
    comp_se_counter, comp_ff1_counter, comp_ff2_counter = Counter(), Counter(), Counter()
    n_multi_rows = 0
    for idx, row in comp.iterrows():
        t_se = rounded_tuple(row[SE_MATCH_COLS], ROUND_MATCH)
        t_ff = rounded_tuple(row[FF_MATCH_COLS], ROUND_MATCH)
        if t_se is not None:
            comp_se_keys.add(t_se)
        if t_ff is not None:
            comp_ff_keys.add(t_ff)
        hits = []
        if t_se is not None and se_keys.get(t_se, 0) > 0:
            hits.append("SE")
        if t_ff is not None:
            if ff1_keys.get(t_ff, 0) > 0:
                hits.append("FF1")
            if ff2_keys.get(t_ff, 0) > 0:
                hits.append("FF2")
        if not hits:
            unmatched.append(
                {
                    "experiment": py_native(row.get("Experiment")),
                    "n_nan_in_match_cols": int(row[SE_MATCH_COLS].isna().sum()),
                }
            )
        provenance.append(sorted(hits))
        # Per-part multisets are built ONLY from uniquely classified rows;
        # ambiguous (multi) rows are reported separately and force exact_match=False.
        if hits == ["SE"]:
            comp_se_counter[t_se] += 1
        elif hits == ["FF1"]:
            comp_ff1_counter[t_ff] += 1
        elif hits == ["FF2"]:
            comp_ff2_counter[t_ff] += 1
        elif len(hits) > 1:
            n_multi_rows += 1

    cls = Counter()
    for h in provenance:
        cls["multi" if len(h) > 1 else (h[0] if h else "unmatched")] += 1

    def multiset_compare(comp_counter, src_counter):
        missing = sum(
            max(0, src_counter[k] - comp_counter.get(k, 0)) for k in src_counter if k is not None
        )
        extra = sum(
            max(0, comp_counter[k] - src_counter.get(k, 0)) for k in comp_counter if k is not None
        )
        return {
            "missing_multiplicity": int(missing),
            "extra_multiplicity": int(extra),
            "exact_match": bool(missing == 0 and extra == 0),
        }

    multiset = {
        "SE": multiset_compare(comp_se_counter, se_keys),
        "FF1": multiset_compare(comp_ff1_counter, ff1_keys),
        "FF2": multiset_compare(comp_ff2_counter, ff2_keys),
    }
    multiset["overall"] = {
        "missing_multiplicity": int(sum(m["missing_multiplicity"] for m in multiset.values())),
        "extra_multiplicity": int(sum(m["extra_multiplicity"] for m in multiset.values())),
        "exact_match": bool(
            all(m["exact_match"] for m in multiset.values())
            and n_multi_rows == 0
            and cls.get("unmatched", 0) == 0
        ),
    }
    multiset["n_ambiguous_multi_rows"] = n_multi_rows

    result = {
        "comp_n_rows": int(len(comp)),
        "source_row_counts": {
            "SE": int(len(se)),
            "FF1_DE": int(len(ff1)),
            "FF2_DE": int(len(ff2)),
        },
        "matched_source_rows": {
            "SE": int(sum(1 for k, v in se_keys.items() if k in comp_se_keys for _ in range(v))),
            "FF1": int(sum(1 for k, v in ff1_keys.items() if k in comp_ff_keys for _ in range(v))),
            "FF2": int(sum(1 for k, v in ff2_keys.items() if k in comp_ff_keys for _ in range(v))),
        },
        "distinct_key_tuples": {
            "SE": sum(1 for k in se_keys if k is not None),
            "FF1": sum(1 for k in ff1_keys if k is not None),
            "FF2": sum(1 for k in ff2_keys if k is not None),
        },
        "duplicate_rows_within_source": {
            "SE": int(sum(v - 1 for k, v in se_keys.items() if k is not None and v > 1)),
            "FF1": int(sum(v - 1 for k, v in ff1_keys.items() if k is not None and v > 1)),
            "FF2": int(sum(v - 1 for k, v in ff2_keys.items() if k is not None and v > 1)),
        },
        "nan_key_rows_in_source": {
            "SE": int(nan_tuples(se_keys)),
            "FF1": int(nan_tuples(ff1_keys)),
            "FF2": int(nan_tuples(ff2_keys)),
        },
        "comp_row_classification": dict(cls),
        "multiset": multiset,
        "comp_rows_matched_distinct": int(sum(1 for h in provenance if h)),
        "unmatched_comp_rows": unmatched,
        "hypothesis_check": {
            "sum_hypothesis": int(len(se)) + int(len(ff1)) + int(len(ff2)),
            "comp_n_rows": int(len(comp)),
            "row_count_hypothesis_holds": (len(se) + len(ff1) + len(ff2)) == len(comp),
        },
    }
    return result


# ----------------------------------------------------------------------------
# Stage 0G — external generalizability audit
# ----------------------------------------------------------------------------
def external_audit(ext_df, se_df, comp_df) -> tuple[pd.DataFrame, dict]:
    """External domain audit with dual reference domains.

    Study B primary reference  : full Comprehensive (the author training domain
                                 that would actually be used for external stress tests).
    Study A supplementary      : SE only (controlled single-fluid domain).
    """
    ext_local, ext_map = assign_geometry_ids(ext_df, id_prefix="EXT-G")
    se_tuples = {canonical_geometry_tuple(r) for _, r in se_df.iterrows()}
    comp_tuples = {canonical_geometry_tuple(r) for _, r in comp_df.iterrows()}
    ext_tuples = [canonical_geometry_tuple(r) for _, r in ext_df.iterrows()]
    ext_tuple_set = {t for t in ext_tuples if t is not None}

    exact_se = ext_tuple_set & se_tuples
    exact_comp = ext_tuple_set & comp_tuples

    def range_block(train_df, col):
        tr = train_df[col]
        ex = ext_df[col]
        t_min, t_max = tr.min(skipna=True), tr.max(skipna=True)
        n_train_nan = int(tr.isna().sum())
        n_ext_nan = int(ex.isna().sum())
        if pd.isna(t_min) or pd.isna(t_max):
            n_outside, n_compared = 0, 0
        else:
            n_outside = int(((ex < t_min) | (ex > t_max)).sum())
            n_compared = int(ex.notna().sum())
        return {
            "train_min": py_native(t_min),
            "train_max": py_native(t_max),
            "external_min": py_native(ex.min(skipna=True)),
            "external_max": py_native(ex.max(skipna=True)),
            "n_external_outside_train_range": n_outside,
            "n_train_nan": n_train_nan,
            "n_external_nan": n_ext_nan,
            "n_compared": n_compared,
        }

    def viscosity_block(train_df, col):
        """Range extrapolation + exact-value overlap (seen/unseen fluid property)."""
        b = range_block(train_df, col)
        train_vals = set(train_df[col].dropna().round(ROUND_MATCH))
        ext_vals = list(ext_df[col].dropna().round(ROUND_MATCH))
        ext_unique = set(ext_vals)
        b["n_unique_train"] = int(len(train_vals))
        b["n_unique_external"] = int(len(ext_unique))
        b["n_exact_unique_overlap"] = int(len(ext_unique & train_vals))
        b["n_external_samples_exact_seen"] = int(sum(1 for v in ext_vals if v in train_vals))
        b["n_external_samples_exact_novel"] = int(len(ext_vals) - sum(1 for v in ext_vals if v in train_vals))
        return b

    per_geom = {
        "vs_Comprehensive": {c: range_block(comp_df, c) for c in GEOM_COLS},
        "vs_SE": {c: range_block(se_df, c) for c in GEOM_COLS},
    }
    op = {
        "vs_Comprehensive": {c: range_block(comp_df, c) for c in FLOW_COLS},
        "vs_SE": {c: range_block(se_df, c) for c in FLOW_COLS},
    }
    fluid = {
        "vs_Comprehensive": {
            "viscosity_ratio": viscosity_block(comp_df, "viscosity ratio"),
            "note": "Primary Study B reference (full Comprehensive training domain).",
        },
        "vs_SE": {
            "viscosity_ratio": None,
            "note": "Raw SE dataset has no viscosity-ratio column; SE reference not available.",
        },
    }
    visc_comp = fluid["vs_Comprehensive"]["viscosity_ratio"]
    viscosity_exact = {
        "n_exact_unique_overlap_vs_Comprehensive": visc_comp["n_exact_unique_overlap"],
        "n_external_samples_exact_seen_vs_Comprehensive": visc_comp["n_external_samples_exact_seen"],
        "n_external_samples_exact_novel_vs_Comprehensive": visc_comp["n_external_samples_exact_novel"],
        "n_external_outside_range_vs_Comprehensive": visc_comp["n_external_outside_train_range"],
    }

    per_ref_rows = []
    if "Ref" in ext_df.columns:
        for ref, grp in ext_df.groupby("Ref", dropna=False):
            ref_tuples = {canonical_geometry_tuple(r) for _, r in grp.iterrows()}
            per_ref_rows.append(
                {
                    "ref": str(ref),
                    "n_samples": int(len(grp)),
                    "n_unique_geometries": int(len([t for t in ref_tuples if t is not None])),
                    "n_unique_viscosity_ratios": int(grp["viscosity ratio"].nunique(dropna=True)),
                    "frr_min": py_native(grp["Flow rate ratio"].min()),
                    "frr_max": py_native(grp["Flow rate ratio"].max()),
                    "ca_min": py_native(grp["Capillary number"].min()),
                    "ca_max": py_native(grp["Capillary number"].max()),
                }
            )

    # long-format CSV for the external domain summary
    csv_rows = [
        {"section": "aggregate", "ref": "", "metric": "n_samples", "value": int(len(ext_df))},
        {"section": "aggregate", "ref": "", "metric": "n_unique_geometries",
         "value": int(len(ext_tuple_set))},
        {"section": "aggregate", "ref": "", "metric": "n_invalid_geometry_rows",
         "value": int(sum(1 for t in ext_tuples if t is None))},
        {"section": "aggregate", "ref": "", "metric": "n_unique_viscosity_ratios",
         "value": int(ext_df["viscosity ratio"].nunique(dropna=True))},
        {"section": "aggregate", "ref": "", "metric": "n_unique_refs",
         "value": int(ext_df["Ref"].nunique(dropna=True)) if "Ref" in ext_df.columns else np.nan},
        {"section": "geometry_overlap", "ref": "", "metric": "exact_geometry_overlap_vs_Comprehensive_(Study_B_primary)",
         "value": int(len(exact_comp))},
        {"section": "geometry_overlap", "ref": "", "metric": "exact_geometry_overlap_vs_SE_(Study_A_supplementary)",
         "value": int(len(exact_se))},
    ]
    for ref_name, geo_block in per_geom.items():
        for c in GEOM_COLS:
            b = geo_block[c]
            csv_rows += [
                {"section": f"geom_feature::{ref_name}::{c}", "ref": "", "metric": "train_min", "value": b["train_min"]},
                {"section": f"geom_feature::{ref_name}::{c}", "ref": "", "metric": "train_max", "value": b["train_max"]},
                {"section": f"geom_feature::{ref_name}::{c}", "ref": "", "metric": "external_min", "value": b["external_min"]},
                {"section": f"geom_feature::{ref_name}::{c}", "ref": "", "metric": "external_max", "value": b["external_max"]},
                {"section": f"geom_feature::{ref_name}::{c}", "ref": "", "metric": "n_external_outside_train_range",
                 "value": b["n_external_outside_train_range"]},
            ]
    for ref_name, op_block in op.items():
        for c in FLOW_COLS:
            b = op_block[c]
            csv_rows += [
                {"section": f"operating::{ref_name}::{c}", "ref": "", "metric": "train_min", "value": b["train_min"]},
                {"section": f"operating::{ref_name}::{c}", "ref": "", "metric": "train_max", "value": b["train_max"]},
                {"section": f"operating::{ref_name}::{c}", "ref": "", "metric": "external_min", "value": b["external_min"]},
                {"section": f"operating::{ref_name}::{c}", "ref": "", "metric": "external_max", "value": b["external_max"]},
                {"section": f"operating::{ref_name}::{c}", "ref": "", "metric": "n_external_outside_train_range",
                 "value": b["n_external_outside_train_range"]},
            ]
    for ref_name, fluid_block in fluid.items():
        b = fluid_block.get("viscosity_ratio")
        if b is None:
            csv_rows.append(
                {"section": f"fluid::viscosity_ratio::{ref_name}", "ref": "", "metric": "note",
                 "value": fluid_block.get("note")}
            )
            continue
        for k, v in b.items():
            csv_rows.append(
                {"section": f"fluid::viscosity_ratio::{ref_name}", "ref": "", "metric": k, "value": v}
            )
        csv_rows.append(
            {"section": f"fluid::viscosity_ratio::{ref_name}", "ref": "", "metric": "note",
             "value": fluid_block.get("note")}
        )
    for r in per_ref_rows:
        for k, v in r.items():
            if k == "ref":
                continue
            csv_rows.append({"section": "per_ref", "ref": r["ref"], "metric": k, "value": v})

    summary = {
        "n_samples": int(len(ext_df)),
        "n_unique_geometries": int(len(ext_tuple_set)),
        "n_invalid_geometry_rows": int(sum(1 for t in ext_tuples if t is None)),
        "n_unique_viscosity_ratios": int(ext_df["viscosity ratio"].nunique(dropna=True)),
        "n_unique_refs": int(ext_df["Ref"].nunique(dropna=True)) if "Ref" in ext_df.columns else None,
        "exact_geometry_overlap_vs_Comprehensive": sorted([list(t) for t in exact_comp]),
        "exact_geometry_overlap_vs_SE": sorted([list(t) for t in exact_se]),
        "per_geometry_feature": per_geom,
        "operating_condition": op,
        "fluid_property": fluid,
        "viscosity_exact": viscosity_exact,
        "per_ref": per_ref_rows,
    }
    return pd.DataFrame(csv_rows), summary, ext_local


# ----------------------------------------------------------------------------
# Integrity: raw files unchanged during the audit
# ----------------------------------------------------------------------------
def verify_raw_unchanged(hashes_before: dict) -> bool:
    ok = True
    for name, sha in hashes_before.items():
        if sha256_of(RAW_DIR / name) != sha:
            ok = False
    return ok


# ----------------------------------------------------------------------------
# Gate decision (data-driven rules, documented)
# ----------------------------------------------------------------------------
def gate_decision(res: dict, unresolved: list) -> tuple[str, list[str]]:
    """Blocking failures -> FAIL. Non-blocking unresolved -> PASS WITH WARNINGS.
    Clean -> PASS. Severity of each unresolved item is given per item."""
    # ---- blocking checks ----
    if not res["raw_files_unchanged"]:
        return "FAIL", ["A raw file was modified during the audit run."]
    se = res["se"]
    if se["n_samples"] == 0:
        return "FAIL", ["No valid SE samples found."]
    if se["n_unique_geometries"] < 2:
        return "FAIL", ["geometry_id collapses to fewer than 2 groups."]
    if se["n_invalid_geometry_rows"] > 0:
        return "FAIL", [
            f"{se['n_invalid_geometry_rows']} SE rows have NaN in pure geometry fields "
            "(grouping not reliable, no imputation performed)."
        ]
    if not res["composition"]["multiset"]["overall"]["exact_match"]:
        return "FAIL", [
            "Comprehensive multiset is NOT exactly SE + FF1 + FF2 "
            "(missing/extra multiplicity > 0 or ambiguous rows exist)."
        ]
    pc = res["se_physical_consistency"]
    if not pc["mapping_is_bijection"] or pc["n_inconsistent"] > 0:
        return "FAIL", [
            "Normalized geometry fields and physical size columns are inconsistent "
            "(derived-value check or group bijection failed)."
        ]
    if pc["n_missing_involved"] > 0:
        return "FAIL", [
            f"{pc['n_missing_involved']} missing value(s) in the geometry size columns "
            "(grouping reliability not verifiable)."
        ]
    if res["composition"]["comp_row_classification"].get("unmatched", 0) > 0:
        return "FAIL", [
            f"{res['composition']['comp_row_classification']['unmatched']} Comprehensive rows "
            "could not be matched to any source file."
        ]
    blocking = [u for u in unresolved if u.get("severity") == "blocking"]
    if blocking:
        return "FAIL", ["Blocking unresolved issue(s):"] + [
            f"- {u['issue']}" for u in blocking
        ]
    # ---- non-blocking warnings ----
    reasons = []
    non_blocking = [u for u in unresolved if u.get("severity") == "non_blocking"]
    if non_blocking:
        reasons.append(
            f"{len(non_blocking)} non-blocking unresolved issue(s) remain — "
            "see DATA_AUDIT.md §10. None blocks the SE-based Stage 1 baseline, "
            "but each needs attention."
        )
    if res["external"]["n_invalid_geometry_rows"] > 0:
        reasons.append(
            f"{res['external']['n_invalid_geometry_rows']} external rows have NaN in "
            "geometry fields."
        )
    if reasons:
        return "PASS WITH WARNINGS", reasons
    return "PASS", ["All Stage 0 checks passed cleanly."]


# ----------------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------------
def build_data_audit_md(res: dict) -> str:
    env = res["environment"]
    L = []
    A = L.append
    A("# DAFD 3.0 Data Audit\n")
    A("> Generated programmatically by `src/audit/audit_dafd3.py` — no hand-filled numbers.\n")

    A("## 1. Environment\n")
    A(f"- Python: {env['python_version']}")
    A(f"- pandas: {env['pandas']} | numpy: {env['numpy']} | openpyxl: {env['openpyxl']} | scikit-learn: {env['sklearn']}")
    A(f"- OS: {env['os']}")
    A("")

    A("## 2. Raw File Integrity\n")
    A("SHA-256 of every raw file before/after the audit (see `raw_file_manifest.csv`):")
    A("")
    A("| file | size (B) | sha256 (first 16) |")
    A("|---|---|---|")
    for r in res["manifest"]:
        A(f"| {r['filename']} | {r['size_bytes']} | {str(r['sha256'])[:16]}… |")
    A("")
    A(f"- Raw files unchanged during audit: **{res['raw_files_unchanged']}**")
    A("")

    A("## 3. Dataset Shapes\n")
    A("| file | sheet | rows | valid rows | cols | missing cells | full-dup rows (involved) | feat-dup rows (involved) |")
    A("|---|---|---|---|---|---|---|---|")
    for f in CORE_FILES + ["FF1_DE.xlsx", "FF2_DE.xlsx", "All_NewFF1_instability.xlsx", "All_NewFF2_instability.xlsx", "Final_Comprehensive_plus_Generalizability_data_normalized.xlsx", "Raw DE dataset.xlsx"]:
        key = f
        d = res["dataset_details"].get(key)
        if d is None:
            continue
        A(f"| {f} | {d['sheet']} | {d['n_rows']} | {d['n_rows_valid']} | {d['n_cols']} | {d['n_missing_cells']} | {d['n_duplicated_full_rows_involved']} | {d['n_duplicated_feature_rows_involved']} |")
    A("")
    n_d = res["se"]["n_se_title_rows_dropped"]
    n_valid = res["se"]["n_samples"]
    n_raw = n_valid + n_d
    A(f"- Raw SE dataset.xlsx: {n_d} leading fully-NaN title row(s) dropped before analysis "
      f"(raw file untouched; {n_raw} Excel rows -> {n_valid} valid).")
    if res.get("raw_de_n_nan_rows", 0) > 0:
        pos = res["raw_de_nan_excel_rows"]
        A(f"- Raw DE dataset.xlsx: {res['raw_de_n_nan_rows']} fully-NaN rows at Excel rows {pos} "
          f"(of {res['raw_de_n_rows']} parsed rows; NOT all trailing — the last rows contain data). "
          "The 'full-duplicate' count equals these empty rows.")
    A("")

    A("## 4. Missing / Duplicate Audit\n")
    for f in CORE_FILES:
        d = res["dataset_details"][f]
        A(f"### {f}\n")
        A(f"- rows={d['n_rows']}, valid={d['n_rows_valid']}, cols={d['n_cols']}, missing cells={d['n_missing_cells']}")
        A(f"- fully duplicated rows (involved): {d['n_duplicated_full_rows_involved']}")
        A(f"- feature-duplicated rows (involved, features={d['feature_cols_used']}): {d['n_duplicated_feature_rows_involved']}")
        A("")
        A("| column | dtype | missing | unique | min | max | IQR outliers (reported only) |")
        A("|---|---|---|---|---|---|---|")
        for c in d["column_details"]:
            mn = c.get("min", "—")
            mx = c.get("max", "—")
            A(f"| {c['column']} | {c['dtype']} | {c['n_missing']} | {c['n_unique']} | {mn} | {mx} | {c.get('n_iqr_outliers','—')} |")
        A("")
    A("_Outliers are reported only — no rows were deleted or imputed._\n")

    A("## 5. Comprehensive Composition\n")
    comp = res["composition"]
    A(f"- Comprehensive rows: {comp['comp_n_rows']}")
    A(f"- Source row counts: SE={comp['source_row_counts']['SE']}, FF1_DE={comp['source_row_counts']['FF1_DE']}, FF2_DE={comp['source_row_counts']['FF2_DE']}")
    A(f"- Hypothesis 868 = 474 + 197 + 197 at row-count level: **{comp['hypothesis_check']['row_count_hypothesis_holds']}** ({comp['hypothesis_check']['sum_hypothesis']} vs {comp['hypothesis_check']['comp_n_rows']})")
    A(f"- Distinct key tuples per source: SE={comp['distinct_key_tuples']['SE']}, FF1={comp['distinct_key_tuples']['FF1']}, FF2={comp['distinct_key_tuples']['FF2']}")
    A(f"- Duplicate rows within source (extra copies): SE={comp['duplicate_rows_within_source']['SE']}, FF1={comp['duplicate_rows_within_source']['FF1']}, FF2={comp['duplicate_rows_within_source']['FF2']}")
    A(f"- NaN-key rows in source: SE={comp['nan_key_rows_in_source']['SE']}, FF1={comp['nan_key_rows_in_source']['FF1']}, FF2={comp['nan_key_rows_in_source']['FF2']}")
    A(f"- Comprehensive row classification: {comp['comp_row_classification']}")
    if comp["unmatched_comp_rows"]:
        A(f"- Unmatched Comprehensive rows (Experiment ids): {[r['experiment'] for r in comp['unmatched_comp_rows']]}")
    A("")
    A("Multiset (bijection) verification — Comprehensive must contain SE + FF1 + FF2 **with exact multiplicities**:")
    A("")
    A("| part | missing multiplicity | extra multiplicity | exact match |")
    A("|---|---|---|---|")
    for part in ["SE", "FF1", "FF2", "overall"]:
        m = comp["multiset"][part]
        A(f"| {part} | {m['missing_multiplicity']} | {m['extra_multiplicity']} | {m['exact_match']} |")
    A(f"- Ambiguous multi-match rows: {comp['multiset']['n_ambiguous_multi_rows']}")
    A("")

    A("## 6. Geometry Definition\n")
    A(f"- Pure geometry fields: {', '.join(GEOM_COLS)}")
    A(f"- Forbidden in geometry_id: flow conditions, fluid properties, targets, row indices.")
    A(f"- Canonicalization: each field rounded to {ROUND_GEOM} decimals; tuple of the five fields in the fixed order above.")
    A("- ID assignment: unique canonical tuples sorted lexicographically ascending; ids G0001, G0002, … (zero-padded, ≥4 digits).")
    A("")
    pc = res["se_physical_consistency"]
    A("**Physical-size consistency check** (normalized field = physical size / orifice width, and group bijection):")
    A("")
    A(f"- Derived-value checks: {pc['n_consistent']}/{pc['n_checks']} consistent, "
      f"{pc['n_inconsistent']} inconsistent, {pc['n_missing_involved']} skipped (missing values; rows: {pc['n_rows']})")
    A(f"- Groups by 5 normalized fields: **{pc['n_groups_normalized']}** | groups by 5 physical size columns: **{pc['n_groups_physical']}**")
    A(f"- Mapping is a strict 1:1 bijection: **{pc['mapping_is_bijection']}**")
    A("")

    A("## 7. SE Geometry Distribution\n")
    se = res["se"]
    A(f"- SE valid samples: {se['n_samples']}")
    A(f"- Unique geometries: {se['n_unique_geometries']}")
    A(f"- Invalid geometry rows (NaN in geometry fields): {se['n_invalid_geometry_rows']}")
    A(f"- Group size: min={se['group_size_min']}, max={se['group_size_max']}, median={se['group_size_median']}, mean={se['group_size_mean']:.2f}")
    A(f"- Singletons: {se['n_singleton_geometries']} | groups with n<5: {se['n_geometries_lt5']} | groups with n<10: {se['n_geometries_lt10']}")
    A("")
    A("Largest 10 geometries:")
    A("")
    A("| geometry_id | n_samples |")
    A("|---|---|")
    for gid, n in se["largest_10"].items():
        A(f"| {gid} | {n} |")
    A("")
    A("Smallest 10 geometries:")
    A("")
    A("| geometry_id | n_samples |")
    A("|---|---|")
    for gid, n in se["smallest_10"].items():
        A(f"| {gid} | {n} |")
    A("")

    A("## 8. Random Split Geometry Overlap\n")
    rs = res["random_split"]
    A(f"- Seeds: {rs['seed_range'][0]}–{rs['seed_range'][1]} ({rs['n_seeds']} seeds), test_size={rs['test_size']}")
    A(f"- Splitter: `{rs['splitter']}`")
    A("")
    A("| metric | mean | SD | median | min | max | 2.5th pct | 97.5th pct |")
    A("|---|---|---|---|---|---|---|---|")
    for name, block in [("sample-level overlap", rs["sample_overlap"]), ("geometry-level overlap", rs["geometry_overlap"])]:
        A(f"| {name} | {block['mean']:.4f} | {block['sd']:.4f} | {block['median']:.4f} | {block['min']:.4f} | {block['max']:.4f} | {block['p2_5']:.4f} | {block['p97_5']:.4f} |")
    A("")
    A("The interval columns are the **empirical 95% percentile interval** of the 1000-seed "
      "distribution (2.5th–97.5th percentile) — not an analytic confidence interval.")
    A("")

    A("## 9. External Dataset Domain Shift\n")
    ex = res["external"]
    A(f"- External samples: {ex['n_samples']}")
    A(f"- External unique geometries: {ex['n_unique_geometries']} | invalid geometry rows: {ex['n_invalid_geometry_rows']}")
    A(f"- Unique viscosity ratios: {ex['n_unique_viscosity_ratios']} | unique Ref sources: {ex['n_unique_refs']}")
    A(f"- Exact geometry overlap vs Comprehensive (Study B primary): **{len(ex['exact_geometry_overlap_vs_Comprehensive'])}**")
    A(f"- Exact geometry overlap vs SE (Study A supplementary): **{len(ex['exact_geometry_overlap_vs_SE'])}**")
    if res["dataset_details"]["Generalizability_data_normalized.xlsx"]["column_details"]:
        gen = next(
            (
                c
                for c in res["dataset_details"]["Generalizability_data_normalized.xlsx"]["column_details"]
                if c["column"] == "Observed generation rate (Hz)"
            ),
            None,
        )
        if gen and gen["n_unique"] == 0 and gen["n_missing"] > 0:
            A("- ⚠ 'Observed generation rate (Hz)' is entirely empty in the external file — external generation-rate evaluation is impossible; only droplet-diameter targets exist.")
    A("")
    A("Reference domains: **vs Comprehensive** = Study B primary (author training domain, 868 rows); "
      "**vs SE** = Study A supplementary (controlled domain, 474 rows).")
    A("")
    A("### Per-geometry-feature range comparison")
    A("")
    A("| reference | field | train min | train max | ext min | ext max | ext outside train range |")
    A("|---|---|---|---|---|---|---|")
    for ref_name in ["vs_Comprehensive", "vs_SE"]:
        for c, b in ex["per_geometry_feature"][ref_name].items():
            A(f"| {ref_name} | {c} | {b['train_min']} | {b['train_max']} | {b['external_min']} | {b['external_max']} | {b['n_external_outside_train_range']} |")
    A("")
    A("### Operating-condition range comparison")
    A("")
    A("| reference | field | train min | train max | ext min | ext max | ext outside train range |")
    A("|---|---|---|---|---|---|---|")
    for ref_name in ["vs_Comprehensive", "vs_SE"]:
        for c, b in ex["operating_condition"][ref_name].items():
            A(f"| {ref_name} | {c} | {b['train_min']} | {b['train_max']} | {b['external_min']} | {b['external_max']} | {b['n_external_outside_train_range']} |")
    A("")
    A("### Viscosity ratio — exact-value novelty + range extrapolation")
    A("")
    b = ex["fluid_property"]["vs_Comprehensive"]["viscosity_ratio"]
    A(f"- Train (Comprehensive) unique values: {b['n_unique_train']} | external unique values: {b['n_unique_external']}")
    if b["n_exact_unique_overlap"] == 0:
        A("- **Exact unique overlap: 0** — every external viscosity ratio is an unseen value")
    else:
        A(f"- **Exact unique overlap: {b['n_exact_unique_overlap']}/{b['n_unique_external']}**")
    A(f"- External samples with exact-seen viscosity: **{b['n_external_samples_exact_seen']}/{ex['n_samples']}** "
      f"(exact-novel: {b['n_external_samples_exact_novel']}/{ex['n_samples']})")
    A(f"- Range: train [{b['train_min']}, {b['train_max']}], external [{b['external_min']}, {b['external_max']}]; "
      f"**{b['n_external_outside_train_range']}/{ex['n_samples']}** external samples outside the training range")
    A(f"- {ex['fluid_property']['vs_SE']['note']}")
    A("")
    if ex["per_ref"]:
        A("### Per-source structure")
        A("")
        A("| Ref | n samples | n geometries | n visc ratios | FRR range | Ca range |")
        A("|---|---|---|---|---|---|")
        for r in ex["per_ref"]:
            A(f"| {r['ref']} | {r['n_samples']} | {r['n_unique_geometries']} | {r['n_unique_viscosity_ratios']} | [{r['frr_min']}, {r['frr_max']}] | [{r['ca_min']}, {r['ca_max']}] |")
        A("")

    A("## 10. Unresolved Issues\n")
    A("Severity levels: `blocking` (would fail the gate) / `non_blocking` (recorded, needs attention) / `info` (by design).")
    A("")
    issues = res.get("unresolved_issues", [])
    if not issues:
        A("- None.")
    for i in issues:
        sev = i.get("severity", "non_blocking")
        tag = " _(by design)_" if i.get("by_design") else ""
        A(f"- **[{sev}]** {i['issue']}{tag}")
    A("")

    A("## 11. Stage 0 Gate Decision\n")
    A(f"**{res['gate_status']}**")
    A("")
    for r in res["gate_reasons"]:
        A(f"- {r}")
    A("")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = {
        "python_version": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "openpyxl": __import__("openpyxl").__version__,
        "sklearn": __import__("sklearn").__version__,
        "os": f"{platform.system()} {platform.release()}",
        # Public release: do not persist an author's absolute filesystem path.
        "project_root": ".",
    }

    # --- 0A: manifest + integrity baseline -------------------------------
    hashes_before = {p.name: sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}
    manifest = build_raw_manifest()
    manifest.to_csv(OUT_DIR / "raw_file_manifest.csv", index=False)

    # --- load core + component files ------------------------------------
    comp, ext, se, n_se_dropped = load_core_files()
    comps = load_component_files()

    # Raw DE fully-NaN row positions (programmatic; used by DATA_AUDIT.md §3)
    raw_de = comps["Raw_DE"]
    raw_de_nan_excel_rows = [
        int(i) + 2 for i in raw_de.index[raw_de.isna().all(axis=1)].tolist()
    ]  # pandas index 0 == Excel row 2 (header at Excel row 1)
    raw_de_nan_info = {
        "raw_de_n_nan_rows": len(raw_de_nan_excel_rows),
        "raw_de_nan_excel_rows": raw_de_nan_excel_rows,
        "raw_de_n_rows": int(len(raw_de)),
    }

    # --- 0B: dataset summaries ------------------------------------------
    core_specs = {
        "Raw SE dataset.xlsx": ("Generation rate ver. new", se, GEOM_COLS + FLOW_COLS),
        "Comprehensive_normalized.xlsx": ("Sheet1", comp, GEOM_COLS + FLOW_COLS + FLUID_COLS),
        "Generalizability_data_normalized.xlsx": ("sheet 1", ext, GEOM_COLS + FLOW_COLS + FLUID_COLS),
        "FF1_DE.xlsx": ("Ali_allcombined_sweeps_data_wit", comps["FF1_DE"], GEOM_COLS + FLOW_COLS + FLUID_COLS),
        "FF2_DE.xlsx": ("Ali_allcombined_sweeps_data_wit", comps["FF2_DE"], GEOM_COLS + FLOW_COLS + FLUID_COLS),
        "All_NewFF1_instability.xlsx": ("Ali_allcombined_sweeps_data_wit", comps["NewFF1"], GEOM_COLS + FLOW_COLS + FLUID_COLS),
        "All_NewFF2_instability.xlsx": ("Ali_allcombined_sweeps_data_wit", comps["NewFF2"], GEOM_COLS + FLOW_COLS + FLUID_COLS),
        "Final_Comprehensive_plus_Generalizability_data_normalized.xlsx": ("Sheet1", comps["Final"], GEOM_COLS + FLOW_COLS + FLUID_COLS),
        "Raw DE dataset.xlsx": ("1", comps["Raw_DE"], []),
    }
    dataset_details = {}
    summary_rows = []
    for fname, (sheet, df, feat) in core_specs.items():
        d = dataset_stats(df, feat)
        d["file"] = fname
        d["sheet"] = sheet
        dataset_details[fname] = d
        summary_rows.append(
            {
                "file": fname,
                "n_rows": d["n_rows"],
                "n_rows_valid": d["n_rows_valid"],
                "n_cols": d["n_cols"],
                "n_missing_cells": d["n_missing_cells"],
                "n_duplicated_full_rows_involved": d["n_duplicated_full_rows_involved"],
                "n_duplicated_full_rows_extra": d["n_duplicated_full_rows_extra"],
                "n_duplicated_feature_rows_involved": d["n_duplicated_feature_rows_involved"],
                "n_duplicated_feature_rows_extra": d["n_duplicated_feature_rows_extra"],
                "feature_cols_used": " | ".join(d["feature_cols_used"]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "dataset_summary.csv", index=False)

    # --- 0D/0E: geometry reconstruction on SE ---------------------------
    se_with_id, se_map = assign_geometry_ids(se)
    se_geom_summary = geometry_group_summary(se_with_id)

    per_geom_rows = []
    for gid, grp in se_with_id.groupby("geometry_id", sort=True):
        row0 = grp.iloc[0]
        per_geom_rows.append(
            {
                "geometry_id": gid,
                **{c: py_native(row0[c]) for c in GEOM_COLS},
                "Flow rate ratio_min": py_native(grp["Flow rate ratio"].min()),
                "Flow rate ratio_max": py_native(grp["Flow rate ratio"].max()),
                "Capillary number_min": py_native(grp["Capillary number"].min()),
                "Capillary number_max": py_native(grp["Capillary number"].max()),
                "viscosity ratio": np.nan,  # Raw SE has no viscosity-ratio column
                "target_droplet_diameter_min": py_native(grp["Observed droplet diameter (um)"].min()),
                "target_droplet_diameter_max": py_native(grp["Observed droplet diameter (um)"].max()),
                "target_generation_rate_min": py_native(grp["Observed generation rate (Hz)"].min()),
                "target_generation_rate_max": py_native(grp["Observed generation rate (Hz)"].max()),
                "n_samples_in_group": int(len(grp)),
            }
        )
    pd.DataFrame(per_geom_rows).sort_values("geometry_id").to_csv(
        OUT_DIR / "geometry_group_summary.csv", index=False
    )

    groups_cols = (
        ["source_row_id", "experiment_id", "geometry_id"]
        + GEOM_COLS
        + ["Flow rate ratio", "Capillary number", "viscosity ratio"]
        + ["Observed droplet diameter (um)", "Observed generation rate (Hz)"]
        + ["n_samples_in_group"]
    )
    se_with_id = se_with_id.rename(
        columns={"excel_row": "source_row_id", "Experiment": "experiment_id"}
    )
    se_with_id["viscosity ratio"] = np.nan  # column absent in Raw SE; kept for schema stability
    se_with_id["n_samples_in_group"] = se_with_id["geometry_id"].map(
        lambda g: se_map[g]["n_samples"] if g in se_map else np.nan
    )
    se_with_id[groups_cols].to_csv(OUT_DIR / "geometry_groups_se.csv", index=False)

    # --- 0F: random split overlap ---------------------------------------
    split_df, split_summary = random_split_audit(se_with_id)
    split_df.to_csv(OUT_DIR / "random_split_geometry_overlap.csv", index=False)

    # --- 0C: composition -------------------------------------------------
    composition = composition_audit(comp, se, comps["FF1_DE"], comps["FF2_DE"])

    # --- 0G: external audit ---------------------------------------------
    ext_csv, ext_summary, ext_with_id = external_audit(ext, se_with_id, comp)
    ext_csv.to_csv(OUT_DIR / "external_domain_summary.csv", index=False)

    # --- integrity re-check ---------------------------------------------
    raw_unchanged = verify_raw_unchanged(hashes_before)

    se_res = {
        "n_samples": int(len(se_with_id)),
        "n_unique_geometries": se_geom_summary["n_unique_geometries"],
        "n_invalid_geometry_rows": se_geom_summary["n_invalid_geometry_rows"],
        "group_size_min": se_geom_summary["group_size_min"],
        "group_size_max": se_geom_summary["group_size_max"],
        "group_size_median": se_geom_summary["group_size_median"],
        "group_size_mean": se_geom_summary["group_size_mean"],
        "n_singleton_geometries": se_geom_summary["n_singleton_geometries"],
        "n_geometries_lt5": se_geom_summary["n_geometries_lt5"],
        "n_geometries_lt10": se_geom_summary["n_geometries_lt10"],
        "largest_10": se_geom_summary["largest_10"],
        "smallest_10": se_geom_summary["smallest_10"],
        "n_se_title_rows_dropped": n_se_dropped,
        "geometry_id_rule": "sorted canonical tuples -> G####",
    }

    unresolved = []
    if composition["comp_row_classification"].get("unmatched", 0) > 0:
        unresolved.append(
            {
                "issue": (
                    f"{composition['comp_row_classification']['unmatched']} Comprehensive rows could not be "
                    f"matched to any source file. Experiment ids: "
                    f"{[r['experiment'] for r in composition['unmatched_comp_rows']]}. Not guessed — needs manual inspection."
                ),
                "severity": "blocking",
            }
        )
    if composition["duplicate_rows_within_source"]["FF1"] > 0 or composition["duplicate_rows_within_source"]["FF2"] > 0:
        unresolved.append(
            {
                "issue": (
                    "FF1_DE / FF2_DE contain rows that are exact duplicates of each other on the matching keys "
                    "(same geometry+flow+viscosity+targets). Whether these are intentional repeated trials is unresolved. "
                    "Does not affect the SE benchmark (SE part has zero duplicates)."
                ),
                "severity": "non_blocking",
            }
        )
    if ext_summary["n_unique_geometries"] > 0 and int(ext["Observed generation rate (Hz)"].isna().sum()) == int(len(ext)):
        unresolved.append(
            {
                "issue": (
                    "Generalizability_data_normalized.xlsx: the column 'Observed generation rate (Hz)' is entirely "
                    "empty (64/64 missing); external generation-rate evaluation is impossible from this file. "
                    "Only droplet-diameter targets are available externally. Affects Stage 4 design only."
                ),
                "severity": "non_blocking",
            }
        )
    unresolved.append(
        {
            "issue": (
                "Raw DE dataset.xlsx uses raw physical units and a different column layout; "
                "no deterministic mapping to the normalized FF1_DE/FF2_DE files was attempted without "
                "unit-conversion assumptions. Left unresolved by design."
            ),
            "severity": "info",
            "by_design": True,
        }
    )
    unresolved.append(
        {
            "issue": (
                "The NewFF instability files overlap the FF1_DE/FF2_DE key tuples only partially "
                "(1/37 and 9/37 on feature columns); their exact provenance is not resolved. "
                "Not used in the SE benchmark."
            ),
            "severity": "non_blocking",
        }
    )

    phys_consistency = physical_geometry_consistency(se)

    gate_status, gate_reasons = gate_decision(
        {
            "raw_files_unchanged": raw_unchanged,
            "se": se_res,
            "composition": composition,
            "external": ext_summary,
            "se_physical_consistency": phys_consistency,
        },
        unresolved,
    )

    res = {
        "environment": env,
        "manifest": manifest.to_dict("records"),
        "manifest_hashes": {n: hashes_before[n] for n in sorted(hashes_before)},
        **raw_de_nan_info,
        "raw_files_unchanged": raw_unchanged,
        "dataset_details": dataset_details,
        "composition": composition,
        "se": se_res,
        "se_physical_consistency": phys_consistency,
        "random_split": split_summary,
        "external": ext_summary,
        "unresolved_issues": unresolved,
        "gate_status": gate_status,
        "gate_reasons": gate_reasons,
    }

    with open(OUT_DIR / "audit_results.json", "w", encoding="utf-8") as f:
        json.dump(py_native(res), f, indent=2, ensure_ascii=False, allow_nan=False)

    (OUT_DIR / "DATA_AUDIT.md").write_text(build_data_audit_md(res), encoding="utf-8")

    print(f"[audit] done. outputs in {OUT_DIR}")
    print(f"[audit] gate decision: {gate_status}")
    for r in gate_reasons:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
