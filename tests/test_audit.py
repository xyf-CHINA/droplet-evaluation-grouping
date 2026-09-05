"""Stage 0 audit tests — minimal, meaningful, no over-engineering.

These tests verify the mechanical contracts of the audit layer:
hash determinism, geometry_id stability rules, overlap bounds, and
read-only behavior toward `data/raw/`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit import audit_dafd3 as aud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REQUIRED_DAFD_FILES = {
    "All_NewFF1_instability.xlsx",
    "All_NewFF2_instability.xlsx",
    "Comprehensive_normalized.xlsx",
    "FF1_DE.xlsx",
    "FF2_DE.xlsx",
    "Final_Comprehensive_plus_Generalizability_data_normalized.xlsx",
    "Generalizability_data_normalized.xlsx",
    "Raw DE dataset.xlsx",
    "Raw SE dataset.xlsx",
}
pytestmark = pytest.mark.skipif(
    not all((RAW_DIR / name).is_file() for name in REQUIRED_DAFD_FILES),
    reason="download the public DAFD 3.0 files described in data/README.md",
)


def make_synthetic_se(n_geoms=5, n_per=4, seed=0):
    """Synthetic SE-like frame with geometry, flow, fluid and target columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_geoms):
        geom = (100 + 10 * g, 2.0, 3.0, 2.5, 2.0)  # 5 pure geometry fields
        for _ in range(n_per):
            rows.append(
                {
                    "Orifice width (um)": geom[0],
                    "Normalized channel depth": geom[1],
                    "Normalized continuous inlet": geom[2],
                    "Normalized dispersed inlet": geom[3],
                    "Normalized outlet width": geom[4],
                    "Flow rate ratio": rng.uniform(1, 10),
                    "Capillary number": rng.uniform(0.01, 0.5),
                    "viscosity ratio": rng.uniform(0.5, 2),
                    "Observed droplet diameter (um)": rng.uniform(100, 300),
                    "Observed generation rate (Hz)": rng.uniform(10, 100),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. SHA-256 computation is repeatable
# ---------------------------------------------------------------------------
def test_sha256_repeatable():
    f = sorted(RAW_DIR.glob("*.xlsx"))[0]
    assert aud.sha256_of(f) == aud.sha256_of(f)


# ---------------------------------------------------------------------------
# 2. geometry_id is insensitive to row order
# ---------------------------------------------------------------------------
def test_geometry_id_row_order_invariance():
    df = make_synthetic_se()
    perm = np.random.default_rng(7).permutation(len(df))
    df_perm = df.iloc[perm].reset_index(drop=True)

    id_a = aud.assign_geometry_ids(df)[0].sort_values(
        ["Orifice width (um)", "Normalized channel depth"]).reset_index(drop=True)["geometry_id"]
    id_b = aud.assign_geometry_ids(df_perm)[0].sort_values(
        ["Orifice width (um)", "Normalized channel depth"]).reset_index(drop=True)["geometry_id"]
    # Same physical row -> same id regardless of input order.
    pd.testing.assert_series_equal(id_a, id_b)


# ---------------------------------------------------------------------------
# 3–5. Flow / fluid columns must not influence geometry_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("col", ["Flow rate ratio", "Capillary number", "viscosity ratio"])
def test_flow_fluid_columns_do_not_change_geometry_id(col):
    df = make_synthetic_se()
    ids_before = aud.assign_geometry_ids(df)[0]["geometry_id"]
    df[col] = df[col] * 100.0  # drastic change of a non-geometry column
    ids_after = aud.assign_geometry_ids(df)[0]["geometry_id"]
    pd.testing.assert_series_equal(ids_before, ids_after)


# ---------------------------------------------------------------------------
# 6. Changing a pure geometry field must change the canonical tuple
# ---------------------------------------------------------------------------
def test_geometry_field_change_changes_tuple():
    df = make_synthetic_se()
    t_before = aud.canonical_geometry_tuple(df.iloc[0])
    df.loc[0, "Orifice width (um)"] += 1.0
    t_after = aud.canonical_geometry_tuple(df.iloc[0])
    assert t_before != t_after


# ---------------------------------------------------------------------------
# 7. Random-split overlap outputs are bounded in [0, 1]
# ---------------------------------------------------------------------------
def test_random_split_overlap_in_unit_interval():
    df = make_synthetic_se(n_geoms=8, n_per=6)
    df, _ = aud.assign_geometry_ids(df)
    geom = df["geometry_id"].to_numpy()
    n = len(df)
    from sklearn.model_selection import train_test_split

    for seed in range(30):
        tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=seed)
        m = aud.random_split_metrics(tr, te, geom)
        assert 0.0 <= m["sample_overlap_frac"] <= 1.0
        assert 0.0 <= m["geometry_overlap_frac"] <= 1.0


# ---------------------------------------------------------------------------
# 8. Raw files are bit-identical after read-only audit operations
# ---------------------------------------------------------------------------
def test_raw_files_unchanged_by_audit_reads():
    hashes_before = {p.name: aud.sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}
    # Touch every raw file through the audit's own read-only loaders.
    aud.build_raw_manifest()
    aud.load_core_files()
    aud.load_component_files()
    hashes_after = {p.name: aud.sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}
    assert hashes_before == hashes_after


# ===========================================================================
# Real-data invariant tests (Stage 0.1)
#
# These encode the paper-critical invariants of the OFFICIAL DAFD 3.0 data.
# As long as the raw files are not replaced, they must keep passing.
# ===========================================================================

def _load_all():
    comp, ext, se, _ = aud.load_core_files()
    comps = aud.load_component_files()
    return comp, ext, se, comps


def test_se_normalized_physical_geometry_bijection():
    """The 5 normalized geometry fields and the 5 physical size columns must
    both form 35 groups with a strict 1:1 mapping (not a rounding artifact)."""
    _, _, se, _ = _load_all()
    r = aud.physical_geometry_consistency(se)
    assert r["n_groups_normalized"] == r["n_groups_physical"] == 35
    assert r["n_inconsistent"] == 0
    assert r["mapping_is_bijection"]


def test_comprehensive_multiset_exact_match():
    """Comprehensive must equal SE + FF1 + FF2 with exact multiplicities."""
    comp, _, se, comps = _load_all()
    r = aud.composition_audit(comp, se, comps["FF1_DE"], comps["FF2_DE"])
    assert r["multiset"]["overall"]["missing_multiplicity"] == 0
    assert r["multiset"]["overall"]["extra_multiplicity"] == 0
    assert r["multiset"]["overall"]["exact_match"]


def test_external_geometry_overlap_vs_comprehensive_zero():
    """External geometries must not exactly overlap the Comprehensive training domain."""
    comp, ext, se, _ = _load_all()
    _, summ, _ = aud.external_audit(ext, se, comp)
    assert len(summ["exact_geometry_overlap_vs_Comprehensive"]) == 0


def test_external_viscosity_exact_overlap_vs_comprehensive_zero():
    """External viscosity ratios must be exact-novel relative to Comprehensive."""
    comp, ext, se, _ = _load_all()
    _, summ, _ = aud.external_audit(ext, se, comp)
    assert summ["viscosity_exact"]["n_exact_unique_overlap_vs_Comprehensive"] == 0
    assert summ["viscosity_exact"]["n_external_samples_exact_seen_vs_Comprehensive"] == 0
