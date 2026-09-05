"""Cross-module integration tests.

These tests pin the contracts for
provenance correctness, sampler uniformity, strict JSON, locked model spec,
finite-value enforcement, and protocol invariants.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.audit import audit_dafd3 as aud
from src.common import data as cdata
from src.common import metrics as cmetrics
from src.common import model_spec
from src.evaluation import protocol_g_sampler as pgs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "Raw SE dataset.xlsx").is_file()
    or not (RAW_DIR / "Comprehensive_normalized.xlsx").is_file(),
    reason="download the public DAFD 3.0 files described in data/README.md",
)


# ---------------------------------------------------------------------------
# Provenance: extracted source_row_id must equal the audit's excel_row
# ---------------------------------------------------------------------------
def test_source_row_id_matches_audit_excel_row():
    _comp, _ext, se, _n_dropped = aud.load_core_files()
    se_df = cdata.load_se_extracted()
    assert len(se_df) == 474
    se_audit = se.copy()
    se_audit["se_experiment_id"] = se_audit["Experiment"].astype(int)
    mapping = se_audit.set_index("se_experiment_id")["excel_row"]
    assert (se_df.set_index("se_experiment_id")["source_row_id"] == mapping).all()
    assert se_df["source_row_id"].min() == 3
    assert se_df["source_row_id"].max() == 476


def test_extraction_counts_and_constant_viscosity():
    se_df = cdata.load_se_extracted()
    assert len(se_df) == 474
    assert se_df["geometry_id"].nunique() == 35
    assert se_df["viscosity ratio"].nunique() == 1


# ---------------------------------------------------------------------------
# Protocol G sampler contracts
# ---------------------------------------------------------------------------
def _sizes():
    se_df = cdata.load_se_extracted()
    return se_df["geometry_id"].value_counts().sort_index().to_list()


def test_sampler_feasibility_distinctness_determinism():
    sizes = _sizes()
    cnt, n_feasible, subs = pgs.collect_distinct(sizes, 95, 100, 0)
    assert n_feasible == 1720220  # exact DP count
    assert all(sum(sizes[i] for i in s) == 95 for s in subs)
    assert len(set(subs)) == 100
    _, _, again = pgs.collect_distinct(sizes, 95, 100, 0)
    assert subs == again  # same seed -> identical draw
    _, _, other = pgs.collect_distinct(sizes, 95, 100, 1)
    assert set(subs) != set(other)  # independent draw


def test_sampler_uniformity_sanity():
    sizes = _sizes()
    n_feasible, probs = pgs.theoretical_inclusion(sizes, 95)
    assert n_feasible == 1720220
    # real invariant: sum_g p_g * n_g == E[test size] == 95 (each feasible
    # subset has exactly 95 samples by construction)
    assert abs(sum(probs[i] * sizes[i] for i in range(len(sizes))) - 95) < 1e-9
    cnt, _, subs = pgs.collect_distinct(sizes, 95, 400, 7)
    obs = np.zeros(len(sizes), dtype=int)
    for s in subs:
        for i in s:
            obs[i] += 1
    for i in range(len(sizes)):
        p = probs[i]
        sd = math.sqrt(400 * p * (1 - p))
        # 4-SD tolerance at n=400: non-flaky, catches gross sampler bias
        assert abs(obs[i] - 400 * p) <= max(4 * sd, 1.0), (
            f"geometry {i}: obs={obs[i]} expected={400 * p:.1f}"
        )


def test_sampler_dp_matches_exhaustive_enumeration():
    """Tiny toy case: DP counts / inclusion probabilities / per-subset
    probabilities must EXACTLY match brute-force enumeration."""
    sizes = [1, 2, 2, 3, 4]
    target = 5
    n = len(sizes)

    # brute force over all 2^n subsets
    brute_subsets, brute_ng = set(), {i: 0 for i in range(n)}
    for mask in range(1 << n):
        sub = tuple(i for i in range(n) if mask >> i & 1)
        if sum(sizes[i] for i in sub) == target:
            brute_subsets.add(sub)
            for i in sub:
                brute_ng[i] += 1
    n_brute = len(brute_subsets)
    assert n_brute > 0

    # 1) DP feasible count equals brute force
    cnt = pgs.feasible_counts(sizes, target)
    assert cnt[0][target] == n_brute

    # 2) per-geometry inclusion probabilities equal brute force
    n_dp, probs = pgs.theoretical_inclusion(sizes, target)
    assert n_dp == n_brute
    for i in range(n):
        assert abs(probs[i] - brute_ng[i] / n_brute) < 1e-12

    # 3) collect_distinct with n_want == n_brute returns ALL feasible subsets
    _, n_feasible, subs = pgs.collect_distinct(sizes, target, n_brute, 0)
    assert n_feasible == n_brute
    assert set(subs) == brute_subsets

    # 4) the DP walk assigns every feasible subset probability exactly 1/N_feasible
    for sub in brute_subsets:
        p = 1.0
        v = target
        for i in range(n):
            s = sizes[i]
            if v < s:
                continue
            inc = cnt[i + 1][v - s]
            exc = cnt[i + 1][v]
            tot = inc + exc
            if tot == 0:
                continue
            if i in sub:
                p *= inc / tot
                v -= s
            else:
                p *= exc / tot
        assert abs(p - 1.0 / n_brute) < 1e-12, f"subset {sub}: walk prob {p}"


# ---------------------------------------------------------------------------
# Strict JSON serialization (NaN/inf -> null)
# ---------------------------------------------------------------------------
def test_strict_json_serialization():
    obj = {"a": float("nan"), "b": float("inf"), "c": np.float64("nan"), "d": [1, 2.5]}
    clean = aud.py_native(obj)
    assert clean["a"] is None and clean["b"] is None and clean["c"] is None
    text = json.dumps(clean, allow_nan=False)  # raises on any NaN/inf
    assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# Locked model spec
# ---------------------------------------------------------------------------
def test_model_spec_locked_literal():
    assert model_spec.LOCKED_XGB_PARAMS == dict(
        n_estimators=100,
        learning_rate=0.3,
        max_depth=6,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=0,
    )


# ---------------------------------------------------------------------------
# Metric / fit_eval guards
# ---------------------------------------------------------------------------
def test_mape_zero_guard():
    with pytest.raises(ValueError):
        cmetrics.mape_pct([0.0, 1.0], [1.0, 1.0])


def test_fit_eval_raises_on_nonfinite():
    from src.common.modeling import fit_eval

    se_df = cdata.load_se_extracted()
    bad = se_df.copy()
    bad.loc[0, "Orifice width (um)"] = np.nan
    with pytest.raises(ValueError):
        fit_eval(bad, np.arange(100, 474), np.arange(100), "T", 0, disjoint=False)


def test_fit_eval_raises_on_geometry_split():
    from src.common.modeling import fit_eval

    se_df = cdata.load_se_extracted()
    g = se_df["geometry_id"].iloc[0]
    te = np.nonzero((se_df["geometry_id"] == g).to_numpy())[0][:2]
    tr = np.setdiff1d(np.arange(len(se_df)), te)
    with pytest.raises(ValueError):
        fit_eval(se_df, tr, te, "T", 0, disjoint=True)
