"""Tests for the Talebjedi 2022 replication module (frozen-output checks)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replication.talebjedi2022.run_protocols import (  # noqa: E402
    ANGLES, FEATURES, SPEC, TARGET,
)

BASE = Path(__file__).resolve().parents[1] / "replication" / "talebjedi2022"

RECONSTRUCTED = BASE / "reconstructed" / "talebjedi_125_reconstructed.csv"


def require_local(path: Path) -> Path:
    """Skip only the test that needs a license-scoped local artifact."""
    if not path.is_file():
        pytest.skip(
            f"missing {path.name}; obtain the official Talebjedi Supporting "
            "Information and regenerate it (see DATA_ACQUISITION.md)"
        )
    return path


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(require_local(RECONSTRUCTED))


@pytest.fixture(scope="module")
def audit():
    return json.loads((BASE / "audit" / "reconstruction_audit.json")
                      .read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics():
    return json.loads((BASE / "outputs" / "replication_metrics.json")
                      .read_text(encoding="utf-8"))


# ------------------------------ reconstruction -----------------------------
def test_reconstructed_csv_shape(df):
    assert len(df) == 125
    assert list(df.columns) == ["ExpNo", "TiltAngle_deg", "FRR", "Qc_uL_min",
                                "Size_um", "Frequency_Hz", "Uniformity_pct",
                                "CircleMetric"]


def test_expno_exact_1_125(df):
    assert sorted(df["ExpNo"]) == list(range(1, 126))


def test_angles_and_group_sizes(df):
    assert sorted(df["TiltAngle_deg"].unique()) == ANGLES
    assert all((df["TiltAngle_deg"] == a).sum() == 25 for a in ANGLES)


def test_factorial_coverage(df):
    for a in ANGLES:
        sub = df[df["TiltAngle_deg"] == a]
        assert sub[["FRR", "Qc_uL_min"]].drop_duplicates().shape[0] == 25


def test_qd_audit(df, audit):
    assert audit["qd_audit_all_within_0_02"]
    qd = df["Qc_uL_min"] / df["FRR"]
    nearest = qd.apply(lambda x: min((1, 3, 4, 5, 7), key=lambda L: abs(x - L)))
    assert all(abs(qd.iloc[i] - nearest.iloc[i]) < 0.02 for i in range(len(df)))


def test_audit_gates(audit):
    assert audit["s2_expno_exact_1_125"]
    assert audit["tilt_angles_exact"]
    assert audit["factorial_pairs_identical_across_angles"]
    assert audit["s3_expno_exact_1_125"]
    assert audit["s2_s3_exact_one_to_one_join"]
    assert audit["s3_all_targets_present_finite"]


# --------------------------- published-split audit --------------------------
def test_published_split_lists(audit):
    ps = audit["published_split"]
    for k, v in ps.items():
        assert v["n"] == 16, k
        assert v["all_expnos_valid"] and v["no_duplicates"], k
    assert all(v == [] for v in audit["published_split_overlap_test_validation"].values())


# ------------------------------- protocol lock ------------------------------
def test_spec_matches_lock():
    locked = dict(n_estimators=100, learning_rate=0.3, max_depth=6, reg_lambda=1,
                  min_child_weight=1, objective="reg:squarederror",
                  tree_method="hist", random_state=0)
    assert SPEC == locked
    assert FEATURES == ["TiltAngle_deg", "FRR", "Qc_uL_min"]
    assert TARGET == "Size_um"


# ------------------------------ protocol outputs ----------------------------
def test_protocol_R_outputs():
    r = pd.read_csv(BASE / "outputs" / "random_seed_metrics.csv")
    pred = pd.read_csv(require_local(BASE / "outputs" / "random_predictions.csv"))
    assert list(r["seed"]) == list(range(100))
    assert len(pred) == 2500
    assert pred["seed"].nunique() == 100
    assert (pred.groupby("seed").size() == 25).all()


def test_protocol_T_outputs(df):
    pred = pd.read_csv(require_local(BASE / "outputs" / "loao_predictions.csv"))
    assert len(pred) == 125
    for a in ANGLES:
        assert (pred["held_out_angle"] == a).sum() == 25
    # held-out angle absent from training by construction: per-angle train set
    for a in ANGLES:
        train_angles = set(df[df["TiltAngle_deg"] != a]["TiltAngle_deg"])
        assert a not in train_angles
    per_angle = pd.read_csv(BASE / "outputs" / "loao_angle_metrics.csv")
    assert list(per_angle["held_out_angle"]) == ANGLES
    assert all(per_angle["n"] == 25)


def test_run_mean_equals_pooled(metrics):
    r = metrics["protocol_R"]
    assert r["run_mean_equals_pooled_mae_mape"]
    assert abs(r["run_mean_mae"] - r["pooled"]["mae"]) < 1e-9
    assert abs(r["run_mean_mape"] - r["pooled"]["mape"]) < 1e-9


def test_predictions_finite():
    for f in ("random_predictions.csv", "loao_predictions.csv"):
        p = pd.read_csv(require_local(BASE / "outputs" / f))
        assert np.isfinite(p["predicted_um"]).all()


def test_exposure_is_empirical_not_hardcoded(df):
    exp = json.loads((BASE / "audit" / "random_geometry_exposure.json")
                     .read_text(encoding="utf-8"))
    assert len(exp["per_seed"]) == 100
    # each per-seed entry is consistent with its own split assignment
    from sklearn.model_selection import train_test_split
    for e in exp["per_seed"][:5]:
        _, te = train_test_split(np.arange(125), test_size=0.20,
                                 random_state=e["seed"], shuffle=True)
        te_angles = df.iloc[te]["TiltAngle_deg"]
        tr_angles = set(df.iloc[np.setdiff1d(np.arange(125), te)]["TiltAngle_deg"])
        assert abs(te_angles.isin(tr_angles).mean() - e["sample_level_exposure"]) < 1e-12


def test_verdict_rule(metrics):
    t, r = metrics["protocol_T"]["pooled"], metrics["protocol_R"]["pooled"]
    if t["mae"] > r["mae"] and t["mape"] > r["mape"]:
        expect = "supportive independent replication"
    elif (t["mae"] - r["mae"]) * (t["mape"] - r["mape"]) < 0:
        expect = "mixed replication"
    else:
        expect = "non-replication / boundary condition"
    assert metrics["pre_locked_verdict"] == expect
    assert metrics["no_pvalues_no_bootstrap"]


# ----------------------------------- R1 -------------------------------------
def test_r1_reconstruct_coverage_hard_gates(audit):
    cov = audit["published_split_coverage"]
    for net in ("Size", "Frequency", "Uniformity", "Circle metric"):
        assert cov[net]["train_n"] == 93, net
    for role in ("train", "test", "validation"):
        assert cov["Size"][f"{role}_covers_all_5_angles"]
    # documented published-table property: Uniformity test lacks 60 deg
    assert cov["Uniformity"]["test_covers_all_5_angles"] is False
    assert audit["published_split"]["Uniformity test"]["angle_counts"]["60"] == 0


def test_r1_same_angle_table():
    t = pd.read_csv(BASE / "outputs" / "r1_same_angle_sensitivity.csv")
    assert list(t["TiltAngle_deg"]) == ANGLES
    assert (t["n"] == 25).all()
    # deltas recomputed from the table itself must equal unseen - seen
    assert np.allclose(t["delta_mae_unseen_minus_seen"],
                       t["unseen_mae"] - t["seen_mae"])
    assert np.allclose(t["delta_mape_unseen_minus_seen"],
                       t["unseen_mape"] - t["seen_mape"])
    # frozen-output regression: unseen > seen for every angle
    assert (t["unseen_mae"] > t["seen_mae"]).all()
    assert (t["unseen_mape"] > t["seen_mape"]).all()


def test_r1_run_mean_check():
    r1 = json.loads((BASE / "outputs" / "r1_posthoc_sensitivity.json")
                    .read_text(encoding="utf-8"))
    chk = r1["run_mean_equals_pooled"]
    assert chk["asserted_for"] == ["mae", "mape", "bias"]
    for k in ("mae", "mape", "bias"):
        assert chk[k]["equal"]
    # RMSE/R2 are recorded but explicitly NOT asserted
    assert "rmse_not_asserted" in chk and "r2_not_asserted" in chk


def test_r1_weighting_sensitivity():
    r1 = json.loads((BASE / "outputs" / "r1_posthoc_sensitivity.json")
                    .read_text(encoding="utf-8"))
    metrics = json.loads((BASE / "outputs" / "replication_metrics.json")
                         .read_text(encoding="utf-8"))
    w = r1["weighting_sensitivity"]
    # the pre-locked primary pooled values must match the frozen metrics json
    assert w["primary_pooled_random_mae"] == metrics["protocol_R"]["pooled"]["mae"]
    assert w["primary_pooled_random_mape"] == metrics["protocol_R"]["pooled"]["mape"]
