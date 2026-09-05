"""
Talebjedi 2022 replication — Protocol R (random sample-wise) and
Protocol T (leave-one-tilt-angle-out), exactly per
replication/talebjedi2022/PROTOCOL_LOCK.md.

Design is fully locked before this script was run. No p-values, no bootstrap,
no tuning, no extra features/models. The pre-locked interpretation rule is
applied mechanically and reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "replication" / "talebjedi2022"
DATA_CSV = BASE / "reconstructed" / "talebjedi_125_reconstructed.csv"
OUT = BASE / "outputs"
AUDIT = BASE / "audit"

SPEC = dict(n_estimators=100, learning_rate=0.3, max_depth=6, reg_lambda=1,
            min_child_weight=1, objective="reg:squarederror", tree_method="hist",
            random_state=0)
FEATURES = ["TiltAngle_deg", "FRR", "Qc_uL_min"]
TARGET = "Size_um"
ANGLES = [30, 60, 90, 120, 150]
N_SPLITS = 100


def mape(y, p):
    return float(np.mean(np.abs(y - p) / np.abs(y)) * 100)


def bias(y, p):
    return float(np.mean(y - p))


def fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    model = XGBRegressor(**SPEC)
    model.fit(scaler.transform(Xtr), ytr)
    pred = model.predict(scaler.transform(Xte))
    return pred, model


def metric_block(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "mape": mape(y, p),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)),
        "bias": bias(y, p),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_CSV)

    # ---------- locked-input gates ----------
    assert len(df) == 125, f"expected 125 rows, got {len(df)}"
    assert sorted(df["TiltAngle_deg"].unique()) == ANGLES
    assert all((df["TiltAngle_deg"] == a).sum() == 25 for a in ANGLES)
    assert list(df.columns) == ["ExpNo", "TiltAngle_deg", "FRR", "Qc_uL_min",
                                "Size_um", "Frequency_Hz", "Uniformity_pct",
                                "CircleMetric"]

    # ========================= Protocol R ====================================
    exposure, r_rows, r_metrics = [], [], []
    for seed in range(N_SPLITS):
        tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.20,
                                          random_state=seed, shuffle=True)
        assert len(tr_idx) == 100 and len(te_idx) == 25
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        # exposure audit — computed from the split assignments BEFORE fitting
        tr_angles = set(tr["TiltAngle_deg"])
        te_angles = te["TiltAngle_deg"]
        exposure.append({
            "seed": seed,
            "sample_level_exposure": float(te_angles.isin(tr_angles).mean()),
            "geometry_level_exposure": float(
                te_angles[te_angles.isin(tr_angles)].nunique() / te_angles.nunique()
            ),
            "n_test_angles_seen": int(te_angles[te_angles.isin(tr_angles)].nunique()),
            "n_test_angles": int(te_angles.nunique()),
        })
        pred, model = fit_predict(tr[FEATURES], tr[TARGET], te[FEATURES])
        for k, v in SPEC.items():
            assert model.get_params()[k] == v, f"spec drift: {k}"
        y, p = te[TARGET].to_numpy(), pred
        m = metric_block(y, p)
        m["seed"] = seed
        r_metrics.append(m)
        for expno, obs, pr in zip(te["ExpNo"], y, p):
            r_rows.append({"ExpNo": int(expno), "seed": seed,
                           "observed_um": float(obs), "predicted_um": float(pr)})

    r_rows_df = pd.DataFrame(r_rows)
    r_metrics_df = pd.DataFrame(r_metrics)
    r_pooled = metric_block(r_rows_df["observed_um"], r_rows_df["predicted_um"])
    # equal test size (25) -> unweighted run means must equal pooled for
    # MAE/MAPE within numerical tolerance
    eq_mae = abs(float(r_metrics_df["mae"].mean()) - r_pooled["mae"]) < 1e-9
    eq_mape = abs(float(r_metrics_df["mape"].mean()) - r_pooled["mape"]) < 1e-9
    assert eq_mae and eq_mape, "run-mean vs pooled mismatch (should be exact)"

    # ========================= Protocol T ====================================
    t_rows, t_metrics = [], []
    for a in ANGLES:
        tr = df[df["TiltAngle_deg"] != a]
        te = df[df["TiltAngle_deg"] == a]
        assert len(tr) == 100 and len(te) == 25
        assert set(tr["TiltAngle_deg"]).isdisjoint({a}), "angle overlap != 0"
        pred, model = fit_predict(tr[FEATURES], tr[TARGET], te[FEATURES])
        y, p = te[TARGET].to_numpy(), pred
        m = metric_block(y, p)
        m["held_out_angle"] = a
        m["n"] = int(len(y))
        t_metrics.append(m)
        for expno, obs, pr in zip(te["ExpNo"], y, p):
            t_rows.append({"ExpNo": int(expno), "held_out_angle": a,
                           "observed_um": float(obs), "predicted_um": float(pr)})

    t_rows_df = pd.DataFrame(t_rows)
    t_pooled = metric_block(t_rows_df["observed_um"], t_rows_df["predicted_um"])
    assert len(t_rows_df) == 125

    # ---------- finite-prediction gate ----------
    assert np.isfinite(r_rows_df["predicted_um"]).all()
    assert np.isfinite(t_rows_df["predicted_um"]).all()

    # ---------- pre-locked interpretation (mechanical, descriptive) ----------
    r_mae, r_mape = r_pooled["mae"], r_pooled["mape"]
    t_mae, t_mape = t_pooled["mae"], t_pooled["mape"]
    if t_mae > r_mae and t_mape > r_mape:
        verdict = "supportive independent replication"
    elif (t_mae - r_mae) * (t_mape - r_mape) < 0:
        verdict = "mixed replication"
    else:
        verdict = "non-replication / boundary condition"

    summary = {
        "protocol": "talebjedi2022 replication (per PROTOCOL_LOCK.md)",
        "protocol_R": {
            "n_splits": N_SPLITS,
            "pooled": r_pooled,
            "run_mean_mae": float(r_metrics_df["mae"].mean()),
            "run_mean_mape": float(r_metrics_df["mape"].mean()),
            "run_mean_equals_pooled_mae_mape": bool(eq_mae and eq_mape),
            "exposure_mean_sample_level": float(
                np.mean([e["sample_level_exposure"] for e in exposure])),
            "exposure_mean_geometry_level": float(
                np.mean([e["geometry_level_exposure"] for e in exposure])),
        },
        "protocol_T": {
            "pooled": t_pooled,
            "per_angle": t_metrics,
            "angle_overlap_zero_all_folds": True,
        },
        "pre_locked_verdict": verdict,
        "no_pvalues_no_bootstrap": True,
    }

    # ---------- outputs ----------
    r_metrics_df.to_csv(OUT / "random_seed_metrics.csv", index=False)
    r_rows_df.to_csv(OUT / "random_predictions.csv", index=False)
    pd.DataFrame(t_metrics).to_csv(OUT / "loao_angle_metrics.csv", index=False)
    t_rows_df.to_csv(OUT / "loao_predictions.csv", index=False)
    (OUT / "replication_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (AUDIT / "random_geometry_exposure.json").write_text(
        json.dumps({"per_seed": exposure,
                    "summary": summary["protocol_R"]["exposure_mean_sample_level"],
                    "note": "computed from split assignments before any fit"},
                   indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"[run_protocols] R pooled MAE {r_mae:.4f} MAPE {r_mape:.4f}% | "
          f"T pooled MAE {t_mae:.4f} MAPE {t_mape:.4f}%")
    print(f"[run_protocols] R exposure: sample {summary['protocol_R']['exposure_mean_sample_level']:.4f} "
          f"geometry {summary['protocol_R']['exposure_mean_geometry_level']:.4f}")
    print(f"[run_protocols] pre-locked verdict: {verdict}")
    print("[run_protocols] outputs written.")


if __name__ == "__main__":
    main()
