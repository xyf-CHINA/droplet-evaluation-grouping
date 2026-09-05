"""
R1 audit-only post-hoc analyses — reads the EXISTING protocol outputs
(replication/talebjedi2022/outputs/*) and NEVER retrains.

1. run-mean == pooled confirmation: MAE/MAPE and the linear signed bias must
   match pooled within 1e-9 (equal test size 25); RMSE and R2 are computed
   for documentation only and are deliberately NOT asserted.
2. Post-hoc descriptive same-angle sensitivity (no retraining): each
   original sample's absolute and percentage errors are averaged across the
   random splits in which it appeared as test (seen-angle error), then
   aggregated per tilt angle and compared with the LOAO unseen-angle
   per-angle MAE/MAPE.
3. Weighting sensitivity: sample-equal Random MAE/MAPE (each sample counted
   once) vs the primary 2,500 repeated-prediction pooled values. The primary
   R metrics are NOT replaced.

All outputs are labeled post hoc descriptive; no p-values, no bootstrap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "replication" / "talebjedi2022"
OUT = BASE / "outputs"
ANGLES = [30, 60, 90, 120, 150]


def main() -> None:
    df = pd.read_csv(BASE / "reconstructed" / "talebjedi_125_reconstructed.csv")
    rp = pd.read_csv(OUT / "random_predictions.csv")
    rsm = pd.read_csv(OUT / "random_seed_metrics.csv")
    loao = pd.read_csv(OUT / "loao_angle_metrics.csv")
    metrics = json.loads((OUT / "replication_metrics.json").read_text(encoding="utf-8"))

    # ================= 1. run-mean == pooled (MAE/MAPE/bias only) ============
    r_pooled = metrics["protocol_R"]["pooled"]
    run_means = {
        "mae": float(rsm["mae"].mean()),
        "mape": float(rsm["mape"].mean()),
        "bias": float(rsm["bias"].mean()),
        "rmse": float(rsm["rmse"].mean()),
        "r2": float(rsm["r2"].mean()),
    }
    eq = {
        k: abs(run_means[k] - r_pooled[k]) < 1e-9
        for k in ("mae", "mape", "bias")
    }
    assert all(eq.values()), f"run-mean != pooled for MAE/MAPE/bias: {eq}"
    # RMSE/R2: computed for documentation only — with equal test sizes their
    # run-mean still differs from the pooled value because they are nonlinear.
    run_mean_check = {
        "asserted_for": ["mae", "mape", "bias"],
        "mae": {"run_mean": run_means["mae"], "pooled": r_pooled["mae"],
                "equal": eq["mae"]},
        "mape": {"run_mean": run_means["mape"], "pooled": r_pooled["mape"],
                 "equal": eq["mape"]},
        "bias": {"run_mean": run_means["bias"], "pooled": r_pooled["bias"],
                 "equal": eq["bias"]},
        "rmse_not_asserted": {"run_mean": run_means["rmse"],
                              "pooled": r_pooled["rmse"]},
        "r2_not_asserted": {"run_mean": run_means["r2"],
                            "pooled": r_pooled["r2"]},
        "note": "equal per-run test size (25) makes MAE/MAPE/bias linear means "
                "match pooled exactly; RMSE and R2 are nonlinear and must NOT "
                "be assumed equal.",
    }

    # ============ 2. post-hoc same-angle sensitivity (no retraining) =========
    assert len(rp) == 2500
    # each sample is test in ~20 of the 100 splits on average (binomial over
    # seeds); the mean must be exactly 20 = 2500 / 125
    appearances = rp.groupby("ExpNo").size()
    assert abs(appearances.mean() - 20.0) < 1e-12
    assert (appearances > 0).all() and (appearances < 100).all()
    rp = rp.copy()
    rp["abs_err"] = (rp["observed_um"] - rp["predicted_um"]).abs()
    rp["pct_err"] = rp["abs_err"] / rp["observed_um"] * 100
    sample = (rp.groupby("ExpNo")
                .agg(abs_err_mean=("abs_err", "mean"), pct_err_mean=("pct_err", "mean"))
                .reset_index()
                .merge(df[["ExpNo", "TiltAngle_deg"]], on="ExpNo"))
    assert len(sample) == 125
    seen = (sample.groupby("TiltAngle_deg")
                  .agg(n=("ExpNo", "size"),
                       seen_mae=("abs_err_mean", "mean"),
                       seen_mape=("pct_err_mean", "mean"))
                  .reindex(ANGLES)
                  .reset_index())
    assert list(seen["TiltAngle_deg"]) == ANGLES
    assert (seen["n"] == 25).all()

    unseen = loao.set_index("held_out_angle").loc[ANGLES][["mae", "mape"]]
    table = pd.DataFrame({
        "TiltAngle_deg": ANGLES,
        "n": 25,
        "seen_mae": seen["seen_mae"].round(4).to_numpy(),
        "seen_mape": seen["seen_mape"].round(4).to_numpy(),
        "unseen_mae": unseen["mae"].round(4).to_numpy(),
        "unseen_mape": unseen["mape"].round(4).to_numpy(),
    })
    table["delta_mae_unseen_minus_seen"] = (
        table["unseen_mae"] - table["seen_mae"]).round(4)
    table["delta_mape_unseen_minus_seen"] = (
        table["unseen_mape"] - table["seen_mape"]).round(4)
    table.to_csv(OUT / "r1_same_angle_sensitivity.csv", index=False)

    # ============ 3. weighting sensitivity (sample-equal vs pooled) ==========
    sample_equal = {
        "mae": float(sample["abs_err_mean"].mean()),
        "mape": float(sample["pct_err_mean"].mean()),
    }
    weighting = {
        "sample_equal_random_mae": sample_equal["mae"],
        "sample_equal_random_mape": sample_equal["mape"],
        "primary_pooled_random_mae": r_pooled["mae"],
        "primary_pooled_random_mape": r_pooled["mape"],
        "note": "weighting sensitivity only; the 2,500 repeated-prediction "
                "pooled values remain the pre-locked primary R metrics.",
    }

    summary = {
        "label": "R1 post hoc descriptive sensitivity (audit-only; no retraining)",
        "run_mean_equals_pooled": run_mean_check,
        "same_angle_table": table.round(4).to_dict(orient="records"),
        "weighting_sensitivity": weighting,
        "no_pvalues_no_bootstrap": True,
    }
    (OUT / "r1_posthoc_sensitivity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("[r1] same-angle table written: outputs/r1_same_angle_sensitivity.csv")
    print("[r1] run-mean==pooled (MAE/MAPE/bias):",
          all(eq.values()), "| RMSE/R2 run-mean vs pooled recorded, not asserted")
    print("[r1] weighting sensitivity: sample-equal MAE %.4f vs pooled %.4f"
          % (sample_equal["mae"], r_pooled["mae"]))
    print("[r1] STOP — descriptive only, primary metrics unchanged.")


if __name__ == "__main__":
    main()
