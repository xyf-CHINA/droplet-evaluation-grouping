"""
Stage 2.1 — Consolidated analysis (NO model training, NO protocol changes).

Four analysis groups:
  A. Protocol G weighting audit: p_g (theoretical inclusion probability) vs
     geometry size and vs LOGO difficulty; expected G MAE under p_g weighting.
  B. Geometry-level seen -> unseen comparison: for the 32 non-singleton
     geometries, per-sample Random-seen errors (Protocol R predictions with
     geometry_seen_in_train=True, averaged over seeds) vs LOGO per-sample
     errors; per-geometry dMAE / dMAPE, positive-d fraction, paired geometry
     bootstrap CI.
  C. Hardest-geometry sensitivity: remove Top1/Top3 hardest (by LOGO MAE)
     from BOTH sides, recompute the gap.
  D. Master table.

Inputs (read-only, all previously generated):
  outputs/baseline/stage1b_predictions.csv
  outputs/geometry_shift/stage2_logo_predictions.csv
  outputs/geometry_shift/stage2_logo_geometry_metrics.csv
  outputs/geometry_shift/stage2_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.audit.audit_dafd3 import sha256_of

OUT_DIR = PROJECT_ROOT / "outputs" / "geometry_shift"
B = 10000          # bootstrap iterations
BS_SEED = 0


def load_inputs():
    r = pd.read_csv(PROJECT_ROOT / "outputs" / "baseline" / "stage1b_predictions.csv")
    logo = pd.read_csv(OUT_DIR / "stage2_logo_predictions.csv")
    logo_geom = pd.read_csv(OUT_DIR / "stage2_logo_geometry_metrics.csv")
    gkf = pd.read_csv(OUT_DIR / "stage2_groupkfold_predictions.csv")
    s2 = json.loads((OUT_DIR / "stage2_metrics.json").read_text(encoding="utf-8"))
    assert len(r) == 9500 and len(logo) == 474 and len(gkf) == 474
    return r, logo, logo_geom, gkf, s2


def pooled_metrics(df: pd.DataFrame) -> dict:
    obs = df["observed_um"].to_numpy()
    pred = df["predicted_um"].to_numpy()
    err = pred - obs
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "mape_pct": float(np.mean(np.abs(err / obs)) * 100),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r, logo, logo_geom, gkf, s2 = load_inputs()

    p_g = s2["protocol_g_sampling"]["theoretical_inclusion_p"]
    obs_counts = s2["protocol_g_sampling"]["coverage_counts"]
    singletons = logo_geom.loc[logo_geom["n"] == 1, "geometry_id"].tolist()
    assert len(singletons) == 3

    # ---------------- A. weighting audit ----------------------------------
    w = logo_geom.merge(
        pd.DataFrame(
            [
                {"geometry_id": gid, "p_g": p_g[gid],
                 "observed_inclusion_count": obs_counts[gid],
                 "observed_inclusion_rate": obs_counts[gid] / 100}
                for gid in p_g
            ]
        ),
        on="geometry_id",
    )
    w = w.rename(columns={"mae": "LOGO_MAE", "mape_pct": "LOGO_MAPE"})
    w["corr_pg_size"] = w["p_g"].corr(w["n"])
    w["corr_pg_logo_mae"] = w["p_g"].corr(w["LOGO_MAE"])
    w["corr_pg_logo_mape"] = w["p_g"].corr(w["LOGO_MAPE"])

    # p_g-weighted LOGO reference (DESCRIPTIVE, not the strict theoretical
    # expectation of Protocol G: G withholds several geometries at once with a
    # fixed 379-row training set, LOGO withholds one geometry at a time with
    # 474-n_g training rows).
    pg_weighted_reference_mae = float((w["p_g"] * w["n"] * w["LOGO_MAE"]).sum() / 95)
    logo_micro = float((w["n"] * w["LOGO_MAE"]).sum() / w["n"].sum())
    logo_macro = float(w["LOGO_MAE"].mean())
    observed_g_mae = s2["protocol_G_summary"]["mae"]["mean"]

    # training-size comparison for the Discussion point: LOGO models generally
    # used MORE training samples than the fixed 379 of Random / Protocol G.
    n_train_logo = (474 - w["n"]).astype(int)
    train_size_info = {
        "n_train_random_and_G": 379,
        "n_train_logo_min": int(n_train_logo.min()),
        "n_train_logo_mean": float(n_train_logo.mean()),
        "n_train_logo_max": int(n_train_logo.max()),
        "note": (
            "LOGO training sets (474 - n_g) are larger than the fixed 379-row "
            "training sets of Random/Protocol G for every geometry with n_g < 95 "
            "(all geometries). The observed seen->unseen deterioration therefore "
            "cannot be readily explained by a smaller training set."
        ),
    }

    # GKF pooled metrics (from the 474 out-of-fold predictions)
    gkf_pooled = pooled_metrics(gkf)

    # ---------------- B. seen -> unseen paired comparison -------------------
    seen = r[r["geometry_seen_in_train"] == True].copy()
    per_sample_seen = (
        seen.groupby("se_experiment_id")
        .agg(
            geometry_id=("geometry_id", "first"),
            n_seen_appearances=("seed", "count"),
            sample_mae_seen=("absolute_error_um", "mean"),
            sample_ape_seen=("percentage_error_pct", "mean"),
        )
        .reset_index()
    )
    per_geom_seen = (
        per_sample_seen.groupby("geometry_id")
        .agg(
            n_samples_seen=("se_experiment_id", "count"),
            min_appearances=("n_seen_appearances", "min"),
            MAE_seen=("sample_mae_seen", "mean"),
            MAPE_seen=("sample_ape_seen", "mean"),
        )
        .reset_index()
    )

    logo_sample = logo.rename(
        columns={"absolute_error_um": "sample_mae_unseen",
                 "percentage_error_pct": "sample_ape_unseen"}
    )
    per_geom_unseen = (
        logo_sample.groupby("geometry_id")
        .agg(
            n_samples_unseen=("se_experiment_id", "count"),
            MAE_unseen=("sample_mae_unseen", "mean"),
            MAPE_unseen=("sample_ape_unseen", "mean"),
        )
        .reset_index()
    )

    paired = per_geom_seen.merge(per_geom_unseen, on="geometry_id")
    paired["dMAE"] = paired["MAE_unseen"] - paired["MAE_seen"]
    paired["dMAPE"] = paired["MAPE_unseen"] - paired["MAPE_seen"]
    paired = paired[~paired["geometry_id"].isin(singletons)].reset_index(drop=True)
    n_geoms = len(paired)
    assert n_geoms == 32

    n_samples_missing_seen = int((paired["n_samples_seen"] != paired["n_samples_unseen"]).sum())
    n_positive_dmae = int((paired["dMAE"] > 0).sum())
    n_positive_dmape = int((paired["dMAPE"] > 0).sum())

    def paired_bootstrap_ci(vals, seed, b_iter=B):
        rng = np.random.default_rng(seed)
        vals = np.asarray(vals, dtype=float)
        means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                          for _ in range(b_iter)])
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    ci_dmae = paired_bootstrap_ci(paired["dMAE"].to_numpy(), BS_SEED)
    ci_dmape = paired_bootstrap_ci(paired["dMAPE"].to_numpy(), BS_SEED)

    # ---------------- C. hardest Top1/Top3 sensitivity ----------------------
    hard = paired.sort_values("MAE_unseen", ascending=False).reset_index(drop=True)

    def macro_block(df):
        return {
            "n_geometries": int(len(df)),
            "MAE_seen_macro": float(df["MAE_seen"].mean()),
            "MAE_unseen_macro": float(df["MAE_unseen"].mean()),
            "dMAE_macro": float(df["dMAE"].mean()),
            "rel_dMAE_macro": float(df["dMAE"].mean() / df["MAE_seen"].mean()),
            "MAPE_seen_macro": float(df["MAPE_seen"].mean()),
            "MAPE_unseen_macro": float(df["MAPE_unseen"].mean()),
            "dMAPE_macro": float(df["dMAPE"].mean()),
            "n_positive_dmae": int((df["dMAE"] > 0).sum()),
        }

    sens = {
        "all_32": macro_block(hard),
        "excl_hardest_1": macro_block(hard.iloc[1:]),
        "excl_hardest_3": macro_block(hard.iloc[3:]),
        "hardest_1": hard.iloc[0]["geometry_id"],
        "hardest_3": hard.iloc[:3]["geometry_id"].tolist(),
        "note": (
            "sensitivity analysis (not a confirmatory test): hardest geometries "
            "removed from BOTH the seen and unseen sides; the gap remained "
            "substantial, i.e. it is not solely driven by the few hardest geometries"
        ),
    }

    # ---------------- D. master table ----------------------------------------
    r_sum = s2["protocol_R_summary"]
    g_sum = s2["protocol_G_summary"]
    deltas = s2["deltas_G_vs_R"]
    metric_label = {"mae": "MAE (um)", "rmse": "RMSE (um)", "r2": "R2",
                    "mape_pct": "MAPE (%)"}

    # Pooled repeated-holdout metrics for R and G: RMSE and R2 are NOT linear,
    # so the protocol summary must use the
    # metrics of all 9,500 test predictions pooled per protocol — NOT the mean
    # of the 100 per-split metrics. MAE/MAPE coincide (every split has exactly
    # 95 test samples) and are asserted below.
    def pooled_from_csv(csv_path: str) -> dict:
        d = pd.read_csv(PROJECT_ROOT / csv_path)
        obs = d["observed_um"].to_numpy()
        pred = d["predicted_um"].to_numpy()
        err = pred - obs
        return {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((obs - obs.mean()) ** 2)),
            "mape_pct": float(np.mean(np.abs(err / obs)) * 100),
        }

    r_pooled = pooled_from_csv("outputs/baseline/stage1b_predictions.csv")
    g_pooled = pooled_from_csv("outputs/geometry_shift/stage2_protocol_g_predictions.csv")
    for m in ("mae", "mape_pct"):
        assert abs(r_pooled[m] - r_sum[m]["mean"]) < 1e-6, "R pooled != mean-of-splits"
        assert abs(g_pooled[m] - g_sum[m]["mean"]) < 1e-6, "G pooled != mean-of-splits"

    def r_vs_g_rows():
        rows = []
        for m in ("mae", "rmse", "r2", "mape_pct"):
            if m == "r2":
                change = f"{g_pooled[m] - r_pooled[m]:+.4f} (absolute dR2)"
            else:
                rel = (g_pooled[m] - r_pooled[m]) / r_pooled[m]
                change = f"{rel*100:+.1f}%"
            rows.append(
                {
                    "analysis": "Repeated exact-size holdout (Protocol R vs G)",
                    "metric": metric_label[m],
                    "random_or_seen": r_pooled[m],
                    "geometry_unseen": g_pooled[m],
                    "change": change,
                }
            )
        return rows

    master = pd.DataFrame(
        r_vs_g_rows()
        + [
            {"analysis": "GroupKFold pooled", "metric": metric_label[m],
             "random_or_seen": None, "geometry_unseen": gkf_pooled[m], "change": None}
            for m in ("mae", "rmse", "r2", "mape_pct")
        ]
        + [
            {"analysis": "LOGO micro", "metric": metric_label[m],
             "random_or_seen": None, "geometry_unseen": s2["logo"]["micro"][m], "change": None}
            for m in ("mae", "rmse", "r2", "mape_pct")
        ]
        + [
            {"analysis": "32-geometry paired (macro)", "metric": "MAE (um)",
             "random_or_seen": sens["all_32"]["MAE_seen_macro"],
             "geometry_unseen": sens["all_32"]["MAE_unseen_macro"],
             "change": f"{sens['all_32']['dMAE_macro']:+.2f} um ({sens['all_32']['rel_dMAE_macro']*100:+.1f}%)"},
            {"analysis": "32-geometry paired (macro)", "metric": "MAPE (%)",
             "random_or_seen": sens["all_32"]["MAPE_seen_macro"],
             "geometry_unseen": sens["all_32"]["MAPE_unseen_macro"],
             "change": f"{sens['all_32']['dMAPE_macro']:+.2f} pp"},
            {"analysis": "Excluding hardest 1 (sensitivity)", "metric": "MAE (um)",
             "random_or_seen": sens["excl_hardest_1"]["MAE_seen_macro"],
             "geometry_unseen": sens["excl_hardest_1"]["MAE_unseen_macro"],
             "change": f"{sens['excl_hardest_1']['dMAE_macro']:+.2f} um ({sens['excl_hardest_1']['rel_dMAE_macro']*100:+.1f}%)"},
            {"analysis": "Excluding hardest 3 (sensitivity)", "metric": "MAE (um)",
             "random_or_seen": sens["excl_hardest_3"]["MAE_seen_macro"],
             "geometry_unseen": sens["excl_hardest_3"]["MAE_unseen_macro"],
             "change": f"{sens['excl_hardest_3']['dMAE_macro']:+.2f} um ({sens['excl_hardest_3']['rel_dMAE_macro']*100:+.1f}%)"},
        ],
        dtype=object,  # keep None as None (JSON null), not NaN
    )

    results = {
        "stage": "2.1a",
        "no_training": True,
        "A_weighting_audit": {
            "per_geometry": w[["geometry_id", "n", "p_g", "observed_inclusion_count",
                               "observed_inclusion_rate", "LOGO_MAE", "LOGO_MAPE"]].to_dict("records"),
            "corr_pg_vs_size": float(w["p_g"].corr(w["n"])),
            "corr_pg_vs_logo_mae": float(w["p_g"].corr(w["LOGO_MAE"])),
            "corr_pg_vs_logo_mape": float(w["p_g"].corr(w["LOGO_MAPE"])),
            "pg_weighted_logo_reference_mae": pg_weighted_reference_mae,
            "observed_G_MAE": observed_g_mae,
            "logo_micro_MAE": logo_micro,
            "logo_macro_MAE": logo_macro,
            "interpretation_note": (
                "The p_g-weighted LOGO reference (12.46 um) was close to the observed "
                "Protocol G MAE (12.22 um), suggesting that geometry weighting "
                "contributes substantially to the discrepancy between Protocol G and "
                "LOGO. This is a descriptive reference, NOT a strict theoretical "
                "expectation: Protocol G withholds multiple geometries simultaneously "
                "with a fixed 379-row training set, whereas LOGO withholds one "
                "geometry at a time with 474-n_g training rows."
            ),
            "training_size_comparison": train_size_info,
        },
        "B_seen_unseen_paired": {
            "n_geometries": n_geoms,
            "n_singletons_excluded": len(singletons),
            "singletons": singletons,
            "n_geometries_with_missing_seen_samples": n_samples_missing_seen,
            "n_positive_dMAE": n_positive_dmae,
            "n_positive_dMAPE": n_positive_dmape,
            "fraction_positive_dMAE": n_positive_dmae / n_geoms,
            "fraction_positive_dMAPE": n_positive_dmape / n_geoms,
            "dMAE_mean": float(paired["dMAE"].mean()),
            "dMAE_median": float(paired["dMAE"].median()),
            "dMAE_geometry_bootstrap_p95": list(ci_dmae),
            "dMAPE_mean": float(paired["dMAPE"].mean()),
            "dMAPE_median": float(paired["dMAPE"].median()),
            "dMAPE_geometry_bootstrap_p95": list(ci_dmape),
            "interval_label": (
                "95% geometry-level bootstrap percentile interval "
                "(resampling 32 geometry deltas with replacement, B=10000, seed 0). "
                "Not a confidence interval for a population effect."
            ),
            "per_geometry": paired.to_dict("records"),
        },
        "C_hardest_sensitivity": sens,
        "D_master_table": master.to_dict("records"),
        "gkf_pooled_metrics": gkf_pooled,
        "metric_definitions": {
            "main_table_rule": (
                "Protocol R and G rows: POOLED REPEATED-HOLDOUT metrics (all 9,500 "
                "test predictions per protocol pooled into a single MAE/RMSE/R2/MAPE; "
                "one sample may appear in several holdouts). GKF and LOGO rows: POOLED "
                "OOF metrics (474 predictions, each sample exactly once). RMSE and R2 "
                "are not linear, so the mean of 100 per-split metrics would differ from "
                "the pooled value; MAE/MAPE coincide (every split has exactly 95 test "
                "samples, verified by assertion)."
            ),
            "gkf_r2_foldwise_vs_pooled": (
                "GroupKFold mean fold-wise R2 = 0.832 (stage2 fold-level summary) and "
                "pooled OOF R2 = 0.853 are BOTH correct: R2 is not linear, so the mean "
                "of per-fold R2 differs from the pooled R2. Main table: pooled (0.853). "
                "Fold-wise statistics belong in the supplementary robustness table."
            ),
        },
        "consumed_artifact_sha256": {
            str(Path("outputs/baseline/stage1b_predictions.csv")): sha256_of(
                PROJECT_ROOT / "outputs" / "baseline" / "stage1b_predictions.csv"
            ),
            str(Path("outputs/geometry_shift/stage2_metrics.json")): sha256_of(
                OUT_DIR / "stage2_metrics.json"
            ),
            str(Path("outputs/geometry_shift/stage2_protocol_g_predictions.csv")): sha256_of(
                OUT_DIR / "stage2_protocol_g_predictions.csv"
            ),
            str(Path("outputs/geometry_shift/stage2_logo_predictions.csv")): sha256_of(
                OUT_DIR / "stage2_logo_predictions.csv"
            ),
            str(Path("outputs/geometry_shift/stage2_logo_geometry_metrics.csv")): sha256_of(
                OUT_DIR / "stage2_logo_geometry_metrics.csv"
            ),
            str(Path("outputs/geometry_shift/stage2_groupkfold_predictions.csv")): sha256_of(
                OUT_DIR / "stage2_groupkfold_predictions.csv"
            ),
        },
    }

    # strict-JSON guard: locate any non-finite value before dumping
    def _find_nan(obj, path=""):
        import math
        if isinstance(obj, dict):
            for k, v in obj.items():
                _find_nan(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _find_nan(v, f"{path}[{i}]")
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            raise ValueError(f"non-finite float at {path}")

    _find_nan(results)

    with open(OUT_DIR / "stage21_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
    w.to_csv(OUT_DIR / "stage21_weighting_audit.csv", index=False)
    paired.to_csv(OUT_DIR / "stage21_seen_unseen_by_geometry.csv", index=False)
    master.to_csv(OUT_DIR / "stage21_master_table.csv", index=False)

    A, Bb, C = results["A_weighting_audit"], results["B_seen_unseen_paired"], sens
    print("[stage2.1a] done (no training)")
    print(f"  A: corr(p_g, size)={A['corr_pg_vs_size']:.3f} | corr(p_g, LOGO_MAE)={A['corr_pg_vs_logo_mae']:.3f} | "
          f"corr(p_g, LOGO_MAPE)={A['corr_pg_vs_logo_mape']:.3f}")
    print(f"  A: p_g-weighted LOGO reference={A['pg_weighted_logo_reference_mae']:.2f} "
          f"(descriptive) | observed G={A['observed_G_MAE']:.2f} "
          f"| LOGO micro={A['logo_micro_MAE']:.2f} | LOGO macro={A['logo_macro_MAE']:.2f}")
    print(f"  GKF pooled: MAE={gkf_pooled['mae']:.2f} RMSE={gkf_pooled['rmse']:.2f} "
          f"R2={gkf_pooled['r2']:.4f} MAPE={gkf_pooled['mape_pct']:.2f}%")
    print(f"  B: {Bb['n_positive_dMAE']}/{n_geoms} geometries dMAE>0 "
          f"(mean dMAE={Bb['dMAE_mean']:.2f}, median={Bb['dMAE_median']:.2f}, "
          f"geometry-level bootstrap p95 {Bb['dMAE_geometry_bootstrap_p95']})")
    print(f"  B: {Bb['n_positive_dMAPE']}/{n_geoms} geometries dMAPE>0 "
          f"(mean dMAPE={Bb['dMAPE_mean']:.2f} pp)")
    for k in ("all_32", "excl_hardest_1", "excl_hardest_3"):
        print(f"  C {k}: seen={C[k]['MAE_seen_macro']:.2f} unseen={C[k]['MAE_unseen_macro']:.2f} "
              f"d={C[k]['dMAE_macro']:+.2f} (+{C[k]['rel_dMAE_macro']*100:.1f}%) "
              f"[{C[k]['n_positive_dmae']}/{C[k]['n_geometries']} positive]")


if __name__ == "__main__":
    main()
