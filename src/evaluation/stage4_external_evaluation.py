"""
Stage 4 — External domain stress test (XGBoost only).

Scientific question: does the aggregate external performance hide
source/domain-specific failure?

Design (locked BEFORE running):
    * ONLY the locked core XGBoost. No RF/MLP, no tuning,
      no threshold hunting, no model selection.
    * Train: 868 Comprehensive rows (author training domain).
    * Test: 64 official Generalizability rows (never seen in training;
      geometry+fluid+operating+source shift simultaneously).
    * Target: Normalized droplet diameter; denormalized by Hydraulic
      diameter; train-only StandardScaler.
    * Reports: overall MAE/RMSE/R2/MAPE/signed bias;
      by Ref/source (n, MAE, median AE, MAPE, signed bias — no R2 for n=6);
      by external geometry (n, MAE, median AE, MAPE, bias — NO per-geometry R2);
      fluid stratification PRE-LOCKED: viscosity outside vs within the
      Comprehensive min-max (Stage 0 audited: 12/64 outside, 52/64 inside) —
      a data-driven, not result-driven, stratification.
    * Identical-input repeated conditions (audited: 2 pairs / 4 rows) are
      KEPT and reported; descriptive results only, no CI claims.
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
import sklearn.metrics
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.audit.audit_dafd3 import assign_geometry_ids, py_native, sha256_of
from src.common.metrics import mape_pct, require_finite
from src.common.model_spec import (
    DENORM_COL,
    FEATURE_COLS,
    LOCKED_XGB_PARAMS,
    OBS_COL,
    TARGET_COL,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "external_shift"
AUDIT_JSON = PROJECT_ROOT / "outputs" / "audit" / "audit_results.json"


def block(y_obs, y_pred, with_r2: bool = True) -> dict:
    """Group metrics. R2 is only computed where the locked design allows it:
    overall yes; by-source only when n >= 10; per-geometry NEVER (small groups,
    unstable R2)."""
    err = y_pred - y_obs
    out = {
        "n": int(len(y_obs)),
        "mae": float(np.mean(np.abs(err))),
        "median_ae": float(np.median(np.abs(err))),
        "mape_pct": mape_pct(y_obs, y_pred),
        "signed_bias": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }
    if with_r2:
        out["r2"] = float(sklearn.metrics.r2_score(y_obs, y_pred))
    else:
        out["r2"] = None  # locked design: no R2 for small/geometry subgroups
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {p.name: sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}

    comp = pd.read_excel(RAW_DIR / "Comprehensive_normalized.xlsx", sheet_name="Sheet1")
    ext = pd.read_excel(RAW_DIR / "Generalizability_data_normalized.xlsx", sheet_name="sheet 1")
    assert len(comp) == 868 and len(ext) == 64

    X_comp = require_finite(comp[FEATURE_COLS].to_numpy(), "train features")
    Y_comp = require_finite(comp[TARGET_COL].to_numpy(), "train target")
    X_ext = require_finite(ext[FEATURE_COLS].to_numpy(), "external features")
    Ori_ext = require_finite(ext[DENORM_COL].to_numpy(), "external hydraulic diameter")
    D_ext = require_finite(ext[OBS_COL].to_numpy(), "external observed diameter")

    scaler = StandardScaler().fit(X_comp)  # train domain only
    reg = xgb.XGBRegressor(**LOCKED_XGB_PARAMS).fit(scaler.transform(X_comp), Y_comp)
    pred_norm = reg.predict(scaler.transform(X_ext))
    pred_um = Ori_ext * pred_norm

    ext = ext.copy()
    ext_with_id, _ = assign_geometry_ids(ext, id_prefix="EXT-G")
    ext["geometry_id"] = ext_with_id["geometry_id"]
    ext["observed_um"] = D_ext
    ext["predicted_um"] = pred_um
    ext["absolute_error_um"] = np.abs(pred_um - D_ext)
    ext["percentage_error_pct"] = np.abs((pred_um - D_ext) / D_ext) * 100

    overall = block(D_ext, pred_um)

    # ---- by Ref / source (R2 only when n >= 10, per locked design) -------
    by_ref = []
    for ref, grp in ext.groupby("Ref", dropna=False):
        b = block(grp[OBS_COL].to_numpy(), grp["predicted_um"].to_numpy(),
                  with_r2=(len(grp) >= 10))
        b["ref"] = str(ref)
        by_ref.append(b)

    # ---- by external geometry (NO per-geometry R2, per locked design) -----
    by_geom = []
    for gid, grp in ext.groupby("geometry_id", sort=True):
        b = block(grp[OBS_COL].to_numpy(), grp["predicted_um"].to_numpy(),
                  with_r2=False)
        b["geometry_id"] = gid
        by_geom.append(b)

    # ---- PRE-LOCKED fluid stratification --------------------------------
    # "within training range" is the CLOSED interval [min, max]; the audit's
    # outside-count uses the equivalent strict complement (< min | > max).
    # No boundary ties and no NaN viscosity exist in the current data
    # (asserted below); the 52/12 counts are cross-checked against the audit.
    v_min = float(comp["viscosity ratio"].min())
    v_max = float(comp["viscosity ratio"].max())
    assert ext["viscosity ratio"].notna().all(), "NaN viscosity in external data"
    ext["fluid_stratum"] = np.where(
        ext["viscosity ratio"].between(v_min, v_max, inclusive="both"),
        "within_train_range",
        "outside_train_range",
    )
    by_fluid = []
    for stratum, grp in ext.groupby("fluid_stratum", sort=False):
        b = block(grp[OBS_COL].to_numpy(), grp["predicted_um"].to_numpy())
        b["stratum"] = stratum
        by_fluid.append(b)

    # ---- identical-input repeated conditions (kept, not deleted) ----------
    feat_dup = ext.duplicated(subset=FEATURE_COLS, keep=False)
    dup_rows = ext.loc[feat_dup, ["Experiment", "Ref"] + FEATURE_COLS].sort_values(
        by=FEATURE_COLS
    )
    n_dup_rows = int(feat_dup.sum())
    n_dup_clusters = int(dup_rows.drop_duplicates(subset=FEATURE_COLS).shape[0])

    # ---- gate (computed, no literals) ---------------------------------------
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    audited_outside = audit["external"]["fluid_property"]["vs_Comprehensive"]["viscosity_ratio"][
        "n_external_outside_train_range"
    ]
    audited_dup_rows = audit["dataset_details"]["Generalizability_data_normalized.xlsx"][
        "n_duplicated_feature_rows_involved"
    ]
    fluid_ok = bool(
        len(ext.loc[ext["fluid_stratum"] == "outside_train_range"]) == audited_outside
        and int(feat_dup.sum()) == int(audited_dup_rows)
    )

    # external-never-in-training: computed as the exact 8-feature row overlap
    # between the external file and the Comprehensive training rows.
    def feat_tuples(df):
        return {tuple(round(float(v), 8) for v in row) for row in df[FEATURE_COLS].to_numpy()}

    n_feat_matches = len(feat_tuples(ext) & feat_tuples(comp))

    model_params = dict(LOCKED_XGB_PARAMS)
    model_full_params = py_native(xgb.XGBRegressor(**LOCKED_XGB_PARAMS).get_params(deep=False))

    hashes_after = {p.name: sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}
    metric_frames_finite = bool(
        all(
            np.isfinite([b[k] for b in blocks for k in ("mae", "median_ae", "mape_pct", "signed_bias", "rmse")]).all()
            for blocks in (by_ref, by_geom, by_fluid)
        )
    )
    gate_checks = {
        "train_rows_868": bool(len(comp) == 868),
        "external_rows_64": bool(len(ext) == 64),
        "external_never_in_training": bool(
            n_feat_matches == 0  # computed: no external 8-feature row matches any training row
        ),
        "model_params_match_locked": bool(model_params == LOCKED_XGB_PARAMS),
        "fluid_stratification_matches_audit": fluid_ok,
        "identical_input_clusters_kept": bool(int(feat_dup.sum()) == int(audited_dup_rows)),
        "all_predictions_finite": bool(
            np.isfinite(ext[["predicted_um", "absolute_error_um", "percentage_error_pct"]].to_numpy()).all()
        ),
        "all_metric_blocks_finite": metric_frames_finite,
        "raw_files_unchanged": bool(hashes_before == hashes_after),
    }
    gate_status = "PASS" if all(gate_checks.values()) else "FAIL"

    results = {
        "stage": "4",
        "purpose": "external domain stress test — does the aggregate hide domain-specific failure?",
        "train": "Comprehensive_normalized.xlsx (868 rows), locked XGBoost only",
        "test": "Generalizability_data_normalized.xlsx (64 rows, official external)",
        "model_params": model_params,
        "model_full_params": model_full_params,
        "features": FEATURE_COLS,
        "r2_reporting_rule": (
            "overall R2: reported. by-source R2: only n >= 10. "
            "per-geometry R2: never reported (locked design)."
        ),
        "overall": overall,
        "by_ref": by_ref,
        "by_geometry": by_geom,
        "by_fluid_stratum": {
            "definition": (
                f"PRE-LOCKED stratification: viscosity ratio within vs outside the "
                f"Comprehensive training range [{v_min:.4f}, {v_max:.4f}] "
                f"(Stage 0 audited counts: 52 within / 12 outside)"
            ),
            "strata": by_fluid,
        },
        "identical_input_repeats": {
            "n_rows_involved": n_dup_rows,
            "n_clusters": n_dup_clusters,
            "note": "repeated conditions are KEPT; descriptive results only — "
                    "no independence/CI claims over the 64 rows",
        },
        "level_framing_note": (
            "Level 1: Protocol R (predominantly seen-geometry, condition-level). "
            "Level 2: controlled unseen-geometry generalization (G/GKF/LOGO, 474 SE). "
            "Level 3: mixed external-domain generalization (this stage, 868->64). "
            "Levels 2 and 3 are NOT a simple difficulty ranking — Stage 4 also "
            "changes the training population and mixes geometry+fluid+operating+source shifts."
        ),
        "environment": {
            "xgboost": xgb.__version__,
            "sklearn": __import__("sklearn").__version__,
            "python": sys.version.split()[0],
        },
        "gate": {"status": gate_status, "checks": gate_checks},
    }

    with open(OUT_DIR / "stage4_external_metrics.json", "w", encoding="utf-8") as f:
        json.dump(py_native(results), f, indent=2, ensure_ascii=False, allow_nan=False)

    pred_cols = ["Experiment", "Ref", "geometry_id", "fluid_stratum", "viscosity ratio"]
    ext[pred_cols + ["observed_um", "predicted_um", "absolute_error_um", "percentage_error_pct"]].to_csv(
        OUT_DIR / "stage4_predictions.csv", index=False
    )
    pd.DataFrame(by_ref).to_csv(OUT_DIR / "stage4_by_ref.csv", index=False)
    pd.DataFrame(by_geom).to_csv(OUT_DIR / "stage4_by_geometry.csv", index=False)
    pd.DataFrame(by_fluid).to_csv(OUT_DIR / "stage4_by_fluid.csv", index=False)

    print("[stage4] done | gate:", gate_status)
    print(f"  overall: MAE={overall['mae']:.2f} RMSE={overall['rmse']:.2f} "
          f"R2={overall['r2']:.4f} MAPE={overall['mape_pct']:.2f}% bias={overall['signed_bias']:+.2f}")
    for r in by_ref:
        print(f"  {r['ref']:22s}: n={r['n']:2d} MAE={r['mae']:6.2f} medAE={r['median_ae']:6.2f} "
              f"MAPE={r['mape_pct']:5.1f}% bias={r['signed_bias']:+.2f}")
    for r in by_fluid:
        print(f"  fluid {r['stratum']:20s}: n={r['n']:2d} MAE={r['mae']:6.2f} MAPE={r['mape_pct']:5.1f}% "
              f"bias={r['signed_bias']:+.2f}")


if __name__ == "__main__":
    main()
