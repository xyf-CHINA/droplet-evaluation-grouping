"""
Stage 2 — Evaluation protocol benchmark (frozen model).

Implementation safeguards:
    * Protocol G subsets are drawn UNIFORMLY over all feasible n_test=95
      geometry subsets (src/evaluation/protocol_g_sampler.py);
    * every gate entry is computed (no literals): feasible count via DP,
      subset optimality/distinctness, held-label vs prediction-row cross-check,
      baseline stage1b JSON validated (params + gate status + row counts),
      np.isfinite on all metrics AND predictions;
    * 35/35 coverage and exposure are REPORTED, not gated;
    * LOGO uses positional indices (iloc-safe).

Only the split protocol changes; the model is FROZEN.
Cross-protocol equal ids are NOT a paired design.
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
from sklearn.model_selection import GroupKFold

from src.audit.audit_dafd3 import sha256_of
from src.common.data import load_se_extracted
from src.common.metrics import require_finite, summary_stats
from src.common.model_spec import LOCKED_XGB_PARAMS
from src.common.modeling import fit_eval
from src.evaluation.protocol_g_sampler import (
    collect_distinct,
    feasible_counts,
    theoretical_inclusion,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "geometry_shift"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BASELINE_JSON = PROJECT_ROOT / "outputs" / "baseline" / "stage1b_metrics.json"

G_TARGET_N_TEST = 95      # round(0.2 * 474)
G_N_SUBSETS = 100
G_SUBSET_RNG_SEED = 0
GKF_N_SPLITS = 5
METRIC_COLS = ["mae", "rmse", "r2", "mape_pct"]

PROTOCOL_G_RULE = (
    "Uniform sampling over ALL feasible geometry subsets with n_test == 95 "
    "(DP-counted). Geometry inclusion probabilities p_g are "
    "computed exactly and are NOT assumed equal."
)


def protocol_g(se_df: pd.DataFrame):
    sizes_by_gid = se_df["geometry_id"].value_counts().sort_index()
    gid_list = sizes_by_gid.index.to_list()
    sizes = sizes_by_gid.to_list()

    cnt, n_feasible, subsets = collect_distinct(
        sizes, G_TARGET_N_TEST, G_N_SUBSETS, G_SUBSET_RNG_SEED
    )
    n_feasible2, probs = theoretical_inclusion(sizes, G_TARGET_N_TEST)
    assert n_feasible == n_feasible2

    sample_sizes = [sum(sizes[i] for i in sub) for sub in subsets]
    assert all(s == G_TARGET_N_TEST for s in sample_sizes), "sampled subset not feasible"

    metrics_rows, pred_rows = [], []
    for k, sub in enumerate(subsets):
        held_ids = [gid_list[i] for i in sub]
        held = set(held_ids)
        mask = se_df["geometry_id"].isin(held).to_numpy()
        te = np.nonzero(mask)[0]
        tr = np.nonzero(~mask)[0]
        m, rows = fit_eval(se_df, tr, te, protocol="G", label=k, disjoint=True)
        m.update(
            {
                "seed": k,
                "n_test_geom": len(held),
                "n_train_geom": se_df["geometry_id"].nunique() - len(held),
                "held_out_geometries": "|".join(sorted(held)),
            }
        )
        rows["seed"] = k
        rows["held_out_geometries"] = "|".join(sorted(held))
        metrics_rows.append(m)
        pred_rows.append(rows)
    g_df = pd.DataFrame(metrics_rows)
    g_pred = pd.concat(pred_rows, ignore_index=True)

    # cross-check: prediction rows per fold must carry exactly the held geometries
    held_match = []
    for k, sub in enumerate(subsets):
        held = {gid_list[i] for i in sub}
        row_geoms = set(g_pred.loc[g_pred["seed"] == k, "geometry_id"])
        held_match.append(row_geoms == held)
    coverage = {gid_list[i]: int(sum(1 for sub in subsets if i in sub)) for i in range(len(gid_list))}
    return g_df, g_pred, {
        "n_feasible": int(n_feasible),
        "theoretical_inclusion_p": {gid_list[i]: float(probs[i]) for i in range(len(gid_list))},
        "held_match_all": bool(all(held_match)),
        "coverage_counts": coverage,
        "n_geometries_covered": int(sum(1 for c in coverage.values() if c > 0)),
        "exposure_min": int(min(coverage.values())),
        "exposure_max": int(max(coverage.values())),
    }


def group_kfold(se_df: pd.DataFrame):
    gkf = GroupKFold(n_splits=GKF_N_SPLITS)
    X = se_df[["Orifice width (um)"]].to_numpy()  # shape only; groups drive the split
    rows, pred_rows = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, groups=se_df["geometry_id"])):
        m, pr = fit_eval(se_df, tr, te, protocol="GKF", label=fold, disjoint=True)
        m["fold"] = fold
        pr["fold"] = fold
        rows.append(m)
        pred_rows.append(pr)
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def logo(se_df: pd.DataFrame):
    per_geom_rows, pred_rows = [], []
    for gid in sorted(se_df["geometry_id"].unique()):
        te = np.nonzero((se_df["geometry_id"] == gid).to_numpy())[0]  # positional indices
        tr = np.setdiff1d(np.arange(len(se_df)), te)
        m, rows = fit_eval(se_df, tr, te, protocol="LOGO", label=gid, disjoint=True)
        err = rows["predicted_um"].to_numpy() - rows["observed_um"].to_numpy()
        per_geom_rows.append(
            {
                "geometry_id": gid,
                "n": int(len(te)),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape_pct": m["mape_pct"],
                "median_ae": float(np.median(np.abs(err))),
                "signed_bias": float(np.mean(err)),
            }
        )
        pred_rows.append(rows)
    per_geom = pd.DataFrame(per_geom_rows)
    all_pred = pd.concat(pred_rows, ignore_index=True)
    require_finite(
        all_pred[["observed_um", "predicted_um", "absolute_error_um", "percentage_error_pct"]].to_numpy(),
        "LOGO predictions",
    )
    micro = {
        "mae": float(np.mean(np.abs(all_pred["predicted_um"] - all_pred["observed_um"]))),
        "rmse": float(np.sqrt(np.mean((all_pred["predicted_um"] - all_pred["observed_um"]) ** 2))),
        "r2": float(1 - np.sum((all_pred["observed_um"] - all_pred["predicted_um"]) ** 2)
                    / np.sum((all_pred["observed_um"] - all_pred["observed_um"].mean()) ** 2)),
        "mape_pct": float(np.mean(np.abs((all_pred["observed_um"] - all_pred["predicted_um"])
                                         / all_pred["observed_um"])) * 100),
    }
    macro = {
        "mae": float(per_geom["mae"].mean()),
        "median_ae": float(per_geom["median_ae"].mean()),
        "mape_pct": float(per_geom["mape_pct"].mean()),
        "signed_bias": float(per_geom["signed_bias"].mean()),
    }
    return per_geom, all_pred, micro, macro


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {p.name: sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}

    se_df = load_se_extracted()
    n_geom = int(se_df["geometry_id"].nunique())

    # ---- Protocol R baseline: validate the consumed JSON, not just read it ----
    if not BASELINE_JSON.exists():
        raise RuntimeError(
            f"{BASELINE_JSON} missing — run stage1b first (Step 4 rerun order: "
            "audit -> 1A -> 1B -> 2)"
        )
    r_json = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline_params_match = bool(r_json.get("model_params") == LOCKED_XGB_PARAMS)
    baseline_gate_pass = bool(r_json.get("gate", {}).get("status") == "PASS")
    baseline_rows_match = bool(
        r_json.get("n_se_rows") == len(se_df)
        and r_json.get("n_unique_geometries") == n_geom
    )
    r_summary = r_json["per_seed_summary"]

    # ---- Protocol G -----------------------------------------------------------
    g_df, g_pred, g_info = protocol_g(se_df)
    g_summary = {k: summary_stats(g_df[k]) for k in METRIC_COLS}

    # ---- GroupKFold -------------------------------------------------------------
    gkf_df, gkf_pred = group_kfold(se_df)
    gkf_summary = {k: summary_stats(gkf_df[k]) for k in METRIC_COLS}

    # ---- LOGO --------------------------------------------------------------------
    logo_geom, logo_pred, logo_micro, logo_macro = logo(se_df)

    # ---- deltas (descriptive only) ------------------------------------------------
    deltas = {}
    for k in METRIC_COLS:
        deltas[k] = {
            "R_mean": r_summary[k]["mean"],
            "G_mean": g_summary[k]["mean"],
            "diff_mean": g_summary[k]["mean"] - r_summary[k]["mean"],
            "rel_diff_mean": (g_summary[k]["mean"] - r_summary[k]["mean"]) / r_summary[k]["mean"],
            "R_median": r_summary[k]["median"],
            "G_median": g_summary[k]["median"],
            "diff_median": g_summary[k]["median"] - r_summary[k]["median"],
            "rel_diff_median": (g_summary[k]["median"] - r_summary[k]["median"]) / r_summary[k]["median"],
        }

    hashes_after = {p.name: sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}

    # ---- gates: computed, nothing literal ----------------------------------------
    geom_fold_counts = {}
    for gid, sub in gkf_pred.groupby("geometry_id"):
        geom_fold_counts[gid] = sub["fold"].nunique()
    all_pred_frames = [g_pred, gkf_pred, logo_pred]
    all_metrics_frames = [g_df, gkf_df]
    gate_checks = {
        "baseline_params_match": baseline_params_match,
        "baseline_gate_pass": baseline_gate_pass,
        "baseline_rows_match": baseline_rows_match,
        "g_subsets_all_feasible_n95": bool((g_df["n_test"] == G_TARGET_N_TEST).all()),
        "g_subsets_all_distinct": bool(g_df["held_out_geometries"].nunique() == G_N_SUBSETS),
        "g_held_labels_match_prediction_rows": bool(g_info["held_match_all"]),
        "g_zero_geometry_overlap_enforced_by_fit_eval": bool(
            len(g_df) == G_N_SUBSETS  # fit_eval(disjoint=True) raises otherwise
        ),
        "g_feasible_count_computed": bool(g_info["n_feasible"] == int(feasible_counts(
            se_df["geometry_id"].value_counts().sort_index().to_list(), G_TARGET_N_TEST
        )[0][G_TARGET_N_TEST])),
        "gkf_5_folds": bool(len(gkf_df) == GKF_N_SPLITS),
        "gkf_each_geometry_exactly_one_fold": bool(all(v == 1 for v in geom_fold_counts.values())),
        "logo_covers_all_geometries": bool(len(logo_geom) == n_geom),
        "logo_predictions_474": bool(len(logo_pred) == len(se_df)),
        "all_metrics_finite": bool(
            all(np.isfinite(df[METRIC_COLS].to_numpy(dtype=float)).all() for df in all_metrics_frames)
        ),
        "all_predictions_finite": bool(
            all(
                np.isfinite(
                    df[["observed_um", "predicted_um", "absolute_error_um", "percentage_error_pct"]]
                    .to_numpy(dtype=float)
                ).all()
                for df in all_pred_frames
            )
        ),
        "raw_files_unchanged": bool(hashes_before == hashes_after),
    }
    gate_status = "PASS" if all(gate_checks.values()) else "FAIL"

    results = {
        "stage": "2",
        "version": 2,
        "model_frozen": True,
        "model_params": dict(LOCKED_XGB_PARAMS),
        "protocol_g_rule": PROTOCOL_G_RULE,
        "protocol_g_sampling": {
            "target_n_test": G_TARGET_N_TEST,
            "n_feasible_computed": g_info["n_feasible"],
            "n_sampled": G_N_SUBSETS,
            "subset_rng_seed": G_SUBSET_RNG_SEED,
            "sampling": "uniform over feasible subsets (DP probability walk)",
            "theoretical_inclusion_p": g_info["theoretical_inclusion_p"],
            "n_geometries_covered": g_info["n_geometries_covered"],
            "coverage_counts": g_info["coverage_counts"],
            "exposure_min": g_info["exposure_min"],
            "exposure_max": g_info["exposure_max"],
            "note": "coverage/exposure are reported, not gated",
        },
        "protocol_R_summary": r_summary,
        "protocol_G_summary": g_summary,
        "groupkfold_summary": gkf_summary,
        "groupkfold_fold_sizes": gkf_df[["fold", "n_train", "n_test"]].to_dict("records"),
        "logo": {
            "micro": logo_micro,
            "macro": logo_macro,
            "per_geometry": logo_geom.to_dict("records"),
        },
        "deltas_G_vs_R": deltas,
        "stats_note": (
            "Descriptive only. Cross-protocol equal seed ids are NOT a paired design. "
            "No p-values specified at this stage."
        ),
        "gate": {"status": gate_status, "checks": gate_checks},
    }

    with open(OUT_DIR / "stage2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)

    g_df.to_csv(OUT_DIR / "stage2_protocol_g_seed_metrics.csv", index=False)
    g_pred.to_csv(OUT_DIR / "stage2_protocol_g_predictions.csv", index=False)
    gkf_df.to_csv(OUT_DIR / "stage2_groupkfold_metrics.csv", index=False)
    gkf_pred.to_csv(OUT_DIR / "stage2_groupkfold_predictions.csv", index=False)
    logo_geom.to_csv(OUT_DIR / "stage2_logo_geometry_metrics.csv", index=False)
    logo_pred.to_csv(OUT_DIR / "stage2_logo_predictions.csv", index=False)

    print("[stage2] done | gate:", gate_status)
    print("  Protocol R : MAE=%.2f MAPE=%.2f%% R2=%.3f" % (
        r_summary["mae"]["mean"], r_summary["mape_pct"]["mean"], r_summary["r2"]["mean"]))
    print("  Protocol G : MAE=%.2f MAPE=%.2f%% R2=%.3f" % (
        g_summary["mae"]["mean"], g_summary["mape_pct"]["mean"], g_summary["r2"]["mean"]))
    print("  dMAE=%.2f (rel %.1f%%) | dMAPE=%.2f (rel %.1f%%)" % (
        deltas["mae"]["diff_mean"], deltas["mae"]["rel_diff_mean"] * 100,
        deltas["mape_pct"]["diff_mean"], deltas["mape_pct"]["rel_diff_mean"] * 100))


if __name__ == "__main__":
    main()
