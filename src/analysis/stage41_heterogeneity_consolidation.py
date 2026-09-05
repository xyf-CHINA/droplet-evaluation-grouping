"""
Stage 4.1 — heterogeneity consolidation (read-only, NO training).

Six pre-specified items over the existing 64 Stage 4 predictions:
  1. source x external_geometry cross-tab (nesting structure — source and
     geometry evidence are NOT independent where they nest);
  2. viscosity stratum x source / x geometry cross-tabs (who are the 12
     range-outside samples? confounding check);
  3. viscosity strata target-diameter distribution (+ per-source diameter
     scale, to interpret pooled R2 and MAE-vs-MAPE differences);
  4. signed-error refinement: median bias, MPE, median signed % error,
     underprediction count/proportion (systematic vs average bias);
  5. macro-source and macro-geometry metrics (equal-weight, to show
     sample-weighted aggregates underrepresent small difficult domains);
  6. identical-input cluster sensitivity: primary (64 rows) vs
     cluster-equal-weighted aggregate.

Viscosity findings are descriptive: stratum is confounded with source and
geometry unless the cross-tabs show otherwise. All 64 external samples
represent unseen geometry and exact-novel viscosity conditions.
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

from src.audit.audit_dafd3 import py_native, sha256_of
from src.common.model_spec import FEATURE_COLS

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "external_shift"


def signed_details(obs, pred) -> dict:
    err = pred - obs
    return {
        "n": int(len(obs)),
        "mean_signed_error_um": float(np.mean(err)),
        "median_signed_error_um": float(np.median(err)),
        "mpe_pct": float(np.mean(err / obs) * 100),
        "median_signed_pct_error": float(np.median(err / obs) * 100),
        "n_underpredicted": int((pred < obs).sum()),
        "prop_underpredicted": float((pred < obs).mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ext_raw = pd.read_excel(RAW_DIR / "Generalizability_data_normalized.xlsx", sheet_name="sheet 1")
    pred = pd.read_csv(OUT_DIR / "stage4_predictions.csv")
    assert len(pred) == 64 and len(ext_raw) == 64
    # stable-ID merge (never positional): one-to-one on Experiment
    df = ext_raw.merge(pred, on="Experiment", how="inner", suffixes=("", "_pred"))
    assert len(df) == 64, "stage4 predictions merge is not one-to-one"
    for c in ["Ref", "viscosity ratio"]:
        if f"{c}_pred" not in df.columns:
            continue
        if c == "Ref":
            same = bool((df[c] == df[f"{c}_pred"]).all())
        else:  # numeric: CSV round-trip may differ by float epsilon
            same = bool(np.allclose(
                df[c].to_numpy(dtype=float), df[f"{c}_pred"].to_numpy(dtype=float),
                rtol=1e-12, atol=1e-12,
            ))
        if not same:
            raise ValueError(f"identity mismatch on merged column {c}")
    df = df.drop(columns=[c for c in df.columns if c.endswith("_pred")]).reset_index(drop=True)

    # ---------- 1. source x geometry cross-tab ------------------------------
    cross = pd.crosstab(df["Ref"], df["geometry_id"])
    geom_counts_per_ref = (cross > 0).sum(axis=1).to_dict()
    single_geom_refs = sorted(r for r, k in geom_counts_per_ref.items() if k == 1)
    nested_note = (
        f"Nesting: {', '.join(single_geom_refs)} each contribute exactly ONE "
        "geometry (fully nested). Therefore source-stratified and "
        "geometry-stratified errors are NOT independent evidence for those "
        "domains."
    )

    # ---------- 2. stratum x source / x geometry cross-tabs -------------------
    strat_source = pd.crosstab(df["fluid_stratum"], df["Ref"])
    strat_geom = pd.crosstab(df["fluid_stratum"], df["geometry_id"])
    outside_refs = df.loc[df["fluid_stratum"] == "outside_train_range", "Ref"].value_counts().to_dict()
    outside_geoms = df.loc[df["fluid_stratum"] == "outside_train_range", "geometry_id"].value_counts().to_dict()
    confounded = bool(
        len(outside_refs) <= 2 or len(outside_geoms) <= 2
    )
    n_outside_total = int(sum(outside_refs.values()))
    confound_note = (
        f"The {n_outside_total} range-outside samples come from {len(outside_refs)} "
        f"source(s) and {len(outside_geoms)} geometry(ies). Viscosity-range status is "
        f"therefore {'NOT independent' if confounded else 'largely independent'} of "
        "source and geometry -> interpret the viscosity finding descriptively rather "
        "than causally."
    )

    # ---------- 3. target-diameter distributions ------------------------------
    def diam_block(sub):
        d = sub["observed_um"]
        return {
            "n": int(len(sub)),
            "mean_um": float(d.mean()),
            "median_um": float(d.median()),
            "min_um": float(d.min()),
            "max_um": float(d.max()),
        }

    strata_diams = {
        s: diam_block(df[df["fluid_stratum"] == s])
        for s in ("within_train_range", "outside_train_range")
    }
    per_ref_diams = {r: diam_block(df[df["Ref"] == r]) for r in df["Ref"].unique()}
    scale_note = (
        "Sources span different droplet-diameter scales (see per-source diameters); "
        "pooled external R2 benefits from between-source target variation. Interpretation: "
        "'The pooled external R2 remained high, while source-stratified absolute and "
        "relative errors revealed substantial performance heterogeneity.'"
    )

    # ---------- 4. signed-error refinement ------------------------------------
    overall_signed = signed_details(df["observed_um"].to_numpy(), df["predicted_um"].to_numpy())
    strata_signed = {
        s: signed_details(
            df.loc[df["fluid_stratum"] == s, "observed_um"].to_numpy(),
            df.loc[df["fluid_stratum"] == s, "predicted_um"].to_numpy(),
        )
        for s in ("within_train_range", "outside_train_range")
    }
    for s, b in strata_signed.items():
        b["stratum"] = s
    systematic_note = (
        f"outside-range stratum: {strata_signed['outside_train_range']['n_underpredicted']}/"
        f"{strata_signed['outside_train_range']['n']} underpredicted — supports "
        "'systematic underprediction tendency' if a clear majority, otherwise "
        "'negative average bias'."
    )

    # ---------- 5. macro metrics ------------------------------------------------
    ref_metrics = df.groupby("Ref", sort=True).apply(
        lambda g: pd.Series(
            {
                "mae": np.mean(np.abs(g["predicted_um"] - g["observed_um"])),
                "mape_pct": np.mean(np.abs(g["predicted_um"] - g["observed_um"]) / g["observed_um"]) * 100,
            }
        ),
        include_groups=False,
    )
    geom_metrics = df.groupby("geometry_id", sort=True).apply(
        lambda g: pd.Series(
            {
                "mae": np.mean(np.abs(g["predicted_um"] - g["observed_um"])),
                "mape_pct": np.mean(np.abs(g["predicted_um"] - g["observed_um"]) / g["observed_um"]) * 100,
            }
        ),
        include_groups=False,
    )
    macro = {
        "micro_mae": float(np.mean(np.abs(df["predicted_um"] - df["observed_um"]))),
        "micro_mape_pct": float(np.mean(np.abs(df["predicted_um"] - df["observed_um"]) / df["observed_um"]) * 100),
        "macro_source_mae": float(ref_metrics["mae"].mean()),
        "macro_source_mape_pct": float(ref_metrics["mape_pct"].mean()),
        "macro_geometry_mae": float(geom_metrics["mae"].mean()),
        "macro_geometry_mape_pct": float(geom_metrics["mape_pct"].mean()),
        "note": "macro metrics weight each source/geometry equally — the sample-weighted "
                "aggregate underrepresents small but difficult domains (e.g. Mazutis n=6).",
    }

    # ---------- 6. identical-input cluster sensitivity ---------------------------
    feat_dup = df.duplicated(subset=FEATURE_COLS, keep=False)
    n_dup_rows = int(feat_dup.sum())
    n_dup_clusters = int(df.loc[feat_dup].drop_duplicates(subset=FEATURE_COLS).shape[0])
    cluster = df.groupby(FEATURE_COLS, dropna=False).ngroup()
    df["cluster_id"] = cluster
    primary = {
        "mae": float(np.mean(np.abs(df["predicted_um"] - df["observed_um"]))),
        "mape_pct": float(np.mean(np.abs(df["predicted_um"] - df["observed_um"]) / df["observed_um"]) * 100),
        "signed_bias": float(np.mean(df["predicted_um"] - df["observed_um"])),
    }
    cl = (
        df.groupby("cluster_id")
        .agg(
            mae=("absolute_error_um", "mean"),
            ape=("percentage_error_pct", "mean"),
            bias=("predicted_um", lambda v: np.mean(v - df.loc[v.index, "observed_um"])),
        )
    )
    sensitivity = {
        "mae": float(cl["mae"].mean()),
        "mape_pct": float(cl["ape"].mean()),
        "signed_bias": float(cl["bias"].mean()),
        "n_clusters": int(len(cl)),
        "note": (
            f"each identical-input cluster weighted equally "
            f"({len(cl)} clusters from {len(df)} rows: "
            f"{n_dup_clusters} clusters hold {n_dup_rows} rows)"
        ),
    }

    results = {
        "stage": "4.1",
        "no_training": True,
        "1_source_geometry_crosstab": py_native(cross.to_dict()),
        "1_nesting_note": nested_note,
        "2_stratum_source_crosstab": py_native(strat_source.to_dict()),
        "2_stratum_geometry_crosstab": py_native(strat_geom.to_dict()),
        "2_outside_range_by_ref": py_native(outside_refs),
        "2_outside_range_by_geometry": py_native(outside_geoms),
        "2_confound_note": confound_note,
        "3_strata_diameters": strata_diams,
        "3_per_ref_diameters": per_ref_diams,
        "3_scale_note": scale_note,
        "4_signed_error": {
            "overall": overall_signed,
            "strata": strata_signed,
            "systematic_note": systematic_note,
        },
        "5_macro_metrics": macro,
        "6_cluster_sensitivity": {"primary_64_rows": primary, "cluster_weighted": sensitivity},
        "consumed_artifact_sha256": {
            "outputs/external_shift/stage4_predictions.csv": sha256_of(
                OUT_DIR / "stage4_predictions.csv"
            ),
            "data/raw/Generalizability_data_normalized.xlsx": sha256_of(
                RAW_DIR / "Generalizability_data_normalized.xlsx"
            ),
        },
    }

    with open(OUT_DIR / "stage41_heterogeneity.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
    cross.to_csv(OUT_DIR / "stage41_source_geometry_crosstab.csv")
    strat_source.to_csv(OUT_DIR / "stage41_stratum_source_crosstab.csv")
    strat_geom.to_csv(OUT_DIR / "stage41_stratum_geometry_crosstab.csv")
    pd.DataFrame(strata_signed).T.to_csv(OUT_DIR / "stage41_signed_error_details.csv")
    pd.DataFrame(
        {
            "metric": ["micro_mae", "micro_mape_pct", "macro_source_mae", "macro_source_mape_pct",
                       "macro_geometry_mae", "macro_geometry_mape_pct"],
            "value": [macro[k] for k in
                      ("micro_mae", "micro_mape_pct", "macro_source_mae", "macro_source_mape_pct",
                       "macro_geometry_mae", "macro_geometry_mape_pct")],
        }
    ).to_csv(OUT_DIR / "stage41_macro_metrics.csv", index=False)
    pd.DataFrame(
        {
            "aggregation": ["primary_64_rows", "cluster_weighted"],
            "mae": [primary["mae"], sensitivity["mae"]],
            "mape_pct": [primary["mape_pct"], sensitivity["mape_pct"]],
            "signed_bias": [primary["signed_bias"], sensitivity["signed_bias"]],
        }
    ).to_csv(OUT_DIR / "stage41_cluster_sensitivity.csv", index=False)

    print("[stage4.1] done (no training)")
    print("  source x geometry crosstab:")
    print(cross.to_string())
    print("\n  stratum x source:")
    print(strat_source.to_string())
    print("\n  stratum x geometry:")
    print(strat_geom.to_string())
    print(f"\n  outside-range underprediction: {strata_signed['outside_train_range']['n_underpredicted']}/"
          f"{strata_signed['outside_train_range']['n']}")
    print(f"  macro vs micro: MAE {macro['micro_mae']:.2f} -> macro-source {macro['macro_source_mae']:.2f} | "
          f"MAPE {macro['micro_mape_pct']:.2f}% -> macro-source {macro['macro_source_mape_pct']:.2f}%")
    print(f"  cluster sensitivity: MAE {primary['mae']:.2f} -> {sensitivity['mae']:.2f} | "
          f"MAPE {primary['mape_pct']:.2f}% -> {sensitivity['mape_pct']:.2f}%")


if __name__ == "__main__":
    main()
