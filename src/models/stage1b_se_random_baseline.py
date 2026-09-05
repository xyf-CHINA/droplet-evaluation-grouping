"""
Stage 1B — SE Random-Split baseline (formal Protocol R).

Implementation safeguards:
    * single SE loader (src.common.data.load_se_extracted) — fixes the
      source_row_id off-by-one (true Excel rows 3..476);
    * multiplicity guards (Stage 0 multiset contract enforced at extraction);
    * gates are computed, not literal (audit_results.json is cross-checked);
    * np.isfinite everywhere (NaN AND inf), MAPE has a zero-division guard;
    * model hyperparameters follow the frozen specification in src.common.model_spec.

Protocol R: 100 fixed seeds 0-99, train_test_split(test_size=0.20,
random_state=seed, shuffle=True). The author model specification is kept
UNCHANGED (8 features incl. the SE-constant viscosity ratio; normalized-
diameter target; hydraulic-diameter denormalization; train-only scaler).
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
from sklearn.model_selection import train_test_split

from src.audit.audit_dafd3 import random_split_metrics, sha256_of
from src.common.data import load_se_extracted
from src.common.metrics import require_finite, summary_stats
from src.common.model_spec import LOCKED_XGB_PARAMS, SEEDS_R, TEST_SIZE
from src.common.modeling import ID_COLS, fit_eval

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
AUDIT_JSON = PROJECT_ROOT / "outputs" / "audit" / "audit_results.json"

METRIC_COLS = ["mae", "rmse", "r2", "mape_pct"]
OVERLAP_COLS = [
    "n_train_geom",
    "n_test_geom",
    "n_test_seen_geom_samples",
    "sample_overlap_frac",
    "n_test_geom_seen",
    "geometry_overlap_frac",
]


def one_seed(seed: int, se_df: pd.DataFrame) -> dict:
    idx = np.arange(len(se_df))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, shuffle=True)
    m, rows = fit_eval(se_df, tr, te, protocol="R", label=seed, disjoint=False)
    overlap = random_split_metrics(tr, te, se_df["geometry_id"].to_numpy())
    rows["seed"] = seed
    m["seed"] = seed
    return {"seed": seed, **m, **overlap, "test_rows": rows}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {p.name: sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}

    se_df = load_se_extracted()

    records = [one_seed(seed, se_df) for seed in SEEDS_R]

    hashes_after = {p.name: sha256_of(p) for p in sorted(RAW_DIR.glob("*.xlsx"))}

    seed_df = pd.DataFrame(
        [{k: r[k] for k in ["seed"] + METRIC_COLS + OVERLAP_COLS} for r in records]
    )
    all_test = pd.concat([r["test_rows"] for r in records], ignore_index=True)

    per_seed_summary = {k: summary_stats(seed_df[k]) for k in METRIC_COLS + OVERLAP_COLS}

    # ---- gates: every entry computed, nothing literal ------------------------
    if AUDIT_JSON.exists():
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        n_expected, n_geom_expected = (
            audit["se"]["n_samples"],
            audit["se"]["n_unique_geometries"],
        )
    else:
        n_expected, n_geom_expected = None, None

    n_unique_visc = int(se_df["viscosity ratio"].nunique(dropna=True))
    gate_checks = {
        "n_se_rows_match_audit": bool(n_expected is not None and len(se_df) == n_expected),
        "n_geometries_match_audit": bool(
            n_geom_expected is not None and se_df["geometry_id"].nunique() == n_geom_expected
        ),
        "n_seeds_complete": bool(len(records) == len(SEEDS_R)),
        "train_test_disjointness_enforced_by_fit_eval": bool(
            len(records) == len(SEEDS_R)  # fit_eval raises on any overlap; reaching here = 100 clean runs
        ),
        "all_metrics_finite": bool(
            np.isfinite(seed_df[METRIC_COLS + OVERLAP_COLS].to_numpy(dtype=float)).all()
        ),
        "predictions_finite_and_traceable": bool(
            np.isfinite(
                all_test[
                    ["observed_um", "predicted_um", "absolute_error_um", "percentage_error_pct"]
                ].to_numpy(dtype=float)
            ).all()
            and all_test[ID_COLS + ["seed", "protocol", "fold"]].notna().all().all()
        ),
        "viscosity_ratio_constant_in_se": bool(n_unique_visc == 1),
        "raw_files_unchanged": bool(hashes_before == hashes_after),
    }
    gate_status = "PASS" if all(gate_checks.values()) else "FAIL"

    results = {
        "stage": "1B",
        "protocol": "R",
        "version": 2,
        "purpose": "SE core random-split baseline (formal Protocol R)",
        "data": "474 SE rows extracted from Comprehensive via Stage 0 locked provenance "
                "(src/common/data.py, multiplicity-guarded)",
        "n_se_rows": int(len(se_df)),
        "n_unique_geometries": int(se_df["geometry_id"].nunique()),
        "n_unique_viscosity_ratio_in_se": n_unique_visc,
        "features": list(se_df.columns),
        "target": "Normalized droplet diameter",
        "denormalization": "Hydraulic diameter * predicted normalized diameter",
        "model_params": dict(LOCKED_XGB_PARAMS),  # consumed & compared by stage2
        "split": f"train_test_split(test_size={TEST_SIZE}, random_state=seed, shuffle=True), seeds 0-99",
        "per_seed_summary": per_seed_summary,
        "seen_geometry_summary": {
            "n_predictions": int(len(all_test)),
            "n_seen": int(all_test["geometry_seen_in_train"].sum()),
            "n_unseen": int((~all_test["geometry_seen_in_train"]).sum()),
            "seen_frac": float(all_test["geometry_seen_in_train"].mean()),
            "note": "prediction-row counts are NOT independent samples (pseudoreplication guard, rule 21)",
        },
        "gate": {"status": gate_status, "checks": gate_checks},
    }

    with open(OUT_DIR / "stage1b_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)

    seed_df.to_csv(OUT_DIR / "stage1b_seed_metrics.csv", index=False)
    all_test.to_csv(OUT_DIR / "stage1b_predictions.csv", index=False)

    s = per_seed_summary
    print("[stage1b] done | gate:", gate_status)
    for k in METRIC_COLS:
        print(f"  {k:9s}: mean={s[k]['mean']:.4f} sd={s[k]['sd']:.4f} "
              f"median={s[k]['median']:.4f} [{s[k]['p2_5']:.4f}, {s[k]['p97_5']:.4f}]")
    print(f"  sample-overlap: mean={s['sample_overlap_frac']['mean']:.4f} | "
          f"geometry-overlap: mean={s['geometry_overlap_frac']['mean']:.4f}")


if __name__ == "__main__":
    main()
