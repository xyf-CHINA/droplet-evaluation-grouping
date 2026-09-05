"""
Stage 1A — Author XGBoost pipeline sanity reproduction.

Faithful mirror of the droplet-diameter part of the official
`reference/official_code/BoostedDTs_final.py`, restricted to:

    * Comprehensive_normalized.xlsx (868 rows)
    * 8 input features -> target `Normalized droplet diameter`
    * StandardScaler fit on TRAIN split only
    * xgb.XGBRegressor(tree_method="hist") with DEFAULT hyperparameters
    * train_test_split(test_size=0.2)
    * denormalization: D_pred = Hydraulic_diameter * D_normalized_pred
    * metrics on denormalized diameter: MAE, RMSE, R^2, MAPE (author formula)

Documented deviation from the official script (unavoidable):
    the official script passes NO random_state to train_test_split, so its
    splits are not reproducible run-to-run. We mirror the official 3-session
    loop with FIXED seeds (0, 1, 2) and report mean +- 1.96*SEM in exactly the
    author's reporting style.

Purpose: sanity check only — prove we understand the author's inputs, target,
normalization/denormalization and random split. NOT a paper result.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sklearn.metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.audit.audit_dafd3 import py_native
from src.common.metrics import mape_pct
from src.common.model_spec import (
    DENORM_COL,
    FEATURE_COLS,
    OBS_COL,
    TARGET_COL,
    TEST_SIZE,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "baseline"

SEEDS = (0, 1, 2)  # mirrors the author's 3 training sessions


# Author MAPE formula, shared implementation with zero-division guard.
author_mape = mape_pct


def one_run(seed: int, wb: pd.DataFrame) -> dict:
    X = wb.loc[:, FEATURE_COLS].to_numpy()
    Y = wb.loc[:, TARGET_COL].to_numpy()
    Ori = wb.loc[:, DENORM_COL].to_numpy()
    D = wb.loc[:, OBS_COL].to_numpy()

    X_train, X_test, Y_train, Y_test, Ori_train, Ori_test, D_train, D_test = train_test_split(
        X, Y, Ori, D, test_size=TEST_SIZE, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # random_state pinned at the model level too (project rule 11); the official
    # script passed none, which left run-to-run reproducibility to xgboost internals.
    reg = xgb.XGBRegressor(tree_method="hist", random_state=0).fit(X_train, Y_train)
    y_pred_test = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)

    d_pred_test = Ori_test * y_pred_test
    d_pred_train = Ori_train * y_pred_train

    def block(d_obs, d_pred):
        return {
            "mae": float(sklearn.metrics.mean_absolute_error(d_obs, d_pred)),
            "rmse": float(math.sqrt(sklearn.metrics.mean_squared_error(d_obs, d_pred))),
            "r2": float(sklearn.metrics.r2_score(d_obs, d_pred)),
            "mape_pct": author_mape(d_obs, d_pred),
        }

    return {
        "seed": seed,
        "train": block(D_train, d_pred_train),
        "test": block(D_test, d_pred_test),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_predictions": pd.DataFrame(
            {
                "seed": seed,
                "split": "test",
                "obs_norm": Y_test,
                "pred_norm": y_pred_test,
                "obs_um": D_test,
                "pred_um": d_pred_test,
            }
        ),
        "feature_importances": {
            c: float(v) for c, v in zip(FEATURE_COLS, reg.feature_importances_)
        },
    }


def summarize(runs: list[dict]) -> dict:
    out = {}
    for split in ("train", "test"):
        for m in ("mae", "rmse", "r2", "mape_pct"):
            vals = [r[split][m] for r in runs]
            out[f"{split}_{m}_mean"] = float(np.mean(vals))
            out[f"{split}_{m}_sem95"] = float(
                1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals))
            )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wb = pd.read_excel(RAW_DIR / "Comprehensive_normalized.xlsx", sheet_name="Sheet1")
    assert len(wb) == 868

    runs = [one_run(seed, wb) for seed in SEEDS]

    metrics = {
        "stage": "1A",
        "purpose": "author XGBoost pipeline sanity reproduction (NOT a paper result)",
        "data": "Comprehensive_normalized.xlsx (868 rows)",
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "denormalization": "Hydraulic diameter * predicted normalized diameter",
        "model": "xgb.XGBRegressor(tree_method='hist') with default hyperparameters",
        "split": f"train_test_split(test_size={TEST_SIZE}, random_state=seed, shuffle=True)",
        "seeds": list(SEEDS),
        "seed_note": "official script passes no random_state (non-reproducible); "
                     "fixed seeds 0/1/2 mirror the official 3-session loop",
        "environment": {
            "python": sys.version.split()[0],
            "xgboost": xgb.__version__,
            "xgboost_note": "official script dates from 2022 (xgboost 1.x era); "
                            "core defaults (n_estimators=100, max_depth=6, lr=0.3) unchanged",
            "scikit_learn": __import__("sklearn").__version__,
            "pandas": pd.__version__,
        },
        "per_seed": {str(r["seed"]): {"train": r["train"], "test": r["test"]} for r in runs},
        "summary": summarize(runs),
        "feature_importances_mean": {
            c: float(np.mean([r["feature_importances"][c] for r in runs]))
            for c in FEATURE_COLS
        },
        "model_full_params": py_native(
            xgb.XGBRegressor(tree_method="hist", random_state=0).get_params(deep=False)
        ),
    }

    with open(OUT_DIR / "stage1a_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, allow_nan=False)

    pd.DataFrame(
        [
            {
                "seed": r["seed"],
                "split": s,
                **{m: r[s][m] for m in ("mae", "rmse", "r2", "mape_pct")},
            }
            for r in runs
            for s in ("train", "test")
        ]
    ).to_csv(OUT_DIR / "stage1a_metrics.csv", index=False)

    pd.concat([r["test_predictions"] for r in runs], ignore_index=True).to_csv(
        OUT_DIR / "stage1a_test_predictions.csv", index=False
    )

    s = metrics["summary"]
    print("[stage1a] done")
    for split in ("train", "test"):
        print(
            f"  {split:5s}: MAE={s[f'{split}_mae_mean']:.2f}+-{s[f'{split}_mae_sem95']:.2f} | "
            f"RMSE={s[f'{split}_rmse_mean']:.2f}+-{s[f'{split}_rmse_sem95']:.2f} | "
            f"R2={s[f'{split}_r2_mean']:.4f}+-{s[f'{split}_r2_sem95']:.4f} | "
            f"MAPE={s[f'{split}_mape_pct_mean']:.2f}%+-{s[f'{split}_mape_pct_sem95']:.2f}%"
        )


if __name__ == "__main__":
    main()
