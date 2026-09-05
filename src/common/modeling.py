"""Shared frozen-model training/evaluation helper.

Every protocol trains IDENTICALLY: 8 features, normalized-diameter target,
train-only StandardScaler, LOCKED_XGB_PARAMS, hydraulic-diameter
denormalization. Only the train/test index sets change.
"""
import math

import numpy as np
import pandas as pd
import sklearn.metrics
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from src.common.metrics import mape_pct, require_finite
from src.common.model_spec import (
    DENORM_COL,
    FEATURE_COLS,
    LOCKED_XGB_PARAMS,
    OBS_COL,
    TARGET_COL,
)

ID_COLS = ["comp_experiment_id", "se_experiment_id", "source_row_id", "geometry_id"]


def fit_eval(se_df: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
             protocol: str, label, disjoint: bool = True,
             make_model=None) -> tuple[dict, pd.DataFrame]:
    """Train the FROZEN model on tr, evaluate on te.

    `disjoint=True` asserts the test geometries never appear in training
    (Protocol G / GKF / LOGO); Protocol R passes disjoint=False.

    `make_model` is a no-arg factory returning an unfitted sklearn-style
    regressor. Default (None) is the locked XGBoost spec;
    Stage 3 passes the locked RF / MLP factories.
    """
    X = require_finite(se_df[FEATURE_COLS].to_numpy(), "features")
    Y = require_finite(se_df[TARGET_COL].to_numpy(), "target")
    Ori = require_finite(se_df[DENORM_COL].to_numpy(), "hydraulic diameter")
    D = require_finite(se_df[OBS_COL].to_numpy(), "observed diameter")
    geom = se_df["geometry_id"].to_numpy()
    ids = se_df[ID_COLS]

    if len(np.intersect1d(tr, te)) != 0:
        raise ValueError(f"{protocol}:{label}: train/test index overlap")
    if disjoint and not set(geom[te]).isdisjoint(set(geom[tr])):
        raise ValueError(
            f"{protocol}:{label}: geometry split across train/test "
            f"(violates whole-geometry holdout)"
        )

    scaler = StandardScaler().fit(X[tr])
    if make_model is None:
        reg = xgb.XGBRegressor(**LOCKED_XGB_PARAMS).fit(scaler.transform(X[tr]), Y[tr])
    else:
        reg = make_model().fit(scaler.transform(X[tr]), Y[tr])
    pred_norm = reg.predict(scaler.transform(X[te]))
    obs_um = D[te]
    pred_um = Ori[te] * pred_norm

    r2 = (
        float(sklearn.metrics.r2_score(obs_um, pred_um))
        if len(te) >= 2
        else float("nan")  # R2 undefined for single-sample folds (LOGO singletons)
    )
    m = {
        "mae": float(sklearn.metrics.mean_absolute_error(obs_um, pred_um)),
        "rmse": float(math.sqrt(sklearn.metrics.mean_squared_error(obs_um, pred_um))),
        "r2": r2,
        "mape_pct": mape_pct(obs_um, pred_um),  # zero-guarded
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }

    train_geom = set(geom[tr])
    rows = ids.iloc[te].copy().reset_index(drop=True)
    rows["fold"] = label
    rows["protocol"] = protocol
    rows["split"] = "test"
    rows["geometry_seen_in_train"] = [g in train_geom for g in geom[te]]
    rows["observed_um"] = obs_um
    rows["predicted_um"] = pred_um
    rows["absolute_error_um"] = np.abs(obs_um - pred_um)
    rows["percentage_error_pct"] = np.abs((obs_um - pred_um) / obs_um) * 100
    require_finite(
        rows[["observed_um", "predicted_um", "absolute_error_um", "percentage_error_pct"]].to_numpy(),
        f"{protocol}:{label}: predictions/errors",
    )
    return m, rows
