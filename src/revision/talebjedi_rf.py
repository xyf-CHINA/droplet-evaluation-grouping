"""Targeted post hoc RF extension on the published Talebjedi data.

The training phase is explicit and optional. It regenerates only the fixed
RF extension, retaining the existing XGBoost predictions for comparison.
All experimental rows, mappings and predictions stay in ignored outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .analysis import aggregate, close, compare_splits, dump, export, ranking_summary, read, require

ROOT = Path(__file__).resolve().parents[2]
ANGLES = [30, 60, 90, 120, 150]
FEATURES = ["TiltAngle_deg", "FRR", "Qc_uL_min"]


def validate_data(data):
    require(len(data) == 125 and set(data.ExpNo) == set(range(1, 126)), "Expected 125 unique experiments")
    require(sorted(data.TiltAngle_deg.unique()) == ANGLES, "Expected five tilt angles")
    require(data.groupby("TiltAngle_deg").size().eq(25).all(), "Expected 25 observations per angle")
    require(np.isfinite(data[FEATURES + ["Size_um"]]).all().all() and (data.Size_um > 0).all(),
            "Nonfinite predictors or nonpositive targets")


def ordered_splits(data, r, t):
    """Reconstruct the published protocol and verify ordered baseline members."""
    from sklearn.model_selection import train_test_split
    validate_data(data)
    all_splits = []
    for protocol, pred, split_col, ids in (("R", r, "seed", list(range(100))),
                                         ("T", t, "held_out_angle", ANGLES)):
        require(len(pred) == (2500 if protocol == "R" else 125), "Baseline prediction count")
        require(not pred.duplicated([split_col, "ExpNo"]).any(), "Duplicate baseline prediction")
        require(set(pred[split_col]) == set(ids), "Baseline split identifiers")
        require(set(pred.ExpNo) == set(data.ExpNo), "Incomplete baseline cohort")
        close(pred.observed_um, data.set_index("ExpNo").loc[pred.ExpNo, "Size_um"], "Baseline target mismatch")
        for sid in ids:
            if protocol == "R":
                tr, te = train_test_split(np.arange(125), test_size=.2, random_state=sid, shuffle=True)
            else:
                tr = np.flatnonzero(data.TiltAngle_deg.to_numpy() != sid)
                te = np.flatnonzero(data.TiltAngle_deg.to_numpy() == sid)
            require(data.iloc[te].ExpNo.tolist() == pred.loc[pred[split_col] == sid, "ExpNo"].tolist(),
                    "Ordered baseline test members differ from reconstructed split")
            require(len(tr) == 100 and len(te) == 25 and not set(tr) & set(te), "Invalid training/test partition")
            all_splits.append((protocol, sid, tr, te))
    return all_splits


def fit_extension(data, r, t, out, config=None):
    """Fit exactly 105 RF models. Call only when local retraining is intended."""
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    require(sklearn.__version__ == "1.6.1", "Use the pinned scikit-learn 1.6.1 environment")
    out = Path(out)
    require(not out.exists(), "Refusing to overwrite a previous fit directory")
    params = json.loads(Path(config or ROOT / "configs/revision_rf.json").read_text(encoding="utf-8"))
    require(RandomForestRegressor(**params).get_params(deep=False) == params, "RF full configuration differs")
    splits = ordered_splits(data, r, t)
    records, warning_count = [], 0
    for protocol, sid, train_idx, test_idx in splits:
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            xtrain = scaler.fit_transform(train[FEATURES])
            xtest = scaler.transform(test[FEATURES])
            reg = RandomForestRegressor(**params).fit(xtrain, train.Size_um)
            predicted = reg.predict(xtest)
        require(scaler.n_samples_seen_ == 100 and np.isfinite(predicted).all(), "Invalid fit or predictions")
        warning_count += len(caught)
        for exp, angle, observed, value in zip(test.ExpNo, test.TiltAngle_deg, test.Size_um, predicted):
            records.append(dict(protocol=protocol, split_id=int(sid), ExpNo=int(exp), angle=int(angle),
                observed_um=float(observed), predicted_um=float(value), absolute_error_um=abs(float(value)-float(observed)),
                percentage_error_pct=100*abs(float(value)-float(observed))/abs(float(observed)), model="RF"))
    require(len(records) == 2625, "Incomplete RF extension")
    out.mkdir(parents=True, exist_ok=False)
    export(pd.DataFrame(records), out / "predictions.csv")
    dump(out / "fit_summary.json", dict(fits=105, predictions=2625, warning_count=warning_count,
                                       sklearn=sklearn.__version__, params=params))


def summarize(data, r, t, rf, out):
    """Recompute four model summaries and all 105 same-split comparisons."""
    validate_data(data)
    ordered_splits(data, r, t)
    out = Path(out)
    require(not out.exists(), "Refusing to overwrite an aggregate directory")
    require(len(rf) == 2625 and set(rf.protocol) == {"R", "T"}, "Incomplete RF predictions")
    summaries, differences = [], []
    for protocol, xgb, col in (("R", r, "seed"), ("T", t, "held_out_angle")):
        xgb = xgb.rename(columns={col: "split_id"}).copy()
        xgb["angle"] = data.set_index("ExpNo").loc[xgb.ExpNo, "TiltAngle_deg"].to_numpy()
        other = rf[rf.protocol == protocol].copy()
        keys = ["split_id", "ExpNo"]
        a, b = [df.sort_values(keys).reset_index(drop=True) for df in (xgb, other)]
        require(a[keys + ["angle", "observed_um"]].equals(b[keys + ["angle", "observed_um"]]),
                "XGB/RF original observation, angle or target mismatch")
        for model, df in (("XGB", xgb), ("RF", other)):
            row = aggregate(df, protocol, model, "ExpNo", "angle")
            require((row["appearance_min"], row["appearance_max"]) ==
                    ((13, 32) if protocol == "R" else (1, 1)), "Talebjedi appearance frequencies differ")
            if protocol == "T":
                for metric in ("mae_um", "mape_pct", "rmse_um"):
                    close(row["pooled_" + metric], row["sample_equal_" + metric], "T aggregation mismatch")
            summaries.append(row)
        differences.append(compare_splits(xgb, other, protocol, "split_id", "ExpNo"))
    differences = pd.concat(differences, ignore_index=True)
    ranking = ranking_summary(differences)
    # Match the frozen extension's reported fields; do not add quartile
    # analyses merely because the generic DAFD helper can calculate them.
    for key in ("q25_delta_mae_um", "q75_delta_mae_um"):
        ranking["split_descriptive"]["R"].pop(key)
    out.mkdir(parents=True, exist_ok=False)
    export(pd.DataFrame(summaries), out / "model_summary.csv")
    export(differences, out / "split_differences.csv")
    dump(out / "ranking_summary.json", ranking)
    return dict(model_summary=summaries, ranking=ranking)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "summarize"))
    parser.add_argument("--data", type=Path, default=Path("replication/talebjedi2022/reconstructed/talebjedi_125_reconstructed.csv"))
    parser.add_argument("--protocol-r", type=Path, default=Path("replication/talebjedi2022/outputs/random_predictions.csv"))
    parser.add_argument("--protocol-t", type=Path, default=Path("replication/talebjedi2022/outputs/loao_predictions.csv"))
    parser.add_argument("--rf-predictions", type=Path, default=Path("outputs/revision/talebjedi_rf_fit/predictions.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data, r, t = [read(p) for p in (args.data, args.protocol_r, args.protocol_t)]
    if args.phase == "fit":
        fit_extension(data, r, t, args.output or Path("outputs/revision/talebjedi_rf_fit"))
    else:
        summarize(data, r, t, read(args.rf_predictions), args.output or Path("outputs/revision/talebjedi"))
    print("Talebjedi revision phase complete: " + args.phase)


if __name__ == "__main__":
    main()
