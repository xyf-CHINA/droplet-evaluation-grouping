"""Portable revision-analysis formulas, without training or private data.

All comparisons pair models only inside the same protocol and test split.
Repeated test appearances are descriptive, not independent observations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ATOL = 1e-10


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, atol=ATOL, rtol=0), message)


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def export(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def errors(df, sample_col, group_col):
    needed = [sample_col, group_col, "observed_um", "predicted_um"]
    require(not df[needed].isna().any().any(), "Missing required prediction fields")
    require(np.isfinite(df[["observed_um", "predicted_um"]]).all().all(), "Nonfinite predictions")
    require((df.observed_um > 0).all(), "MAPE requires positive observed diameters")
    require(df.groupby(sample_col)[[group_col, "observed_um"]].nunique().eq(1).all().all(),
            "Original-observation identity is not constant")
    result = df.copy()
    result["ae"] = np.abs(df.predicted_um - df.observed_um)
    result["ape"] = 100 * result.ae / np.abs(df.observed_um)
    result["se"] = result.ae ** 2
    if "absolute_error_um" in df:
        close(result.ae, df.absolute_error_um, "Stored/reconstructed absolute errors differ")
    if "percentage_error_pct" in df:
        close(result.ape, df.percentage_error_pct, "Stored/reconstructed percentage errors differ")
    return result


def aggregate(df, protocol, model, sample_col, group_col):
    """Return one aggregate row; no row-level values are exported here."""
    e = errors(df, sample_col, group_col)
    s = e.groupby(sample_col, sort=True).agg(
        count=("ae", "size"), ae=("ae", "mean"), ape=("ape", "mean"), se=("se", "mean"))
    return dict(protocol=protocol, model=model, n_predictions=len(e), n_samples=len(s),
                n_groups=int(e[group_col].nunique()), appearance_min=int(s["count"].min()),
                appearance_max=int(s["count"].max()), pooled_mae_um=float(e.ae.mean()),
                sample_equal_mae_um=float(s.ae.mean()), pooled_mape_pct=float(e.ape.mean()),
                sample_equal_mape_pct=float(s.ape.mean()), pooled_rmse_um=float(np.sqrt(e.se.mean())),
                sample_equal_rmse_um=float(np.sqrt(s.se.mean())))


def compare_splits(xgb, rf, protocol, split_col, sample_col):
    """Pair XGB and RF within a protocol; never pair Protocol R with G/T."""
    keys = [split_col, sample_col]
    for df in (xgb, rf):
        require(not df.duplicated(keys).any(), "Duplicate test member within a split")
    a = xgb.sort_values(keys).reset_index(drop=True)
    b = rf.sort_values(keys).reset_index(drop=True)
    require(a[keys + ["observed_um"]].equals(b[keys + ["observed_um"]]),
            "XGB/RF test identities or observed targets differ")
    require(np.isfinite(a[["observed_um", "predicted_um"]]).all().all(), "Nonfinite XGB values")
    require(np.isfinite(b[["observed_um", "predicted_um"]]).all().all(), "Nonfinite RF values")
    require((a.observed_um > 0).all(), "Nonpositive denominator")
    records = []
    for sid, left in a.groupby(split_col, sort=True):
        right = b[b[split_col] == sid]
        ae_a = np.abs(left.predicted_um.to_numpy() - left.observed_um.to_numpy())
        ae_b = np.abs(right.predicted_um.to_numpy() - right.observed_um.to_numpy())
        delta = float(ae_a.mean() - ae_b.mean())
        records.append(dict(protocol=protocol, split_id=int(sid), n_test=len(left),
            mae_xgb_um=float(ae_a.mean()), mae_rf_um=float(ae_b.mean()), delta_mae_um=delta,
            mape_xgb_pct=float((100 * ae_a / np.abs(left.observed_um.to_numpy())).mean()),
            mape_rf_pct=float((100 * ae_b / np.abs(right.observed_um.to_numpy())).mean()),
            winner="XGB" if delta < -ATOL else "RF" if delta > ATOL else "tie"))
    return pd.DataFrame(records)


def ranking_summary(split_differences):
    """Include all splits and the leave-one-split-out/extreme-split checks."""
    result = {}
    for protocol, frame in split_differences.groupby("protocol", sort=True):
        d = frame.delta_mae_um.to_numpy()
        require(len(d) > 1 and np.isfinite(d).all(), "Invalid split differences")
        omitted = (d.sum() - d) / (len(d) - 1)
        total = np.abs(d).sum()
        shares = np.sort(np.abs(d))[::-1] / total if total else np.zeros(len(d))
        result[protocol] = dict(n_splits=len(d), mean_delta_mae_um=float(d.mean()),
            median_delta_mae_um=float(np.median(d)), sd_delta_mae_um=float(d.std(ddof=1)),
            min_delta_mae_um=float(d.min()), max_delta_mae_um=float(d.max()),
            q25_delta_mae_um=float(np.quantile(d, .25)), q75_delta_mae_um=float(np.quantile(d, .75)),
            xgb_lower_count=int((d < -ATOL).sum()), rf_lower_count=int((d > ATOL).sum()),
            numerical_tie_count=int((np.abs(d) <= ATOL).sum()),
            leave_one_split_out_mean_range_um=[float(omitted.min()), float(omitted.max())],
            top_abs_contribution_shares={str(k): float(shares[:k].sum()) for k in (1, 5, 10) if k <= len(d)})
        # The five-angle extension reported all five deltas, not a new
        # leave-one-angle-out influence or quartile sensitivity analysis.
        if protocol == "T":
            for key in ("q25_delta_mae_um", "q75_delta_mae_um", "leave_one_split_out_mean_range_um",
                        "top_abs_contribution_shares"):
                result[protocol].pop(key)
    return dict(post_hoc=True, atol=ATOL, rtol=0, delta_definition="MAE_XGB - MAE_RF",
                split_descriptive=result,
                interpretation="Fixed pipelines; descriptive counts, not win probabilities or causal effects.")


def ecdf(errors, weights, thresholds):
    """Right-continuous weighted empirical CDF, including equality at tau."""
    errors, weights, thresholds = map(np.asarray, (errors, weights, thresholds))
    require(len(errors) == len(weights) and len(errors) > 0, "CDF input lengths")
    require(np.isfinite(errors).all() and np.isfinite(weights).all() and np.isfinite(thresholds).all(),
            "CDF inputs must be finite")
    require((weights >= 0).all(), "CDF weights must be nonnegative")
    order = np.argsort(errors, kind="stable")
    return np.r_[0., np.cumsum(weights[order])][np.searchsorted(errors[order], thresholds, side="right")]


def tolerance_curves(r, g, sample_col="source_row_id"):
    """Indicator first, within-sample average second, equal sample average last.

    The exact full-tail curve is regenerated locally; only its summary is
    included in the public release. No application tolerance is selected.
    """
    frames = {"R": r, "G": g}
    thresholds = np.unique(np.r_[0., r.absolute_error_um, g.absolute_error_um])
    table = pd.DataFrame({"threshold_um": thresholds})
    info = dict(threshold_count=len(thresholds), units="micrometers", difference_definition="F_R - F_G",
                post_hoc=True, protocols={})
    close(ecdf([0., 20., 9.], [.25, .25, .5], [10.]), [.75], "Threshold aggregation regression")
    for protocol, df in frames.items():
        ae = df.absolute_error_um.to_numpy()
        close(ae, np.abs(df.observed_um - df.predicted_um), "Tolerance AE reconstruction")
        counts = df.groupby(sample_col)[sample_col].transform("size").to_numpy()
        n = df[sample_col].nunique()
        weights = 1. / (n * counts)
        close(weights.sum(), 1., "Sample-equal CDF weights")
        f = ecdf(ae, weights, thresholds)
        direct = np.zeros(len(thresholds))
        for _, sample in df.groupby(sample_col):
            direct += np.searchsorted(np.sort(sample.absolute_error_um), thresholds, side="right") / len(sample) / n
        close(f, direct, "Independent sample-wise CDF calculation")
        require((np.diff(f) >= -ATOL).all() and f.min() >= -ATOL and f.max() <= 1 + ATOL,
                "CDF monotonicity/range")
        close(f[-1], 1., "Complete-tail endpoint")
        area = float(np.sum(np.diff(thresholds) * (1 - f[:-1])))
        sample_equal_mae = float(df.groupby(sample_col).absolute_error_um.mean().mean())
        close(area, sample_equal_mae, "Survival integral equals sample-equal MAE")
        table[protocol + "_sample_equal_fraction"] = f
        info["protocols"][protocol] = dict(
            quantiles_um={"p" + str(int(q * 100)): float(thresholds[np.flatnonzero(f >= q)[0]]) for q in (.5, .9, .95, .99)},
            max_error_um=float(ae.max()), min_error_um=float(ae.min()), sample_equal_MAE_um=sample_equal_mae,
            survival_integral_um=area, distinct_error_count=len(np.unique(ae)),
            largest_repeated_error_count=int(pd.Series(ae).value_counts().max()))
    gap = table.R_sample_equal_fraction - table.G_sample_equal_fraction
    table["difference_R_minus_G"] = gap
    def point(i):
        return dict(threshold_um=float(thresholds[i]), R_fraction=float(table.R_sample_equal_fraction.iloc[i]),
                    G_fraction=float(table.G_sample_equal_fraction.iloc[i]), difference=float(gap.iloc[i]))
    info["maximum_observed_gap"] = point(int(np.argmax(gap)))
    info["minimum_observed_gap"] = point(int(np.argmin(gap)))
    signs = np.where(gap > ATOL, 1, np.where(gap < -ATOL, -1, 0))
    nz = np.flatnonzero(signs)
    info["sign_changes"] = [dict(previous_nonzero_threshold_um=float(thresholds[a]), threshold_um=float(thresholds[b]),
                                **{"from": int(signs[a]), "to": int(signs[b])})
                            for a, b in zip(nz[:-1], nz[1:]) if signs[a] != signs[b]]
    info["full_upper_um"] = float(thresholds[-1])
    info["both_equal_one_from_um"] = float(thresholds[-1])
    info["note"] = "The maximum gap is a descriptive extremum, not an application tolerance or deployment success rate."
    return table, info
