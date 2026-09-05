"""Data-free regressions for revision formulas and released aggregate claims."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.public_release_audit import path_problems
from src.revision.analysis import ATOL, aggregate, compare_splits, ecdf, ranking_summary, tolerance_curves

RESULTS = Path(__file__).resolve().parents[1] / "results/revision"


def test_indicator_before_sample_average():
    # A has errors 0 and 20, B has one error 9: at tau=10 the answer is .75,
    # not the result of first thresholding A's mean absolute error.
    assert np.isclose(ecdf([0, 20, 9], [.25, .25, .5], [10])[0], .75, atol=ATOL, rtol=0)


def test_aggregate_does_not_weight_frequent_observations_equally_with_rare_ones():
    frame = pd.DataFrame(dict(sample=[1, 1, 2], group=[1, 1, 2], observed_um=[100., 100., 100.],
                              predicted_um=[100., 120., 109.]))
    got = aggregate(frame, "R", "XGB", "sample", "group")
    assert got["pooled_mae_um"] == pytest.approx(29/3)
    assert got["sample_equal_mae_um"] == pytest.approx(9.5)


def test_mismatched_pairing_rejected():
    a = pd.DataFrame(dict(seed=[0, 0], sample=[1, 2], observed_um=[10., 20.], predicted_um=[11., 21.]))
    b = a.copy()
    b.loc[1, "sample"] = 3
    with pytest.raises(ValueError, match="identities"):
        compare_splits(a, b, "R", "seed", "sample")


def test_duplicate_member_rejected():
    a = pd.DataFrame(dict(seed=[0, 0], sample=[1, 1], observed_um=[10., 10.], predicted_um=[11., 12.]))
    with pytest.raises(ValueError, match="Duplicate"):
        compare_splits(a, a, "R", "seed", "sample")


def test_negative_or_zero_percentage_denominator_rejected():
    a = pd.DataFrame(dict(sample=[1], group=[1], observed_um=[0.], predicted_um=[1.]))
    with pytest.raises(ValueError, match="positive"):
        aggregate(a, "R", "XGB", "sample", "group")


def test_threshold_curve_covers_tail_and_reconciles_integral():
    def frame(pred):
        f = pd.DataFrame(dict(source_row_id=[1, 1, 2], observed_um=[100., 100., 100.], predicted_um=pred))
        f["absolute_error_um"] = abs(f.predicted_um-f.observed_um)
        return f
    curve, summary = tolerance_curves(frame([100., 120., 109.]), frame([105., 130., 112.]))
    assert curve.threshold_um.iloc[-1] == 30.
    assert np.allclose(curve.iloc[-1][["R_sample_equal_fraction", "G_sample_equal_fraction"]], 1.)
    for protocol in ("R", "G"):
        d = summary["protocols"][protocol]
        assert d["survival_integral_um"] == pytest.approx(d["sample_equal_MAE_um"])


def test_all_model_orderings_and_unfavorable_angle_counts_preserved():
    d = pd.read_csv(RESULTS / "dafd_model_summary.csv")
    for weight in ("pooled_mae_um", "sample_equal_mae_um"):
        assert list(d[d.protocol == "R"].sort_values(weight).model) == ["XGB", "RF", "MLP"]
        assert list(d[d.protocol == "G"].sort_values(weight).model) == ["RF", "XGB", "MLP"]
    t = pd.read_csv(RESULTS / "talebjedi_split_differences.csv")
    assert len(t[t.protocol == "T"]) == 5
    assert t[t.protocol == "T"].winner.eq("RF").sum() == 2
    summary = ranking_summary(t)
    assert "leave_one_split_out_mean_range_um" not in summary["split_descriptive"]["T"]


def test_geometry_omission_rows_do_not_invent_intervals():
    rows = json.loads((RESULTS / "revision_aggregate_summary.json").read_text())["dafd"]["geometry_seen_unseen"]
    assert [r["n_geometries"] for r in rows] == [32, 31, 29]
    assert rows[0]["bootstrap_B"] == 10000 and rows[0]["bootstrap_seed"] == 0
    assert rows[0]["bootstrap_percentile_interval_um"] is not None
    for row in rows[1:]:
        assert row["bootstrap_percentile_interval_um"] is None
        assert row["bootstrap_B"] is None and row["bootstrap_seed"] is None


def test_release_allowlist_does_not_open_row_level_outputs():
    for f in RESULTS.iterdir():
        assert path_problems("results/revision/" + f.name) == []
    assert path_problems("results/revision/predictions.csv")
    assert path_problems("results/revision/sample_errors.csv")
    for f in RESULTS.glob("*.csv"):
        columns = set(pd.read_csv(f, nrows=0).columns)
        assert not columns.intersection({"source_row_id", "ExpNo", "observed_um", "predicted_um", "sequence_id", "image_id"})
