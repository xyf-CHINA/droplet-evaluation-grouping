"""Stage 4B post-hoc audit-only test: pin the locked reporting rounding
convention of the frozen r1_same_angle_sensitivity.csv delta columns.

Not a bug report. The frozen pipeline computes each delta column as
round(round(unseen, 4) - round(seen, 4), 4) — per-angle component
summaries are rounded to 4 decimal places first, then subtracted. The
alternative full-precision convention round(unseen_full - seen_full, 4)
differs in exactly 4 of 10 cells by 0.0001. This test (1) pins the
locked convention, (2) proves the maximum discrepancy of the alternative
is 0.0001, and (3) shows the component columns are convention-invariant.

It reads frozen artifacts only and writes nothing; no frozen output is
modified or replaced.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import pytest

MOD = Path(__file__).resolve().parents[1] / "replication" / "talebjedi2022"
ANGLES = ["30", "60", "90", "120", "150"]

# Requires the locally reconstructed SI table and predictions; skip on a
# fresh clone until the user obtains the official source and regenerates
# them (see DATA_ACQUISITION.md).
for _req in ("reconstructed/talebjedi_125_reconstructed.csv",
             "outputs/random_predictions.csv",
             "outputs/loao_predictions.csv"):
    if not (MOD / _req).exists():
        pytest.skip(
            "missing %s: obtain the official Talebjedi Supporting "
            "Information and regenerate (see DATA_ACQUISITION.md)" % _req,
            allow_module_level=True)


def _load(p):
    return list(csv.DictReader(Path(p).read_text().splitlines()))


def _frozen_pipeline_seen():
    """Row-equal weighting exactly as in r1_posthoc_sensitivity.py:
    per-row mean over test seeds, then per-angle mean over the 25 rows."""
    rec = {r["ExpNo"]: r["TiltAngle_deg"]
           for r in _load(MOD / "reconstructed" / "talebjedi_125_reconstructed.csv")}
    rand = _load(MOD / "outputs" / "random_predictions.csv")
    per_row = defaultdict(lambda: {"err": [], "pct": []})
    for r in rand:
        e = abs(float(r["observed_um"]) - float(r["predicted_um"]))
        per_row[r["ExpNo"]]["err"].append(e)
        per_row[r["ExpNo"]]["pct"].append(e / float(r["observed_um"]) * 100)
    seen = {}
    for a in ANGLES:
        exps = [e for e, ang in rec.items() if ang == a]
        assert len(exps) == 25
        seen[a] = {
            "mae": sum(sum(per_row[e]["err"]) / len(per_row[e]["err"])
                       for e in exps) / 25,
            "mape": sum(sum(per_row[e]["pct"]) / len(per_row[e]["pct"])
                        for e in exps) / 25,
        }
    return seen


def _frozen_pipeline_unseen():
    """Per-fold pooled metrics straight from the frozen LOAO predictions
    (25 rows per held-out angle), full precision."""
    loao_p = _load(MOD / "outputs" / "loao_predictions.csv")
    un = {}
    for a in ANGLES:
        rows = [r for r in loao_p if r["held_out_angle"] == a]
        assert len(rows) == 25
        un[a] = {
            "mae": sum(abs(float(r["observed_um"]) - float(r["predicted_um"]))
                       for r in rows) / 25,
            "mape": sum(abs(float(r["observed_um"]) - float(r["predicted_um"]))
                        / float(r["observed_um"]) for r in rows) / 25 * 100,
        }
    return un


SEEN, UNSEEN = _frozen_pipeline_seen(), _frozen_pipeline_unseen()
FROZEN = {r["TiltAngle_deg"]: r
          for r in _load(MOD / "outputs" / "r1_same_angle_sensitivity.csv")}


def test_locked_rounding_convention_pinned():
    """Every frozen delta cell equals round(round(unseen,4)-round(seen,4),4)."""
    for a in ANGLES:
        for metric, col in (("mae", "delta_mae_unseen_minus_seen"),
                            ("mape", "delta_mape_unseen_minus_seen")):
            expected = round(round(UNSEEN[a][metric], 4) - round(SEEN[a][metric], 4), 4)
            actual = float(FROZEN[a][col])
            assert actual == expected, f"{a} {col}: frozen={actual} != convention={expected}"


def test_full_precision_alternative_max_discrepancy_0001():
    """Audit-only comparison: the full-precision alternative differs from
    the frozen values in exactly the 4 known cells, each by 0.0001, with
    max absolute discrepancy == 0.0001."""
    expected_alternative = {
        ("30", "delta_mape_unseen_minus_seen"): 6.0832,
        ("60", "delta_mape_unseen_minus_seen"): 7.9844,
        ("90", "delta_mape_unseen_minus_seen"): 6.1675,
        ("150", "delta_mae_unseen_minus_seen"): 2.5294,
    }
    discrepancies = {}
    for a in ANGLES:
        for metric, col in (("mae", "delta_mae_unseen_minus_seen"),
                            ("mape", "delta_mape_unseen_minus_seen")):
            full = round(UNSEEN[a][metric] - SEEN[a][metric], 4)
            frozen = float(FROZEN[a][col])
            discrepancies[(a, col)] = abs(frozen - full)
            if (a, col) in expected_alternative:
                assert full == expected_alternative[(a, col)], (a, col, full)
                # 0.0001 is not exactly representable in binary float
                assert math.isclose(discrepancies[(a, col)], 0.0001, abs_tol=1e-9), (a, col)
            else:
                assert frozen == full, (a, col, "unexpected convention difference")
    assert len(expected_alternative) == 4
    assert math.isclose(max(discrepancies.values()), 0.0001, abs_tol=1e-9)


def test_component_columns_convention_invariant():
    """The per-angle component summaries are identical under both
    conventions — only the derivative delta cells can differ."""
    for a in ANGLES:
        assert round(SEEN[a]["mae"], 4) == float(FROZEN[a]["seen_mae"]), a
        assert round(SEEN[a]["mape"], 4) == float(FROZEN[a]["seen_mape"]), a
        assert round(UNSEEN[a]["mae"], 4) == float(FROZEN[a]["unseen_mae"]), a
        assert round(UNSEEN[a]["mape"], 4) == float(FROZEN[a]["unseen_mape"]), a
