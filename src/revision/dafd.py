"""Recompute all DAFD revision summaries from locally generated predictions.

Usage: python -m src.revision.dafd --output outputs/revision/dafd
No models are fitted. The complete tolerance curve remains a local output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis import aggregate, compare_splits, dump, export, ranking_summary, read, require, tolerance_curves


def run_dafd(r, g, stage3, out):
    out = Path(out)
    require(not out.exists(), "Refusing to overwrite an existing output directory")
    require(set(stage3.model) == {"RF", "MLP"} and len(stage3) == 38000, "Expected complete RF/MLP predictions")
    summaries, differences, xgbs = [], [], {}
    identity = ["seed", "source_row_id", "comp_experiment_id", "se_experiment_id", "geometry_id",
                "fold", "split", "geometry_seen_in_train", "observed_um"]
    for protocol, xgb in (("R", r), ("G", g)):
        xgb = xgb.sort_values(["seed", "source_row_id"]).reset_index(drop=True)
        xgbs[protocol] = xgb
        for model in ("XGB", "RF", "MLP"):
            df = xgb if model == "XGB" else stage3[(stage3.model == model) & (stage3.protocol == protocol)].sort_values(
                ["seed", "source_row_id"]).reset_index(drop=True)
            require(len(df) == 9500 and set(df.seed) == set(range(100)), "Expected 100 DAFD splits")
            require(df.groupby("seed").size().eq(95).all(), "Expected 95 test rows per split")
            require(not df.duplicated(["seed", "source_row_id"]).any(), "Duplicate split/sample")
            require(df[identity].equals(xgb[identity]), "Model test identity mismatch")
            require(set(df.protocol) == {protocol} and set(df.split) == {"test"}, "Protocol labels differ")
            require(df.source_row_id.nunique() == 474 and df.geometry_id.nunique() == 35, "DAFD cohort mismatch")
            if protocol == "G":
                require(not df.geometry_seen_in_train.any(), "G geometry exposure is nonzero")
            result = aggregate(df, protocol, model, "source_row_id", "geometry_id")
            require((result["appearance_min"], result["appearance_max"]) ==
                    ((7, 32) if protocol == "R" else (2, 53)), "Test-appearance frequencies differ")
            summaries.append(result)
            if model == "RF":
                differences.append(compare_splits(xgb, df, protocol, "seed", "source_row_id"))
    cohort_fields = ["geometry_id", "observed_um"]
    cohorts = [xgbs[p].drop_duplicates("source_row_id").set_index("source_row_id")[cohort_fields].sort_index()
               for p in ("R", "G")]
    require(cohorts[0].equals(cohorts[1]), "R and G do not represent the same original cohort")
    summary = pd.DataFrame(summaries)
    difference = pd.concat(differences, ignore_index=True)
    ranking = ranking_summary(difference)
    curve, tolerance = tolerance_curves(xgbs["R"], xgbs["G"])
    out.mkdir(parents=True, exist_ok=False)
    export(summary, out / "model_summary.csv")
    export(difference, out / "split_differences.csv")
    dump(out / "ranking_summary.json", ranking)
    export(curve, out / "tolerance_curves.csv")
    dump(out / "tolerance_summary.json", tolerance)
    return dict(model_summary=summaries, ranking=ranking, tolerance=tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-r", type=Path, default=Path("outputs/baseline/stage1b_predictions.csv"))
    parser.add_argument("--protocol-g", type=Path, default=Path("outputs/geometry_shift/stage2_protocol_g_predictions.csv"))
    parser.add_argument("--stage3", type=Path, default=Path("outputs/geometry_shift/stage3_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/revision/dafd"))
    args = parser.parse_args()
    run_dafd(read(args.protocol_r), read(args.protocol_g), read(args.stage3), args.output)
    print("DAFD revision summaries complete; no model training performed.")


if __name__ == "__main__":
    main()
