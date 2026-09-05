"""V2.2 post hoc descriptive sample-equal sensitivity analysis.

Reads the frozen Protocol R/G prediction CSVs without modifying them. Two
independent paths calculate prediction-row errors, collapse errors within each
original observation, and then average the 474 observation-level values with
equal weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TOLERANCE = 1e-10
EXPECTED_ROWS = 9_500
EXPECTED_RUNS = 100
EXPECTED_ROWS_PER_RUN = 95
EXPECTED_SAMPLES = 474
EXPECTED_GEOMETRIES = 35
EXPECTED_APPEARANCE_RANGES = {"R": (7, 32), "G": (2, 53)}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "dafd" / "sample_equal_recomputed"

DEFAULT_INPUTS = {
    "R": {
        "path": PROJECT_ROOT / "results" / "dafd" / "predictions" / "protocol_R_predictions.csv",
        "sha256": "785003573283D2247715B386015B1BECF64D39B9F0C8F179FC608B02B6E98E58",
    },
    "G": {
        "path": PROJECT_ROOT / "results" / "dafd" / "predictions" / "protocol_G_predictions.csv",
        "sha256": "A14C085C655C27BA35B80BA440F2DCD52E52DF798B2C3CE0A9CADBF279150BF8",
    },
}

REQUIRED_COLUMNS = {
    "source_row_id",
    "geometry_id",
    "seed",
    "observed_um",
    "predicted_um",
    "absolute_error_um",
    "percentage_error_pct",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative_increase(new: float, reference: float) -> float:
    return 100.0 * (new / reference - 1.0)


def path_a_pandas(protocol: str, path: Path) -> dict:
    """Path A: use stored AE/APE columns and pandas groupby aggregation."""

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    require(not missing, f"{protocol}: missing columns: {sorted(missing)}")
    require(len(df) == EXPECTED_ROWS, f"{protocol}: expected 9500 rows, got {len(df)}")
    require(df[list(REQUIRED_COLUMNS)].notna().all().all(), f"{protocol}: missing required values")

    for column in ("observed_um", "predicted_um", "absolute_error_um", "percentage_error_pct"):
        require(np.isfinite(df[column].to_numpy(dtype=float)).all(), f"{protocol}: non-finite {column}")

    run_sizes = df.groupby("seed", sort=True).size()
    require(len(run_sizes) == EXPECTED_RUNS, f"{protocol}: expected 100 runs")
    require((run_sizes == EXPECTED_ROWS_PER_RUN).all(), f"{protocol}: run size not equal to 95")
    require(not df.duplicated(["seed", "source_row_id"]).any(), f"{protocol}: duplicate sample within run")

    sample_groups = df.groupby("source_row_id", sort=True)
    require(sample_groups.ngroups == EXPECTED_SAMPLES, f"{protocol}: expected 474 samples")
    require((sample_groups["observed_um"].nunique() == 1).all(), f"{protocol}: observed target varies within sample")
    require((sample_groups["geometry_id"].nunique() == 1).all(), f"{protocol}: geometry varies within sample")
    require((df["observed_um"] > 0).all(), f"{protocol}: non-positive MAPE denominator")
    require(df["geometry_id"].nunique() == EXPECTED_GEOMETRIES, f"{protocol}: expected 35 geometries")

    reconstructed_ae = (df["observed_um"] - df["predicted_um"]).abs()
    reconstructed_ape = reconstructed_ae / df["observed_um"] * 100.0
    max_stored_ae_diff = float((df["absolute_error_um"] - reconstructed_ae).abs().max())
    max_stored_ape_diff = float((df["percentage_error_pct"] - reconstructed_ape).abs().max())
    require(max_stored_ae_diff <= TOLERANCE, f"{protocol}: stored AE discrepancy {max_stored_ae_diff}")
    require(max_stored_ape_diff <= TOLERANCE, f"{protocol}: stored APE discrepancy {max_stored_ape_diff}")

    appearances = sample_groups.size().rename("appearances")
    require(int(appearances.sum()) == EXPECTED_ROWS, f"{protocol}: appearance counts do not sum to 9500")
    observed_range = (int(appearances.min()), int(appearances.max()))
    require(
        observed_range == EXPECTED_APPEARANCE_RANGES[protocol],
        f"{protocol}: appearance range {observed_range} != {EXPECTED_APPEARANCE_RANGES[protocol]}",
    )

    sample_errors = sample_groups[["absolute_error_um", "percentage_error_pct"]].mean()
    sample_meta = sample_groups[["observed_um"]].first().join(sample_groups[["geometry_id"]].first())
    sample_table = sample_meta.join(appearances).join(sample_errors).reset_index()
    sample_table.insert(0, "protocol", protocol)

    return {
        "rows": len(df),
        "runs": len(run_sizes),
        "samples": sample_groups.ngroups,
        "geometries": int(df["geometry_id"].nunique()),
        "min_appearances": observed_range[0],
        "max_appearances": observed_range[1],
        "pooled_mae_um": float(df["absolute_error_um"].mean()),
        "pooled_mape_pct": float(df["percentage_error_pct"].mean()),
        "sample_equal_mae_um": float(sample_errors["absolute_error_um"].mean()),
        "sample_equal_mape_pct": float(sample_errors["percentage_error_pct"].mean()),
        "max_stored_ae_discrepancy": max_stored_ae_diff,
        "max_stored_ape_discrepancy": max_stored_ape_diff,
        "sample_table": sample_table,
    }


def path_b_csv(protocol: str, path: Path) -> dict:
    """Path B: reconstruct AE/APE and aggregate with csv + math.fsum."""

    errors: dict[str, dict[str, list[float] | str | float]] = defaultdict(
        lambda: {"ae": [], "ape": [], "geometry_id": "", "observed_um": math.nan}
    )
    run_samples: dict[str, set[str]] = defaultdict(set)
    geometry_ids: set[str] = set()
    row_count = 0
    max_stored_ae_diff = 0.0
    max_stored_ape_diff = 0.0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"{protocol}: missing CSV header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        require(not missing, f"{protocol}: missing columns in path B: {sorted(missing)}")

        for row in reader:
            row_count += 1
            sample_id = row["source_row_id"]
            geometry_id = row["geometry_id"]
            seed = row["seed"]
            observed = float(row["observed_um"])
            predicted = float(row["predicted_um"])
            stored_ae = float(row["absolute_error_um"])
            stored_ape = float(row["percentage_error_pct"])

            require(observed > 0.0, f"{protocol}: non-positive observed value")
            require(sample_id not in run_samples[seed], f"{protocol}: duplicate {sample_id} in run {seed}")
            run_samples[seed].add(sample_id)
            geometry_ids.add(geometry_id)

            ae = abs(observed - predicted)
            ape = ae / observed * 100.0
            max_stored_ae_diff = max(max_stored_ae_diff, abs(stored_ae - ae))
            max_stored_ape_diff = max(max_stored_ape_diff, abs(stored_ape - ape))

            record = errors[sample_id]
            if record["geometry_id"]:
                require(record["geometry_id"] == geometry_id, f"{protocol}: geometry mismatch for {sample_id}")
                require(float(record["observed_um"]) == observed, f"{protocol}: observed mismatch for {sample_id}")
            else:
                record["geometry_id"] = geometry_id
                record["observed_um"] = observed
            record["ae"].append(ae)  # type: ignore[union-attr]
            record["ape"].append(ape)  # type: ignore[union-attr]

    require(row_count == EXPECTED_ROWS, f"{protocol}: path B expected 9500 rows")
    require(len(run_samples) == EXPECTED_RUNS, f"{protocol}: path B expected 100 runs")
    require(all(len(values) == EXPECTED_ROWS_PER_RUN for values in run_samples.values()), f"{protocol}: path B run size mismatch")
    require(len(errors) == EXPECTED_SAMPLES, f"{protocol}: path B expected 474 samples")
    require(len(geometry_ids) == EXPECTED_GEOMETRIES, f"{protocol}: path B expected 35 geometries")
    require(max_stored_ae_diff <= TOLERANCE, f"{protocol}: path B AE discrepancy")
    require(max_stored_ape_diff <= TOLERANCE, f"{protocol}: path B APE discrepancy")

    sample_rows = []
    for sample_id in sorted(errors, key=lambda value: int(value)):
        record = errors[sample_id]
        ae_values = record["ae"]
        ape_values = record["ape"]
        appearances = len(ae_values)  # type: ignore[arg-type]
        sample_rows.append(
            {
                "source_row_id": sample_id,
                "geometry_id": record["geometry_id"],
                "observed_um": float(record["observed_um"]),
                "appearances": appearances,
                "absolute_error_um": math.fsum(ae_values) / appearances,  # type: ignore[arg-type]
                "percentage_error_pct": math.fsum(ape_values) / appearances,  # type: ignore[arg-type]
            }
        )

    appearance_values = [int(row["appearances"]) for row in sample_rows]
    observed_range = (min(appearance_values), max(appearance_values))
    require(sum(appearance_values) == EXPECTED_ROWS, f"{protocol}: path B appearance sum mismatch")
    require(observed_range == EXPECTED_APPEARANCE_RANGES[protocol], f"{protocol}: path B appearance range mismatch")

    return {
        "sample_equal_mae_um": math.fsum(float(row["absolute_error_um"]) for row in sample_rows)
        / EXPECTED_SAMPLES,
        "sample_equal_mape_pct": math.fsum(float(row["percentage_error_pct"]) for row in sample_rows)
        / EXPECTED_SAMPLES,
        "max_stored_ae_discrepancy": max_stored_ae_diff,
        "max_stored_ape_discrepancy": max_stored_ape_diff,
        "sample_rows": sample_rows,
    }


def write_outputs(
    results: dict,
    sample_tables: list[pd.DataFrame],
    input_hashes: dict[str, str],
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        output_dir / "sample_equal_summary.csv",
        output_dir / "sample_equal_sample_level.csv",
        output_dir / "sample_equal_audit.json",
    ]
    if not overwrite:
        for target in targets:
            require(not target.exists(), f"Refusing to overwrite existing output: {target}")

    summary_rows = []
    for protocol in ("R", "G"):
        result = results[protocol]
        summary_rows.append(
            {
                "protocol": protocol,
                "n_rows": result["rows"],
                "n_runs": result["runs"],
                "n_samples": result["samples"],
                "n_geometries": result["geometries"],
                "min_appearances": result["min_appearances"],
                "max_appearances": result["max_appearances"],
                "pooled_mae_um": result["pooled_mae_um"],
                "pooled_mape_pct": result["pooled_mape_pct"],
                "sample_equal_mae_um": result["sample_equal_mae_um"],
                "sample_equal_mape_pct": result["sample_equal_mape_pct"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(targets[0], index=False, float_format="%.15g")
    pd.concat(sample_tables, ignore_index=True).to_csv(targets[1], index=False, float_format="%.15g")

    r_result = results["R"]
    g_result = results["G"]
    audit = {
        "status": "PASS",
        "analysis_role": "post hoc descriptive sample-equal sensitivity analysis",
        "aggregation_order": "prediction-level AE/APE -> within-sample mean AE/APE -> equal-weight mean over 474 samples",
        "tolerance": TOLERANCE,
        "input_sha256": input_hashes,
        "protocols": {
            protocol: {key: value for key, value in results[protocol].items() if key != "sample_table"}
            for protocol in ("R", "G")
        },
        "contrasts": {
            "sample_equal_mae_difference_um_G_minus_R": g_result["sample_equal_mae_um"]
            - r_result["sample_equal_mae_um"],
            "sample_equal_mae_relative_increase_pct": relative_increase(
                g_result["sample_equal_mae_um"], r_result["sample_equal_mae_um"]
            ),
            "sample_equal_mape_difference_pp_G_minus_R": g_result["sample_equal_mape_pct"]
            - r_result["sample_equal_mape_pct"],
            "sample_equal_mape_relative_increase_pct": relative_increase(
                g_result["sample_equal_mape_pct"], r_result["sample_equal_mape_pct"]
            ),
        },
    }
    with targets[2].open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-r",
        type=Path,
        default=DEFAULT_INPUTS["R"]["path"],
        help="Protocol R prediction CSV.",
    )
    parser.add_argument(
        "--protocol-g",
        type=Path,
        default=DEFAULT_INPUTS["G"]["path"],
        help="Protocol G prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for recomputed outputs.",
    )
    parser.add_argument(
        "--skip-locked-hash-check",
        action="store_true",
        help="Allow equivalent user-generated prediction files with different byte hashes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.protocol_r).is_file() or not Path(args.protocol_g).is_file():
        raise SystemExit(
            "Protocol R/G prediction CSVs are not distributed with this "
            "repository. Obtain the DAFD 3.0 source data (see "
            "DATA_ACQUISITION.md), regenerate the predictions with "
            "src/models/stage1b_se_random_baseline.py (R) and "
            "src/evaluation/stage2_protocol_benchmark.py (G), then pass the "
            "generated files with --protocol-r and --protocol-g (see the "
            "exact command in DATA_ACQUISITION.md). You may instead copy "
            "them to the default results/dafd/predictions paths."
        )
    inputs = {
        "R": {"path": args.protocol_r, "sha256": DEFAULT_INPUTS["R"]["sha256"]},
        "G": {"path": args.protocol_g, "sha256": DEFAULT_INPUTS["G"]["sha256"]},
    }
    input_hashes = {}
    results = {}
    sample_tables = []

    for protocol in ("R", "G"):
        path = Path(inputs[protocol]["path"])
        require(path.is_file(), f"Missing input: {path}")
        actual_hash = sha256(path)
        if not args.skip_locked_hash_check:
            require(actual_hash == inputs[protocol]["sha256"], f"{protocol}: input hash mismatch")
        input_hashes[protocol] = actual_hash

        path_a = path_a_pandas(protocol, path)
        path_b = path_b_csv(protocol, path)

        max_sample_ae_diff = 0.0
        max_sample_ape_diff = 0.0
        path_a_by_sample = path_a["sample_table"].set_index("source_row_id")
        require(set(path_a_by_sample.index.astype(str)) == {row["source_row_id"] for row in path_b["sample_rows"]}, f"{protocol}: A/B sample set mismatch")
        for row in path_b["sample_rows"]:
            source_row_id = int(row["source_row_id"])
            a_row = path_a_by_sample.loc[source_row_id]
            max_sample_ae_diff = max(
                max_sample_ae_diff,
                abs(float(a_row["absolute_error_um"]) - float(row["absolute_error_um"])),
            )
            max_sample_ape_diff = max(
                max_sample_ape_diff,
                abs(float(a_row["percentage_error_pct"]) - float(row["percentage_error_pct"])),
            )
        aggregate_ae_diff = abs(path_a["sample_equal_mae_um"] - path_b["sample_equal_mae_um"])
        aggregate_ape_diff = abs(path_a["sample_equal_mape_pct"] - path_b["sample_equal_mape_pct"])
        require(max_sample_ae_diff <= TOLERANCE, f"{protocol}: A/B sample AE discrepancy")
        require(max_sample_ape_diff <= TOLERANCE, f"{protocol}: A/B sample APE discrepancy")
        require(aggregate_ae_diff <= TOLERANCE, f"{protocol}: A/B aggregate MAE discrepancy")
        require(aggregate_ape_diff <= TOLERANCE, f"{protocol}: A/B aggregate MAPE discrepancy")

        path_a.update(
            {
                "path_b_sample_equal_mae_um": path_b["sample_equal_mae_um"],
                "path_b_sample_equal_mape_pct": path_b["sample_equal_mape_pct"],
                "max_A_B_sample_ae_discrepancy": max_sample_ae_diff,
                "max_A_B_sample_ape_discrepancy": max_sample_ape_diff,
                "A_B_aggregate_mae_discrepancy": aggregate_ae_diff,
                "A_B_aggregate_mape_discrepancy": aggregate_ape_diff,
            }
        )
        results[protocol] = path_a
        sample_tables.append(path_a["sample_table"])

    require(
        set(results["R"]["sample_table"]["source_row_id"])
        == set(results["G"]["sample_table"]["source_row_id"]),
        "Protocol R/G sample sets differ",
    )
    write_outputs(results, sample_tables, input_hashes, args.output_dir, args.overwrite)

    print("PASS: post hoc sample-equal sensitivity analysis")
    for protocol in ("R", "G"):
        print(
            f"{protocol}: MAE={results[protocol]['sample_equal_mae_um']:.10f}; "
            f"MAPE={results[protocol]['sample_equal_mape_pct']:.10f}; "
            f"appearances={results[protocol]['min_appearances']}-{results[protocol]['max_appearances']}"
        )
    print(
        "Contrasts: "
        f"MAE +{relative_increase(results['G']['sample_equal_mae_um'], results['R']['sample_equal_mae_um']):.2f}%; "
        f"MAPE +{relative_increase(results['G']['sample_equal_mape_pct'], results['R']['sample_equal_mape_pct']):.2f}%"
    )


if __name__ == "__main__":
    main()
