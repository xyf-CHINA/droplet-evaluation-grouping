"""Recompute aggregate SM4 protocol summaries from fold metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "video" / "fold_metrics.csv"
METRICS = ("mAP50", "mAP50_95", "class_0_AP50", "class_1_AP50")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"protocol", "fold", *METRICS}
        require(reader.fieldnames is not None, "metrics CSV has no header")
        require(
            not required.difference(reader.fieldnames),
            "metrics CSV has missing columns",
        )
        rows = []
        for source in reader:
            row = {"protocol": source["protocol"], "fold": int(source["fold"])}
            for metric in METRICS:
                value = float(source[metric])
                require(math.isfinite(value), f"non-finite {metric}")
                row[metric] = value
            rows.append(row)
    require(len(rows) == 10, "expected ten fold rows")
    require(
        {row["protocol"] for row in rows} == {"F", "V"},
        "protocols must be F and V",
    )
    for protocol in ("F", "V"):
        folds = sorted(row["fold"] for row in rows if row["protocol"] == protocol)
        require(folds == list(range(5)), f"{protocol}: folds must be 0..4")
    require(
        len({(row["protocol"], row["fold"]) for row in rows}) == 10,
        "duplicate fold row",
    )
    return rows


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "sample_sd": statistics.stdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(rows: list[dict]) -> dict:
    summaries = {}
    for protocol in ("F", "V"):
        selected = [row for row in rows if row["protocol"] == protocol]
        summaries[protocol] = {
            metric: describe([row[metric] for row in selected])
            for metric in METRICS
        }
    criterion = (
        summaries["V"]["mAP50"]["mean"]
        < summaries["F"]["mAP50"]["mean"]
        and summaries["V"]["mAP50_95"]["mean"]
        < summaries["F"]["mAP50_95"]["mean"]
    )
    return {
        "protocol_summaries": summaries,
        "directional_criterion": {
            "definition": (
                "both mean(V mAP@0.5) < mean(F mAP@0.5) and "
                "mean(V mAP@0.5:0.95) < mean(F mAP@0.5:0.95)"
            ),
            "satisfied": criterion,
        },
        "paired_tests_performed": "none; equally numbered F/V folds are not paired",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(load_rows(args.input))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
