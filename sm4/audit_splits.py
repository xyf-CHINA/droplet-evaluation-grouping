"""Independently audit F/V split CSVs without importing their builder."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"{path} has no header")
        required = {"protocol", "fold", "role", "image_id", "video_id"}
        missing = required.difference(reader.fieldnames)
        require(not missing, f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def audit_protocol(
    rows: list[dict[str, str]], protocol: str, folds: int
) -> list[dict]:
    require(rows, f"{protocol}: empty split")
    require(
        all(row["protocol"] == protocol for row in rows),
        f"{protocol}: protocol mismatch",
    )
    image_ids = {row["image_id"] for row in rows}
    require(len(rows) == len(image_ids) * folds, f"{protocol}: unexpected row count")
    val_counts = Counter(row["image_id"] for row in rows if row["role"] == "val")
    train_counts = Counter(
        row["image_id"] for row in rows if row["role"] == "train"
    )
    require(
        set(val_counts) == image_ids
        and all(value == 1 for value in val_counts.values()),
        f"{protocol}: each image must be validation exactly once",
    )
    require(
        set(train_counts) == image_ids
        and all(value == folds - 1 for value in train_counts.values()),
        f"{protocol}: each image must be training exactly folds-1 times",
    )

    summary = []
    for fold in range(folds):
        current = [row for row in rows if int(row["fold"]) == fold]
        train = {row["image_id"] for row in current if row["role"] == "train"}
        val = {row["image_id"] for row in current if row["role"] == "val"}
        require(
            train | val == image_ids and not (train & val),
            f"{protocol} fold {fold}: partition error",
        )
        train_videos = {
            row["video_id"] for row in current if row["role"] == "train"
        }
        val_videos = {row["video_id"] for row in current if row["role"] == "val"}
        overlap = len(train_videos & val_videos)
        if protocol == "V":
            require(overlap == 0, f"V fold {fold}: sequence overlap is not zero")
        summary.append(
            {
                "protocol": protocol,
                "fold": fold,
                "train_n": len(train),
                "val_n": len(val),
                "val_sequence_count": len(val_videos),
                "sequence_overlap": overlap,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-f", type=Path, required=True)
    parser.add_argument("--protocol-v", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    f_summary = audit_protocol(read_csv(args.protocol_f), "F", args.folds)
    v_summary = audit_protocol(read_csv(args.protocol_v), "V", args.folds)
    require(
        [row["val_n"] for row in f_summary]
        == [row["val_n"] for row in v_summary],
        "F/V validation sizes do not match fold by fold",
    )
    print("PASS: split audit")
    for row in f_summary + v_summary:
        print(row)


if __name__ == "__main__":
    main()
