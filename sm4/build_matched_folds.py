"""Build size-matched frame-wise and sequence-isolated outer folds.

This public version implements the split construction used for SM4. It accepts
a user-supplied manifest and contains no study image or sequence identifiers.

The input manifest must contain ``image_id``, ``video_id``, and ``frame_id``.
Optional ``n_class0_boxes`` and ``n_class1_boxes`` columns enable the original
check that every validation fold contains both classes. The output split CSVs
retain identifiers from the input manifest and should remain local when the
input dataset is not public.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


REQUIRED_COLUMNS = {"image_id", "video_id", "frame_id"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "manifest has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        require(not missing, f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    require(rows, "manifest is empty")
    ids = [row["image_id"] for row in rows]
    require(all(ids), "image_id contains an empty value")
    require(len(ids) == len(set(ids)), "image_id values must be unique")
    require(all(row["video_id"] for row in rows), "video_id contains an empty value")
    return sorted(rows, key=lambda row: row["image_id"])


def class_counts(
    rows_by_id: dict[str, dict[str, str]], ids: list[str]
) -> tuple[int, int] | None:
    first = next(iter(rows_by_id.values()))
    columns = {"n_class0_boxes", "n_class1_boxes"}
    if not columns.issubset(first):
        return None
    return (
        sum(int(rows_by_id[item]["n_class0_boxes"]) for item in ids),
        sum(int(rows_by_id[item]["n_class1_boxes"]) for item in ids),
    )


def build_folds(
    rows: list[dict[str, str]], n_splits: int, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    ids = [row["image_id"] for row in rows]
    videos = [row["video_id"] for row in rows]
    by_id = {row["image_id"]: row for row in rows}
    require(len(set(videos)) >= n_splits, "fewer sequence groups than folds")

    gkf = GroupKFold(n_splits=n_splits)
    v_val: dict[int, list[str]] = {}
    for fold, (_, val_idx) in enumerate(gkf.split(ids, groups=videos)):
        val_ids = [ids[index] for index in val_idx]
        counts = class_counts(by_id, val_ids)
        if counts is not None:
            require(all(value > 0 for value in counts), f"V fold {fold} lacks a class")
        val_set = set(val_ids)
        train_videos = {
            by_id[item]["video_id"] for item in ids if item not in val_set
        }
        val_videos = {by_id[item]["video_id"] for item in val_ids}
        require(not (train_videos & val_videos), f"V fold {fold} has sequence overlap")
        v_val[fold] = val_ids

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ids))
    f_val: dict[int, list[str]] = {}
    offset = 0
    for fold in range(n_splits):
        size = len(v_val[fold])
        indices = permutation[offset : offset + size]
        offset += size
        f_val[fold] = [ids[index] for index in indices]
        counts = class_counts(by_id, f_val[fold])
        if counts is not None:
            require(all(value > 0 for value in counts), f"F fold {fold} lacks a class")
    require(offset == len(ids), "validation blocks do not cover the manifest")

    def rows_for(protocol: str, validation: dict[int, list[str]]) -> list[dict]:
        output: list[dict] = []
        for fold in range(n_splits):
            val_set = set(validation[fold])
            for image_id in ids:
                source = by_id[image_id]
                output.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "role": "val" if image_id in val_set else "train",
                        "image_id": image_id,
                        "video_id": source["video_id"],
                        "frame_id": source["frame_id"],
                    }
                )
        return output

    f_rows = rows_for("F", f_val)
    v_rows = rows_for("V", v_val)
    summary = []
    for protocol, validation in (("F", f_val), ("V", v_val)):
        for fold in range(n_splits):
            val_set = set(validation[fold])
            train_set = set(ids).difference(val_set)
            val_videos = {by_id[item]["video_id"] for item in val_set}
            train_videos = {by_id[item]["video_id"] for item in train_set}
            summary.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "train_n": len(train_set),
                    "val_n": len(val_set),
                    "val_sequence_count": len(val_videos),
                    "sequence_overlap": len(val_videos & train_videos),
                }
            )
    return f_rows, v_rows, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    require(args.folds >= 2, "--folds must be at least 2")
    targets = [
        args.output_dir / "protocol_F_folds.csv",
        args.output_dir / "protocol_V_folds.csv",
        args.output_dir / "split_summary.json",
    ]
    if not args.overwrite:
        require(
            not any(path.exists() for path in targets),
            "refusing to overwrite existing output",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    f_rows, v_rows, summary = build_folds(
        load_manifest(args.manifest), args.folds, args.seed
    )
    fields = ["protocol", "fold", "role", "image_id", "video_id", "frame_id"]
    write_csv(targets[0], f_rows, fields)
    write_csv(targets[1], v_rows, fields)
    targets[2].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {targets[0]}")
    print(f"Wrote {targets[1]}")
    print(f"Wrote {targets[2]}")


if __name__ == "__main__":
    main()
