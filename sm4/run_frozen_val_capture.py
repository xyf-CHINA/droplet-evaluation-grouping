"""Run the pinned YOLOv5 evaluator and capture full-precision metrics.

This public version retains the evaluation and class-AP capture logic used in
SM4. Paths must be supplied explicitly, and the summary JSON omits file paths,
checkpoint hashes, prediction hashes, and data filenames.

Local YOLO outputs may still contain data filenames. Keep the run directory
outside version control when using data that are not public.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


YOLO_COMMIT = "915bbf294bb74c859f0b41f1c23bc395014ea679"
VAL_PY_SHA256 = "e8db56f1e83b80f7b53d5f5e01faf5372988dca75b9bc62cabfc50c77ae37fa2"
METRICS_PY_SHA256 = "e4451751430c5aa0a97c268c57a1538a84cfc8f4d45b094f6bea3a62a71125d3"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_yolov5(yolo_dir: Path):
    require((yolo_dir / "val.py").is_file(), f"YOLOv5 val.py not found under {yolo_dir}")
    require(sha256(yolo_dir / "val.py") == VAL_PY_SHA256, "pinned val.py SHA-256 mismatch")
    metrics_path = yolo_dir / "utils" / "metrics.py"
    require(metrics_path.is_file(), "YOLOv5 utils/metrics.py is missing")
    require(sha256(metrics_path) == METRICS_PY_SHA256, "pinned metrics.py SHA-256 mismatch")
    sys.path.insert(0, str(yolo_dir.resolve()))
    val = importlib.import_module("val")
    general = importlib.import_module("utils.general")
    return val, general.check_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="fold_eval")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    require(args.weights.is_file(), "weights file not found")
    require(not args.out_json.exists(), "refusing to overwrite --out-json")
    require(not args.log.exists(), "refusing to overwrite --log")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    val, check_dataset = load_yolov5(args.yolo_dir)
    capture: dict[str, object] = {}
    original_ap_per_class = val.ap_per_class

    def capture_ap_per_class(*capture_args, **capture_kwargs):
        result = original_ap_per_class(*capture_args, **capture_kwargs)
        capture["ap"] = result[5]
        capture["ap_class"] = result[6]
        return result

    val.ap_per_class = capture_ap_per_class
    data_dict = check_dataset(str(args.data))
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    root_logger.handlers = [handler]
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            results, _maps, _timing = val.run(
                data=data_dict,
                weights=str(args.weights),
                batch_size=6,
                imgsz=960,
                conf_thres=0.001,
                iou_thres=0.6,
                max_det=300,
                task="val",
                device="",
                workers=8,
                single_cls=False,
                augment=False,
                verbose=False,
                save_txt=True,
                save_hybrid=False,
                save_conf=True,
                save_json=False,
                project=str(args.project),
                name=args.name,
                exist_ok=True,
                half=False,
                dnn=False,
            )
    finally:
        root_logger.handlers = saved_handlers
        val.ap_per_class = original_ap_per_class
    args.log.write_text(buffer.getvalue(), encoding="utf-8", errors="replace")

    mp, mr, map50, map5095 = (float(value) for value in results[:4])
    skipped = not capture
    class_values = {
        "class_0_AP50": None,
        "class_1_AP50": None,
        "class_0_AP50_95": None,
        "class_1_AP50_95": None,
    }
    if not skipped:
        ap_matrix = capture["ap"]
        ap_class = capture["ap_class"]
        row_by_class = {
            int(class_id): index for index, class_id in enumerate(ap_class)
        }
        require({0, 1}.issubset(row_by_class), "class AP capture is incomplete")
        class_values = {
            "class_0_AP50": float(ap_matrix[row_by_class[0], 0]),
            "class_1_AP50": float(ap_matrix[row_by_class[1], 0]),
            "class_0_AP50_95": float(ap_matrix[row_by_class[0]].mean()),
            "class_1_AP50_95": float(ap_matrix[row_by_class[1]].mean()),
        }

    output = {
        "artifact": "metrics_full_precision",
        "release_adaptation": True,
        "yolov5_commit": YOLO_COMMIT,
        "evaluator_source_sha256": {
            "val.py": VAL_PY_SHA256,
            "utils/metrics.py": METRICS_PY_SHA256,
        },
        "evaluation_config": {
            "imgsz": 960,
            "batch_size": 6,
            "precision": "FP32",
            "conf_thres": 0.001,
            "iou_thres": 0.6,
            "max_det": 300,
            "task": "val",
            "workers": 8,
            "single_cls": False,
            "augment": False,
        },
        "metrics": {
            "precision": mp,
            "recall": mr,
            "mAP50": map50,
            "mAP50_95": map5095,
            **class_values,
        },
        "evaluator_skipped_ap_per_class": skipped,
    }
    args.out_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print("Run directory and log may contain private filenames; do not commit them.")


if __name__ == "__main__":
    main()
