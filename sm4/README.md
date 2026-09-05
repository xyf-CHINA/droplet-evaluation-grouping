# SM4 video-evaluation workflow

SM4 examines how the grouping unit affects evaluation in a video-detection
task. It compares frame-wise five-fold evaluation (Protocol F) with
acquisition-sequence-isolated five-fold evaluation (Protocol V).

The two reported metrics changed in different directions. SM4 is therefore
interpreted as a cross-task extension rather than a direct replication of the
regression result.

## Included scripts

- `build_matched_folds.py` constructs Protocol V with GroupKFold, then creates
  Protocol F with the same validation-fold sizes.
- `audit_splits.py` checks coverage, sample counts, train-validation
  disjointness, and sequence separation in Protocol V.
- `run_frozen_val_capture.py` calls the specified YOLOv5 evaluator and records
  full-precision aggregate and class AP metrics.
- `summarize_metrics.py` calculates descriptive summaries and the
  pre-specified directional criterion from the released fold metrics.

These scripts were adapted for public use. They retain the grouping, split
checks, evaluation, and aggregation logic used in the study while using
configurable paths and omitting dataset identifiers and local run controls.

`configs/sm4_experiment.yaml` and `configs/sm4_hyperparameters.yaml` are
frozen provenance records. They are not configuration files consumed
automatically by the public scripts. The experiment record distinguishes the
requested worker counts from the effective counts observed under the pinned
YOLOv5 implementation and records the `PIN_MEMORY=false` environment setting
used for both training and evaluation.

## Local data handling

The source videos and annotations used for the reported SM4 results are not
publicly available or distributed externally.

Generated manifests, dataset YAML files, predictions, logs, and checkpoints
may contain dataset identifiers. Keep them outside this repository or under an
ignored local directory such as `sm4/runs/`.

## Split construction

Prepare a local manifest with these columns:

```text
image_id,video_id,frame_id,n_class0_boxes,n_class1_boxes
```

The final two columns are optional. When present, they allow the script to
check that both classes occur in every validation fold.

Run:

```bash
python sm4/build_matched_folds.py --manifest PRIVATE_MANIFEST.csv --output-dir sm4/runs/splits
python sm4/audit_splits.py --protocol-f sm4/runs/splits/protocol_F_folds.csv --protocol-v sm4/runs/splits/protocol_V_folds.csv
```

## YOLOv5

YOLOv5 is an external dependency and is not included. Obtain the specified
revision from the official Ultralytics repository and follow its license terms:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout 915bbf294bb74c859f0b41f1c23bc395014ea679
```

Install the upstream requirements and the appropriate PyTorch/CUDA build for
your system. Training settings are recorded in `configs/sm4_experiment.yaml`.
Initialization weights and study-trained checkpoints are not included.

The evaluator wrapper requires explicit paths. Write all run outputs to a
local ignored directory. For example:

```bash
python sm4/run_frozen_val_capture.py --yolo-dir PATH/TO/yolov5 --data PRIVATE_FOLD.yaml --weights PRIVATE_LAST.pt --project sm4/runs/eval --name fold_eval --out-json sm4/runs/metrics.json --log sm4/runs/eval.log
```

## Recalculate Table S5 summaries

```bash
python sm4/summarize_metrics.py
```

Fold labels identify runs only. Protocol F and Protocol V use different
validation samples and should not be analyzed as paired folds.
