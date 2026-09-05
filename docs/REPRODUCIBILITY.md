# Reproducibility

The DAFD and Talebjedi analyses can be rerun after users obtain the cited
published source files. The SM4 dataset is not distributed, so this repository
provides its workflow and aggregate results rather than the original inputs.

## Environment

Public-data analyses were verified on Windows with Python 3.12.3 and the
complete active dependency closure in `requirements-lock.txt`.
`requirements.txt` lists the direct dependencies for users who prefer fresh
dependency resolution.

SM4 used Python 3.12.3, PyTorch 2.4.1 with CUDA 11.8, torchvision 0.19.1, and
YOLOv5 v7.0 at commit `915bbf294bb74c859f0b41f1c23bc395014ea679`.

Install PyTorch and YOLOv5 separately because the appropriate PyTorch build
depends on the local system.

## DAFD analyses

After downloading the DAFD files listed in `data/README.md`, run the commands
in the root README. The scripts reconstruct the 474-row single-emulsion cohort.

They compare observation-level random and geometry-isolated partitions, run
grouped and cross-model analyses, and evaluate the published 64-row
generalizability dataset.

The post hoc sample-equal sensitivity analysis is rerun from two locally
generated DAFD prediction files: Protocol R is produced by
`stage1b_se_random_baseline.py`, and Protocol G is produced by
`stage2_protocol_benchmark.py`. Pass those outputs explicitly as shown in
`DATA_ACQUISITION.md`, or copy them to the documented canonical paths.

```bash
python src/analysis/sample_equal_sensitivity.py --protocol-r outputs/baseline/stage1b_predictions.csv --protocol-g outputs/geometry_shift/stage2_protocol_g_predictions.csv
```

## Talebjedi replication

The included aggregate reference outputs allow the aggregate-result tests to
run without the publisher PDF. Row-level tests are skipped until the
reconstructed table has been rebuilt locally.

To rebuild them from the PDF, place it at the location specified in
`data/README.md`, then run the three Talebjedi scripts listed in the root README.

## SM4 video evaluation

Because the SM4 videos and annotations cannot be shared, the original detector
training cannot be reproduced from this repository alone.

The repository provides the grouping algorithm, split checks, evaluation
wrapper, configuration, fold-level metrics, and summary calculations needed to
review the design and reproduce the reported aggregate summaries.

`results/video/fold_metrics.csv` can reproduce the protocol means, sample
standard deviations, medians, minima, maxima, and the pre-specified directional
criterion:

```bash
python sm4/summarize_metrics.py
```

Passing the public tests confirms that the released code and fixtures behave
as expected. It does not recreate detector training from the SM4 source data.
