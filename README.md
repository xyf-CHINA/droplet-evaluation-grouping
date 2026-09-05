# Evaluation Grouping Changes Generalization Estimates and Comparative Model Assessment in Droplet Microfluidics

This repository provides analysis code, configuration files, and aggregate
verification results for the study by Yunfeng Xing, Qiang Tang, and Yuwen Liu.
Software authorship is recorded separately in `CITATION.cff`.

Version **1.1.1** provides scientific calculation code, configurations, tests,
and aggregate verification results. It includes the post hoc fixed-pipeline
comparisons and original-observation-equal absolute-error threshold
calculations. Paper figures, drawing/layout scripts, and document builders
are outside its scope. See [Public package scope](docs/PUBLIC_PACKAGE_SCOPE.md).

This release updates the package contents and dependencies while preserving
the scientific calculations, model settings, and retained numerical results.
Its preparation did not rerun model training.

## Contents

- DAFD 3.0 data checks and geometry-group reconstruction;
- observation-level random, geometry-isolated, GroupKFold, and LOGO evaluation;
- cross-model and external-domain analyses;
- replication using Talebjedi et al. (2022);
- the post hoc sample-equal sensitivity analysis;
- DAFD model-ordering comparisons and full-tail error-threshold curves;
- a targeted post hoc random-forest comparison on the Talebjedi dataset;
- SM4 code for frame-wise and sequence-isolated YOLOv5 evaluation; and
- aggregate reference results for checking the calculations.

## Data availability

The DAFD 3.0 and Talebjedi et al. (2022) source files are available from their
original repositories. They are not redistributed here; download instructions
and exact local file locations are provided in
[DATA_ACQUISITION.md](DATA_ACQUISITION.md) and
[data/README.md](data/README.md).

Row-level predictions, per-geometry LOGO metrics (including singleton groups),
and reconstructed experimental rows are generated locally. The public package
provides aggregate reference summaries for comparison. Source-data reuse terms
remain applicable; see [Third-party notices](THIRD_PARTY_NOTICES.md).

The video dataset used in SM4 was collected and annotated in the laboratory.
It is not publicly available or distributed externally. The repository
provides the SM4 workflow, configuration, and aggregate fold-level results,
but not files that reveal image or acquisition-sequence identity.

These materials support review and reuse of the workflow, but they cannot
reproduce the original SM4 training. See
[Data availability](docs/DATA_AVAILABILITY.md).

## Repository structure

- `src/audit/` — DAFD data checks and geometry-group reconstruction.
- `src/common/` — shared data, model, and metric utilities.
- `src/models/` — XGBoost, random-forest, and multilayer-perceptron analyses.
- `src/evaluation/` — random, geometry-isolated, grouped, and external-domain protocols.
- `src/analysis/` — summary and heterogeneity analyses.
- `src/replication/talebjedi2022/` — Talebjedi reconstruction and analysis code.
- `replication/talebjedi2022/` — aggregate replication results and frozen-provenance records; the row-level tables are regenerated locally (see DATA_ACQUISITION.md).
- `sm4/` — code and documentation for sequence-grouped video evaluation.
- `results/` — aggregate reference results.
- `tests/` — automated tests.
- `src/revision/` — portable implementations of the manuscript-revision analyses.
- `results/revision/` — frozen aggregate model comparisons and threshold summaries.
- `DATA_ACQUISITION.md` — source locations and end-to-end local regeneration steps.
- `data/README.md` — the required published source-file inventory.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

`requirements-lock.txt` records the active Windows/Python 3.12 dependency
closure checked for this release, preserving the study's scientific dependency
versions. Unused plotting dependencies are excluded. Use
`requirements.txt` instead when only the direct dependency pins are wanted
and fresh dependency resolution is acceptable.

Tests that use the DAFD spreadsheets or the locally reconstructed
Talebjedi tables are skipped when those files are absent.
All remaining code and included-result checks still run.

## Public-data analyses

Download the source files listed in [data/README.md](data/README.md) and place
them as described in [DATA_ACQUISITION.md](DATA_ACQUISITION.md). From the
repository root, run:

```bash
python src/audit/audit_dafd3.py
python src/models/stage1a_author_xgboost_reproduction.py
python src/models/stage1b_se_random_baseline.py
python src/evaluation/stage2_sampler_validation.py
python src/evaluation/stage2_protocol_benchmark.py
python src/analysis/stage21_consolidated_analysis.py
python src/models/stage3_model_independence.py
python src/evaluation/stage4_external_evaluation.py
python src/analysis/stage41_heterogeneity_consolidation.py
```

Run the sample-equal sensitivity analysis directly from the locally generated
Protocol R and Protocol G outputs:

```bash
python src/analysis/sample_equal_sensitivity.py --protocol-r outputs/baseline/stage1b_predictions.csv --protocol-g outputs/geometry_shift/stage2_protocol_g_predictions.csv
```

The script also supports canonical default input paths under
`results/dafd/predictions/`; see [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for
the output descriptions and frozen checksum records.

For the Talebjedi replication, run:

```bash
python src/replication/talebjedi2022/reconstruct_si_tables.py
python src/replication/talebjedi2022/run_protocols.py
python src/replication/talebjedi2022/r1_posthoc_sensitivity.py
```

The revision analyses use these locally regenerated predictions and do not
include third-party experimental rows in the repository. Instructions,
scope, and the explicit optional RF fitting command are in
[Revision analyses](docs/REVISION_ANALYSES.md).

## SM4 workflow

See [sm4/README.md](sm4/README.md) for setup and usage. YOLOv5 is not included.
The reported analyses used YOLOv5 v7.0 at commit
`915bbf294bb74c859f0b41f1c23bc395014ea679`.

## Documentation

- [Data acquisition and local regeneration](DATA_ACQUISITION.md)
- [Data availability](docs/DATA_AVAILABILITY.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Reproducibility scope](REPRODUCIBILITY_SCOPE.md)
- [Public package scope](docs/PUBLIC_PACKAGE_SCOPE.md)
- [Paper-to-code map](docs/PAPER_TO_CODE_MAP.md)
- [Code provenance](docs/SOURCE_PROVENANCE.md)
- [Software verification](docs/VALIDATION.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Pinned public-data environment](requirements-lock.txt)

## Citation

Citation metadata are provided in `CITATION.cff`. Cite the software as
Xing, Y. and Tang, Q., *droplet-evaluation-grouping*, version 1.1.1,
[GitHub repository](https://github.com/xyf-CHINA/droplet-evaluation-grouping),
and cite the associated manuscript when using its scientific results.

The [concept DOI](https://doi.org/10.5281/zenodo.22100497) identifies the
collection of all archived versions. A version-specific DOI for v1.1.1 will
be added after archival publication.

## License

The MIT License applies only to the original source code, tests, configuration
files, and code-oriented documentation in this repository. It does not apply
to files under `results/`, `replication/talebjedi2022/audit/`,
`replication/talebjedi2022/outputs/`, nor does it cover
third-party data or software, publisher materials, pretrained weights, or
manuscript text or paper figures. The included aggregate non-code files are
provided for scientific verification
and citation. This repository grants no additional reuse rights beyond those
provided by the applicable source-data, manuscript, or publisher terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
