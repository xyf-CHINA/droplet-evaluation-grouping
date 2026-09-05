# DAFD prediction tables (NOT redistributed)

The row-level prediction tables `protocol_R_predictions.csv` and
`protocol_G_predictions.csv` are **not distributed** in this repository.
They contain source-linked rows (source row identifiers and observed
targets) derived from the DAFD 3.0 dataset, whose OSF project applies no
license (see THIRD_PARTY_NOTICES.md).

## Schema (for local regeneration)

| Column | Meaning |
|---|---|
| comp_experiment_id | DAFD Comprehensive experiment identifier |
| se_experiment_id | single-emulsion subset identifier |
| source_row_id | original row index in the SE extraction |
| geometry_id | reconstructed geometry group (G0001-G0035) |
| fold | split index (0-99) |
| protocol | R or G |
| split | train/test |
| geometry_seen_in_train | whether the row's geometry is represented in training |
| observed_um | observed droplet diameter (µm) |
| predicted_um | model prediction (µm) |
| absolute_error_um / percentage_error_pct | row errors |
| seed | split seed |
| held_out_geometries | (G only) geometry set held out in that run |

## Regeneration

1. Obtain the DAFD 3.0 source (see DATA_ACQUISITION.md).
2. Run `python src/models/stage1b_se_random_baseline.py` (Protocol R
   predictions) and `python src/evaluation/stage2_protocol_benchmark.py`
   (Protocol G predictions).
3. Copy `outputs/baseline/stage1b_predictions.csv` to
   `protocol_R_predictions.csv` and
   `outputs/geometry_shift/stage2_protocol_g_predictions.csv` to
   `protocol_G_predictions.csv` in this directory. Regeneration is
   deterministic; the copies are byte-identical to the frozen files.

## Frozen provenance (SHA-256 of the study's local copies, recorded 2026-08-25)

- protocol_R_predictions.csv
  `785003573283D2247715B386015B1BECF64D39B9F0C8F179FC608B02B6E98E58`
- protocol_G_predictions.csv
  `A14C085C655C27BA35B80BA440F2DCD52E52DF798B2C3CE0A9CADBF279150BF8`

These hashes are also pinned in
`src/analysis/sample_equal_sensitivity.py` (DEFAULT_INPUTS) and in
`results/dafd/sample_equal_audit.json`.
