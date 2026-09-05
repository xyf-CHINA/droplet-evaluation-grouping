# Data acquisition

This repository does not redistribute third-party source data. To
reproduce the public-data analyses, obtain the following sources and
place them in the locations documented below. All local source copies
are git-ignored.

## 1. DAFD 3.0 (Lashkaripour et al., 2024)

- Source: Open Science Framework, https://osf.io/938rs/.
- For the complete audit and all DAFD analyses, download these nine files
  without renaming them:

  1. `All_NewFF1_instability.xlsx`
  2. `All_NewFF2_instability.xlsx`
  3. `Comprehensive_normalized.xlsx`
  4. `FF1_DE.xlsx`
  5. `FF2_DE.xlsx`
  6. `Final_Comprehensive_plus_Generalizability_data_normalized.xlsx`
  7. `Generalizability_data_normalized.xlsx`
  8. `Raw DE dataset.xlsx`
  9. `Raw SE dataset.xlsx`

  The core 474-observation analyses use `Comprehensive_normalized.xlsx`
  together with `Raw SE dataset.xlsx`; the external-domain analysis also uses
  the 64-row `Generalizability_data_normalized.xlsx` file.
- License: none applied (`node_license: null` as of 2026-08-25) — the
  source is used for local analysis only and is never redistributed.
- Place under `data/raw/` (git-ignored). Expected checksums are listed
  in `data/DAFD3_FILE_MANIFEST.csv`.

Regeneration and direct use of the (not redistributed) row-level prediction
tables:

```bash
python src/models/stage1b_se_random_baseline.py
python src/evaluation/stage2_protocol_benchmark.py
python src/analysis/sample_equal_sensitivity.py --protocol-r outputs/baseline/stage1b_predictions.csv --protocol-g outputs/geometry_shift/stage2_protocol_g_predictions.csv
```

`stage1b_se_random_baseline.py` writes
`outputs/baseline/stage1b_predictions.csv` (Protocol R, 9500 rows) and
`stage2_protocol_benchmark.py` writes
`outputs/geometry_shift/stage2_protocol_g_predictions.csv` (Protocol G,
9500 rows). The sample-equal command above reads those generated files
directly. Alternatively, copy them to the default canonical paths documented
in `results/dafd/predictions/README.md`. Regeneration is deterministic, and
the frozen SHA-256 values and column schema are recorded there.

## 2. Talebjedi et al. (2022) Supporting Information

- Source: ACS Figshare,
  https://acs.figshare.com/articles/journal_contribution/20499180
  (CC BY-NC 4.0; attribution required; non-commercial use).
- Download the Supporting Information PDF and keep it in
  `replication/talebjedi2022/source/` (git-ignored).

Reconstruct the 125-row experimental table locally:

```bash
python src/replication/talebjedi2022/reconstruct_si_tables.py
```

This writes
`replication/talebjedi2022/reconstructed/talebjedi_125_reconstructed.csv`
(git-ignored; frozen SHA-256 in `replication/talebjedi2022/outputs/README.md`).
Then run the replication protocols:

```bash
python src/replication/talebjedi2022/run_protocols.py
```

## 3. SM4 video dataset

- The laboratory droplet-video dataset (1,000 frames, 17 acquisition
  sequences) is **private and is not available for external
  distribution**. The repository provides the workflow, configuration
  (`configs/sm4_experiment.yaml`), split-construction/audit scripts, the
  frozen evaluator wrapper, and the frozen aggregate fold metrics in
  `results/video/`.
