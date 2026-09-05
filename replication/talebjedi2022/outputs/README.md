# Talebjedi 2022 replication outputs

## Distributed here

Aggregate results only (frozen values cited in the manuscript, SM3.4):

| File | Content |
|---|---|
| `replication_metrics.json` | frozen primary metrics (random MAE / LOAO MAE, per-angle) |
| `random_seed_metrics.csv` | per-seed metrics across R=100 random-split runs |
| `loao_angle_metrics.csv` | per-angle leave-one-angle-out metrics |
| `r1_posthoc_sensitivity.json` | R1 post-hoc descriptive sensitivity |
| `r1_same_angle_sensitivity.csv` | R1 same-angle descriptive sensitivity |

## Not distributed

Row-level prediction tables, which contain source-linked rows from the
Talebjedi et al. (2022) Supporting Information (CC BY-NC 4.0, see
THIRD_PARTY_NOTICES.md), and the full reconstructed 125-row experimental
table are regenerated locally via DATA_ACQUISITION.md:

- `reconstructed/talebjedi_125_reconstructed.csv`
- `outputs/random_predictions.csv`
- `outputs/loao_predictions.csv`

## Frozen provenance (SHA-256 of the study's local copies, recorded 2026-08-25)

- `replication/talebjedi2022/reconstructed/talebjedi_125_reconstructed.csv`
  `55c17dac883d2c037ab860556597f384225a0bee7744963f835b8a92bc069a58`
- `replication/talebjedi2022/outputs/loao_predictions.csv`
  `a4d68c2ba652721491e0ae0b2ba5ece7c360a3b1df0870490078e9eadbfdc23d`
- `replication/talebjedi2022/outputs/random_predictions.csv`
  `59e1c8729a858b44b3f0f7c4d0485b01629624ccd2160e797fcd8a325f2eb842`

Regenerated files must match these hashes to be treated as the frozen
study inputs for the post-hoc scripts.
