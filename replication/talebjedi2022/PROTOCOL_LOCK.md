# Talebjedi 2022 Replication — Protocol Lock

*Frozen pre-specified replication protocol. Content is taken verbatim from
the manuscript methods (Section 2.7 and SM3.4); no new analysis, no new
statistics, no tunable choices. This document exists so that
`run_protocols.py` has a real, tracked protocol reference.*

## Data

- Source: Talebjedi, B., Abouei Mehrizi, A., Talebjedi, B., Mohseni, S.S.,
  Tasnim, N., Hoorfar, M. (2022). Machine learning-aided microdroplets
  breakup characteristic prediction in flow-focusing microdevices by
  incorporating variations of cross-flow tilt angles. Langmuir 38,
  10465-10477. https://doi.org/10.1021/acs.langmuir.2c01255
- Supporting Information: https://acs.figshare.com/articles/journal_contribution/20499180
  (CC BY-NC 4.0; obtain and reconstruct locally, see DATA_ACQUISITION.md).
- Reconstructed table: 125 experiments; five tilt-angle configurations
  (30, 60, 90, 120, 150 degrees); 25 identical operating-condition
  combinations per angle.
- Target: droplet size (µm). Predictors: tilt angle, flow-rate ratio,
  continuous-phase flow rate (published design inputs only; the
  dispersed-phase flow rate was used solely for a reconstruction
  consistency audit and is not a predictor).

## Protocol R (random sample-wise)

- 100 fixed sample-wise 80/20 splits (seeds 0-99); 100 train / 25 test
  rows per split.
- Angle exposure is an empirical audit result (100% in this dataset),
  not a protocol definition.

## Protocol T (leave-one-tilt-angle-out)

- Five folds; each fold holds out one complete tilt-angle
  configuration (100 train / 25 test rows; held-out angle absent from
  training; overlap = 0, verified).

## Model

- Fixed XGBoost specification shared with the DAFD reference analysis
  (100 trees, learning rate 0.3, max depth 6, L2 penalty 1.0, min child
  weight 1.0, squared-error objective, histogram tree method, random
  state 0).
- Dataset-specific predictor set and target definition are retained; no
  hyperparameter tuning; inputs standardized on each training partition
  only.

## Metrics

- Primary: MAE and MAPE. Secondary: pooled RMSE, pooled R², signed bias.
- Protocol R pooled over 2,500 repeated test predictions; Protocol T
  pooled over 125 out-of-fold predictions.
- Run/fold means of MAE and MAPE equal pooled values (every run/fold has
  exactly 25 test rows; verified programmatically). RMSE and R² are
  nonlinear, so only pooled values are reported.

## Primary replication criterion (pre-specified)

- Supportive replication requires **concordant increases in both MAE and
  MAPE** under leave-one-tilt-angle-out evaluation relative to
  sample-wise random evaluation. The criterion was defined before model
  fitting and before inspection of the results.

## Post hoc analyses (do not contribute to the primary criterion)

- Same-angle sensitivity: seen-angle versus unseen-angle MAE/MAPE for
  the five tilt configurations — descriptive only.
- Sample-equal weighting sensitivity — descriptive only.
- No p-values and no bootstrap intervals across the five
  configurations.
