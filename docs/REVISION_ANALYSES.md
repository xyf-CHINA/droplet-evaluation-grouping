# Manuscript-revision analyses

These post hoc analyses extend the original release without changing the
original datasets, DAFD fits, or evaluation partitions. They compare fixed
candidate pipelines, not equally optimized learning algorithms. The original
v1.0.0 tag and archive are not replaced.

## DAFD comparisons and threshold curves

First obtain the DAFD sources and run the public regeneration commands in
[DATA_ACQUISITION.md](../DATA_ACQUISITION.md), through Stage 3. Then run:

```bash
python -m src.revision.dafd --output outputs/revision/dafd
```

This command reads the existing Protocol R and G XGBoost predictions and
Stage 3 RF/MLP predictions. It fits no model. Optional `--protocol-r`,
`--protocol-g`, and `--stage3` arguments accept equivalent local files.
Each model uses 100 test sets of 95 observations per protocol. All six
model/protocol summaries are retained, including the MLP results.

Model differences pair XGBoost with RF only within the same protocol and
the same test set. The reported split counts are descriptive counts, not
estimated population win probabilities. R and G split identifiers are
never treated as cross-protocol pairs. Pooled and original-observation-equal
aggregates are reported separately; extreme-split and one-split-omitted
checks do not replace the complete result.

For each threshold, the code first evaluates the absolute-error indicator
for every prediction, averages within each original observation, and then
averages equally over all 474 observations. It does not threshold the
per-observation mean error. The complete curve includes every distinct
observed error through the full tail; the integral of its survival curve
is checked against sample-equal MAE. No application tolerance, industrial
acceptance rate, or deployment success probability is inferred.

The full threshold-node table is regenerated in ignored `outputs/`; the
public package includes its aggregate summary and calculation code. Rendering
that table as a manuscript figure is outside the current package scope.

The combined aggregate JSON also preserves the original 32-geometry
seen/unseen summary and its one- and three-hardest-geometry omission checks.
These values are mapped from the frozen Stage 2.1 output, not recomputed
as a new analysis. Only the complete 32-geometry result has the original
10,000-resample, seed-0 descriptive bootstrap percentile interval. No
bootstrap interval is supplied or inferred for the two omission subsets.

## Targeted Talebjedi RF extension

Reconstruct the 125-row experimental table and regenerate the original
XGBoost R/T predictions using the existing public commands. To reproduce
the additional 105 fixed RF fits explicitly, run:

```bash
python -m src.revision.talebjedi_rf fit
python -m src.revision.talebjedi_rf summarize
```

The first command fits RF only: 100 random 100/25 splits and five
leave-one-angle-out 100/25 splits. It verifies the exact ordered test
members against the existing XGBoost prediction cache. The RF parameters
are the complete frozen 18-parameter configuration in
`configs/revision_rf.json`; use `requirements-lock.txt`, including
scikit-learn 1.6.1. StandardScaler is fitted on the 100 training observations
only. The target here is droplet size in micrometers, not the DAFD
hydraulic-diameter-normalized target. XGBoost is not refitted by this command.

The second command performs aggregation only. `--data`, `--protocol-r`,
`--protocol-t`, `--rf-predictions`, and `--output` support local artifacts.
All five held-out-angle differences are reported. The aggregate ordering
change does not imply RF is better at every angle: RF has lower MAE at
only two of the five held-out angles in the frozen results.

## Included outputs and unchanged privacy boundaries

`results/revision/` contains six DAFD and four Talebjedi model summaries,
protocol-internal split-level aggregate errors, full descriptive ordering
summaries, and the DAFD threshold summary. No experimental row, prediction
row, original-observation error table, ordered membership list, per-fit
scaler record, source spreadsheet, or machine-local provenance path is
included. Split indices in aggregate tables do not disclose membership.

DAFD and Talebjedi analyses are reproducible after obtaining the published
sources; they are not runnable from aggregate results alone. The private
SM4 data remain unavailable for external distribution, and the revision
does not change the SM4 workflow or release any additional SM4 information.

The rights statements in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
apply to the new aggregate outputs. These outputs are not relicensed as
MIT software. See [REVISION_VALIDATION.md](REVISION_VALIDATION.md) for the
checks actually performed on this release preparation.
