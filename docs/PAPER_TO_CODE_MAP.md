# Paper-to-code map

This map links manuscript calculations and SM1-SM4 to source code and aggregate
results. The `src/revision/` rows map the post hoc model comparisons and
threshold analyses.
Paper figures, drawing/layout scripts, and document builders are outside the
current package scope; see [PUBLIC_PACKAGE_SCOPE.md](PUBLIC_PACKAGE_SCOPE.md).

| Manuscript component | Code or result |
|---|---|
| DAFD provenance and geometry reconstruction; SM1 | `src/audit/audit_dafd3.py`, `src/common/data.py` |
| Reference reproduction and Protocol R | `src/models/stage1a_author_xgboost_reproduction.py`, `src/models/stage1b_se_random_baseline.py` |
| Exact-size Protocol G sampler; SM2 | `src/evaluation/protocol_g_sampler.py`, `src/evaluation/stage2_sampler_validation.py` |
| Protocol G, GroupKFold, and LOGO | `src/evaluation/stage2_protocol_benchmark.py` |
| Seen/unseen geometry and weighting analyses | `src/analysis/stage21_consolidated_analysis.py`, `src/analysis/sample_equal_sensitivity.py` |
| Cross-model robustness | `src/models/stage3_model_independence.py` |
| Revised DAFD fixed-pipeline comparisons | `src/revision/dafd.py`, `src/revision/analysis.py`, `results/revision/dafd_model_summary.csv` |
| Original-observation-equal threshold curves | `src/revision/analysis.py`, `results/revision/tolerance_summary.json` |
| Targeted Talebjedi RF comparison | `src/revision/talebjedi_rf.py`, `configs/revision_rf.json`, `results/revision/talebjedi_model_summary.csv` |
| External-domain evaluation | `src/evaluation/stage4_external_evaluation.py`, `src/analysis/stage41_heterogeneity_consolidation.py` |
| Talebjedi cross-dataset replication | `src/replication/talebjedi2022/` and `replication/talebjedi2022/` |
| SM4 split construction and checks | `sm4/build_matched_folds.py`, `sm4/audit_splits.py` |
| SM4 evaluation wrapper | `sm4/run_frozen_val_capture.py` |
| SM4 Table S5 calculations | `sm4/summarize_metrics.py`, `results/video/fold_metrics.csv` |

Detailed predictions, reconstructed experimental rows, and per-geometry LOGO
metrics (including singleton groups) are generated locally by the documented
workflows. The distributed result files provide aggregate references; see
[DATA_ACQUISITION.md](../DATA_ACQUISITION.md).

Because the SM4 source data are not distributed, the public scripts omit
dataset identifiers and local execution details while retaining the grouping,
evaluation, and aggregation logic.
