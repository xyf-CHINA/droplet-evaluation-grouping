# Reproducibility scope

Version 1.1.1 provides scientific code, configurations, tests, source-acquisition
instructions, and aggregate reference results. Its reproducibility scope is
the reported calculations and evaluation procedures.

## The repository provides

- analysis code for the public-data analyses (DAFD 3.0 single-emulsion
  subset; external-domain evaluation);
- split-construction and grouped-evaluation procedures (exact-size
  geometry holdout, GroupKFold, LOGO);
- the Talebjedi et al. (2022) reconstruction procedure, reconstruction
  audit, checksums, and aggregate replication results;
- a public adaptation of the SM4 sequence-grouped video-detection
  workflow (fold construction, split audits, frozen evaluator wrapper,
  metric summarization);
- configuration files, pinned dependency records, and aggregate frozen results;
- a read-only release audit (`scripts/public_release_audit.py`).

## The repository does NOT provide

- paper figures, drawing/layout-only scripts, or manuscript/document builders;
- laboratory videos, extracted frames, or YOLO annotations;
- image, sequence, or frame identifiers for the SM4 dataset;
- private manifests, private split assignments, or exclusion records;
- trained model weights or checkpoints (`best.pt`, `last.pt`);
- logs, caches, or machine-specific paths from the internal runs;
- DAFD 3.0 source files, row-level prediction tables, or per-geometry LOGO
  metrics, including singleton groups (the recorded OSF source has no
  redistribution license; see THIRD_PARTY_NOTICES.md);
- the Talebjedi Supporting Information or the full reconstructed
  125-row experimental table, including row-level experiment-to-split or
  angle mappings (CC BY-NC 4.0; rebuild locally, see DATA_ACQUISITION.md).

## What "reproducible" means here

- **Public-data analyses (DAFD regression, external-domain, Talebjedi
  replication)**: obtain the published sources via DATA_ACQUISITION.md,
  regenerate the required detailed tables locally, and run the documented
  analyses. The frozen aggregate summaries provide numerical references for
  comparison. This package update did not repeat model training or constitute
  a new end-to-end reproduction.
- **The SM4 workflow is reproducible as a procedure**: the split
  construction, audits, evaluator wrapper, and aggregation logic are
  public. Re-running the SM4 training/evaluation requires the private
  laboratory video dataset, which is intentionally not distributed, so
  the published SM4 aggregate metrics are provided as frozen results
  rather than being regenerable from this repository alone.

## Boundaries

- Reproducing calculations and aggregate numbers does not mean recreating the
  exact colors, typography, panel arrangements, or Word/PDF table layouts used
  in the manuscript. Numerical processing is retained even when its output is
  later used in a paper figure or table.
- The root MIT License covers original code only; third-party data
  terms are documented in THIRD_PARTY_NOTICES.md.
- If a public adaptation produces numbers that differ from the frozen
  manuscript values, the repository and the manuscript are NOT
  automatically reconciled: the discrepancy must be diagnosed against
  the frozen workflow before any statement is made.
