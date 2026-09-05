# Code provenance

The DAFD and Talebjedi modules preserve the numerical calculation logic used
for the manuscript. Public-release changes are limited to repository-relative
paths, command-line input/output options, clear guidance when license-scoped
inputs are absent, corrected references to `PROTOCOL_LOCK.md`, and test skips
when locally reconstructed third-party tables are unavailable. These changes
do not alter metric definitions, partition rules, random seeds, estimators, or
the numerical calculations.

The modules in `src/revision/` are portable public adaptations of the frozen
post hoc DAFD ranking/threshold calculations and targeted Talebjedi RF
extension. They omit internal file inventories, local paths, and private
run-management dependencies. Aggregation is independently compared against
the frozen predictions and summaries at absolute tolerance 1e-10 with zero
relative tolerance. The explicit-column fix for pandas' `GroupBy.observed`
attribute-name collision is retained; it changes no scientific formula.

Version 1.1.1 updates package contents, dependency declarations, documentation,
and release safeguards. The scientific calculations, model configurations,
and retained aggregate reference values are unchanged. Detailed per-geometry
LOGO metrics remain available through local regeneration and are excluded from
the distributed reference results. This package preparation did not rerun
training; validation of aggregate calculations is distinct from a new
end-to-end reproduction.

The scripts under `sm4/` are non-byte-identical public adaptations of the
internal execution workflow. They retain the grouping, split-checking,
evaluation, and aggregation logic used in the study while using configurable
paths and omitting dataset identifiers, private manifests, and local
run-management code. The public scripts therefore document and rerun the
released procedure; they are not represented as an archival copy of the
private laboratory driver.
