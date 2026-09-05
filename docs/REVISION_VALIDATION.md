# Revision validation

The revision aggregation functions were independently compared with frozen
DAFD and Talebjedi predictions and existing summaries at absolute tolerance
1e-10 and zero relative tolerance. These checks concern the calculation of
reported results from existing predictions.

The tests cover indicator-before-aggregation, unequal test-appearance
weighting, exact pairing, duplicate rejection, positive MAPE denominators,
complete error tails, unfavorable model/angle results, and the absence of
invented geometry-omission intervals. Public-result checks also enforce the
permitted aggregate-output scope.

Run the revision calculation tests and the package audit from the repository
root:

```bash
python -m pytest -q tests/test_revision_analysis.py
python scripts/public_release_audit.py
```

See [REVISION_ANALYSES.md](REVISION_ANALYSES.md) for the commands that regenerate
DAFD comparison and threshold summaries and the targeted Talebjedi RF results.
Data-dependent checks require the locally obtained and reconstructed inputs.

Version 1.1.1 preserves the calculation logic, model settings, and retained
aggregate reference values. Its preparation did not rerun training. The RF
fitting entry point is supplied for reproduction, but its 105 fits were not
rerun for this package update. Frozen outputs remain the reference; aggregate
checks do not establish a fresh end-to-end training reproduction.
