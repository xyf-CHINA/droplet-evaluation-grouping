# Software verification

The test suite checks numerical calculations, included aggregate Talebjedi
outputs, frozen sample-equal audit values, SM4 summaries, and package safeguards.
Checks requiring the nine DAFD spreadsheets, locally generated DAFD prediction
tables, or the reconstructed Talebjedi row-level table are skipped when those
inputs are absent.

After obtaining and regenerating the license-scoped public-data inputs, rerun
the same suite to enable the skipped row-level and end-to-end checks. No fixed
"full suite" count is claimed here because available checks depend on those
local artifacts; the release criterion is zero failures.

The SM4 summary calculation reproduced the protocol means, sample standard
deviations, medians, minima, and maxima from the included fold metrics. It also
confirmed that the pre-specified directional criterion was not met.

Run the available tests with:

```bash
python -m pytest -q
```

Run the canonical privacy/license release gate with either equivalent entry
point:

```bash
python scripts/public_release_audit.py
python tools/release_audit.py
```

These checks cover the code and files distributed in the repository. They do
not recreate detector training from the SM4 source videos and annotations.
Version 1.1.1 is a package update and did not rerun model training. The installed
Windows/Python 3.12.3 dependency metadata were checked to confirm that the
retained pins form the complete active dependency closure after removal of
unused plotting packages; no scientific dependency version was changed.

On 2026-09-05, the v1.1.1 package-only test run completed with **57 passed,
33 skipped, and zero failures**. The skips were the expected checks requiring
source spreadsheets or locally regenerated row-level inputs. This is not a
claim that the unavailable-data checks or detector training were rerun.
