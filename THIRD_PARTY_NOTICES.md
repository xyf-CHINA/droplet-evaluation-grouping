# Third-party notices

The MIT License covers only the original code in this repository.
Third-party software, data, and publications remain subject to their own
licenses and terms. The repository redistributes none of the third-party
source data described below.

## DAFD 3.0 data (Lashkaripour et al., 2024)

- Source: Open Science Framework project
  https://osf.io/938rs/ and the DAFD 3.0 publication
  (Lashkaripour, A., McIntyre, D.P., Calhoun, S.G.K., Krauth, K.,
  Densmore, D.M., Fordyce, P.M., 2024. Design automation of microfluidic
  single and double emulsion droplets with machine learning. Nat.
  Commun. 15, 83. https://doi.org/10.1038/s41467-023-44068-3).
- License / reuse status: **the OSF project applies no license
  (`node_license: null` as of 2026-08-25)**. No redistribution terms
  were identified.
- Repository treatment: the source files are not included and are not
  redistributed. Row-level prediction tables and per-geometry LOGO metrics,
  including singleton groups, are generated locally and are not included;
  only broader aggregate summaries, input
  manifests, regeneration scripts, and checksum records are published.
  Users obtain the source from OSF and regenerate predictions locally
  (see DATA_ACQUISITION.md).

## Talebjedi et al. (2022) Supporting Information

- Publication: Talebjedi, B., Abouei Mehrizi, A., Talebjedi, B.,
  Mohseni, S.S., Tasnim, N., Hoorfar, M., 2022. Machine learning-aided
  microdroplets breakup characteristic prediction in flow-focusing
  microdevices by incorporating variations of cross-flow tilt angles.
  Langmuir 38, 10465-10477.
  https://doi.org/10.1021/acs.langmuir.2c01255
- Source: Supporting Information hosted by ACS Figshare,
  https://acs.figshare.com/articles/journal_contribution/20499180
- License: **CC BY-NC 4.0**
  (https://creativecommons.org/licenses/by-nc/4.0/). Attribution to the
  original authors is required; use is limited to non-commercial
  purposes.
- Repository treatment: the original Supporting Information and the full
  reconstructed 125-row experimental table are not redistributed.
  Row-level experiment-to-split and angle mappings are also generated only
  in the user's ignored local workspace and are not distributed. The
  repository provides the reconstruction procedure
  (`src/replication/talebjedi2022/reconstruct_si_tables.py`), the
  reconstruction audit and checksums, aggregate protocol results, and
  descriptive sensitivity summaries. Users obtain the Supporting
  Information from the Figshare link above and rebuild the table locally
  (see DATA_ACQUISITION.md).

## Ultralytics YOLOv5

- The SM4 workflow uses YOLOv5 as an external dependency; its source
  code and weights are not distributed here. The study used YOLOv5 v7.0
  at commit `915bbf294bb74c859f0b41f1c23bc395014ea679`.
- License: the `LICENSE` file at the pinned YOLOv5 v7.0 commit specifies
  **GPL-3.0**. No YOLOv5 source code is copied into this repository —
  the SM4 adapter pins an external checkout by commit and by SHA-256 of
  `val.py` and `utils/metrics.py` and does not vendor any YOLOv5 files.
  The root MIT License therefore applies to the code in this repository;
  use of YOLOv5 itself by third parties is governed by its own license.
  Obtain it from https://github.com/ultralytics/yolov5.

## Aggregate outputs and other non-code materials

The root MIT License does not apply to the aggregate outputs under `results/`
or to the audit and output files under `replication/talebjedi2022/`. These
files are provided for scientific verification and citation. Their inclusion
does not grant rights in the underlying source material, and this repository
grants no additional reuse rights beyond those already provided by the
applicable source terms.

- Aggregate outputs derived from DAFD 3.0 are not relicensed under MIT. The
  DAFD source project had no identified redistribution license as recorded
  above.
- Aggregate and audit outputs derived from Talebjedi et al. (2022) are not
  relicensed under MIT. Use of the source material remains subject to CC
  BY-NC 4.0, including its attribution and non-commercial requirements.
- The files under `results/video/` contain only aggregate SM4 verification
  outputs. The private laboratory videos, annotations, sequence identities,
  split manifests, and model weights are not distributed. No separate reuse
  license is granted for these aggregate outputs unless explicitly stated.

## Dependencies

Python packages retain their own licenses. Listing or using a
dependency does not incorporate it into this repository's MIT License.
