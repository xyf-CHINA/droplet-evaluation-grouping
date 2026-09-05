# Input data

This repository does not redistribute the published source files or the SM4
video dataset.

## DAFD 3.0

Download the published DAFD 3.0 package from:

<https://osf.io/938rs/>

For the complete audit and all DAFD analyses, place these nine files under
`data/raw/` without renaming them:

1. `All_NewFF1_instability.xlsx`
2. `All_NewFF2_instability.xlsx`
3. `Comprehensive_normalized.xlsx`
4. `FF1_DE.xlsx`
5. `FF2_DE.xlsx`
6. `Final_Comprehensive_plus_Generalizability_data_normalized.xlsx`
7. `Generalizability_data_normalized.xlsx`
8. `Raw DE dataset.xlsx`
9. `Raw SE dataset.xlsx`

Expected sizes, worksheet names, and SHA-256 values are recorded in
`DAFD3_FILE_MANIFEST.csv`.

The core 474-observation analyses use `Comprehensive_normalized.xlsx` and
`Raw SE dataset.xlsx`. The external-domain analysis also uses
`Generalizability_data_normalized.xlsx`.

DAFD 3.0 remains subject to the terms of its source repository.

## Talebjedi et al. (2022)

Download the Supporting Information for:

> Talebjedi et al., *Langmuir* 38 (2022), 10465–10477.
>
> DOI: <https://doi.org/10.1021/acs.langmuir.2c01255>

The publisher-hosted contribution is available at:

<https://acs.figshare.com/articles/journal_contribution/20499180>

Save the PDF as:

`replication/talebjedi2022/source/Talebjedi2022_SI.pdf`

Expected SHA-256:

`54aac690fd8619a67e3a29eae9529f723c20ed9e38e24dcb3bbc917ff6ef8940`

The source PDF is not included. The reconstructed table is rebuilt locally
from the PDF (see DATA_ACQUISITION.md); the repository contains the
corresponding aggregate derived results.

## SM4 video data

The SM4 source images and annotations are not publicly available.

If you adapt the workflow using data that you are authorized to access, keep
those files outside the repository or in an ignored local directory.

Do not commit source images, annotations, identifiers, split files, dataset
YAML files, per-image outputs, model weights, checkpoints, or logs.
