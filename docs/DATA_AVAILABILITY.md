# Data availability

## Published source data

The DAFD 3.0 spreadsheets and the Talebjedi et al. (2022) Supporting
Information are third-party publications. They are neither redistributed nor
relicensed here.

`data/README.md` lists the source links, expected filenames, and checksums.

## SM4 video dataset

The SM4 video dataset was collected and annotated in the laboratory. It is not
publicly available or distributed externally, and is not included here.

Files that reveal image or acquisition-sequence identity are also omitted,
including:

- source videos, extracted frames, and YOLO annotations;
- filenames and image, sequence, or frame identifiers;
- image-level training and validation split lists;
- sample-selection and exclusion records;
- per-image predictions;
- trained checkpoint files (`best.pt` and `last.pt`); and
- logs, caches, and machine-specific commands or paths.

The repository contains the SM4 workflow, configuration and metric definitions,
fold-size summaries, and aggregate results. These materials support review and
reuse of the workflow, but not training on the original video dataset.

## Derived results

DAFD prediction summaries and Talebjedi reconstruction outputs were generated
from published sources. For SM4, only the aggregate results reported in the
manuscript are included.

These derived files do not grant access to or rights in the underlying source
data.
