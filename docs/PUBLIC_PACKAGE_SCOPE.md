# Public code package scope

Version 1.1.1 distributes scientific calculation code and aggregate verification
materials for the reported evaluation protocols.

## Included

- Data-source instructions and reconstruction/checking code.
- Partition construction, training/evaluation entry points and configurations.
- Metric calculations, fixed-pipeline comparisons, sample-equal aggregation,
  threshold-node calculations and other reported numerical analyses.
- Aggregate reference outputs, tests and reproducibility documentation.

## Local inputs and detailed outputs

Users obtain the DAFD and Talebjedi source materials from the cited providers
and regenerate detailed outputs locally. These include row-level predictions,
the complete per-geometry LOGO metrics table (including singleton groups), and
reconstructed experimental rows and their split/angle mappings. The public
package retains broader aggregate reference summaries. Acquisition instructions
and source-specific reuse terms are in [DATA_ACQUISITION.md](../DATA_ACQUISITION.md)
and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The private SM4 videos, frames, annotations, sequence identities, split
manifests, and model weights are excluded. Its public code and aggregate fold
metrics support inspection and reuse of the evaluation procedure.

## Scientific and presentation scope

The numerical analyses needed to calculate reported metrics, rankings, and
threshold curves are included. Paper figures, image/panel assembly, layout-only
scripts, Word table formatting, and manuscript/document builders are excluded.
Exact paper typography, colors, and panel layouts are outside the package scope.

Version 1.1.1 preserves the scientific calculation logic, model settings, and
retained reference values. This package preparation did not rerun training.
Citation metadata identify version 1.1.1; its version-specific DOI will be
added after archival publication.

## Safeguards

The release audit checks the package for excluded paper assets, private-data
identifiers, credentials, machine-specific paths, and disallowed detailed data
outputs. The private-data boundaries and source-specific non-code terms apply
throughout the package.
