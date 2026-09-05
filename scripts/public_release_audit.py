#!/usr/bin/env python3
"""Canonical, read-only audit for the public repository.

The audit checks the tracked tree and every file path in every commit
reachable from local refs. It never deletes or rewrites anything. Paper
assets and production code are rejected in both the current tree and history.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 512 * 1024

TEXT_EXTENSIONS = {
    ".cff",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SPECIAL_TEXT_NAMES = {".gitattributes", ".gitignore", "LICENSE"}
PAPER_FIGURE_EXTENSIONS = {".pdf", ".png"}

# Data-bearing records are fail-closed: every public CSV/JSON/YAML must be
# explicitly reviewed and listed here. Adding a new aggregate record therefore
# requires a deliberate policy update rather than silently widening release
# scope.
ALLOWED_DATA_RECORDS = {
    ".zenodo.json",
    "configs/revision_rf.json",
    "results/revision/dafd_model_summary.csv",
    "results/revision/dafd_split_differences.csv",
    "results/revision/dafd_ranking_summary.json",
    "results/revision/tolerance_summary.json",
    "results/revision/talebjedi_model_summary.csv",
    "results/revision/talebjedi_split_differences.csv",
    "results/revision/talebjedi_ranking_summary.json",
    "results/revision/revision_aggregate_summary.json",
    "configs/sm4_experiment.yaml",
    "configs/sm4_hyperparameters.yaml",
    "data/DAFD3_FILE_MANIFEST.csv",
    "replication/talebjedi2022/audit/group_sizes.csv",
    "replication/talebjedi2022/audit/random_geometry_exposure.json",
    "replication/talebjedi2022/audit/reconstruction_audit.json",
    "replication/talebjedi2022/outputs/loao_angle_metrics.csv",
    "replication/talebjedi2022/outputs/r1_posthoc_sensitivity.json",
    "replication/talebjedi2022/outputs/r1_same_angle_sensitivity.csv",
    "replication/talebjedi2022/outputs/random_seed_metrics.csv",
    "replication/talebjedi2022/outputs/replication_metrics.json",
    "results/dafd/sample_equal_audit.json",
    "results/dafd/sample_equal_summary.csv",
    "results/dafd/stage1b_seed_metrics.csv",
    "results/dafd/stage2_groupkfold_metrics.csv",
    "results/dafd/stage2_protocol_g_seed_metrics.csv",
    "results/dafd/stage21_master_table.csv",
    "results/dafd/stage3_seed_metrics.csv",
    "results/dafd/stage4_by_fluid.csv",
    "results/dafd/stage4_by_geometry.csv",
    "results/dafd/stage4_by_ref.csv",
    "results/dafd/stage41_macro_metrics.csv",
    "results/video/fold_metrics.csv",
    "results/video/split_summary.csv",
}

# Code/document paths must also be explicitly reviewed before distribution.
ALLOWED_OTHER_FILES = {
    ".gitattributes", ".gitignore", "CITATION.cff", "CHANGELOG.md", "DATA_ACQUISITION.md",
    "LICENSE", "README.md", "REPRODUCIBILITY_SCOPE.md", "THIRD_PARTY_NOTICES.md",
    "conftest.py", "data/README.md", "data/raw/.gitkeep", "pytest.ini",
    "requirements-lock.txt", "requirements.txt",
    "docs/DATA_AVAILABILITY.md", "docs/PAPER_TO_CODE_MAP.md", "docs/PUBLIC_PACKAGE_SCOPE.md",
    "docs/REPRODUCIBILITY.md", "docs/REVISION_ANALYSES.md", "docs/REVISION_VALIDATION.md",
    "docs/SOURCE_PROVENANCE.md", "docs/VALIDATION.md",
    "replication/talebjedi2022/PROTOCOL_LOCK.md", "replication/talebjedi2022/outputs/README.md",
    "replication/talebjedi2022/source/.gitkeep", "results/dafd/predictions/README.md",
    "scripts/public_release_audit.py", "tools/release_audit.py",
    "sm4/README.md", "sm4/__init__.py", "sm4/audit_splits.py", "sm4/build_matched_folds.py",
    "sm4/run_frozen_val_capture.py", "sm4/summarize_metrics.py",
    "src/__init__.py", "src/analysis/__init__.py", "src/analysis/sample_equal_sensitivity.py",
    "src/analysis/stage21_consolidated_analysis.py", "src/analysis/stage41_heterogeneity_consolidation.py",
    "src/audit/__init__.py", "src/audit/audit_dafd3.py", "src/common/__init__.py",
    "src/common/data.py", "src/common/metrics.py", "src/common/model_spec.py", "src/common/modeling.py",
    "src/evaluation/__init__.py", "src/evaluation/protocol_g_sampler.py",
    "src/evaluation/stage2_protocol_benchmark.py", "src/evaluation/stage2_sampler_validation.py",
    "src/evaluation/stage4_external_evaluation.py", "src/models/__init__.py",
    "src/models/stage1a_author_xgboost_reproduction.py", "src/models/stage1b_se_random_baseline.py",
    "src/models/stage3_model_independence.py", "src/replication/__init__.py",
    "src/replication/talebjedi2022/__init__.py", "src/replication/talebjedi2022/r1_posthoc_sensitivity.py",
    "src/replication/talebjedi2022/reconstruct_si_tables.py", "src/replication/talebjedi2022/run_protocols.py",
    "src/revision/__init__.py", "src/revision/analysis.py", "src/revision/dafd.py", "src/revision/talebjedi_rf.py",
    "tests/test_audit.py", "tests/test_pipeline_integration.py", "tests/test_public_release.py",
    "tests/test_revision_analysis.py", "tests/test_talebjedi_replication.py",
    "tests/test_talebjedi_rounding_convention.py",
}

# Binary, source-data, model, archive, and office formats are never public
# repository inputs. PNG/PDF are handled separately and are not accepted in
# the current tree, including below figures/.
FORBIDDEN_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".ckpt",
    ".db",
    ".doc",
    ".docx",
    ".feather",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".mat",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".ppt",
    ".pptx",
    ".pt",
    ".pth",
    ".rar",
    ".sav",
    ".sqlite",
    ".svg",
    ".tar",
    ".tif",
    ".tiff",
    ".wav",
    ".webp",
    ".xls",
    ".xlsm",
    ".xlsx",
    ".zip",
}

FORBIDDEN_DIRECTORY_NAMES = {
    "archive",
    "checkpoints",
    "data_primary_clean",
    "failed_runs",
    "private",
    "runtime",
    "scientific_runs",
    "weights",
    "work",
    "manuscript",
}

CONTENT_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:[/\\]+Users[/\\]+", re.IGNORECASE),
    "mapped drive path": re.compile(r"\b[A-Z]:\\(?:[^\s\\]+\\)+", re.IGNORECASE),
    "Unix home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "real-style video identifier": re.compile(r"\bVideo_\d", re.IGNORECASE),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer token": re.compile(r"\bbearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),
    "assigned credential": re.compile(
        r"\b(?:api[_-]?key|apikey|password|secret|authorization)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}

REQUIRED_DOCUMENTS = {
    "CITATION.cff",
    "DATA_ACQUISITION.md",
    "LICENSE",
    "README.md",
    "REPRODUCIBILITY_SCOPE.md",
    "THIRD_PARTY_NOTICES.md",
    "configs/sm4_hyperparameters.yaml",
    "configs/sm4_experiment.yaml",
    "docs/SOURCE_PROVENANCE.md",
    "docs/VALIDATION.md",
    "requirements-lock.txt",
    "CHANGELOG.md",
}

# These probes must remain ignored. ``git check-ignore --no-index`` checks the
# rules without creating any private-looking files.
IGNORE_PROBES = {
    "PRIVATE_MANIFEST_DO_NOT_COMMIT.csv",
    "PRIVATE_FOLD_DO_NOT_COMMIT.yaml",
    "sm4/runs/private_result.csv",
    "sm4/weights/private.pt",
    "sm4/checkpoints/private.pt",
    "results/dafd/predictions/protocol_R_predictions.csv",
    "replication/talebjedi2022/reconstructed/reconstructed_rows.csv",
    "replication/talebjedi2022/audit/published_split_angle_audit.csv",
    "replication/talebjedi2022/outputs/random_predictions.csv",
    "replication/talebjedi2022/outputs/loao_predictions.csv",
    "figures/Figure_1.pdf",
    "figures/revision/Figure_S2.pdf",
    "figures/paper_layout.py",
    "results/dafd/stage2_logo_geometry_metrics.csv",
}


def _run_git(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _normalise_relative_path(value: str | Path | PurePosixPath) -> PurePosixPath:
    raw = str(value).replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return PurePosixPath(raw)


def path_problems(value: str | Path | PurePosixPath) -> list[str]:
    """Return public-release policy violations for one repository path."""

    rel = _normalise_relative_path(value)
    text = rel.as_posix()
    lower = text.lower()
    parts = tuple(part.lower() for part in rel.parts)
    suffix = rel.suffix.lower()
    problems: list[str] = []

    if not parts or text in {"", "."}:
        return ["empty repository path"]
    if rel.is_absolute() or ".." in rel.parts:
        return ["path is not repository-relative"]

    if text not in ALLOWED_DATA_RECORDS | ALLOWED_OTHER_FILES:
        problems.append("file is not in the reviewed public file list")

    if set(parts) & FORBIDDEN_DIRECTORY_NAMES:
        problems.append("forbidden private/working directory")

    if re.search(r"(?:^|/)PRIVATE_(?:MANIFEST|FOLD)", text, re.IGNORECASE):
        problems.append("private manifest/fold filename")
    if re.search(r"(?:^|/)Video_\d", text, re.IGNORECASE):
        problems.append("real-style private video identifier in filename")

    if lower.startswith("data/raw/") and lower != "data/raw/.gitkeep":
        problems.append("raw DAFD/source-data path")
    if lower.startswith("replication/talebjedi2022/source/") and lower != (
        "replication/talebjedi2022/source/.gitkeep"
    ):
        problems.append("publisher-source data path")
    if lower.startswith("sm4/datasets/") or lower.startswith("sm4/splits/"):
        problems.append("private SM4 dataset/split path")
    if lower.startswith("sm4/runs/"):
        problems.append("private SM4 run path")

    if lower.startswith("results/dafd/predictions/") and suffix == ".csv":
        problems.append("DAFD row-level prediction path")
    if lower.startswith("results/dafd/sample_equal_recomputed/"):
        problems.append("DAFD regenerated row-level output path")
    if lower.startswith("replication/talebjedi2022/reconstructed/"):
        problems.append("Talebjedi reconstructed row-level path")
    if lower in {
        "replication/talebjedi2022/audit/published_split_angle_audit.csv",
        "replication/talebjedi2022/outputs/random_predictions.csv",
        "replication/talebjedi2022/outputs/loao_predictions.csv",
    }:
        problems.append("Talebjedi row-level prediction/source-mapping path")

    if suffix in {".csv", ".json", ".yaml", ".yml"} and text not in (
        ALLOWED_DATA_RECORDS
    ):
        problems.append("data-bearing record is not in the reviewed public allowlist")

    if parts[0] == "figures":
        problems.append("paper figures and production assets are outside the current package scope")
    elif suffix in PAPER_FIGURE_EXTENSIONS:
        problems.append("PNG/PDF paper asset outside the current package scope")

    if suffix in FORBIDDEN_EXTENSIONS:
        problems.append(f"forbidden file format {suffix}")
    elif (
        suffix not in TEXT_EXTENSIONS
        and suffix not in PAPER_FIGURE_EXTENSIONS
        and rel.name not in SPECIAL_TEXT_NAMES
        and rel.name != ".gitkeep"
    ):
        problems.append(f"unsupported tracked file format {suffix or '<none>'}")

    return problems


def historical_path_problems(value: str, oid: str) -> list[str]:
    """Apply the same distribution policy to every historical file."""
    return path_problems(value)


def content_problems(text: str) -> list[str]:
    """Return secret/private-content pattern labels found in UTF-8 text."""

    return [label for label, pattern in CONTENT_PATTERNS.items() if pattern.search(text)]


def semantic_problems(relative: str, text: str) -> list[str]:
    """Check production-only imports and data granularity, not only filenames."""
    problems = []
    if relative.endswith(".py"):
        try:
            parsed = ast.parse(text)
        except SyntaxError:
            return ["Python source does not parse"]
        excluded_modules = {"matplotlib", "seaborn", "plotly", "PIL", "docx", "pptx", "reportlab"}
        for node in ast.walk(parsed):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [(node.module or "").split(".")[0]]
            if set(imported) & excluded_modules:
                problems.append("paper-production dependency imported by public source")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "savefig":
                problems.append("figure-export call in public source")
    elif relative.endswith(".csv"):
        rows = csv.DictReader(io.StringIO(text))
        fields = set(rows.fieldnames or [])
        if fields & {"observed_um", "predicted_um", "source_row_id", "image_id", "video_id", "frame_id"}:
            problems.append("row-level observation or identity fields in distributed CSV")
        count_fields = fields & {"n", "n_test"}
        if {"mae", "mape_pct"}.issubset(fields) and count_fields:
            for row in rows:
                try:
                    counts = [float(row[name]) for name in count_fields]
                    if not all(math.isfinite(value) and value >= 1 and value.is_integer() for value in counts):
                        raise ValueError("invalid aggregate sample count")
                    singleton = min(counts) == 1
                except (TypeError, ValueError):
                    problems.append("invalid aggregate sample count")
                    break
                if singleton:
                    problems.append("singleton metrics can reconstruct an individual observation/prediction")
                    break
    elif relative.endswith(".json"):
        try:
            data = json.loads(text)
        except ValueError:
            return ["JSON does not parse"]
        forbidden_keys = {"locked_display_values", "zoom_upper_um", "wording_notes", "voided_claim"}
        def inspect(value):
            if isinstance(value, dict):
                if set(value) & forbidden_keys:
                    problems.append("paper-presentation or editorial field in distributed JSON")
                for nested in value.values():
                    inspect(nested)
            elif isinstance(value, list):
                for nested in value:
                    inspect(nested)
        inspect(data)
    if relative in {"requirements.txt", "requirements-lock.txt"}:
        if re.search(r"^\s*(?:matplotlib|seaborn|plotly|python-docx|python-pptx|reportlab)\s*[=<>]", text, re.M | re.I):
            problems.append("paper-production package listed as an analysis dependency")
    return sorted(set(problems))


def tracked_paths() -> list[str]:
    result = _run_git("ls-files")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [line for line in result.stdout.splitlines() if line.strip()]


def reachable_history_entries() -> list[tuple[str, str]]:
    """Enumerate every path/blob pair in every commit reachable from local refs."""
    result = _run_git("rev-list", "--all")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git rev-list failed")
    entries: set[tuple[str, str]] = set()
    for commit in result.stdout.splitlines():
        listing = _run_git("ls-tree", "-r", commit)
        if listing.returncode:
            raise RuntimeError("Historical tree cannot be read")
        for line in listing.stdout.splitlines():
            header, path = line.split("\t", 1)
            mode, kind, oid = header.split()
            if kind != "blob" or mode != "100644":
                raise RuntimeError("Non-regular historical file or submodule")
            entries.add((oid, path))
    return sorted(entries)


def _object_metadata(oids: set[str]) -> dict[str, tuple[str, int]]:
    if not oids:
        return {}
    result = _run_git(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_text="\n".join(sorted(oids)) + "\n",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git cat-file --batch-check failed")
    metadata: dict[str, tuple[str, int]] = {}
    for line in result.stdout.splitlines():
        oid, object_type, size = line.split()
        metadata[oid] = (object_type, int(size))
    return metadata


def _read_blob(oid: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", oid],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _is_text_path(path: str) -> bool:
    rel = _normalise_relative_path(path)
    return rel.suffix.lower() in TEXT_EXTENSIONS or rel.name in SPECIAL_TEXT_NAMES


def _check_markdown_links(path: Path, text: str) -> list[str]:
    problems: list[str] = []
    for match in re.finditer(r"\]\(([^)#]+)(?:#[^)]+)?\)", text):
        target = match.group(1).strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        candidate = (path.parent / target).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            problems.append(f"link escapes repository: {target}")
            continue
        if not candidate.exists():
            problems.append(f"broken internal link: {target}")
    return problems


def _self_check() -> list[str]:
    """Exercise policy sentinels without exempting this source file."""

    failures: list[str] = []
    forbidden_examples = (
        "results/dafd/predictions/protocol_G_predictions.csv",
        "replication/talebjedi2022/reconstructed/rows.csv",
        "replication/talebjedi2022/outputs/loao_predictions.csv",
        "sm4/PRIVATE_MANIFEST_001.csv",
        "private/data.csv",
        "manuscript.docx",
        "supplement.pdf",
        "figures/Figure_1.pdf",
        "figures/paper_layout.py",
    )
    for example in forbidden_examples:
        if not path_problems(example):
            failures.append(f"self-check failed to reject path: {example}")
    for example in ("src/common/metrics.py", "results/video/fold_metrics.csv"):
        if path_problems(example):
            failures.append(f"self-check rejected allowed path: {example}")

    for path in ("figures/Figure_1.pdf", "results/dafd/stage2_logo_geometry_metrics.csv"):
        if not historical_path_problems(path, "0" * 40):
            failures.append(f"self-check permitted excluded historical file: {path}")

    private_text_examples = (
        "C:" + "\\Users\\" + "person\\private",
        "/" + "home/person/private",
        "Video_" + "1234",
        "gh" + "p_" + "A" * 24,
    )
    for example in private_text_examples:
        if not content_problems(example):
            failures.append("self-check failed to reject private text sentinel")
    return failures


def audit_repository() -> tuple[list[str], int, int]:
    """Audit the current tree and every file reachable from local refs."""

    problems = _self_check()
    status = _run_git("status", "--porcelain")
    if status.returncode != 0:
        problems.append(status.stderr.strip() or "git status --porcelain failed")
    elif status.stdout.strip():
        problems.append(
            "working tree/index is not clean; commit the intended release tree "
            "before running the release gate"
        )
    files = tracked_paths()

    for relative in files:
        path = ROOT / relative
        for reason in path_problems(relative):
            problems.append(f"tracked path {relative}: {reason}")
        if not path.is_file():
            problems.append(f"tracked path missing from working tree: {relative}")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(
                f"tracked file too large: {relative} "
                f"({path.stat().st_size} > {MAX_FILE_BYTES} bytes)"
            )
        if not _is_text_path(relative):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"tracked text is not UTF-8: {relative}")
            continue
        for label in content_problems(text):
            problems.append(f"tracked content {relative}: {label}")
        for label in semantic_problems(relative, text):
            problems.append(f"tracked semantics {relative}: {label}")
        if path.suffix.lower() == ".md":
            for reason in _check_markdown_links(path, text):
                problems.append(f"Markdown {relative}: {reason}")

    for required in sorted(REQUIRED_DOCUMENTS):
        if required not in files:
            problems.append(f"missing required tracked document: {required}")

    for probe in sorted(IGNORE_PROBES):
        result = _run_git("check-ignore", "-q", "--no-index", probe)
        if result.returncode != 0:
            problems.append(f"private-path ignore rule missing: {probe}")

    history_entries = reachable_history_entries()
    metadata = _object_metadata({oid for oid, _ in history_entries})
    scanned_text_blobs: set[tuple[str, str]] = set()
    for oid, relative in history_entries:
        object_type, size = metadata.get(oid, ("unknown", 0))
        if object_type != "blob":
            continue
        for reason in historical_path_problems(relative, oid):
            problems.append(f"reachable history path {relative}: {reason}")
        if size > MAX_FILE_BYTES:
            problems.append(
                f"reachable history blob too large: {relative} "
                f"({size} > {MAX_FILE_BYTES} bytes)"
            )
        if not _is_text_path(relative) or (oid, relative) in scanned_text_blobs:
            continue
        scanned_text_blobs.add((oid, relative))
        data = _read_blob(oid)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"reachable history text is not UTF-8: {relative}")
            continue
        for label in content_problems(text):
            problems.append(f"reachable history content {relative}: {label}")
        for label in semantic_problems(relative, text):
            problems.append(f"reachable history semantics {relative}: {label}")

    return sorted(set(problems)), len(files), len(history_entries)


def main() -> int:
    try:
        problems, tracked_count, history_count = audit_repository()
    except (OSError, RuntimeError) as exc:
        print(f"PUBLIC RELEASE AUDIT: ERROR: {exc}")
        return 2

    if problems:
        print("PUBLIC RELEASE AUDIT: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(
        "PUBLIC RELEASE AUDIT: PASS "
        f"({tracked_count} tracked files; {history_count} reachable history entries)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
