"""Checks for the public SM4 workflow and release safeguards."""

import json
from pathlib import Path

import numpy as np
import pytest

from sm4.build_matched_folds import build_folds
from sm4.audit_splits import audit_protocol
from sm4.summarize_metrics import DEFAULT_INPUT, load_rows, summarize
from scripts.public_release_audit import (
    content_problems,
    historical_path_problems,
    path_problems,
    semantic_problems,
)
from src.analysis.sample_equal_sensitivity import DEFAULT_INPUTS, path_a_pandas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EQUAL_AUDIT = PROJECT_ROOT / "results" / "dafd" / "sample_equal_audit.json"

# Frozen sample-equal values (post hoc analysis, manuscript SM2/SM5).
FROZEN_SAMPLE_EQUAL = {
    "R": (8.3199943046, 8.1609108288),
    "G": (11.3466035325, 11.8676271627),
}


@pytest.mark.parametrize(
    "relative_path",
    [
        "results/dafd/predictions/protocol_R_predictions.csv",
        "results/dafd/sample_equal_recomputed/per_sample_errors.csv",
        "replication/talebjedi2022/reconstructed/talebjedi_125.csv",
        "replication/talebjedi2022/audit/published_split_angle_audit.csv",
        "replication/talebjedi2022/outputs/random_predictions.csv",
        "replication/talebjedi2022/outputs/loao_predictions.csv",
        "sm4/PRIVATE_MANIFEST_001.csv",
        "sm4/datasets/images/frame.png",
        "data/raw/Comprehensive_normalized.xlsx",
        "manuscript.pdf",
        "figures/source.xlsx",
        "figures/Figure_1.png",
        "figures/Figure_2.pdf",
        "results/video/private_frame_ids.csv",
        "configs/unreviewed_private_manifest.json",
        "results/dafd/stage2_logo_geometry_metrics.csv",
        "src/unreviewed_plot.py",
    ],
)
def test_public_release_policy_rejects_private_or_row_level_paths(relative_path):
    assert path_problems(relative_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "results/dafd/stage2_groupkfold_metrics.csv",
        "src/common/metrics.py",
        "configs/sm4_experiment.yaml",
        "results/video/fold_metrics.csv",
    ],
)
def test_public_release_policy_allows_code_and_public_aggregates(relative_path):
    assert path_problems(relative_path) == []


def test_historical_figures_have_no_exemption():
    for path in ("figures/Figure_1.pdf", "figures/revision/Figure_S2.pdf"):
        assert path_problems(path)
        assert historical_path_problems(path, "0" * 40)


def test_same_name_or_size_cannot_exempt_a_new_figure():
    path = "figures/revision/Figure_S2.pdf"
    assert path_problems(path)
    assert historical_path_problems(path, "1" * 40)


def test_paper_directories_cannot_hide_code_or_data_assets():
    for path in ("figures/paper_layout.py", "figures/source.csv", "manuscript/build.py", "paper.png"):
        assert path_problems(path)


def test_singleton_aggregate_is_not_treated_as_non_row_level():
    sample = "geometry_id,n,mae,mape_pct,signed_bias\nexample,1,2,10,2\n"
    assert semantic_problems("results/dafd/stage2_groupkfold_metrics.csv", sample)
    assert not semantic_problems("results/dafd/stage2_groupkfold_metrics.csv", sample.replace(",1,", ",5,"))


def test_production_code_and_presentation_fields_are_rejected():
    for source in ("import matplotlib.pyplot as plt", "from docx import Document", "fig.savefig('out.pdf')"):
        assert semantic_problems("src/common/metrics.py", source)
    assert not semantic_problems("src/common/metrics.py", "import numpy as np")
    for key in ("zoom_upper_um", "locked_display_values", "wording_notes"):
        assert semantic_problems("results/revision/tolerance_summary.json", json.dumps({"nested": {key: 1}}))
    assert semantic_problems("results/video/fold_metrics.csv", "image_id,score\nsynthetic,0.5\n")


@pytest.mark.parametrize("count_name", ["n", "n_test"])
def test_singleton_error_summaries_do_not_require_signed_bias(count_name):
    sample = f"mae,rmse,r2,mape_pct,n_train,{count_name},fold\n2,2,,10,473,1,0\n"
    assert semantic_problems("results/dafd/stage2_groupkfold_metrics.csv", sample)
    assert not semantic_problems("results/dafd/stage2_groupkfold_metrics.csv", sample.replace(",473,1,", ",473,5,"))


@pytest.mark.parametrize("count", ["nan", "inf", "0", "-1", "1.5"])
def test_invalid_counts_cannot_bypass_singleton_checks(count):
    sample = f"mae,mape_pct,n_test\n2,10,{count}\n"
    assert semantic_problems("results/dafd/stage2_groupkfold_metrics.csv", sample)


def test_no_presentation_fields_in_reference_json():
    for path in (PROJECT_ROOT / "results").rglob("*.json"):
        assert semantic_problems(path.relative_to(PROJECT_ROOT).as_posix(), path.read_text(encoding="utf-8")) == []


def test_history_semantics_are_checked_for_each_path_even_with_shared_blob(monkeypatch):
    from types import SimpleNamespace
    from scripts import public_release_audit as audit

    blob = "a" * 40
    text = b"import matplotlib.pyplot as plt\n"
    monkeypatch.setattr(audit, "_self_check", lambda: [])
    monkeypatch.setattr(audit, "tracked_paths", lambda: [])
    monkeypatch.setattr(audit, "REQUIRED_DOCUMENTS", set())
    monkeypatch.setattr(audit, "_run_git", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(audit, "reachable_history_entries", lambda: [(blob, "README.md"), (blob, "src/common/metrics.py")])
    monkeypatch.setattr(audit, "_object_metadata", lambda _: {blob: ("blob", len(text))})
    monkeypatch.setattr(audit, "_read_blob", lambda _: text)
    problems, _, _ = audit.audit_repository()
    assert any("history semantics src/common/metrics.py" in problem for problem in problems)


def test_public_release_policy_scans_auditor_and_detects_private_content():
    # Sentinels are assembled so the test source itself does not contain a
    # complete credential or workstation path.
    assert content_problems("C:" + "\\Users\\" + "person\\private")
    assert content_problems("Video_" + "1234")
    assert content_problems("gh" + "p_" + "A" * 24)


def synthetic_manifest() -> list[dict[str, str]]:
    rows = []
    for sequence in range(5):
        for frame in range(4):
            rows.append(
                {
                    "image_id": f"synthetic_{sequence}_{frame}",
                    "video_id": f"group_{sequence}",
                    "frame_id": str(frame),
                    "n_class0_boxes": "1",
                    "n_class1_boxes": "1",
                }
            )
    return rows


def test_sm4_split_builder_is_size_matched_and_sequence_isolated():
    f_rows, v_rows, summary = build_folds(synthetic_manifest(), 5, 0)
    assert len(f_rows) == len(v_rows) == 100
    f_sizes = [row["val_n"] for row in summary if row["protocol"] == "F"]
    v_sizes = [row["val_n"] for row in summary if row["protocol"] == "V"]
    assert f_sizes == v_sizes == [4, 4, 4, 4, 4]
    assert all(
        row["sequence_overlap"] == 0
        for row in summary
        if row["protocol"] == "V"
    )
    audited_f = audit_protocol(f_rows, "F", 5)
    audited_v = audit_protocol(v_rows, "V", 5)
    assert [row["val_n"] for row in audited_f] == [4, 4, 4, 4, 4]
    assert all(row["sequence_overlap"] == 0 for row in audited_v)


def test_video_summary_reproduces_reported_values():
    result = summarize(load_rows(DEFAULT_INPUT))
    f = result["protocol_summaries"]["F"]
    v = result["protocol_summaries"]["V"]
    assert np.isclose(f["mAP50"]["mean"], 0.8614585478192867)
    assert np.isclose(f["mAP50"]["sample_sd"], 0.004853307960522305)
    assert np.isclose(v["mAP50"]["mean"], 0.8430584018004812)
    assert np.isclose(v["mAP50_95"]["mean"], 0.4913909222321885)
    assert result["directional_criterion"]["satisfied"] is False


def test_sample_equal_frozen_audit_locked_values():
    """The frozen audit JSON pins the manuscript's sample-equal numbers."""
    audit = json.loads(SAMPLE_EQUAL_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    for protocol, (mae, mape) in FROZEN_SAMPLE_EQUAL.items():
        p = audit["protocols"][protocol]
        assert np.isclose(p["sample_equal_mae_um"], mae, atol=1e-10)
        assert np.isclose(p["sample_equal_mape_pct"], mape, atol=1e-10)


def test_sample_equal_inputs_and_locked_results():
    for protocol, expected in FROZEN_SAMPLE_EQUAL.items():
        path = Path(DEFAULT_INPUTS[protocol]["path"])
        if not path.is_file():
            pytest.skip(
                "DAFD prediction tables absent; regenerate via "
                "stage1b_se_random_baseline.py + stage2_protocol_benchmark.py "
                "(see DATA_ACQUISITION.md)"
            )
        result = path_a_pandas(protocol, path)
        assert np.isclose(result["sample_equal_mae_um"], expected[0], atol=1e-10)
        assert np.isclose(result["sample_equal_mape_pct"], expected[1], atol=1e-10)
