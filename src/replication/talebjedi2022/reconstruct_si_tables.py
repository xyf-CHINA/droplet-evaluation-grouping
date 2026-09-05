"""
Talebjedi 2022 SI reconstruction (Phase A/B) — position-aware table parser.

Reads replication/talebjedi2022/source/Talebjedi2022_SI.pdf (frozen copy),
extracts Table S2 (125-row factorial design: Exp No / Tilt Angle / FRR / Qc),
Table S3 (125-row outputs: Exp No / Size / Frequency / Uniformity / Circle
metric) and Table S4 (randomly selected test/validation experiment numbers),
joins S2+S3 on Exp No 1-125, and runs the reconstruction audits:

- Exp No exactly 1..125, no gaps/duplicates; S2/S3 exact one-to-one join
- tilt angles exactly {30, 60, 90, 120, 150}, 25 rows each
- complete 25-condition factorial coverage within every angle
- Qd rounding audit: Qd_est = Qc / FRR vs published levels {1,3,4,5,7}
  (Qd is used for this audit ONLY; it is not a primary predictor)
- published-split audit: Table S4 test/validation Exp Nos mapped to tilt
  angles per network (Size / Frequency / Uniformity / Circle metric)

No model is trained here. Exit code 1 on any audit failure.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SI_PDF = PROJECT_ROOT / "replication" / "talebjedi2022" / "source" / "Talebjedi2022_SI.pdf"
OUT = PROJECT_ROOT / "replication" / "talebjedi2022"

NUM_RE = re.compile(r"^\d+(\.\d+)?$")

# (page -> (y_lo, y_hi)) table regions and column x-bands (derived from the
# PDF layout; see reconstruction_audit.json layout note).
# yhi = 720 (not 718): page-top rows sit at y1 = 718.1 in this PDF.
S2_REGIONS = {3: (70.0, 391.5), 4: (70.0, 720.0), 5: (70.0, 720.0), 6: (440.0, 720.0)}
S2_BANDS = {"expno": (165, 205), "tilt": (222, 248), "frr": (295, 340), "qc": (395, 420)}
S3_REGIONS = {6: (70.0, 385.0), 7: (70.0, 720.0), 8: (70.0, 720.0), 9: (430.0, 720.0)}
S3_BANDS = {"expno": (120, 150), "size": (165, 200), "freq": (240, 285),
            "unif": (335, 370), "circle": (425, 455)}
S4_REGIONS = {9: (70.0, 336.0)}
S4_BANDS = {"size_t": (95, 140), "size_v": (145, 200), "freq_t": (215, 245),
            "freq_v": (250, 315), "unif_t": (335, 365), "unif_v": (365, 425),
            "circ_t": (450, 495), "circ_v": (495, 545)}

Qd_LEVELS = {1, 3, 4, 5, 7}


def positioned_numeric_lines(pageno):
    out = []
    for layout in extract_pages(SI_PDF):
        if layout.pageid != pageno:
            continue
        for el in layout:
            if isinstance(el, LTTextContainer):
                for tl in el:
                    if isinstance(tl, LTTextLine):
                        t = tl.get_text().strip()
                        if NUM_RE.match(t):
                            out.append((round(tl.x0, 1), round(tl.y1, 1), t))
        break
    return out


def snap_rows(lines, ytol=3.5):
    """Group lines into rows by y1 (top coordinate), descending."""
    groups = []
    for x0, y1, t in sorted(lines, key=lambda r: -r[1]):
        placed = False
        for g in groups:
            if abs(g["y"] - y1) <= ytol:
                g["cells"].append((x0, t))
                placed = True
                break
        if not placed:
            groups.append({"y": y1, "cells": [(x0, t)]})
    return groups


def extract_region(page, ylo, yhi, bands):
    lines = [l for l in positioned_numeric_lines(page)
             if ylo <= l[1] <= yhi]
    rows = []
    for g in snap_rows(lines):
        cells = {}
        for x0, t in g["cells"]:
            for name, (xlo, xhi) in bands.items():
                if xlo <= x0 <= xhi and name not in cells:
                    cells[name] = t
                    break
        if cells:
            rows.append(cells)
    return rows


def main() -> None:
    # ---------- Table S2 ----------
    s2_rows = []
    for page, (ylo, yhi) in sorted(S2_REGIONS.items()):
        s2_rows.extend(extract_region(page, ylo, yhi, S2_BANDS))
    # R1 hard gates: enforce row count and uniqueness BEFORE dict construction
    assert len(s2_rows) == 125, f"S2 raw extracted rows != 125: {len(s2_rows)}"
    _raw_s2_exp = [int(r["expno"]) for r in s2_rows]
    assert len(_raw_s2_exp) == len(set(_raw_s2_exp)), "S2 raw ExpNo duplicates"
    s2 = {}
    for r in s2_rows:
        if set(r) != set(S2_BANDS):
            raise RuntimeError(f"S2 row missing/extra cells: {sorted(r)} @ {r}")
        exp = int(r["expno"])
        s2[exp] = {"tilt": int(r["tilt"]), "frr": float(r["frr"]), "qc": float(r["qc"])}

    # ---------- Table S3 ----------
    s3_rows = []
    for page, (ylo, yhi) in sorted(S3_REGIONS.items()):
        s3_rows.extend(extract_region(page, ylo, yhi, S3_BANDS))
    # R1 hard gates: enforce row count and uniqueness BEFORE dict construction
    assert len(s3_rows) == 125, f"S3 raw extracted rows != 125: {len(s3_rows)}"
    _raw_s3_exp = [int(r["expno"]) for r in s3_rows]
    assert len(_raw_s3_exp) == len(set(_raw_s3_exp)), "S3 raw ExpNo duplicates"
    s3 = {}
    for r in s3_rows:
        if set(r) != set(S3_BANDS):
            raise RuntimeError(f"S3 row missing/extra cells: {sorted(r)} @ {r}")
        exp = int(r["expno"])
        s3[exp] = {"size": float(r["size"]), "freq": float(r["freq"]),
                   "unif": float(r["unif"]), "circle": float(r["circle"])}

    # ---------- Table S4 ----------
    s4_cells = {}
    for page, (ylo, yhi) in sorted(S4_REGIONS.items()):
        for r in extract_region(page, ylo, yhi, S4_BANDS):
            for name, t in r.items():
                s4_cells.setdefault(name, []).append(int(t))
    s4 = {
        "Size": {"test": s4_cells.get("size_t", []), "validation": s4_cells.get("size_v", [])},
        "Frequency": {"test": s4_cells.get("freq_t", []), "validation": s4_cells.get("freq_v", [])},
        "Uniformity": {"test": s4_cells.get("unif_t", []), "validation": s4_cells.get("unif_v", [])},
        "Circle metric": {"test": s4_cells.get("circ_t", []), "validation": s4_cells.get("circ_v", [])},
    }

    # =============================== audits ==================================
    audit = {"layout_note": (
        "Column x-bands per page region were read off the PDF text layout "
        "(pdfminer positioned lines) and are fixed in the parser source. "
        "Row grid: 15 pt spacing, 3.5 pt snap tolerance. Lines outside the "
        "declared bands (e.g. the rotated ACS footer at x~588) are ignored."
    )}

    # S2 structural audits
    expnos = sorted(s2)
    audit["s2_n_rows"] = len(s2)
    audit["s2_expno_exact_1_125"] = expnos == list(range(1, 126))
    angles = sorted(set(r["tilt"] for r in s2.values()))
    audit["tilt_angles_exact"] = angles == [30, 60, 90, 120, 150]
    audit["per_angle_counts"] = {a: sum(1 for r in s2.values() if r["tilt"] == a)
                                 for a in angles}

    # factorial coverage: within each angle, 25 unique (FRR, Qc) conditions
    # identical across angles
    per_angle_pairs = {}
    for a in angles:
        pairs = sorted({(r["frr"], r["qc"]) for r in s2.values() if r["tilt"] == a})
        per_angle_pairs[a] = pairs
    audit["per_angle_unique_conditions"] = {a: len(v) for a, v in per_angle_pairs.items()}
    audit["factorial_pairs_identical_across_angles"] = (
        len({tuple(v) for v in per_angle_pairs.values()}) == 1
    )

    # Qd rounding audit (Qd_est = Qc / FRR vs published levels)
    qd_dev = []
    per_angle_qd = defaultdict(list)
    for exp, r in s2.items():
        qd_est = r["qc"] / r["frr"]
        nearest = min(Qd_LEVELS, key=lambda L: abs(qd_est - L))
        dev = abs(qd_est - nearest)
        qd_dev.append(dev)
        per_angle_qd[r["tilt"]].append(nearest)
    audit["qd_audit_max_abs_dev"] = max(qd_dev)
    audit["qd_audit_all_within_0_02"] = all(d < 0.02 for d in qd_dev)
    audit["qd_audit_per_angle_level_counts"] = {
        a: {L: per_angle_qd[a].count(L) for L in sorted(Qd_LEVELS)} for a in angles
    }

    # S3 structural audits + join
    audit["s3_n_rows"] = len(s3)
    audit["s3_expno_exact_1_125"] = sorted(s3) == list(range(1, 126))
    audit["s3_all_targets_present_finite"] = all(
        all(math.isfinite(v) for v in r.values()) for r in s3.values()
    )
    audit["s2_s3_exact_one_to_one_join"] = sorted(s2) == sorted(s3)

    # S4 audits (R1 hardened: exact 16/16 per role, disjoint, residual train
    # set computed explicitly, and train/test/validation must EACH cover all
    # five angles)
    _all_exp = set(range(1, 126))
    s4_train = {}
    for net, roles in s4.items():
        assert len(roles["test"]) == 16, f"{net} test n != 16"
        assert len(roles["validation"]) == 16, f"{net} validation n != 16"
        assert set(roles["test"]).isdisjoint(set(roles["validation"])), \
            f"{net} test/validation not disjoint"
        s4_train[net] = sorted(_all_exp - set(roles["test"]) - set(roles["validation"]))
        assert len(s4_train[net]) == 93, f"{net} residual train n != 93"

    s4_audit = {}
    for net, roles in s4.items():
        for role, lst in roles.items():
            s4_audit[f"{net} {role}"] = {
                "n": len(lst),
                "all_expnos_valid": all(1 <= e <= 125 for e in lst),
                "no_duplicates": len(set(lst)) == len(lst),
                "angle_counts": {
                    a: sum(1 for e in lst if s2[e]["tilt"] == a) for a in angles
                },
            }
    audit["published_split"] = s4_audit
    audit["published_split_overlap_test_validation"] = {
        net: sorted(set(roles["test"]) & set(roles["validation"]))
        for net, roles in s4.items()
    }

    def _angle_set(expnos):
        return {s2[e]["tilt"] for e in expnos}

    audit["published_split_coverage"] = {}
    for net, roles in s4.items():
        train = s4_train[net]
        audit["published_split_coverage"][net] = {
            "train_n": len(train),
            "train_angle_counts": {
                a: sum(1 for e in train if s2[e]["tilt"] == a) for a in angles
            },
            "train_covers_all_5_angles": _angle_set(train) == set(angles),
            "test_covers_all_5_angles": _angle_set(roles["test"]) == set(angles),
            "validation_covers_all_5_angles": _angle_set(roles["validation"]) == set(angles),
        }

    # =============================== outputs =================================
    (OUT / "reconstructed").mkdir(parents=True, exist_ok=True)
    (OUT / "audit").mkdir(parents=True, exist_ok=True)

    csv_lines = ["ExpNo,TiltAngle_deg,FRR,Qc_uL_min,Size_um,Frequency_Hz,Uniformity_pct,CircleMetric"]
    for exp in range(1, 126):
        a = s2[exp]
        b = s3[exp]
        csv_lines.append(
            f"{exp},{a['tilt']},{a['frr']:.4f},{a['qc']:.0f},"
            f"{b['size']:.4f},{b['freq']:.4f},{b['unif']:.4f},{b['circle']:.4f}"
        )
    (OUT / "reconstructed" / "talebjedi_125_reconstructed.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    (OUT / "audit" / "group_sizes.csv").write_text(
        "TiltAngle_deg,n\n"
        + "\n".join(f"{a},{audit['per_angle_counts'][a]}" for a in angles) + "\n",
        encoding="utf-8",
    )

    split_lines = ["network,role,expno,TiltAngle_deg"]
    for net, roles in s4.items():
        for role, lst in roles.items():
            for e in lst:
                split_lines.append(f"{net},{role},{e},{s2[e]['tilt']}")
    (OUT / "audit" / "published_split_angle_audit.csv").write_text(
        "\n".join(split_lines) + "\n", encoding="utf-8"
    )

    (OUT / "audit" / "reconstruction_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    # =============================== hard gate ================================
    fail = []
    if not audit["s2_expno_exact_1_125"]:
        fail.append("S2 ExpNo not exactly 1..125")
    if not audit["tilt_angles_exact"]:
        fail.append("tilt angles not exactly {30,60,90,120,150}")
    if any(v != 25 for v in audit["per_angle_counts"].values()):
        fail.append(f"per-angle counts not all 25: {audit['per_angle_counts']}")
    if any(v != 25 for v in audit["per_angle_unique_conditions"].values()):
        fail.append("factorial coverage incomplete")
    if not audit["factorial_pairs_identical_across_angles"]:
        fail.append("condition sets differ across angles")
    if not audit["qd_audit_all_within_0_02"]:
        fail.append(f"Qd audit deviation too large: {audit['qd_audit_max_abs_dev']:.4f}")
    if not audit["s3_expno_exact_1_125"] or not audit["s2_s3_exact_one_to_one_join"]:
        fail.append("S3/S2 join broken")
    if not audit["s3_all_targets_present_finite"]:
        fail.append("S3 targets missing/non-finite")
    for k, v in s4_audit.items():
        if not (v["all_expnos_valid"] and v["no_duplicates"] and v["n"] > 0):
            fail.append(f"published split list invalid: {k}")
    # R1 hard gate: the PRIMARY network (Size — the replication target) must
    # have all-5-angle coverage in train/test/validation. Coverage of the
    # other three networks is recorded in published_split_coverage as a
    # factual property of the published table (e.g. the Uniformity test set
    # contains no 60-degree experiments — a finding, not a reconstruction
    # error).
    for role in ("train", "test", "validation"):
        if not audit["published_split_coverage"]["Size"][f"{role}_covers_all_5_angles"]:
            fail.append(f"Size {role} does not cover all 5 angles")
    for net, cov in audit["published_split_coverage"].items():
        if cov["train_n"] != 93:
            fail.append(f"{net} residual train n != 93")

    if fail:
        print("RECONSTRUCTION AUDIT FAILED:")
        for f in fail:
            print("  -", f)
        sys.exit(1)

    print(f"[reconstruct] S2 rows={len(s2)} S3 rows={len(s3)} "
          f"S4={ {net: {r: len(lst) for r, lst in roles.items()} for net, roles in s4.items()} }")
    print("[reconstruct] all audits PASS (Qd max dev %.4f)" % audit["qd_audit_max_abs_dev"])
    print("[reconstruct] outputs: reconstructed CSV, reconstruction_audit.json, "
          "group_sizes.csv, published_split_angle_audit.csv")


if __name__ == "__main__":
    main()
