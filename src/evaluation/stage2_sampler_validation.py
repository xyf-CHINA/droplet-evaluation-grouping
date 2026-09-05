"""
Stage 2 pre-flight: Protocol G sampler validation (NO MODEL TRAINING).

Validates the uniform feasible-subset sampler before any
model rerun:
    * N_feasible computed exactly by DP
    * theoretical inclusion probability p_g = N_g / N_feasible per geometry
    * 100 distinct subsets drawn with fixed seed; all must sum to exactly 95
    * determinism: same seed -> identical subsets
    * observed inclusion frequencies vs theoretical probabilities (per geometry)
    * coverage and exposure distribution (reported, not gated)

Outputs:
    outputs/geometry_shift/stage2_sampler_validation.json
    outputs/geometry_shift/stage2_sampler_validation.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math

import numpy as np
import pandas as pd

from src.common.data import load_se_extracted
from src.evaluation.protocol_g_sampler import collect_distinct, theoretical_inclusion

OUT_DIR = PROJECT_ROOT / "outputs" / "geometry_shift"
TARGET = 95
N_SAMPLES = 100
SEED = 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    se_df = load_se_extracted()
    sizes_by_gid = se_df["geometry_id"].value_counts().sort_index()
    gid_list = sizes_by_gid.index.to_list()
    sizes = sizes_by_gid.to_list()
    n_total = int(se_df["geometry_id"].nunique())

    cnt, n_feasible, subsets = collect_distinct(sizes, TARGET, N_SAMPLES, SEED)
    _, probs = theoretical_inclusion(sizes, TARGET)

    # determinism: same seed -> identical collection;
    # cross-seed distinctness: two seeded draws share ~no subsets
    # (NOT an independence claim — just a distinctness check)
    _, _, subsets_repeat = collect_distinct(sizes, TARGET, N_SAMPLES, SEED)
    _, _, subsets_seed1 = collect_distinct(sizes, TARGET, N_SAMPLES, 1)
    deterministic = subsets == subsets_repeat
    overlap_seed01 = len(set(subsets) & set(subsets_seed1))

    sums = [sum(sizes[i] for i in sub) for sub in subsets]
    all_feasible = all(s == TARGET for s in sums)
    all_distinct = len(set(subsets)) == N_SAMPLES

    observed = {i: 0 for i in range(len(sizes))}
    for sub in subsets:
        for i in sub:
            observed[i] += 1

    table_rows = []
    for i in range(len(sizes)):
        gid = gid_list[i]
        p = probs[i]
        expected = p * N_SAMPLES
        obs = observed[i]
        sd = math.sqrt(N_SAMPLES * p * (1 - p))
        z = (obs - expected) / sd if sd > 0 else 0.0
        table_rows.append(
            {
                "geometry_id": gid,
                "group_size": sizes[i],
                "theoretical_p": round(p, 6),
                "expected_count": round(expected, 2),
                "observed_count": obs,
                "deviation": obs - expected,
                "z_score": round(z, 3),
            }
        )
    table = pd.DataFrame(table_rows)

    coverage = int(sum(1 for v in observed.values() if v > 0))
    results = {
        "target_n_test": TARGET,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "n_geometries_total": n_total,
        "n_feasible_computed": int(n_feasible),
        "all_sampled_subsets_feasible_n95": bool(all_feasible),
        "all_sampled_subsets_distinct": bool(all_distinct),
        "deterministic_same_seed": bool(deterministic),
        "cross_seed_distinctness_overlap": int(overlap_seed01),
        "cross_seed_distinctness_note": "distinctness only, NOT an independence claim",
        "n_geometries_covered": coverage,
        "exposure_min": int(min(observed.values())),
        "exposure_max": int(max(observed.values())),
        "max_abs_deviation": float(table["deviation"].abs().max()),
        "max_abs_z": float(table["z_score"].abs().max()),
        "per_geometry": table.to_dict("records"),
    }

    with open(OUT_DIR / "stage2_sampler_validation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
    table.to_csv(OUT_DIR / "stage2_sampler_validation.csv", index=False)

    print("[sampler validation] done (no model trained)")
    print(f"  N_feasible = {n_feasible} | all n_test=95: {all_feasible} | "
          f"100 distinct: {all_distinct} | deterministic: {deterministic}")
    print(f"  cross-seed distinctness check: overlap = {overlap_seed01}/100 (expect small)")
    print(f"  coverage: {coverage}/{n_total} | exposure: "
          f"{min(observed.values())}..{max(observed.values())} "
          f"(expected counts range {min(p*N_SAMPLES for p in probs.values()):.2f}.."
          f"{max(p*N_SAMPLES for p in probs.values()):.2f})")
    print(f"  max |observed - expected| = {results['max_abs_deviation']:.2f} "
          f"(max |z| = {results['max_abs_z']:.2f})")
    print("  worst 3 by |z|:")
    for r in table.reindex(table["z_score"].abs().sort_values(ascending=False).index).head(3).to_dict("records"):
        print(f"    {r['geometry_id']}: p={r['theoretical_p']} exp={r['expected_count']} "
              f"obs={r['observed_count']} z={r['z_score']}")


if __name__ == "__main__":
    main()
