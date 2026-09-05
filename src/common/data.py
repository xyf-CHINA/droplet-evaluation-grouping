"""Single source of truth for loading and extracting the 474-SE population.

Reuses the Stage 0 audit loader (audit_dafd3.load_core_files) so there is
exactly ONE implementation of the raw-file parsing, title-row dropping,
column renaming and excel_row recovery. The extraction enforces the Stage 0
multiset guarantees (exact 1:1, no duplicates) instead of re-deriving them.
"""
from collections import Counter

import numpy as np
import pandas as pd

from src.audit.audit_dafd3 import (
    ROUND_MATCH,
    SE_MATCH_COLS,
    assign_geometry_ids,
    rounded_tuple,
)
from src.common.metrics import require_finite
from src.common.model_spec import (
    DENORM_COL,
    FEATURE_COLS,
    OBS_COL,
    TARGET_COL,
)


def load_se_extracted() -> pd.DataFrame:
    """Extract the 474 SE rows from Comprehensive via the Stage 0 locked
    provenance rule, with multiplicity guards. Returns a DataFrame with:
    comp_experiment_id, se_experiment_id, source_row_id (true Excel row),
    geometry_id, the 8 features, target, denorm and obs columns.
    """
    from src.audit.audit_dafd3 import load_core_files

    comp, _ext, se, n_dropped = load_core_files()
    se_with_id, _ = assign_geometry_ids(se)

    # fail-fast: matching key columns must not contain NaN (no None-key
    # matching system — a missing key means the data is broken)
    if se_with_id[SE_MATCH_COLS].isna().any().any():
        raise ValueError("NaN in Raw SE matching key columns — fail-fast")

    # multiplicity guard: SE source keys must be unique (Stage 0: verified 1:1)
    se_keys = Counter(
        rounded_tuple(r[SE_MATCH_COLS], ROUND_MATCH) for _, r in se_with_id.iterrows()
    )
    dup = {k: v for k, v in se_keys.items() if k is not None and v > 1}
    if dup:
        raise ValueError(
            f"SE matching keys duplicated in Raw SE (Stage 0 contract violated): {dup}"
        )
    key_to_se = {
        rounded_tuple(r[SE_MATCH_COLS], ROUND_MATCH): r
        for _, r in se_with_id.iterrows()
    }

    rows = []
    comp_key_counts = Counter()
    for _, r in comp.iterrows():
        t = rounded_tuple(r[SE_MATCH_COLS], ROUND_MATCH)
        if t not in key_to_se:
            continue
        comp_key_counts[t] += 1
        if comp_key_counts[t] > 1:
            raise ValueError(
                f"Comprehensive contains duplicate rows for SE key {t} "
                "(Stage 0 multiset contract violated)"
            )
        s = key_to_se[t]
        exp = r.get("Experiment")
        if pd.isna(exp):
            raise ValueError(f"Comprehensive Experiment id missing for SE key {t}")
        se_exp = s.get("Experiment")
        if pd.isna(se_exp):
            raise ValueError(f"Raw SE Experiment id missing for key {t}")
        rows.append(
            {
                "comp_experiment_id": int(exp),
                "se_experiment_id": int(se_exp),
                "source_row_id": int(s["excel_row"]),  # audit-corrected true Excel row
                "geometry_id": s["geometry_id"],
                **{c: r[c] for c in FEATURE_COLS},
                TARGET_COL: r[TARGET_COL],
                DENORM_COL: r[DENORM_COL],
                OBS_COL: r[OBS_COL],
            }
        )

    if len(rows) != len(se):
        raise ValueError(
            f"SE extraction size mismatch: got {len(rows)} rows, "
            f"expected {len(se)} (Stage 0 contract violated)"
        )
    se_df = pd.DataFrame(rows)
    require_finite(se_df[FEATURE_COLS + [TARGET_COL, DENORM_COL, OBS_COL]].to_numpy(),
                   "SE extracted features/targets")
    return se_df
