"""Protocol G sampler — uniform over feasible subsets.

Estimand: every geometry subset whose total sample count equals the target
(95 = round(0.2 * 474)) is drawn with EQUAL probability. This is NOT uniform
over geometries: each geometry's theoretical inclusion probability
p_g = N_g / N_feasible is computed exactly by 0/1-subset-sum DP and can differ
between geometries. The validator (stage2_sampler_validation.py) checks the
observed inclusion frequencies against these theoretical values.
"""
from __future__ import annotations

import numpy as np


def feasible_counts(sizes: list, target: int) -> list:
    """cnt[i][v] = number of subsets of sizes[i:] summing to v (bounded 0/1 DP)."""
    N = len(sizes)
    cnt = [[0] * (target + 1) for _ in range(N + 1)]
    cnt[N][0] = 1
    for i in range(N - 1, -1, -1):
        s = sizes[i]
        row = cnt[i]
        nxt = cnt[i + 1]
        for v in range(target + 1):
            row[v] = nxt[v] + (nxt[v - s] if v >= s else 0)
    return cnt


def sample_uniform_subset(sizes: list, target: int, cnt: list, rng: np.random.Generator) -> tuple:
    """Draw ONE feasible subset uniformly via the DP counting table (probability walk)."""
    subset = []
    v = target
    for i in range(len(sizes)):
        s = sizes[i]
        if v >= s:
            inc = cnt[i + 1][v - s]
            exc = cnt[i + 1][v]
            tot = inc + exc
            if tot == 0:
                continue
            if rng.random() < inc / tot:
                subset.append(i)
                v -= s
    assert v == 0, "sampler failed to hit the target sum"
    return tuple(subset)


def collect_distinct(sizes: list, target: int, n_want: int, seed: int):
    """Sample n_want DISTINCT feasible subsets with a fixed seed.

    Returns (cnt, n_feasible, sorted list of subset tuples).
    """
    cnt = feasible_counts(sizes, target)
    n_feasible = cnt[0][target]
    if n_feasible == 0:
        raise ValueError(f"no feasible subset sums to {target}")
    rng = np.random.default_rng(seed)
    found = set()
    guard = 0
    while len(found) < n_want:
        guard += 1
        if guard > 200_000:
            raise RuntimeError(
                f"could not collect {n_want} distinct feasible subsets "
                f"(only {len(found)} found)"
            )
        found.add(sample_uniform_subset(sizes, target, cnt, rng))
    return cnt, n_feasible, sorted(found)


def theoretical_inclusion(sizes: list, target: int) -> tuple[int, dict]:
    """N_feasible and per-geometry theoretical inclusion probability p_g.

    p_g = (# feasible subsets containing g) / N_feasible, computed exactly
    by re-running the DP with geometry g removed.
    """
    cnt = feasible_counts(sizes, target)
    n_feasible = cnt[0][target]
    probs = {}
    for g in range(len(sizes)):
        sizes_without = sizes[:g] + sizes[g + 1:]
        c2 = feasible_counts(sizes_without, target)
        n_without = c2[0][target]
        probs[g] = (n_feasible - n_without) / n_feasible
    return n_feasible, probs
