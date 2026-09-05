"""Shared metric helpers — single implementation used by every stage."""
import numpy as np


def summary_stats(vals) -> dict:
    """mean / SD(ddof=1) / median / min / max / empirical 2.5-97.5 percentile."""
    a = np.array(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "p2_5": float(np.percentile(a, 2.5)),
        "p97_5": float(np.percentile(a, 97.5)),
    }


def mape_pct(y_obs, y_pred) -> float:
    """Author MAPE formula: mean(|obs - pred| / obs) * 100.

    Raises ValueError if any observed value is 0 (division would be inf);
    never returns inf/NaN silently.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_obs == 0):
        raise ValueError("MAPE undefined: observed droplet diameter contains 0")
    return float(np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100)


def require_finite(arr, name: str):
    """Raise on any NaN/inf — used for features, targets, predictions, metrics."""
    a = np.asarray(arr, dtype=float)
    if not np.isfinite(a).all():
        bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"{name}: {bad} non-finite value(s) (NaN/inf)")
    return a
