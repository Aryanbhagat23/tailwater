"""Season-level summary metrics.

CVaR at level alpha is the mean of the worst ceil(alpha * N) seasons (the
discrete lower-tail mean). With 74 held-out seasons, CVaR-5% is the mean of
the worst 4.
"""
from __future__ import annotations

import math

import numpy as np


def cvar_lower(x, alpha: float = 0.05) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, math.ceil(alpha * len(x)))
    return float(x[:k].mean())


def summarize(water, yields, alpha: float = 0.05) -> dict:
    water, yields = np.asarray(water, float), np.asarray(yields, float)
    mean_y = float(yields.mean())
    return {
        "water": float(water.mean()),
        "water_sd": float(water.std()),
        "mean_yield": mean_y,
        "p5_yield": float(np.percentile(yields, 5)),
        "cvar5_yield": cvar_lower(yields, alpha),
        "worst_yield": float(yields.min()),
        "water_per_yield": float(water.mean() / mean_y) if mean_y > 0 else float("inf"),
        "n": int(len(yields)),
    }


def mean_sd(rows: list[dict]) -> dict:
    """Mean and standard deviation of each metric across training seeds."""
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(v.mean())
        out[k + "_seed_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["seeds"] = len(rows)
    return out
