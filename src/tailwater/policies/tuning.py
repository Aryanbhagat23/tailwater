"""Baseline tuning: sweep every member of a family, pick the best.

"Best" is defined against a yield requirement so the comparison is the one a
grower faces: the least water that still meets the requirement on the
*training* seasons. The chosen member is then scored on held-out seasons by
the protocol. Tuning on the evaluation seasons would flatter the baselines
the same way an untuned baseline flatters the agent.
"""
from __future__ import annotations

import numpy as np

from ..env.irrigation_env import simulate
from ..eval.metrics import summarize
from .baselines import FAMILIES


def sweep(family: str, weather, forecast=None, **env_kwargs) -> list[dict]:
    rows = []
    for pol in FAMILIES[family]():
        res = simulate(pol, weather, forecast, **env_kwargs)
        rows.append({"policy": pol, "label": pol.label(), **summarize(res.water, res.yield_)})
    return rows


def select(rows: list[dict], metric: str = "cvar5_yield", floor: float = 0.95):
    """Least-water member meeting ``metric >= floor``. If none qualifies,
    the member with the highest ``metric`` (ties broken by water)."""
    ok = [r for r in rows if r[metric] >= floor - 1e-12]
    if ok:
        return min(ok, key=lambda r: (r["water"], -r[metric]))
    return max(rows, key=lambda r: (r[metric], -r["water"]))


def pareto(rows: list[dict], metric: str = "cvar5_yield") -> list[dict]:
    """Members not dominated on (less water, higher metric), sorted by water."""
    pts = sorted(rows, key=lambda r: (r["water"], -r[metric]))
    front, best = [], -np.inf
    for r in pts:
        if r[metric] > best + 1e-12:
            front.append(r)
            best = r[metric]
    return front
