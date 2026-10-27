"""Evaluation protocol.

* **Held-out years.** Seasons are split into K folds by calendar year. For
  each fold, baselines are tuned and agents trained on the other K-1 folds
  only, then scored on the held-out fold. Pooling the K held-out folds gives
  an out-of-sample result for every year in the record (74 for Belle Glade),
  which is what makes a 5% tail estimate possible at all.
* **Identical weather.** Every policy is scored on the same seasons with the
  same forecast draw (``EVAL_FORECAST_SEED``). Baseline tuning uses a
  different forecast draw; agents draw fresh forecast noise during training.
* **Multiple seeds.** Each agent configuration is trained with several seeds;
  every reported agent number is a mean over seeds with its standard
  deviation. Tuned baselines are deterministic given the protocol, so they
  have no seed spread.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..env.irrigation_env import simulate
from ..policies.tuning import select, sweep
from .metrics import mean_sd, summarize

FORECAST_NOISE = 0.5
EVAL_FORECAST_SEED = 777
TUNE_FORECAST_SEED = 1
N_FOLDS = 4
FOLD_SEED = 2026


def make_folds(n: int, k: int = N_FOLDS, seed: int = FOLD_SEED):
    """Random year-to-fold assignment (fixed seed), balanced fold sizes."""
    perm = np.random.default_rng(seed).permutation(n)
    test_sets = [np.sort(perm[i::k]) for i in range(k)]
    return [(np.setdiff1d(np.arange(n), te), te) for te in test_sets]


def eval_forecast(weather):
    return weather.forecasts(FORECAST_NOISE, EVAL_FORECAST_SEED)


def tune_and_test(weather, family: str, floors, folds, metric: str = "cvar5_yield"):
    """Tune ``family`` on each fold's training years for each yield floor and
    score the chosen member on the held-out years.

    Returns {floor: {"summary", "water", "yield", "chosen": [labels per fold]}}
    with per-season arrays in the original season order.
    """
    fc_eval = eval_forecast(weather)
    fc_tune = weather.forecasts(FORECAST_NOISE, TUNE_FORECAST_SEED)
    out = {f: {"water": np.zeros(weather.n), "yield": np.zeros(weather.n), "chosen": []}
           for f in floors}
    for tr, te in folds:
        rows = sweep(family, weather.subset(tr), fc_tune[tr])
        for f in floors:
            best = select(rows, metric, f)
            res = simulate(best["policy"], weather.subset(te), fc_eval[te])
            out[f]["water"][te] = res.water
            out[f]["yield"][te] = res.yield_
            out[f]["chosen"].append(best["label"])
    for f in floors:
        out[f]["summary"] = summarize(out[f]["water"], out[f]["yield"])
    return out


def _run_one(args):
    train_fn, cfg, weather, tr, te, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    policy, curve = train_fn(weather.subset(tr), cfg, seed)
    res = simulate(policy, weather.subset(te), eval_forecast(weather)[te])
    return policy, curve, res.water, res.yield_


def cross_validate_agent(train_fn, cfg, weather, folds, seeds, workers: int = 4):
    """Train ``train_fn(train_weather, cfg, seed) -> (policy, curve)`` on every
    fold x seed; return per-seed pooled held-out results and their mean/sd."""
    jobs = [(train_fn, cfg, weather, tr, te, s) for s in seeds for (tr, te) in folds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            outs = list(ex.map(_run_one, jobs))
    else:
        outs = [_run_one(j) for j in jobs]
    per_seed, policies, curves = [], {}, {}
    k = len(folds)
    for i, s in enumerate(seeds):
        water, yields = np.zeros(weather.n), np.zeros(weather.n)
        for j, (tr, te) in enumerate(folds):
            pol, curve, w, y = outs[i * k + j]
            water[te], yields[te] = w, y
            policies[(s, j)] = pol
            curves[(s, j)] = curve
        per_seed.append({"seed": s, "water": water, "yield": yields,
                         "summary": summarize(water, yields)})
    return {"per_seed": per_seed, "summary": mean_sd([p["summary"] for p in per_seed]),
            "policies": policies, "curves": curves}
