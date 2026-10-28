"""Tabular Q-learning over a discretised state.

State: soil-moisture bin x 10-day period of the season x 3-day-forecast
bin x days-since-rain bin. (The growth stage is a function of the period;
an earlier version keyed on stage alone, but stages up to 35 days long hide
the rise of Kc from 0.3 to 1.2 and the agent learned noticeably worse.)

Reward per day: ``-I/100 - kappa * l_t`` with l_t the
day's linearised yield loss computed against the *climatological* stage ETm
(the season's own ETm is unknown until the stage ends).

Training runs a batch of seasons in parallel and applies the Q-learning
update for all of them each day; fields that hit the same (state, action)
cell on the same day contribute their mean TD error.
"""
from __future__ import annotations

import numpy as np

from ..env.irrigation_env import ACTIONS, IrrigationEnv
from ..policies.baselines import Policy
from .common import WATER_UNIT, climatological_stage_etm

MOIST_EDGES = np.arange(5.0, 100.0, 5.0)           # 20 bins of 5 mm
FC_EDGES = np.array([1.0, 5.0, 15.0, 30.0])        # 5 bins
DSR_EDGES = np.array([1.0, 3.0, 7.0])              # 4 bins
PERIOD_DAYS = 10
N_PERIODS = 12
SHAPE = (len(MOIST_EDGES) + 1, N_PERIODS, len(FC_EDGES) + 1, len(DSR_EDGES) + 1)


def encode(obs):
    m = np.digitize(obs["moisture"], MOIST_EDGES)
    f = np.digitize(obs["forecast"], FC_EDGES)
    r = np.digitize(obs["days_since_rain"], DSR_EDGES)
    period = min(obs["day"] // PERIOD_DAYS, N_PERIODS - 1)
    return np.ravel_multi_index((m, np.full_like(m, period), f, r), SHAPE)


class TabularQPolicy(Policy):
    name = "tabular Q"

    def __init__(self, q):
        self.q = q

    def act(self, obs):
        return np.argmax(self.q[encode(obs)], axis=1)


def train_tabular_q(weather, kappa: float = 10.0, episodes: int = 40000, batch: int = 32,
                    alpha: float = 0.5, gamma: float = 1.0, eps_start: float = 0.5,
                    eps_end: float = 0.02, forecast_noise: float = 0.5, seed: int = 0,
                    log_every: int = 0):
    rng = np.random.default_rng(seed)
    env = IrrigationEnv(weather)
    etm_clim = climatological_stage_etm(weather)
    q = np.zeros((int(np.prod(SHAPE)), len(ACTIONS)))
    iters = episodes // batch
    curve = []
    for it in range(iters):
        frac = max(0.0, 1 - it / (0.7 * iters))
        eps = eps_end + (eps_start - eps_end) * frac
        lr = 0.02 + (alpha - 0.02) * frac
        idx = rng.integers(0, weather.n, batch)
        fc = weather.forecasts(forecast_noise, int(rng.integers(1 << 31)))
        obs = env.reset(idx, fc)
        s = encode(obs)
        ret = np.zeros(batch)
        while not env.done:
            greedy = np.argmax(q[s], axis=1)
            a = np.where(rng.random(batch) < eps, rng.integers(0, len(ACTIONS), batch), greedy)
            obs, info = env.step(a)
            r = -(info["irrigation"] / WATER_UNIT
                  + kappa * info["ky"] * (info["etc"] - info["eta"]) / etm_clim[info["stage"]])
            ret += r
            if env.done:
                target = r
                s2 = s
            else:
                s2 = encode(obs)
                target = r + gamma * q[s2].max(1)
            # average the TD errors of all fields that visited the same cell today
            cell = s * len(ACTIONS) + a
            td = np.bincount(cell, target - q[s, a], q.size)
            cnt = np.bincount(cell, minlength=q.size)
            q.ravel()[cnt > 0] += lr * td[cnt > 0] / cnt[cnt > 0]
            s = s2
        y, _, _ = env.final_yields()
        curve.append({"episode": (it + 1) * batch, "return": float(ret.mean()),
                      "water": float(env.water.mean()), "yield": float(y.mean())})
        if log_every and (it + 1) % log_every == 0:
            print(curve[-1])
    return TabularQPolicy(q), curve
