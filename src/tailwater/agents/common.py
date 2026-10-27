"""Pieces shared by the agents: the objective, the yield-loss decomposition
used for dense rewards, and the observation features.

Objective. A season's value is

    kappa * Y  -  W / 100

with Y the FAO-33 relative yield and W the water applied in mm, so
``kappa`` is the value of a full crop expressed in units of 100 mm of water
(kappa = 10: one percent of yield is worth 10 mm). The risk-constrained agent
additionally requires CVaR_5%(Y) >= floor.

Dense rewards. Y is only known at harvest. FAO-33 is multiplicative over
stages, and to first order its loss decomposes over days:

    1 - Y  ~=  L = sum_s Ky_s (1 - ETa_s/ETm_s) = sum_t Ky_s(t) (ETc_t - ETa_t) / ETm_s(t)

Agents get ``-kappa * l_t`` each day, where l_t is day t's term, plus a
terminal correction ``kappa * (Y - 1 + L)`` so that the season's rewards sum to
exactly ``kappa * (Y - 1)``. Shaping changes credit assignment, not the
objective.
"""
from __future__ import annotations

import numpy as np

from ..env.crop import CropParams
from ..env.irrigation_env import ACTIONS
from ..env.water_balance import SoilParams

WATER_UNIT = 100.0  # mm


def daily_loss(etc, eta, ky_day, etm_stage_of_day):
    """Linearised yield-loss contribution of each day (arrays (N, T))."""
    return ky_day * (etc - eta) / np.maximum(etm_stage_of_day, 1e-9)


def stage_etm(etc, stage_of_day, n_stages):
    """Per-season ETm of the stage each day belongs to, (N, T)."""
    out = np.zeros_like(etc)
    for s in range(n_stages):
        m = stage_of_day == s
        out[:, m] = etc[:, m].sum(1, keepdims=True)
    return out


def season_rewards(irr, etc, eta, Y, stage_of_day, ky, kappa, tail_weight=None, nu=0.0):
    """Per-day rewards (N, T) for completed seasons.

    ``tail_weight`` (N,) is lambda/alpha for seasons in the yield tail and 0
    otherwise (see ``cvar_lagrangian.py``). Rows sum exactly to
    ``-W/100 + kappa*(Y-1) + tail_weight*(Y - nu)``.
    """
    n_stages = len(ky)
    w = np.zeros(len(Y)) if tail_weight is None else np.asarray(tail_weight, float)
    l = daily_loss(etc, eta, np.asarray(ky)[stage_of_day][None, :],
                   stage_etm(etc, stage_of_day, n_stages))
    L = l.sum(1)
    R = -irr / WATER_UNIT - (kappa + w[:, None]) * l
    R[:, -1] += kappa * (Y - 1.0 + L) + w * (Y - nu + L)
    return R


def climatological_stage_etm(weather, crop: CropParams = CropParams()):
    """Mean ETm per stage over a set of seasons — what an agent can expect
    before the season (used by the online tabular agent)."""
    etc = weather.et0 * crop.kc(np.arange(crop.season_length))[None, :]
    st = crop.stage_of_day(np.arange(crop.season_length))
    return np.array([etc[:, st == s].sum(1).mean() for s in range(crop.n_stages)])


N_FEATURES = 12


def features(obs, soil: SoilParams = SoilParams(), crop: CropParams = CropParams()):
    """Continuous observation vector for the PPO agent, shape (N, 12)."""
    n = len(obs["moisture"])
    f = np.zeros((n, N_FEATURES))
    f[:, 0] = obs["moisture"] / soil.taw
    f[:, 1] = obs["day"] / crop.season_length
    f[:, 2 + obs["stage"]] = 1.0
    f[:, 7] = np.log1p(obs["forecast"]) / np.log1p(50.0)
    f[:, 8] = np.minimum(obs["days_since_rain"], 15.0) / 15.0
    f[:, 9] = obs["kc"]
    f[:, 10] = obs["et0_yesterday"] / 8.0
    f[:, 11] = obs["etc_yesterday"] / 8.0
    return f


__all__ = ["ACTIONS", "WATER_UNIT", "daily_loss", "season_rewards", "stage_etm",
           "climatological_stage_etm",
           "features", "N_FEATURES"]
