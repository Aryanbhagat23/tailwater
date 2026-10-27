"""Vectorised irrigation-season environment.

A batch of fields, one season each, advanced in lock-step one day at a time.
All fields share the calendar, so the growth stage is common to the batch;
each field has its own weather, forecast and soil moisture.

Decision timing on day d: the policy observes the morning state (moisture
after day d-1, yesterday's rain and ET, a rain forecast for days d..d+2) and
chooses an irrigation amount. Then day d's rain falls and the water balance
advances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import water_balance as wb
from .crop import CropParams
from .weather import WeatherSet
from .yield_model import fao33_relative_yield, linear_stress_yield

ACTIONS = np.array([0.0, 5.0, 10.0, 20.0])  # mm
RAIN_DAY_MM = 2.0  # rain at or above this resets "days since rain"


@dataclass
class SeasonResult:
    water: np.ndarray            # (N,) mm applied
    yield_: np.ndarray           # (N,) relative yield under the env's yield model
    yield_fao33: np.ndarray
    yield_linear: np.ndarray
    eta_stage: np.ndarray        # (N, n_stages)
    etm_stage: np.ndarray
    rain: np.ndarray             # (N,) season rain
    drainage: np.ndarray         # (N,)
    ledger_residual: np.ndarray  # (N,) start + rain + irr - ETa - drainage - final
    guard_bound_days: int
    traj: dict | None = None     # per-day arrays (N, T) when recorded


class IrrigationEnv:
    def __init__(self, weather: WeatherSet, forecast: np.ndarray | None = None,
                 soil: wb.SoilParams = wb.SoilParams(), crop: CropParams = CropParams(),
                 yield_model: str = "fao33"):
        assert weather.length == crop.season_length
        assert yield_model in ("fao33", "linear")
        self.weather = weather
        self.forecast = weather.forecasts(0.0, 0) if forecast is None else forecast
        self.soil, self.crop, self.yield_model = soil, crop, yield_model
        self.kc = crop.kc(np.arange(crop.season_length))
        self.stage = crop.stage_of_day(np.arange(crop.season_length))
        self.ky = np.asarray(crop.ky)

    # ------------------------------------------------------------------
    def reset(self, idx=None, forecast: np.ndarray | None = None):
        self.idx = np.arange(self.weather.n) if idx is None else np.asarray(idx)
        self.fc = (self.forecast if forecast is None else forecast)[self.idx]
        self.rain = self.weather.rain[self.idx]
        self.et0 = self.weather.et0[self.idx]
        n = len(self.idx)
        self.day = 0
        self.moisture = np.full(n, self.soil.initial_fraction * self.soil.taw)
        self.start_moisture = self.moisture.copy()
        self.dsr = np.zeros(n)
        self.rain_y = np.zeros(n)
        self.et0_y = self.et0[:, 0].copy()  # no history on day 0: use today's
        self.etc_y = self.et0_y * self.kc[0]
        self.water = np.zeros(n)
        self.drain = np.zeros(n)
        self.eta_stage = np.zeros((n, self.crop.n_stages))
        self.etm_stage = np.zeros((n, self.crop.n_stages))
        self.stress_days = np.zeros(n)
        self.guard_bound = 0
        return self.observe()

    def observe(self) -> dict:
        d = min(self.day, self.crop.season_length - 1)
        return {
            "day": self.day,
            "stage": int(self.stage[d]),
            "kc": float(self.kc[d]),
            "moisture": self.moisture,
            "forecast": self.fc[:, d],
            "days_since_rain": self.dsr,
            "rain_yesterday": self.rain_y,
            "et0_yesterday": self.et0_y,
            "etc_yesterday": self.etc_y,
        }

    @property
    def done(self) -> bool:
        return self.day >= self.crop.season_length

    def step(self, action_idx=None, mm=None):
        """Apply actions (indices into ACTIONS), or raw amounts in mm for
        diagnostic policies outside the action set. Returns (obs, info)."""
        d = self.day
        irr = ACTIONS[np.asarray(action_idx)] if mm is None else np.broadcast_to(
            np.asarray(mm, dtype=float), self.moisture.shape)
        rain = self.rain[:, d]
        etc = self.kc[d] * self.et0[:, d]
        s = self.stage[d]
        new, f = wb.step(self.moisture, rain, irr, etc, self.soil)
        self.moisture = new
        self.water += irr
        self.drain += f["drainage"]
        self.eta_stage[:, s] += f["eta"]
        self.etm_stage[:, s] += etc
        self.stress_days += 1.0 - f["ks"]
        self.guard_bound += int(f["guard_bound"].sum())
        self.dsr = np.where(rain >= RAIN_DAY_MM, 0.0, self.dsr + 1.0)
        self.rain_y, self.et0_y, self.etc_y = rain, self.et0[:, d], etc
        self.day += 1
        info = {"irrigation": irr, "rain": rain, "etc": etc, "eta": f["eta"],
                "ks": f["ks"], "drainage": f["drainage"], "stage": s, "ky": self.ky[s]}
        return self.observe(), info

    def final_yields(self):
        y33 = fao33_relative_yield(self.eta_stage, self.etm_stage, self.ky)
        ylin = linear_stress_yield(self.stress_days)
        return (y33 if self.yield_model == "fao33" else ylin), y33, ylin

    def result(self, traj=None) -> SeasonResult:
        y, y33, ylin = self.final_yields()
        resid = (self.start_moisture + self.rain.sum(1) + self.water
                 - self.eta_stage.sum(1) - self.drain - self.moisture)
        return SeasonResult(self.water.copy(), y, y33, ylin, self.eta_stage.copy(),
                            self.etm_stage.copy(), self.rain.sum(1), self.drain.copy(),
                            resid, self.guard_bound, traj)


def simulate(policy, weather: WeatherSet, forecast: np.ndarray | None = None,
             record: bool = False, **env_kwargs) -> SeasonResult:
    """Run ``policy`` over every season of ``weather`` (identical weather and
    forecasts for every policy given the same arguments)."""
    env = IrrigationEnv(weather, forecast, **env_kwargs)
    obs = env.reset()
    policy.reset(len(env.idx))
    keys = ("moisture", "irrigation", "rain", "eta", "etc", "ks", "forecast")
    traj = {k: [] for k in keys} if record else None
    while not env.done:
        a = policy.act(obs)
        if record:
            traj["forecast"].append(obs["forecast"].copy())
        obs, info = env.step(mm=a) if getattr(policy, "continuous", False) else env.step(a)
        if record:
            traj["moisture"].append(env.moisture.copy())
            for k in ("irrigation", "rain", "eta", "etc", "ks"):
                traj[k].append(np.broadcast_to(info[k], env.moisture.shape).copy())
    if record:
        traj = {k: np.stack(v, axis=1) for k, v in traj.items()}
    return env.result(traj)
