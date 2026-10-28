"""Robustness (brief step 6): how do trained agents and tuned baselines hold
up when the climate is drier/wetter than the record, or the forecast is
better/worse than the one they were tuned with?

* Climate wetness: 600 seasons from the synthetic generator fitted to the
  full Belle Glade record, with its wetness factor shifted (0.6 = 40% less
  rain). No policy was trained or tuned on these seasons.
* Forecast accuracy: the 74 historical seasons with the forecast noise varied.
  These seasons were (partly) seen in training and tuning, so this sweep
  measures sensitivity to forecast quality, not out-of-sample performance.

Agents: the fold-0 model of every training seed (mean +- sd over seeds).
Baselines: tuned on all 74 historical seasons at the matching CVaR floor.
"""
from _common import load_models, save_json  # noqa: I001

import numpy as np

from tailwater.agents.nn import MLP
from tailwater.agents.ppo import PPOPolicy
from tailwater.data import belle_glade_seasons
from tailwater.env.irrigation_env import simulate
from tailwater.env.weather import SyntheticClimate
from tailwater.eval.metrics import summarize
from tailwater.eval.protocol import FORECAST_NOISE, TUNE_FORECAST_SEED
from tailwater.policies.tuning import select, sweep

WETNESS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
NOISE = [0.0, 0.25, 0.5, 1.0, 2.0]
AGENTS = {"cvar_f90": 0.90, "cvar_f95": 0.95}
FAMILIES = ["threshold", "threshold+forecast", "stage threshold", "ET controller"]


def load_agents(name):
    models = load_models(name)
    return [PPOPolicy(MLP.from_state(st), name) for key, st in sorted(models.items())
            if key.endswith("-0")]


def score(policy, ws, fc):
    r = simulate(policy, ws, fc)
    return summarize(r.water, r.yield_)


def agg(rows):
    keys = ("water", "mean_yield", "cvar5_yield", "worst_yield")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {
        k + "_sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
        for k in keys}


def main():
    hist = belle_glade_seasons()
    climate = SyntheticClimate.fit(hist)
    fc_tune = hist.forecasts(FORECAST_NOISE, TUNE_FORECAST_SEED)
    contenders = {}
    for name, floor in AGENTS.items():
        contenders[f"CVaR-PPO {floor:.2f}"] = load_agents(name)
        for fam in FAMILIES:
            best = select(sweep(fam, hist, fc_tune), "cvar5_yield", floor)
            contenders[f"{fam} {floor:.2f}"] = [best["policy"]]
    out = {"wetness": WETNESS, "noise": NOISE, "climate_sweep": {}, "forecast_sweep": {}}
    for w in WETNESS:
        ws = climate.sample(600, seed=4242, wetness=w)
        fc = ws.forecasts(FORECAST_NOISE, 99)
        rain = float(ws.rain.sum(1).mean())
        for name, pols in contenders.items():
            out["climate_sweep"].setdefault(name, []).append(
                {"wetness": w, "season_rain": rain, **agg([score(p, ws, fc) for p in pols])})
        print(f"wetness {w}: mean season rain {rain:.0f} mm", flush=True)
    for nz in NOISE:
        fc = hist.forecasts(nz, 777)
        for name, pols in contenders.items():
            out["forecast_sweep"].setdefault(name, []).append(
                {"noise": nz, **agg([score(p, hist, fc) for p in pols])})
    save_json(out, "robustness.json")
    for name, rows in out["climate_sweep"].items():
        print(f"{name:28s} " + "  ".join(f"{r['water']:5.0f}/{r['cvar5_yield']:.2f}" for r in rows))


if __name__ == "__main__":
    main()
