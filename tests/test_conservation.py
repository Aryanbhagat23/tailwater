"""The water balance must conserve water exactly, under every policy, on
both synthetic and (when available) historical weather (brief §3.3)."""
import numpy as np
import pytest

from tailwater.env import water_balance as wb
from tailwater.env.irrigation_env import IrrigationEnv, simulate
from tailwater.env.weather import SyntheticClimate, synthetic_iid
from tailwater.policies.baselines import (ConstantRate, ETController, FixedCalendar, RainFed,
                                          Threshold)

POLICIES = [RainFed(), Threshold(35, 5), Threshold(80, 20), FixedCalendar(1, 20),
            ETController(20, 1.0), ConstantRate(12.0)]


@pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.label())
def test_season_ledger_closes(policy):
    ws = synthetic_iid(50, seed=3)
    res = simulate(policy, ws, ws.forecasts(0.5, 0))
    # start + rain + irrigation - crop use - drainage - final moisture == 0
    assert np.abs(res.ledger_residual).max() < 1e-9
    assert res.guard_bound_days == 0  # the non-negativity guard never fires


def test_daily_step_closes_and_stays_in_bounds():
    rng = np.random.default_rng(0)
    soil = wb.SoilParams()
    s = rng.uniform(0, soil.taw, 10000)
    rain = rng.exponential(10, 10000) * (rng.random(10000) < 0.3)
    irr = rng.choice([0, 5, 10, 20], 10000)
    etc = rng.uniform(0, 12, 10000)
    new, f = wb.step(s, rain, irr, etc, soil)
    np.testing.assert_allclose(s + rain + irr - f["eta"] - f["drainage"], new, atol=1e-12)
    assert (new >= 0).all() and (new <= soil.taw + 1e-12).all()
    assert (f["eta"] <= etc + 1e-12).all()
    assert not f["guard_bound"].any()


def test_stress_coefficient_matches_fao56():
    soil = wb.SoilParams(taw=100, p=0.5)
    # no stress above the readily available water, linear to zero below it
    np.testing.assert_allclose(wb.stress_coefficient([100, 50, 25, 0], soil), [1, 1, 0.5, 0])


def test_fitted_synthetic_conserves():
    base = synthetic_iid(200, seed=1)
    clim = SyntheticClimate.fit(base)
    ws = clim.sample(40, seed=2, wetness=0.6)
    env = IrrigationEnv(ws)
    env.reset()
    while not env.done:
        env.step(np.full(ws.n, 1))
    assert np.abs(env.result().ledger_residual).max() < 1e-9
