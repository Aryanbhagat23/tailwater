import numpy as np

from tailwater.env.irrigation_env import ACTIONS, simulate
from tailwater.env.weather import WeatherSet, synthetic_iid
from tailwater.eval.metrics import cvar_lower, summarize
from tailwater.eval.protocol import make_folds
from tailwater.policies.baselines import (FAMILIES, ETController, FixedCalendar, RainFed,
                                          StageThreshold, Threshold, ThresholdForecast)
from tailwater.policies.tuning import pareto, select, sweep


def dry_weather(n=3, et0=6.0):
    return WeatherSet(np.zeros((n, 120)), np.full((n, 120), et0), list(range(n)))


def test_rainfed_uses_no_water_and_fails_in_drought():
    res = simulate(RainFed(), dry_weather())
    assert res.water.sum() == 0
    assert (res.yield_ < 0.2).all()


def test_calendar_applies_exact_schedule():
    res = simulate(FixedCalendar(4, 10.0), dry_weather())
    np.testing.assert_allclose(res.water, 10.0 * 30)


def test_threshold_keeps_moisture_near_trigger():
    res = simulate(Threshold(60, 20.0), dry_weather(), record=True)
    m = res.traj["moisture"]
    assert m.min() > 60 - 2 * 6.0 * 1.2  # never far below trigger
    assert (res.yield_ > 0.999).all()


def test_threshold_forecast_skips_when_rain_expected():
    pol = ThresholdForecast(80, 10.0, skip_mm=5.0)
    pol.reset(2)
    obs = {"moisture": np.array([10.0, 10.0]), "forecast": np.array([0.0, 30.0]), "day": 3}
    assert list(pol.act(obs)) == [2, 0]  # index 2 = 10 mm


def test_stage_threshold_switches_trigger():
    pol = StageThreshold(20, 70, 10.0)
    pol.reset(1)
    m = np.array([50.0])
    assert pol.act({"moisture": m, "stage": 0})[0] == 0
    assert pol.act({"moisture": m, "stage": 2})[0] == 2


def test_et_controller_books_deficit_and_never_sees_moisture():
    pol = ETController(trigger=20, refill=1.0)
    pol.reset(1)
    acts = []
    for d in range(6):
        obs = {"day": d, "etc_yesterday": np.array([6.0]), "rain_yesterday": np.array([0.0]),
               "moisture": np.array([np.nan])}
        acts.append(int(pol.act(obs)[0]))
    # deficit 6,12,18,24 -> irrigate 20 on day 4 (largest action <= 24), carry 4
    assert acts == [0, 0, 0, 0, 3, 0]
    assert pol.deficit[0] == 10.0  # 4 carried + 6 booked on day 5
    assert ACTIONS[3] == 20.0


def test_every_family_is_a_grid():
    for fam, make in FAMILIES.items():
        members = make()
        assert len(members) >= 1
        if fam != "rain-fed":
            assert len({m.label() for m in members}) == len(members) > 5


def test_select_picks_least_water_meeting_floor():
    ws = synthetic_iid(40, seed=5)
    rows = sweep("threshold", ws, ws.forecasts(0.5, 1))
    best = select(rows, "mean_yield", 0.95)
    assert best["mean_yield"] >= 0.95
    assert all(r["water"] >= best["water"] for r in rows if r["mean_yield"] >= 0.95)
    front = pareto(rows, "mean_yield")
    assert all(a["water"] <= b["water"] and a["mean_yield"] < b["mean_yield"]
               for a, b in zip(front, front[1:]))


def test_identical_weather_is_identical():
    ws = synthetic_iid(10, seed=2)
    a = simulate(Threshold(50, 10.0), ws, ws.forecasts(0.5, 9))
    b = simulate(Threshold(50, 10.0), ws, ws.forecasts(0.5, 9))
    np.testing.assert_array_equal(a.water, b.water)


def test_metrics_and_folds():
    y = np.arange(1, 101) / 100.0
    assert cvar_lower(y, 0.05) == np.mean(y[:5])
    s = summarize(np.ones(100), y)
    assert s["worst_yield"] == 0.01 and s["n"] == 100
    folds = make_folds(74, 4)
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests) == list(range(74))  # every year held out exactly once
    for tr, te in folds:
        assert not set(tr) & set(te)
