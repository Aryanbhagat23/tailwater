import numpy as np
import pytest

from tailwater.agents.common import WATER_UNIT, season_rewards
from tailwater.agents.cvar_lagrangian import CVaRLagrangian
from tailwater.agents.nn import MLP, softmax
from tailwater.agents.ppo import PPOConfig, train_ppo
from tailwater.env.crop import CropParams
from tailwater.env.irrigation_env import IrrigationEnv, simulate
from tailwater.env.weather import synthetic_iid


def rollout_arrays(n=16, seed=0):
    ws = synthetic_iid(n, seed)
    env = IrrigationEnv(ws)
    env.reset()
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("irrigation", "etc", "eta")}
    while not env.done:
        _, info = env.step(rng.integers(0, 4, n))
        for k in cols:
            cols[k].append(info[k])
    y, _, _ = env.final_yields()
    return {k: np.stack(v, 1) for k, v in cols.items()}, y, env


def test_dense_rewards_sum_to_objective():
    c, y, env = rollout_arrays()
    w = np.where(y <= np.median(y), 7.0, 0.0)
    nu = float(np.median(y))
    R = season_rewards(c["irrigation"], c["etc"], c["eta"], y, env.stage, env.ky, 10.0, w, nu)
    expected = -c["irrigation"].sum(1) / WATER_UNIT + 10.0 * (y - 1) + w * (y - nu)
    np.testing.assert_allclose(R.sum(1), expected, atol=1e-9)


def test_linearised_loss_is_first_order_fao33():
    # one small deficit in a single stage: linear loss ~= 1 - Y
    c, y, env = rollout_arrays()
    from tailwater.agents.common import daily_loss, stage_etm
    l = daily_loss(c["etc"], c["eta"], env.ky[env.stage][None, :],
                   stage_etm(c["etc"], env.stage, 5)).sum(1)
    assert (l >= 1 - y - 1e-9).all()  # product of factors >= 1 - sum of losses
    mild = l < 0.05
    np.testing.assert_allclose(l[mild], 1 - y[mild], atol=2e-3)


def test_cvar_multiplier_rises_and_falls():
    lag = CVaRLagrangian(floor=0.9, alpha=0.1, lr=1.0)
    bad = np.r_[np.full(9, 1.0), 0.5]
    mask, nu, cvar = lag.tail(bad)
    assert cvar == 0.5 and mask.sum() == 1
    lag.update(cvar)
    assert lag.lam == pytest.approx(0.4)
    lag.update(1.0)
    assert lag.lam == pytest.approx(0.3)
    for _ in range(10):
        lag.update(1.0)
    assert lag.lam == 0.0
    assert CVaRLagrangian(floor=None).update(0.0) == 0.0


def test_mlp_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    net = MLP([5, 8, 3], rng, out_scale=1.0)
    x = rng.standard_normal((4, 5))
    t = rng.standard_normal((4, 3))
    out, hs = net.forward(x)
    grads = net.backward(hs, 2 * (out - t))
    loss = lambda: float(((net(x) - t) ** 2).sum())
    for p, g in zip(net.params, grads):
        i = np.unravel_index(np.argmax(np.abs(g)), g.shape)
        old = p[i]
        p[i] = old + 1e-6
        up = loss()
        p[i] = old - 1e-6
        down = loss()
        p[i] = old
        assert (up - down) / 2e-6 == pytest.approx(g[i], rel=1e-4, abs=1e-6)


def test_ppo_learns_to_irrigate_in_drought():
    from tailwater.env.weather import WeatherSet
    ws = WeatherSet(np.zeros((32, 120)), np.full((32, 120), 5.0), list(range(32)))
    pol, curve = train_ppo(ws, PPOConfig(iterations=60, batch_seasons=32, kappa=10, seed=0))
    res = simulate(pol, ws)
    assert res.yield_.mean() > 0.9  # ~0.42 after only 25 iterations
    assert res.water.mean() < 900
