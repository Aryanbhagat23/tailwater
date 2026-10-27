"""PPO (clipped surrogate, GAE) over the continuous state, optionally with
the CVaR Lagrangian.

Each iteration rolls out a batch of whole seasons (sampled with replacement
from the training weather, fresh forecast noise each time), then builds the
rewards *after* the rollout, when every season's yield is known:

    r_t     = -I_t/100 - (kappa + w) * l_t                (every day)
    r_T    += kappa*(Y - 1 + L) + w*(Y - nu + L)           (harvest correction)

where l_t is the day's linearised yield loss (``common.py``), L its season sum,
and w = lambda/alpha for seasons in the batch's yield tail (0 otherwise). The
season's rewards then sum to exactly

    -W/100 + kappa*(Y - 1) + (lambda/alpha) * min(Y - nu, 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env.irrigation_env import ACTIONS, IrrigationEnv
from ..policies.baselines import Policy
from .common import N_FEATURES, WATER_UNIT, features, season_rewards
from .cvar_lagrangian import CVaRLagrangian
from .nn import MLP, Adam, softmax


@dataclass
class PPOConfig:
    kappa: float = 10.0
    floor: float | None = None
    alpha: float = 0.05
    iterations: int = 300
    batch_seasons: int = 128
    hidden: int = 64
    lr: float = 3e-4
    vf_lr: float = 1e-3
    epochs: int = 4
    minibatch: int = 2048
    clip: float = 0.2
    gae_lambda: float = 0.95
    entropy: float = 0.01
    forecast_noise: float = 0.5
    lam_lr: float = 2.0
    lam_max: float = 20.0
    seed: int = 0


class PPOPolicy(Policy):
    name = "PPO"

    def __init__(self, net: MLP, label: str = "PPO", greedy: bool = True, seed: int = 0):
        self.net, self._label, self.greedy = net, label, greedy
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        logits = self.net(features(obs))
        if self.greedy:
            return np.argmax(logits, 1)
        p = softmax(logits)
        return (p.cumsum(1) > self.rng.random((len(p), 1))).argmax(1)

    def label(self):
        return self._label

    def probs(self, obs):
        return softmax(self.net(features(obs)))


def train_ppo(weather, cfg: PPOConfig = PPOConfig(), log_every: int = 0):
    rng = np.random.default_rng(cfg.seed)
    env = IrrigationEnv(weather)
    T, B, nA = env.crop.season_length, cfg.batch_seasons, len(ACTIONS)
    pi = MLP([N_FEATURES, cfg.hidden, cfg.hidden, nA], rng, out_scale=0.01)
    vf = MLP([N_FEATURES, cfg.hidden, cfg.hidden, 1], rng, out_scale=1.0)
    opt_pi, opt_vf = Adam(pi.params, cfg.lr), Adam(vf.params, cfg.vf_lr)
    lag = CVaRLagrangian(cfg.floor, cfg.alpha, cfg.lam_lr, 0.0, cfg.lam_max)
    ret_mean, ret_std = 0.0, 1.0
    stage_day = env.stage
    curve = []

    for it in range(cfg.iterations):
        idx = rng.integers(0, weather.n, B)
        obs = env.reset(idx, weather.forecasts(cfg.forecast_noise, int(rng.integers(1 << 31))))
        X = np.zeros((T, B, N_FEATURES))
        A = np.zeros((T, B), dtype=int)
        LP = np.zeros((T, B))
        V = np.zeros((T + 1, B))
        irr, etc, eta = np.zeros((B, T)), np.zeros((B, T)), np.zeros((B, T))
        for t in range(T):
            x = features(obs)
            p = softmax(pi(x))
            a = (p.cumsum(1) > rng.random((B, 1))).argmax(1)
            X[t], A[t], LP[t] = x, a, np.log(p[np.arange(B), a] + 1e-12)
            V[t] = vf(x)[:, 0] * ret_std + ret_mean
            obs, info = env.step(a)
            irr[:, t], etc[:, t], eta[:, t] = info["irrigation"], info["etc"], info["eta"]
        Y, _, _ = env.final_yields()

        # rewards, built once yields are known
        w, nu, cvar_b, tail = lag.episode_weights(Y)
        R = season_rewards(irr, etc, eta, Y, stage_day, env.ky, cfg.kappa, w, nu)
        R = R.T  # (T, B)

        adv = np.zeros((T, B))
        last = np.zeros(B)
        for t in reversed(range(T)):
            delta = R[t] + V[t + 1] - V[t]
            last = delta + cfg.gae_lambda * last
            adv[t] = last
        ret = adv + V[:T]
        ret_mean = 0.9 * ret_mean + 0.1 * float(ret.mean()) if it else float(ret.mean())
        ret_std = 0.9 * ret_std + 0.1 * float(ret.std() + 1e-6) if it else float(ret.std() + 1e-6)

        x_f, a_f, lp_f = X.reshape(-1, N_FEATURES), A.ravel(), LP.ravel()
        adv_f = adv.ravel()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        ret_f = (ret.ravel() - ret_mean) / ret_std
        n = len(a_f)
        ent_coef = cfg.entropy * max(0.0, 1.0 - it / cfg.iterations)
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                m = len(mb)
                logits, hs = pi.forward(x_f[mb])
                p = softmax(logits)
                logp_all = np.log(p + 1e-12)
                lp_new = logp_all[np.arange(m), a_f[mb]]
                ratio = np.exp(lp_new - lp_f[mb])
                ad = adv_f[mb]
                unclipped = ratio * ad
                clipped = np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * ad
                active = unclipped <= clipped  # gradient flows through ratio here
                # d(-surrogate)/d logp_new  = -ratio*ad where active
                g_lp = np.where(active, -ratio * ad, 0.0) / m
                onehot = np.zeros_like(p)
                onehot[np.arange(m), a_f[mb]] = 1.0
                g_logits = g_lp[:, None] * (onehot - p)
                # entropy bonus: maximise H = -sum p log p ; dH/dz = -p*(log p + H)
                H = -(p * logp_all).sum(1, keepdims=True)
                g_logits -= ent_coef * (-p * (logp_all + H)) / m
                opt_pi.step(pi.backward(hs, g_logits))

                v, vh = vf.forward(x_f[mb])
                g_v = 2.0 * (v[:, 0] - ret_f[mb])[:, None] / m
                opt_vf.step(vf.backward(vh, g_v))

        lam = lag.update(cvar_b)
        curve.append({"iteration": it + 1, "seasons": (it + 1) * B,
                      "water": float(irr.sum(1).mean()), "mean_yield": float(Y.mean()),
                      "cvar5_yield": cvar_b, "lambda": lam,
                      "objective": float((-irr.sum(1) / WATER_UNIT + cfg.kappa * (Y - 1)).mean())})
        if log_every and (it + 1) % log_every == 0:
            c = curve[-1]
            print(f"it {it+1:4d} water {c['water']:6.1f} meanY {c['mean_yield']:.3f} "
                  f"cvarY {c['cvar5_yield']:.3f} lambda {lam:6.2f}", flush=True)

    tag = f"PPO(kappa={cfg.kappa}" + (f", CVaR>={cfg.floor})" if cfg.floor is not None else ")")
    return PPOPolicy(pi, tag), curve
