"""Baseline irrigation controllers. Every one is parameterised; nothing in a
comparison may use a hand-picked setting (brief §3.1). ``FAMILIES`` defines
the parameter grid that ``tuning.py`` sweeps for each family.

All controllers act in the same discrete action set as the agents
(0 / 5 / 10 / 20 mm), except the diagnostic ``ConstantRate``.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..env.irrigation_env import ACTIONS


def largest_action_at_most(mm):
    """Index of the largest action <= mm (0 if mm < 5)."""
    return np.searchsorted(ACTIONS, np.asarray(mm, dtype=float), side="right") - 1


def action_index(mm: float) -> int:
    idx = np.flatnonzero(ACTIONS == mm)
    if not len(idx):
        raise ValueError(f"{mm} mm is not in the action set {ACTIONS}")
    return int(idx[0])


class Policy:
    name = "policy"
    continuous = False

    def reset(self, n: int):
        self.n = n

    def act(self, obs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def label(self) -> str:
        p = self.params()
        return self.name + ("(" + ", ".join(f"{k}={v}" for k, v in p.items()) + ")" if p else "")


class RainFed(Policy):
    name = "rain-fed"

    def act(self, obs):
        return np.zeros(len(obs["moisture"]), dtype=int)


class FixedCalendar(Policy):
    """Apply ``amount`` mm every ``interval`` days, regardless of conditions."""
    name = "calendar"

    def __init__(self, interval: int, amount: float):
        self.interval, self.amount, self.a = interval, amount, action_index(amount)

    def act(self, obs):
        on = obs["day"] % self.interval == 0
        return np.full(len(obs["moisture"]), self.a if on else 0, dtype=int)

    def params(self):
        return {"interval": self.interval, "amount": self.amount}


class Threshold(Policy):
    """Apply ``amount`` mm whenever soil moisture is below ``trigger`` mm."""
    name = "threshold"

    def __init__(self, trigger: float, amount: float):
        self.trigger, self.amount, self.a = trigger, amount, action_index(amount)

    def act(self, obs):
        return np.where(obs["moisture"] < self.trigger, self.a, 0)

    def params(self):
        return {"trigger": self.trigger, "amount": self.amount}


class ThresholdForecast(Threshold):
    """Threshold rule that skips irrigation when the 3-day rain forecast is at
    least ``skip_mm``. Gives the baselines the same forecast the agent sees, so
    a learned advantage cannot come from information access alone."""
    name = "threshold+forecast"

    def __init__(self, trigger: float, amount: float, skip_mm: float):
        super().__init__(trigger, amount)
        self.skip_mm = skip_mm

    def act(self, obs):
        go = (obs["moisture"] < self.trigger) & (obs["forecast"] < self.skip_mm)
        return np.where(go, self.a, 0)

    def params(self):
        return {**super().params(), "skip_mm": self.skip_mm}


class StageThreshold(Policy):
    """Regulated deficit irrigation: a threshold rule with one trigger for the
    yield-sensitive stages (FAO-33 Ky >= 0.5: flowering, yield formation) and
    another for the tolerant ones. This is the strongest hand-designed rule
    here, because it encodes the same stage sensitivity the agent can learn."""
    name = "stage threshold"
    SENSITIVE = (2, 3)

    def __init__(self, trigger_tolerant: float, trigger_sensitive: float, amount: float):
        self.t_lo, self.t_hi = trigger_tolerant, trigger_sensitive
        self.amount, self.a = amount, action_index(amount)

    def act(self, obs):
        trig = self.t_hi if obs["stage"] in self.SENSITIVE else self.t_lo
        return np.where(obs["moisture"] < trig, self.a, 0)

    def params(self):
        return {"trigger_tolerant": self.t_lo, "trigger_sensitive": self.t_hi,
                "amount": self.amount}


class ETController(Policy):
    """Weather-based 'checkbook' controller, the logic of commercial ET
    controllers: it never sees soil moisture. It books the crop's estimated
    water use (Kc * ET0) minus measured rain as a running deficit, and when the
    deficit reaches ``trigger`` mm it replaces ``refill`` of it (rounded down to
    the action set, at least 5 mm)."""
    name = "ET controller"

    def __init__(self, trigger: float, refill: float = 1.0):
        self.trigger, self.refill = trigger, refill

    def reset(self, n):
        super().reset(n)
        self.deficit = np.zeros(n)

    def act(self, obs):
        if obs["day"] > 0:
            self.deficit = np.maximum(0.0, self.deficit + obs["etc_yesterday"] - obs["rain_yesterday"])
        go = self.deficit >= self.trigger
        a = np.where(go, np.maximum(largest_action_at_most(self.refill * self.deficit), 1), 0)
        self.deficit = np.maximum(0.0, self.deficit - ACTIONS[a])
        return a

    def params(self):
        return {"trigger": self.trigger, "refill": self.refill}


class ConstantRate(Policy):
    """Diagnostic only: a fixed mm/day, outside the action set."""
    name = "constant"
    continuous = True

    def __init__(self, rate: float):
        self.rate = rate

    def act(self, obs):
        return np.full(len(obs["moisture"]), self.rate)

    def params(self):
        return {"rate": self.rate}


def _grid(cls, **axes):
    keys = list(axes)
    return [cls(**dict(zip(keys, vals))) for vals in itertools.product(*axes.values())]


AMOUNTS = (5.0, 10.0, 20.0)
TRIGGERS = tuple(float(t) for t in range(10, 96, 5))

FAMILIES = {
    "rain-fed": lambda: [RainFed()],
    "calendar": lambda: _grid(FixedCalendar, interval=range(1, 11), amount=AMOUNTS),
    "threshold": lambda: _grid(Threshold, trigger=TRIGGERS, amount=AMOUNTS),
    "threshold+forecast": lambda: _grid(ThresholdForecast, trigger=TRIGGERS, amount=AMOUNTS,
                                        skip_mm=(5.0, 10.0, 20.0, 40.0)),
    "stage threshold": lambda: _grid(StageThreshold, trigger_tolerant=TRIGGERS,
                                     trigger_sensitive=TRIGGERS, amount=AMOUNTS),
    "ET controller": lambda: _grid(ETController, trigger=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                                                          40.0, 50.0, 60.0, 70.0, 80.0, 100.0),
                                   refill=(0.5, 0.75, 1.0)),
}
