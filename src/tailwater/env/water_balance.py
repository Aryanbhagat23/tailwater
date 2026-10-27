"""Daily root-zone soil water balance (single bucket, FAO-56 style).

State variable ``S`` is plant-available water in the root zone, in mm,
bounded by ``[0, TAW]`` where TAW is total available water (field capacity
minus wilting point over the rooting depth).

One day, in order:

1. Inputs:     S1 = S + rain + irrigation
2. Drainage:   D  = max(0, S1 - TAW)      (deep percolation + runoff, lumped)
3. Uptake:     Ks = clip(S2 / ((1 - p) * TAW), 0, 1)     (FAO-56 eq. 84)
               ETc = Kc * ET0,  ETa = Ks * ETc
4. Update:     S' = S1 - D - ETa

so  S' = S + rain + irrigation - ETa - D  holds exactly; every mm is
accounted for. Because Ks -> 0 as S -> 0, ETa can never exceed the water in
the bucket for any physically plausible ETc (see ``max_safe_etc``); the
``np.minimum`` guard below exists only so that an absurd input cannot create
negative water, and ``step`` reports how often it bound.

All functions are vectorised: every argument may be a scalar or an array
over a batch of fields/seasons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SoilParams:
    taw: float = 100.0          # total available water in the root zone, mm
    p: float = 0.5              # depletion fraction before stress (FAO-56, maize ~0.55)
    initial_fraction: float = 1.0  # season starts at field capacity (pre-plant irrigation)

    @property
    def stress_point(self) -> float:
        """Moisture below which Ks < 1, mm."""
        return (1.0 - self.p) * self.taw

    def max_safe_etc(self) -> float:
        """ETc (mm/day) above which the uptake guard could bind."""
        return (1.0 - self.p) * self.taw


def stress_coefficient(moisture, soil: SoilParams):
    """FAO-56 water stress coefficient Ks in [0, 1]."""
    return np.clip(np.asarray(moisture, dtype=float) / soil.stress_point, 0.0, 1.0)


def step(moisture, rain, irrigation, etc, soil: SoilParams):
    """Advance the water balance by one day.

    Returns ``(new_moisture, fluxes)`` where ``fluxes`` is a dict with keys
    ``drainage``, ``eta``, ``ks``, ``guard_bound`` (bool array, True where the
    non-negativity guard limited uptake).
    """
    s1 = np.asarray(moisture, dtype=float) + rain + irrigation
    drainage = np.maximum(0.0, s1 - soil.taw)
    s2 = s1 - drainage
    ks = stress_coefficient(s2, soil)
    eta_demand = ks * etc
    eta = np.minimum(eta_demand, s2)
    new = s2 - eta
    return new, {
        "drainage": drainage,
        "eta": eta,
        "ks": ks,
        "guard_bound": eta < eta_demand,
    }
