"""Relative yield models.

``fao33_relative_yield`` — the multiplicative stage model (Rao, Sarma &
Chander 1988 form of the FAO-33 Ky relation, Doorenbos & Kassam 1979):

    Ya/Ym = prod_s [ 1 - Ky_s * (1 - ETa_s / ETm_s) ]

with each factor floored at 0. A deficit in one stage costs yield in
proportion to that stage's sensitivity Ky_s, so partial losses are graded.

``linear_stress_yield`` — the prototype's model, kept for comparison:
``yield = max(0, 1 - stress_days / 30)`` where a stress-day is ``1 - Ks``
accumulated daily. It concentrates all of the response near one stress
level, which is part of why the prototype showed a cliff (brief §3.2).
"""
from __future__ import annotations

import numpy as np


def fao33_relative_yield(eta_by_stage, etm_by_stage, ky):
    """Relative yield from per-stage actual and maximum ET.

    ``eta_by_stage`` / ``etm_by_stage``: arrays (..., n_stages), mm.
    ``ky``: sequence of n_stages sensitivity coefficients.
    """
    eta = np.asarray(eta_by_stage, dtype=float)
    etm = np.asarray(etm_by_stage, dtype=float)
    ratio = np.where(etm > 0, eta / np.where(etm > 0, etm, 1.0), 1.0)
    factors = np.maximum(0.0, 1.0 - np.asarray(ky) * (1.0 - ratio))
    return np.prod(factors, axis=-1)


def linear_stress_yield(stress_days, stress_days_to_zero: float = 30.0):
    return np.maximum(0.0, 1.0 - np.asarray(stress_days, dtype=float) / stress_days_to_zero)
