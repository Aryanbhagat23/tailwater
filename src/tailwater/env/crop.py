"""Crop calendar for a 120-day spring maize season.

Stage lengths follow FAO-33 (Doorenbos & Kassam 1979, maize: establishment
15-25 d, vegetative 25-40 d, flowering 15-20 d, yield formation 35-45 d,
ripening 10-15 d), scaled to a 120-day season. Crop coefficients follow
FAO-56 Table 12 for grain maize (Kc_ini 0.30, Kc_mid 1.20, Kc_end 0.60) with
the FAO-56 four-segment curve (initial / development / mid / late).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STAGE_NAMES = ("establishment", "vegetative", "flowering", "yield formation", "ripening")


@dataclass(frozen=True)
class CropParams:
    season_length: int = 120
    # first day (0-based) of each FAO-33 stage; last stage runs to season end
    stage_starts: tuple = (0, 20, 55, 73, 108)
    # FAO-33 Table 24, maize: vegetative 0.4, flowering 1.5, yield formation 0.5,
    # ripening 0.2. FAO-33 lists no separate establishment value; the
    # vegetative value is used for it.
    ky: tuple = (0.4, 0.4, 1.5, 0.5, 0.2)
    # FAO-56 Kc curve breakpoints (day, Kc)
    kc_points: tuple = ((0, 0.30), (20, 0.30), (55, 1.20), (95, 1.20), (120, 0.60))

    def stage_of_day(self, day):
        return np.searchsorted(np.asarray(self.stage_starts), np.asarray(day), side="right") - 1

    def kc(self, day):
        d, k = zip(*self.kc_points)
        return np.interp(np.asarray(day, dtype=float), d, k)

    @property
    def n_stages(self) -> int:
        return len(self.stage_starts)

    def stage_lengths(self) -> np.ndarray:
        ends = list(self.stage_starts[1:]) + [self.season_length]
        return np.array(ends) - np.array(self.stage_starts)
