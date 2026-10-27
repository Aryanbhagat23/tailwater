"""Weather for the irrigation simulator.

Every source produces a :class:`WeatherSet`: arrays of daily rain and
reference evapotranspiration (ET0) for N seasons of T days, plus labels.
Policies and agents never see the source, only the arrays, so synthetic and
historical weather are interchangeable.

Sources
-------
``load_ghcn_seasons``   NOAA GHCN-Daily station file (PRCP, TMAX, TMIN) cut
                        into one season per calendar year. Days are never
                        reshuffled across years: inter-annual variability
                        (wet years, drought years) is the point.
``synthetic_iid``       The prototype's generator: rain and ET0 drawn i.i.d.
                        per day. Season totals concentrate (no drought years).
``SyntheticClimate``    A stochastic generator fitted to a WeatherSet, with a
                        season-level wetness/ET anomaly so that it reproduces
                        inter-annual variability. Used for fast iteration and
                        for the climate-wetness robustness sweep.

ET0 from temperature uses Hargreaves & Samani (1985) with extraterrestrial
radiation from FAO-56 (Allen et al. 1998) eqs. 21-25.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GSC = 0.0820  # solar constant, MJ m-2 min-1


# --------------------------------------------------------------------------
# Reference evapotranspiration
# --------------------------------------------------------------------------

def extraterrestrial_radiation(lat_deg, doy):
    """Ra, MJ m-2 day-1 (FAO-56 eq. 21). Also returns the intermediates."""
    phi = np.deg2rad(lat_deg)
    j = np.asarray(doy, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)            # eq. 23
    delta = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)         # eq. 24
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))   # eq. 25
    ra = (24.0 * 60.0 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )
    return ra, {"dr": dr, "delta": delta, "ws": ws}


def hargreaves_et0(tmax, tmin, lat_deg, doy):
    """Hargreaves & Samani (1985) reference ET, mm/day.

    ET0 = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * 0.408 * Ra
    (0.408 converts MJ m-2 day-1 to mm/day of evaporated water).
    """
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    ra, _ = extraterrestrial_radiation(lat_deg, doy)
    tmean = 0.5 * (tmax + tmin)
    trange = np.maximum(tmax - tmin, 0.0)
    return np.maximum(0.0, 0.0023 * (tmean + 17.8) * np.sqrt(trange) * 0.408 * ra)


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

@dataclass
class WeatherSet:
    rain: np.ndarray            # (N, T) mm/day
    et0: np.ndarray             # (N, T) mm/day
    labels: list                # length N (e.g. calendar years)
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rain = np.asarray(self.rain, dtype=float)
        self.et0 = np.asarray(self.et0, dtype=float)
        assert self.rain.shape == self.et0.shape and self.rain.ndim == 2
        assert len(self.labels) == self.rain.shape[0]

    @property
    def n(self) -> int:
        return self.rain.shape[0]

    @property
    def length(self) -> int:
        return self.rain.shape[1]

    def subset(self, idx) -> "WeatherSet":
        idx = np.asarray(idx)
        return WeatherSet(self.rain[idx], self.et0[idx], [self.labels[i] for i in idx],
                          self.source, dict(self.meta))

    def forecasts(self, noise: float, seed: int, horizon: int = 3) -> np.ndarray:
        """Rain forecast for days d .. d+horizon-1, issued on the morning of day d.

        Multiplicative log-normal error with unit mean, plus false alarms on
        dry spells; ``noise=0`` is a perfect forecast. The draw is seeded so
        every policy evaluated with the same seed sees identical forecasts.
        """
        n, t = self.rain.shape
        c = np.concatenate([np.zeros((n, 1)), np.cumsum(self.rain, axis=1)], axis=1)
        end = np.minimum(np.arange(t) + horizon, t)
        truth = c[:, end] - c[:, :t]
        if noise <= 0:
            return truth
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, t))
        fc = truth * np.exp(noise * z - 0.5 * noise ** 2)
        false_alarm = rng.random((n, t)) < 0.15 * min(noise, 1.0)
        fc = fc + false_alarm * rng.exponential(5.0, (n, t))
        return fc

    def season_totals(self) -> dict:
        return {"rain": self.rain.sum(1), "et0": self.et0.sum(1)}


# --------------------------------------------------------------------------
# NOAA GHCN-Daily
# --------------------------------------------------------------------------

def load_ghcn_daily(path) -> "pd.DataFrame":
    """Read a GHCN-Daily by-station CSV (AWS/NCEI 'by_station' format).

    Returns a daily-indexed frame with columns prcp (mm), tmax, tmin (degC)
    and evap (mm, if present). Values with a quality flag are dropped.
    """
    import pandas as pd

    path = Path(path)
    cols = ["ID", "DATE", "ELEMENT", "DATA_VALUE", "M_FLAG", "Q_FLAG", "S_FLAG", "OBS_TIME"]
    df = pd.read_csv(path, dtype={"DATE": str, "Q_FLAG": str, "M_FLAG": str}, low_memory=False)
    if list(df.columns[:4]) != cols[:4]:  # headerless NCEI variant
        df = pd.read_csv(path, names=cols, header=None, dtype={"DATE": str, "Q_FLAG": str})
    df = df[df["ELEMENT"].isin(["PRCP", "TMAX", "TMIN", "EVAP"]) & df["Q_FLAG"].isna()]
    wide = df.pivot_table(index="DATE", columns="ELEMENT", values="DATA_VALUE", aggfunc="first")
    wide.index = pd.to_datetime(wide.index, format="%Y%m%d")
    wide = wide.asfreq("D") / 10.0  # tenths of mm / tenths of degC
    wide.columns = [c.lower() for c in wide.columns]
    return wide


def load_ghcn_seasons(path, lat_deg: float, start_mmdd: str = "03-01", length: int = 120,
                      max_missing: int = 6, station: str = "") -> WeatherSet:
    """One season per calendar year starting on ``start_mmdd``.

    A year is kept only if each of PRCP, TMAX, TMIN has at most
    ``max_missing`` missing days in the window. Missing PRCP is filled with 0
    (counted in ``meta``); missing temperatures are linearly interpolated.
    """
    import pandas as pd

    w = load_ghcn_daily(path)
    rains, et0s, years, filled = [], [], [], []
    for year in range(w.index.year.min(), w.index.year.max() + 1):
        start = pd.Timestamp(f"{year}-{start_mmdd}")
        seg = w.loc[start: start + pd.Timedelta(days=length - 1)]
        if len(seg) < length or not {"prcp", "tmax", "tmin"} <= set(seg.columns):
            continue
        miss = seg[["prcp", "tmax", "tmin"]].isna().sum()
        if miss.max() > max_missing:
            continue
        tx = seg["tmax"].interpolate(limit_direction="both").to_numpy()
        tn = seg["tmin"].interpolate(limit_direction="both").to_numpy()
        lo, hi = np.minimum(tx, tn), np.maximum(tx, tn)  # guard against swapped pairs
        doy = seg.index.dayofyear.to_numpy()
        rains.append(seg["prcp"].fillna(0.0).to_numpy())
        et0s.append(hargreaves_et0(hi, lo, lat_deg, doy))
        years.append(year)
        filled.append(int(miss["prcp"]))
    return WeatherSet(np.array(rains), np.array(et0s), years, source=f"GHCN {station}".strip(),
                      meta={"lat": lat_deg, "start": start_mmdd, "prcp_days_filled": filled})


# --------------------------------------------------------------------------
# Synthetic generators
# --------------------------------------------------------------------------

def synthetic_iid(n: int, seed: int, length: int = 120, p_wet: float = 0.2,
                  mean_wet_mm: float = 5.6, et0_mean: float = 6.0, et0_sd: float = 0.9) -> WeatherSet:
    """The prototype's i.i.d. generator (kept to reproduce brief §3.2)."""
    rng = np.random.default_rng(seed)
    wet = rng.random((n, length)) < p_wet
    rain = wet * rng.exponential(mean_wet_mm, (n, length))
    et0 = np.clip(rng.normal(et0_mean, et0_sd, (n, length)), 0.5, None)
    return WeatherSet(rain, et0, [f"iid-{seed}-{i}" for i in range(n)], source="synthetic-iid")


def _smooth(x, window=21):
    k = np.ones(window) / window
    pad = window // 2
    xp = np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])]
    return np.convolve(xp, k, mode="valid")


@dataclass
class SyntheticClimate:
    """Stochastic weather fitted to observed seasons.

    Daily structure: day-of-season wet-day probability and mean wet-day
    amount (gamma distributed), ET0 climatology with AR(1) daily noise and a
    reduction on wet days. Season structure: a log-normal wetness factor that
    scales rain occurrence and amount, and a season ET0 anomaly — these are
    what produce drought and wet years. ``wetness`` shifts the whole climate
    (1.0 = fitted climate, 0.7 = 30% drier) for the robustness sweep.
    """
    p_wet: np.ndarray
    mean_wet: np.ndarray
    gamma_shape: float
    et0_clim: np.ndarray
    et0_daily_sd: float
    et0_ar: float
    wet_day_et0_factor: float
    sigma_wet: float
    sigma_et0: float

    @classmethod
    def fit(cls, ws: WeatherSet, wet_threshold: float = 0.3) -> "SyntheticClimate":
        rain, et0 = ws.rain, ws.et0
        wet = rain >= wet_threshold
        p = _smooth(wet.mean(0))
        amt = np.where(wet, rain, np.nan)
        m = _smooth(np.nan_to_num(np.nanmean(amt, 0), nan=np.nanmean(amt)))
        a = amt[~np.isnan(amt)]
        shape = float(a.mean() ** 2 / a.var())
        clim = _smooth(et0.mean(0))
        resid = et0 - clim
        season_anom = resid.mean(1, keepdims=True)
        daily = resid - season_anom
        ar = float(np.corrcoef(daily[:, :-1].ravel(), daily[:, 1:].ravel())[0, 1])
        dry_mean = et0[~wet].mean()
        wet_factor = float(et0[wet].mean() / dry_mean) if wet.any() else 1.0
        obj = cls(p, m, shape, clim, float(daily.std()), ar, wet_factor, 0.0,
                  float(season_anom.std() / clim.mean()))
        # season wetness: add log-variance beyond what i.i.d. daily draws give
        obs_var = np.var(np.log(rain.sum(1) + 1.0))
        iid_var = np.var(np.log(obj.sample(2000, seed=12345).rain.sum(1) + 1.0))
        obj.sigma_wet = float(np.sqrt(max(obs_var - iid_var, 0.0)))
        return obj

    def sample(self, n: int, seed: int, wetness: float = 1.0) -> WeatherSet:
        rng = np.random.default_rng(seed)
        t = len(self.p_wet)
        w = np.exp(self.sigma_wet * rng.standard_normal((n, 1)) - 0.5 * self.sigma_wet ** 2) * wetness
        # split the wetness factor between occurrence and intensity
        p = np.clip(self.p_wet[None, :] * np.sqrt(w), 0.0, 0.95)
        wet = rng.random((n, t)) < p
        amt = rng.gamma(self.gamma_shape, (self.mean_wet * np.sqrt(w)) / self.gamma_shape, (n, t))
        rain = wet * amt
        e = np.zeros((n, t))
        z = rng.standard_normal((n, t)) * self.et0_daily_sd * np.sqrt(1 - self.et0_ar ** 2)
        e[:, 0] = z[:, 0] / np.sqrt(1 - self.et0_ar ** 2)
        for d in range(1, t):
            e[:, d] = self.et0_ar * e[:, d - 1] + z[:, d]
        anom = 1.0 + self.sigma_et0 * rng.standard_normal((n, 1))
        # drier climates are also sunnier: scale ET0 mildly against wetness
        et0 = (self.et0_clim[None, :] * anom + e) * np.where(wet, self.wet_day_et0_factor, 1.0)
        et0 *= wetness ** -0.1
        return WeatherSet(rain, np.clip(et0, 0.3, None),
                          [f"syn-{seed}-{i}" for i in range(n)], source="synthetic-fitted",
                          meta={"wetness": wetness})
