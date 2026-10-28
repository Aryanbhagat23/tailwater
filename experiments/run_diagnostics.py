"""Environment diagnostics (brief §3.2-3.4, steps 2-3).

1. Inter-annual variability: synthetic i.i.d. vs NOAA Belle Glade vs fitted synthetic.
2. Constant-rate sweeps: cliff (i.i.d. weather + linear yield) vs frontier
   (real weather + FAO-33 yield), with the width of the 'interesting region'.
3. Hargreaves ET0 vs the station's own Class-A pan evaporation.
4. Per-year season table (rain, ET0, rain-fed yield).
"""
from _common import RESULTS, save_json  # noqa: I001
import numpy as np

from tailwater.data import BELLE_GLADE, RAW, belle_glade_seasons
from tailwater.env.irrigation_env import simulate
from tailwater.env.weather import (SyntheticClimate, hargreaves_et0, load_ghcn_daily,
                                   synthetic_iid)
from tailwater.eval.metrics import summarize
from tailwater.eval.protocol import eval_forecast
from tailwater.policies.baselines import ConstantRate, RainFed

RATES = np.round(np.arange(0.0, 7.01, 0.25), 2)


def variability(ws):
    t = ws.season_totals()
    out = {}
    for k, v in t.items():
        out[k] = {"mean": v.mean(), "sd": v.std(), "cv": v.std() / v.mean(),
                  "p5": np.percentile(v, 5), "p95": np.percentile(v, 95)}
    return out


def rate_sweep(ws, yield_model):
    fc = ws.forecasts(0.0, 0)
    rows = []
    for r in RATES:
        res = simulate(ConstantRate(float(r)), ws, fc, yield_model=yield_model)
        rows.append({"rate": float(r), **summarize(res.water, res.yield_)})
    return rows


def region_width(rows, metric="cvar5_yield", lo=0.1, hi=0.9):
    """Water span over which ``metric`` climbs from ``lo`` to ``hi``."""
    w = np.array([r["water"] for r in rows])
    m = np.maximum.accumulate(np.array([r[metric] for r in rows]))
    if m.max() < hi or m.min() > lo:
        return None
    w_lo = np.interp(lo, m + np.arange(len(m)) * 1e-9, w)
    w_hi = np.interp(hi, m + np.arange(len(m)) * 1e-9, w)
    return float(w_hi - w_lo)


def pan_check():
    d = load_ghcn_daily(RAW / f"{BELLE_GLADE['station']}.csv")
    d = d.dropna(subset=["tmax", "tmin", "evap"])
    d = d[(d["evap"] > 0) & (d["evap"] < 20)]
    et0 = hargreaves_et0(d["tmax"].clip(lower=d["tmin"]), d["tmin"], BELLE_GLADE["lat"],
                         d.index.dayofyear)
    d = d.assign(et0=et0)
    monthly = d.groupby(d.index.month)[["et0", "evap"]].mean()
    return {
        "days": int(len(d)),
        "years": [int(d.index.year.min()), int(d.index.year.max())],
        "monthly": [{"month": int(m), "et0": r.et0, "pan": r.evap, "ratio": r.et0 / r.evap}
                    for m, r in monthly.iterrows()],
        "overall_ratio": float(d["et0"].sum() / d["evap"].sum()),
        "annual_et0_mm": float(d["et0"].mean() * 365.25),
    }


def main():
    noaa = belle_glade_seasons()
    iid = synthetic_iid(1000, seed=0)
    fitted = SyntheticClimate.fit(noaa).sample(1000, seed=0)
    out = {
        "variability": {"synthetic_iid": variability(iid), "noaa": variability(noaa),
                        "synthetic_fitted": variability(fitted)},
        "sweeps": {
            "iid_linear": rate_sweep(iid, "linear"),
            "noaa_linear": rate_sweep(noaa, "linear"),
            "noaa_fao33": rate_sweep(noaa, "fao33"),
        },
    }
    out["region_width_mm"] = {k: region_width(v) for k, v in out["sweeps"].items()}
    # how far the bad tail sits below the average season at the same water
    out["max_tail_gap"] = {k: max(r["mean_yield"] - r["cvar5_yield"] for r in v)
                           for k, v in out["sweeps"].items()}
    out["pan_check"] = pan_check()
    rf = simulate(RainFed(), noaa, eval_forecast(noaa))
    out["seasons"] = [{"year": y, "rain": float(noaa.rain[i].sum()), "et0": float(noaa.et0[i].sum()),
                       "rainfed_yield": float(rf.yield_[i])} for i, y in enumerate(noaa.labels)]
    out["n_seasons"] = noaa.n
    out["prcp_days_filled"] = int(sum(noaa.meta["prcp_days_filled"]))
    save_json(out, "diagnostics.json")
    for k, v in out["variability"].items():
        print(f"{k:18s} rain CV {v['rain']['cv']:.3f}  ET0 CV {v['et0']['cv']:.3f}  "
              f"rain p5-p95 {v['rain']['p5']:.0f}-{v['rain']['p95']:.0f}")
    print("interesting-region width (CVaR5 0.1 -> 0.9), mm:", out["region_width_mm"])
    print("max (mean - CVaR5) yield gap on the constant-rate sweep:", out["max_tail_gap"])
    pc = out["pan_check"]
    print(f"Hargreaves/pan ratio {pc['overall_ratio']:.2f} over {pc['days']} days; "
          f"annual Hargreaves ET0 {pc['annual_et0_mm']:.0f} mm")


if __name__ == "__main__":
    main()
