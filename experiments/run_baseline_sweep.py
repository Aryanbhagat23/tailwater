"""Sweep and tune every baseline family (brief §4 step 1).

* ``insample``: every member of every family on all seasons (Pareto fronts
  for the figures; *not* used for any comparison).
* ``tuned``: for each yield floor, the family's least-water member meeting
  CVaR5 >= floor on the training years of each fold, scored on held-out years.
"""
from _common import save_json  # noqa: I001
import numpy as np

from tailwater.data import belle_glade_seasons
from tailwater.eval.protocol import eval_forecast, make_folds, tune_and_test
from tailwater.policies.baselines import FAMILIES
from tailwater.policies.tuning import pareto, sweep

FLOORS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]
KEEP = ("water", "mean_yield", "p5_yield", "cvar5_yield", "worst_yield")


def main():
    ws = belle_glade_seasons()
    folds = make_folds(ws.n)
    fc = eval_forecast(ws)
    out = {"floors": FLOORS, "insample": {}, "tuned": {}}
    for fam in FAMILIES:
        rows = sweep(fam, ws, fc)
        front = {r["label"] for r in pareto(rows)}
        out["insample"][fam] = [{"label": r["label"], "pareto": r["label"] in front,
                                 **{k: r[k] for k in KEEP}} for r in rows]
        tuned = tune_and_test(ws, fam, FLOORS if fam != "rain-fed" else [0.0], folds)
        out["tuned"][fam] = {str(f): {"summary": v["summary"], "chosen": v["chosen"],
                                      "water": v["water"], "yield": v["yield"]}
                             for f, v in tuned.items()}
        print(f"== {fam}")
        for f, v in tuned.items():
            s = v["summary"]
            print(f"  floor {f:.2f}: water {s['water']:6.1f} meanY {s['mean_yield']:.3f} "
                  f"CVaR5 {s['cvar5_yield']:.3f} worst {s['worst_yield']:.3f}  "
                  f"chosen {v['chosen'][0]}")
    save_json(out, "baselines.json")


if __name__ == "__main__":
    main()
