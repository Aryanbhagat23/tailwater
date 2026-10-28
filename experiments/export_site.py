"""Build the static demo site: docs/index.html (GitHub Pages) from
site/template.html plus the data the page needs, inlined as JSON.

The page re-runs the simulator in JavaScript (site/sim.js mirrors
src/tailwater/env and the baselines/PPO forward pass). ``reference`` holds
Python results for every demo policy on every season so that
``node site/check_sim.js`` can verify the port.
"""
from _common import RESULTS, ROOT, load_models  # noqa: I001
import json

import numpy as np

from tailwater.agents.nn import MLP
from tailwater.agents.ppo import PPOPolicy
from tailwater.data import BELLE_GLADE, belle_glade_seasons
from tailwater.env.crop import STAGE_NAMES, CropParams
from tailwater.env.irrigation_env import ACTIONS, simulate
from tailwater.env.water_balance import SoilParams
from tailwater.eval.protocol import (FORECAST_NOISE, TUNE_FORECAST_SEED, eval_forecast,
                                     make_folds)
from tailwater.policies.baselines import RainFed
from tailwater.policies.tuning import select, sweep

DEMO_FLOOR = 0.90
DEMO_AGENT = "cvar_f90"
MAP_AGENTS = ["cvar_f90", "cvar_f95", "ppo_k7", "ppo_k10"]
DEMO_BASELINES = ["ET controller", "stage threshold", "threshold+forecast"]


def r(x, d=2):
    return np.round(np.asarray(x, dtype=float), d).tolist()


def mlp_state(st, d=7):
    return {"W": [np.round(np.array(w), d).tolist() for w in st["W"]],
            "b": [np.round(np.array(b), d).tolist() for b in st["b"]]}


def main():
    ws = belle_glade_seasons()
    folds = make_folds(ws.n)
    fold_of = np.zeros(ws.n, dtype=int)
    for j, (_, te) in enumerate(folds):
        fold_of[te] = j
    fc = eval_forecast(ws)
    fc_tune = ws.forecasts(FORECAST_NOISE, TUNE_FORECAST_SEED)
    crop, soil = CropParams(), SoilParams()

    # baselines tuned per fold (on that fold's training years) at the demo floor
    baselines = {}
    for fam in DEMO_BASELINES:
        per_fold = []
        for tr, _ in folds:
            best = select(sweep(fam, ws.subset(tr), fc_tune[tr]), "cvar5_yield", DEMO_FLOOR)
            per_fold.append({"label": best["label"], "params": best["policy"].params(),
                             "policy": best["policy"]})
        baselines[fam] = per_fold

    agents_json = json.loads((RESULTS / "agents.json").read_text())
    models = {}
    for name in MAP_AGENTS:
        st = load_models(name)
        models[name] = [st[f"0-{j}"] for j in range(len(folds))]

    # Python reference for the JS port: each demo policy on each held-out season
    reference = {}

    def run_fold_policies(make):
        water, yld = np.zeros(ws.n), np.zeros(ws.n)
        for j, (_, te) in enumerate(folds):
            res = simulate(make(j), ws.subset(te), fc[te])
            water[te], yld[te] = res.water, res.yield_
        return {"water": r(water, 4), "yield": r(yld, 6)}

    reference["rain-fed"] = run_fold_policies(lambda j: RainFed())
    for fam in DEMO_BASELINES:
        reference[fam] = run_fold_policies(lambda j, fam=fam: baselines[fam][j]["policy"])
    for name in MAP_AGENTS:
        reference[name] = run_fold_policies(
            lambda j, name=name: PPOPolicy(MLP.from_state(models[name][j])))

    base = json.loads((RESULTS / "baselines.json").read_text())
    diag = json.loads((RESULTS / "diagnostics.json").read_text())
    frontier = {"baselines": {fam: [{"floor": float(f), **{k: v["summary"][k] for k in (
        "water", "mean_yield", "cvar5_yield", "worst_yield")}} for f, v in by.items()]
        for fam, by in base["tuned"].items()},
        "agents": {n: {"kind": a["kind"], "config": a["config"], **{
            k: a["summary"][k] for k in a["summary"]}} for n, a in agents_json.items()}}
    curves = {n: {k: r(v, 4) if isinstance(v, list) else v for k, v in a["curve"].items()}
              for n, a in agents_json.items() if a["kind"] == "ppo"}
    robustness = json.loads((RESULTS / "robustness.json").read_text()) \
        if (RESULTS / "robustness.json").exists() else None

    data = {
        "station": BELLE_GLADE,
        "season": {"start": "03-01", "length": crop.season_length},
        "soil": {"taw": soil.taw, "p": soil.p, "initial_fraction": soil.initial_fraction},
        "crop": {"stage_starts": list(crop.stage_starts), "ky": list(crop.ky),
                 "kc_points": [list(p) for p in crop.kc_points], "stage_names": list(STAGE_NAMES)},
        "actions": ACTIONS.tolist(),
        "years": ws.labels,
        "fold_of": fold_of.tolist(),
        "rain": [r(x, 1) for x in ws.rain],
        "et0": [r(x, 5) for x in ws.et0],
        "forecast": [r(x, 5) for x in fc],
        "demo_floor": DEMO_FLOOR,
        "demo_agent": DEMO_AGENT,
        "baselines": {fam: [{"label": b["label"], "params": b["params"]} for b in v]
                      for fam, v in baselines.items()},
        "models": {n: [mlp_state(m) for m in ms] for n, ms in models.items()},
        "reference": reference,
        "frontier": frontier,
        "curves": curves,
        "diagnostics": {"variability": diag["variability"], "max_tail_gap": diag["max_tail_gap"],
                        "pan_check": {k: diag["pan_check"][k] for k in ("overall_ratio", "days",
                                                                        "annual_et0_mm")},
                        "sweeps": {k: [{kk: row[kk] for kk in ("water", "mean_yield", "cvar5_yield")}
                                       for row in v] for k, v in diag["sweeps"].items()}},
        "robustness": robustness,
        "headtohead": json.loads((RESULTS / "headtohead.json").read_text()),
    }
    blob = json.dumps(data, separators=(",", ":"))
    (ROOT / "site" / "data.json").write_text(blob)
    template = (ROOT / "site" / "template.html").read_text()
    sim = (ROOT / "site" / "sim.js").read_text()
    html = template.replace("/*__DATA__*/null", blob).replace("/*__SIM__*/", sim)
    # fragment for hosts that supply their own document skeleton
    (ROOT / "site" / "build").mkdir(exist_ok=True)
    (ROOT / "site" / "build" / "tailwater.html").write_text(html)
    # full document for GitHub Pages
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    (ROOT / "docs" / ".nojekyll").write_text("")
    print(f"wrote {out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
