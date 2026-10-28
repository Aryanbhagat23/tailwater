"""Train every agent configuration on every fold x seed; score on held-out years.

    python experiments/run_agents.py            # full run (~45 min on 4 cores)
    python experiments/run_agents.py --quick    # 2 seeds, short training
"""
import argparse
import json
import time

from _common import RESULTS, save_json  # noqa: I001  (sets single-thread BLAS before numpy loads)
import numpy as np
from tailwater.agents import registry
from tailwater.data import belle_glade_seasons
from tailwater.eval.protocol import cross_validate_agent, make_folds

CONFIGS = {
    # risk-neutral: maximise kappa*Y - W/100 for several crop values kappa
    "tabular_k10": ("tabular", {"kappa": 10.0}),
    "ppo_k2": ("ppo", {"kappa": 2.0}),
    "ppo_k5": ("ppo", {"kappa": 5.0}),
    "ppo_k6": ("ppo", {"kappa": 6.0}),
    "ppo_k7": ("ppo", {"kappa": 7.0}),
    "ppo_k8": ("ppo", {"kappa": 8.0}),
    "ppo_k10": ("ppo", {"kappa": 10.0}),
    "ppo_k20": ("ppo", {"kappa": 20.0}),
    # risk-constrained: kappa = 2 (water-saving grower) plus a CVaR-5% yield floor
    "cvar_f80": ("ppo", {"kappa": 2.0, "floor": 0.80}),
    "cvar_f90": ("ppo", {"kappa": 2.0, "floor": 0.90}),
    "cvar_f95": ("ppo", {"kappa": 2.0, "floor": 0.95}),
}


def curve_mean(curves):
    keys = [k for k in curves[0][0] if k not in ("iteration", "episode", "seasons")]
    xkey = "seasons" if "seasons" in curves[0][0] else "episode"
    arr = {k: np.mean([[r[k] for r in c] for c in curves], axis=0) for k in keys}
    sd = {k + "_sd": np.std([[r[k] for r in c] for c in curves], axis=0) for k in keys}
    return {"x": [r[xkey] for r in curves[0]], "x_label": xkey, **arr, **sd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    ws = belle_glade_seasons()
    folds = make_folds(ws.n)
    out_path = RESULTS / "agents.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    poldir = RESULTS / "policies"
    poldir.mkdir(exist_ok=True)
    for name, (kind, cfg) in CONFIGS.items():
        if args.only and name not in args.only:
            continue
        cfg = dict(cfg)
        if args.quick:
            cfg.update({"iterations": 40} if kind == "ppo" else {"episodes": 4000})
        t = time.time()
        fn = registry.ppo if kind == "ppo" else registry.tabular
        r = cross_validate_agent(fn, cfg, ws, folds, seeds, workers=args.workers)
        s = r["summary"]
        print(f"{name:12s} {time.time()-t:6.0f}s  water {s['water']:6.1f}±{s['water_seed_sd']:.1f}  "
              f"meanY {s['mean_yield']:.3f}±{s['mean_yield_seed_sd']:.3f}  "
              f"CVaR5 {s['cvar5_yield']:.3f}±{s['cvar5_yield_seed_sd']:.3f}  "
              f"worst {s['worst_yield']:.3f}", flush=True)
        out[name] = {
            "kind": kind, "config": cfg, "seeds": seeds, "summary": s,
            "per_seed": [{"seed": p["seed"], "summary": p["summary"],
                          "water": p["water"], "yield": p["yield"]} for p in r["per_seed"]],
            "curve": curve_mean(list(r["curves"].values())),
        }
        if kind == "ppo":  # keep every fold model of seed 0 for the demo / robustness sweep
            models = {f"{s_}-{j}": pol.net.state() for (s_, j), pol in r["policies"].items()}
            (poldir / f"{name}.json").write_text(json.dumps(models))
        else:
            np.savez_compressed(poldir / f"{name}.npz",
                                **{f"s{s_}_f{j}": pol.q for (s_, j), pol in r["policies"].items()})
        save_json(out, "agents.json")


if __name__ == "__main__":
    main()
