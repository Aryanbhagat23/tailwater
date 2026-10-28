"""Assemble the results: frontier figures, learning curves and the summary
tables in results/summary.md. Reads the JSON written by the other
experiments; trains nothing.

Head-to-head metric. For an agent trained with seed s, let c_s be its
held-out CVaR-5% yield and W_s its held-out mean water. The baseline cost of
that reliability is read off the Pareto envelope of all tuned baselines (any
family, any floor, also held-out), interpolated linearly between envelope
points. The saving is baseline cost - W_s, reported as mean +- sd over seeds.
Positive means the agent buys the same tail protection with less water. The
same comparison is made against the risk-neutral PPO frontier (seed-mean of
each kappa) to ask whether the constraint itself adds anything.
"""
from _common import FIGURES, RESULTS, save_json  # noqa: I001
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a897f", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAYS = {"calendar": "#b5b4ac", "threshold": "#8a897f", "threshold+forecast": "#52514e",
         "stage threshold": "#2e2d2a", "ET controller": "#a3a29a", "rain-fed": "#c9c8c0"}
MARKERS = {"calendar": "s", "threshold": "o", "threshold+forecast": "^",
           "stage threshold": "D", "ET controller": "v", "rain-fed": "x"}
AGENT_LABELS = {"tabular_k10": "Tabular Q (κ=10)", "ppo_k2": "PPO κ=2", "ppo_k5": "PPO κ=5",
                "ppo_k6": "PPO κ=6", "ppo_k7": "PPO κ=7", "ppo_k8": "PPO κ=8",
                "ppo_k10": "PPO κ=10", "ppo_k20": "PPO κ=20", "cvar_f80": "CVaR-PPO ≥0.80",
                "cvar_f90": "CVaR-PPO ≥0.90", "cvar_f95": "CVaR-PPO ≥0.95"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def baseline_points(base):
    pts = []
    for fam, by_floor in base["tuned"].items():
        for f, v in by_floor.items():
            pts.append({"family": fam, "floor": float(f), **v["summary"]})
    return pts


def envelope_cost(points, level, metric="cvar5_yield"):
    """Water needed to reach ``level`` of ``metric`` on the Pareto envelope of
    ``points``, interpolated linearly between neighbouring envelope points
    (a step-function envelope would flatter whichever side sits just past a
    step). NaN outside the envelope's range."""
    pts = sorted(points, key=lambda p: (p["water"], -p[metric]))
    front, best = [], -np.inf
    for p in pts:
        if p[metric] > best + 1e-12:
            front.append((p[metric], p["water"]))
            best = p[metric]
    if not front or level < front[0][0] - 1e-9 or level > front[-1][0] + 1e-9:
        return np.nan
    m, w = zip(*front)
    return float(np.interp(level, m, w))


def head_to_head(agents, bpts, only=None):
    rows = []
    for name, a in agents.items():
        if only and not name.startswith(only):
            continue
        for metric in ("cvar5_yield", "mean_yield"):
            sav = [envelope_cost(bpts, p["summary"][metric], metric) - p["summary"]["water"]
                   for p in a["per_seed"]]
            sav = np.array(sav, dtype=float)
            sav = sav[~np.isnan(sav)] if np.isfinite(sav).any() else sav
            rows.append({"agent": name, "metric": metric, "saving_mean": np.nanmean(sav),
                         "saving_sd": np.nanstd(sav, ddof=1) if len(sav) > 1 else 0.0,
                         "n_beat": int((sav > 0).sum()), "n": len(sav)})
    return rows


def fig_frontier(base, agents, metric, fname, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for fam, by_floor in base["tuned"].items():
        if fam == "rain-fed":
            continue
        pts = sorted({(v["summary"]["water"], v["summary"][metric]) for v in by_floor.values()})
        x, y = zip(*pts)
        ax.plot(x, y, color=GRAYS[fam], lw=1.2, marker=MARKERS[fam], ms=4, label=f"{fam} (tuned)")
    groups = [("cvar_f", BLUE, "CVaR-PPO (κ=2, floor)"), ("ppo_k", ORANGE, "PPO risk-neutral (κ)"),
              ("tabular", AQUA, "Tabular Q")]
    for prefix, color, lab in groups:
        names = [n for n in agents if n.startswith(prefix)]
        if not names:
            continue
        xs = [agents[n]["summary"]["water"] for n in names]
        ys = [agents[n]["summary"][metric] for n in names]
        xe = [agents[n]["summary"]["water_seed_sd"] for n in names]
        ye = [agents[n]["summary"][metric + "_seed_sd"] for n in names]
        if prefix in ("ppo_k", "cvar_f"):
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], color=color, lw=1, alpha=0.6, zorder=4)
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o", color=color, ms=7, lw=1.5, capsize=2,
                    mec="white", mew=1.5, label=lab, zorder=5)
        for n, x_, y_ in zip(names, xs, ys):
            short = AGENT_LABELS[n].replace("CVaR-PPO ", "").replace("PPO ", "").replace("Tabular Q ", "")
            ax.annotate(short, (x_, y_), textcoords="offset points", xytext=(6, -10),
                        fontsize=8, color=INK2)
    ax.set_xlabel("Mean seasonal irrigation, mm (held-out years)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(70, 320)
    ax.set_ylim(0.4 if metric == "cvar5_yield" else 0.6, 1.01)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / fname)
    plt.close(fig)


def fig_rate_sweep(diag):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, key, title in [(axes[0], "iid_linear", "Prototype-style: i.i.d. synthetic weather, linear yield"),
                           (axes[1], "noaa_fao33", "Belle Glade 1925-2001, FAO-33 yield")]:
        rows = diag["sweeps"][key]
        w = [r["water"] for r in rows]
        ax.fill_between(w, [r["cvar5_yield"] for r in rows], [r["mean_yield"] for r in rows],
                        color=BLUE, alpha=0.12, lw=0)
        ax.plot(w, [r["mean_yield"] for r in rows], color=BLUE, label="mean yield")
        ax.plot(w, [r["cvar5_yield"] for r in rows], color=ORANGE, label="CVaR-5% yield")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("Constant irrigation over the season, mm")
        gap = diag["max_tail_gap"][key]
        ax.text(0.97, 0.05, f"max gap {gap:.2f}", transform=ax.transAxes, ha="right",
                color=INK2, fontsize=9)
    axes[0].set_ylabel("Relative yield")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "rate_sweep.png")
    plt.close(fig)


def fig_weather(diag):
    s = diag["seasons"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    yrs = [r["year"] for r in s]
    ax.bar(yrs, [r["rain"] for r in s], color=BLUE, width=0.8)
    ax.set_ylabel("Mar-Jun rain, mm")
    ax.set_title("Belle Glade, FL (GHCN USC00080611): season rainfall by year", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "season_rain.png")
    plt.close(fig)


def fig_curves(agents):
    names = [n for n in ("ppo_k10", "cvar_f90", "cvar_f95") if n in agents]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    colors = {"ppo_k10": ORANGE, "cvar_f90": BLUE, "cvar_f95": AQUA}
    for n in names:
        c = agents[n]["curve"]
        x = np.array(c["x"]) / 1000
        for ax, key in zip(axes, ("water", "cvar5_yield", "lambda")):
            m, sd = np.array(c[key]), np.array(c[key + "_sd"])
            ax.plot(x, m, color=colors[n], lw=1.5, label=AGENT_LABELS[n])
            ax.fill_between(x, m - sd, m + sd, color=colors[n], alpha=0.12, lw=0)
    for ax, t in zip(axes, ("Water per season, mm", "Batch CVaR-5% yield", "Multiplier λ")):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Training seasons (thousands)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png")
    plt.close(fig)


def fmt_row(name, s, sd=True):
    def f(k, d=3):
        v = s[k]
        e = s.get(k + "_seed_sd")
        return f"{v:.{d}f}" + (f" ± {e:.{d}f}" if sd and e is not None else "")
    return (f"| {name} | {f('water', 0)} | {f('mean_yield')} | {f('p5_yield')} | "
            f"{f('cvar5_yield')} | {f('worst_yield')} |")


def main():
    diag, base, agents, rob = load("diagnostics.json"), load("baselines.json"), \
        load("agents.json") or {}, load("robustness.json")
    agents = {k: agents[k] for k in AGENT_LABELS if k in agents}
    fig_rate_sweep(diag)
    fig_weather(diag)
    fig_frontier(base, agents, "cvar5_yield", "frontier_cvar.png", "CVaR-5% yield (worst 4 of 74 years)")
    fig_frontier(base, agents, "mean_yield", "frontier_mean.png", "Mean relative yield")
    fig_curves(agents)
    bpts = baseline_points(base)
    h2h = head_to_head(agents, bpts)
    # the CVaR agents against the risk-neutral PPO frontier (mean over seeds of each kappa)
    rn_pts = [{"water": a["summary"]["water"], "cvar5_yield": a["summary"]["cvar5_yield"],
               "mean_yield": a["summary"]["mean_yield"]} for n, a in agents.items() if n.startswith("ppo_k")]
    h2h_rn = head_to_head(agents, rn_pts, only="cvar")
    save_json({"vs_baselines": h2h, "vs_risk_neutral": h2h_rn}, "headtohead.json")

    L = ["# Results summary", "",
         "All numbers are on held-out years: 4-fold cross-validation over the 74 Belle Glade "
         "seasons (1925-2001), every year scored by a model that never saw it. Agent rows are "
         "mean ± sd over 5 training seeds; tuned baselines are deterministic. Generated by "
         "`experiments/run_frontier.py`.", "",
         "## Agents", "",
         "| Agent | Water (mm) | Mean yield | p5 yield | CVaR-5% | Worst |", "|---|---|---|---|---|---|"]
    for n, a in agents.items():
        L.append(fmt_row(AGENT_LABELS.get(n, n), a["summary"]))
    L += ["", "## Tuned baselines (least water meeting CVaR-5% ≥ floor on training years)", "",
          "| Baseline @ floor | Water (mm) | Mean yield | p5 yield | CVaR-5% | Worst |",
          "|---|---|---|---|---|---|"]
    for fam, by_floor in base["tuned"].items():
        for f, v in by_floor.items():
            if fam == "rain-fed" or float(f) in (0.8, 0.9, 0.95, 1.0):
                L.append(fmt_row(f"{fam} @ {float(f):.2f}", v["summary"], sd=False))
    L += ["", "## Head-to-head: water saved vs the tuned-baseline envelope at the same "
          "held-out tail (linear interpolation along the envelope)", "",
          "| Agent | matched on | saving (mm, mean ± sd over seeds) | seeds with saving > 0 |",
          "|---|---|---|---|"]
    for r in h2h:
        L.append(f"| {AGENT_LABELS.get(r['agent'], r['agent'])} | {r['metric']} | "
                 f"{r['saving_mean']:+.1f} ± {r['saving_sd']:.1f} | {r['n_beat']}/{r['n']} |")
    L += ["", "## Does the constraint beat pricing the crop higher? CVaR-PPO vs the risk-neutral PPO "
          "frontier", "", "Water saved relative to the risk-neutral PPO envelope (κ = 2…20, seed means, "
          "linear interpolation) at the agent's held-out value of the matched metric.", "",
          "| Agent | matched on | saving (mm, mean ± sd over seeds) | seeds with saving > 0 |",
          "|---|---|---|---|"]
    for r in h2h_rn:
        L.append(f"| {AGENT_LABELS.get(r['agent'], r['agent'])} | {r['metric']} | "
                 f"{r['saving_mean']:+.1f} ± {r['saving_sd']:.1f} | {r['n_beat']}/{r['n']} |")
    if rob:
        L += ["", "## Robustness: climate wetness (600 synthetic seasons per level, fitted to "
              "Belle Glade)", "", "Cells: water mm / CVaR-5% yield.", "",
              "| Policy | " + " | ".join(f"wetness {w}" for w in rob["wetness"]) + " |",
              "|---|" + "---|" * len(rob["wetness"])]
        for name, rows in rob["climate_sweep"].items():
            L.append(f"| {name} | " + " | ".join(f"{r['water']:.0f} / {r['cvar5_yield']:.2f}"
                                                  for r in rows) + " |")
        L += ["", "## Robustness: forecast noise (historical seasons)", "",
              "| Policy | " + " | ".join(f"σ={z}" for z in rob["noise"]) + " |",
              "|---|" + "---|" * len(rob["noise"])]
        for name, rows in rob["forecast_sweep"].items():
            L.append(f"| {name} | " + " | ".join(f"{r['water']:.0f} / {r['cvar5_yield']:.2f}"
                                                  for r in rows) + " |")
    v = diag["variability"]
    L += ["", "## Environment diagnostics", "",
          "| Weather | Season rain CV | Season ET0 CV | max (mean − CVaR5) gap, constant-rate sweep |",
          "|---|---|---|---|",
          f"| Synthetic i.i.d. (prototype-style) | {v['synthetic_iid']['rain']['cv']:.3f} | "
          f"{v['synthetic_iid']['et0']['cv']:.3f} | {diag['max_tail_gap']['iid_linear']:.2f} |",
          f"| NOAA Belle Glade | {v['noaa']['rain']['cv']:.3f} | {v['noaa']['et0']['cv']:.3f} | "
          f"{diag['max_tail_gap']['noaa_fao33']:.2f} |",
          f"| Synthetic fitted to Belle Glade | {v['synthetic_fitted']['rain']['cv']:.3f} | "
          f"{v['synthetic_fitted']['et0']['cv']:.3f} | – |", "",
          f"Hargreaves ET0 / Class-A pan evaporation at the station: "
          f"{diag['pan_check']['overall_ratio']:.2f} over {diag['pan_check']['days']} days "
          f"(typical pan coefficients are 0.65-0.85, so Hargreaves likely overestimates ET0 here)."]
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
