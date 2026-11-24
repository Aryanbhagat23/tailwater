"""Extra figures and tables for the written report (report/):

* case study: the driest season (1967), four controllers on identical weather,
  each tuned or trained without that year
* the input/output sequence of that season for a window of days
* the trained policy map (irrigation chosen by soil water and day)
"""
from _common import FIGURES, RESULTS, load_models, save_json  # noqa: I001
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from run_frontier import BLUE, INK2, MUTED  # noqa: E402  (also applies the shared rcParams)
from tailwater.agents.common import features  # noqa: E402
from tailwater.agents.nn import MLP  # noqa: E402
from tailwater.agents.ppo import PPOPolicy  # noqa: E402
from tailwater.data import belle_glade_seasons  # noqa: E402
from tailwater.env.crop import STAGE_NAMES, CropParams  # noqa: E402
from tailwater.env.irrigation_env import ACTIONS, simulate  # noqa: E402
from tailwater.eval.protocol import (FORECAST_NOISE, TUNE_FORECAST_SEED, eval_forecast,  # noqa: E402
                                     make_folds)
from tailwater.policies.baselines import RainFed  # noqa: E402
from tailwater.policies.tuning import select, sweep  # noqa: E402

AGENT, FLOOR = "cvar_f90", 0.90
COLORS = {"Rain-fed": "#8a897f", "ET controller": "#1baf7a", "Stage threshold": "#eb6834",
          "CVaR-PPO (floor 0.90)": BLUE}


def main():
    ws = belle_glade_seasons()
    folds = make_folds(ws.n)
    fold_of = np.zeros(ws.n, dtype=int)
    for j, (_, te) in enumerate(folds):
        fold_of[te] = j
    fc_eval = eval_forecast(ws)
    fc_tune = ws.forecasts(FORECAST_NOISE, TUNE_FORECAST_SEED)
    crop = CropParams()

    i = int(np.argmin(ws.rain.sum(1)))
    year, j = ws.labels[i], int(fold_of[i])
    tr = folds[j][0]
    models = load_models(AGENT)
    pols = {
        "Rain-fed": RainFed(),
        "ET controller": select(sweep("ET controller", ws.subset(tr), fc_tune[tr]), "cvar5_yield", FLOOR)["policy"],
        "Stage threshold": select(sweep("stage threshold", ws.subset(tr), fc_tune[tr]), "cvar5_yield", FLOOR)["policy"],
        "CVaR-PPO (floor 0.90)": PPOPolicy(MLP.from_state(models[f"0-{j}"])),
    }
    one = ws.subset([i])
    runs = {k: simulate(p, one, fc_eval[[i]], record=True) for k, p in pols.items()}

    # ---- case-study figure
    days = np.arange(1, 121)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2.2, 1.6]})
    axes[0].bar(days, ws.rain[i], color="#8fb0cc", width=0.8)
    axes[0].set_ylabel("Rain, mm")
    axes[0].set_title(f"{year} (driest spring on record, {ws.rain[i].sum():.0f} mm): four controllers, "
                      f"identical weather", fontsize=10)
    ax = axes[1]
    ax.axhspan(0, 50, color="#f6e7c8", alpha=0.7, lw=0)
    ax.text(119, 3, "crop under stress (< 50 mm)", ha="right", color=INK2, fontsize=8)
    for k, r in runs.items():
        ax.plot(days, r.traj["moisture"][0], color=COLORS[k], lw=1.6,
                label=f"{k}: {r.water[0]:.0f} mm, yield {r.yield_[0]:.2f}")
    ax.set_ylabel("Soil water, mm")
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    ax = axes[2]
    for n, (k, r) in enumerate(runs.items()):
        irr = r.traj["irrigation"][0]
        on = irr > 0
        ax.scatter(days[on], np.full(on.sum(), n), s=irr[on] * 2.2, color=COLORS[k], lw=0)
    ax.set_yticks(range(len(runs)), list(runs))
    ax.set_ylim(-0.7, len(runs) - 0.3)
    ax.set_xlabel("Day of season (day 1 = 1 March)")
    ax.set_title("Irrigation events (marker area ∝ amount: 5, 10 or 20 mm)", fontsize=9, color=INK2)
    for s in crop.stage_starts[1:]:
        for a in axes:
            a.axvline(s + 0.5, color=MUTED, lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(FIGURES / "case_study_1967.png")
    plt.close(fig)

    # ---- input/output sequence for a window around flowering
    agent_run = runs["CVaR-PPO (floor 0.90)"]
    window = range(52, 66)  # 0-based days 52..65 span the vegetative -> flowering transition
    seq = []
    s_prev = 100.0
    for d in window:
        s_prev = agent_run.traj["moisture"][0, d - 1] if d else 100.0
        seq.append({"day": d + 1, "stage": STAGE_NAMES[crop.stage_of_day(d)],
                    "moisture_morning": s_prev, "forecast_3d": fc_eval[i, d], "et0": ws.et0[i, d],
                    "rain": ws.rain[i, d], "action_mm": agent_run.traj["irrigation"][0, d],
                    "moisture_evening": agent_run.traj["moisture"][0, d]})

    # ---- policy map
    net = PPOPolicy(MLP.from_state(models["0-0"])).net
    m = np.arange(1, 100, 2.0)
    et0 = float(ws.et0.mean())
    grid = np.zeros((len(m), 120))
    for d in range(120):
        obs = {"moisture": m, "day": d, "stage": int(crop.stage_of_day(d)), "kc": float(crop.kc(d)),
               "forecast": np.zeros(len(m)), "days_since_rain": np.full(len(m), 5.0),
               "et0_yesterday": np.full(len(m), et0),
               "etc_yesterday": np.full(len(m), et0 * crop.kc(max(d - 1, 0)))}
        grid[:, d] = ACTIONS[np.argmax(net(features(obs)), 1)]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    cmap = ListedColormap(["#e9eef3", "#86b6ef", "#3987e5", "#184f95"])
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=(0, 120, 0, 100), cmap=cmap,
                   vmin=-2.5, vmax=22.5, interpolation="nearest")
    im.set_norm(matplotlib.colors.BoundaryNorm([-2.5, 2.5, 7.5, 15, 22.5], 4))
    ax.grid(False)
    ax.axhline(50, color="black", lw=0.8, ls=":")
    st = select(sweep("stage threshold", ws.subset(folds[0][0]), fc_tune[folds[0][0]]),
                "cvar5_yield", FLOOR)["policy"]
    trig = [st.t_hi if crop.stage_of_day(d) in (2, 3) else st.t_lo for d in range(120)]
    ax.step(np.arange(120), trig, where="post", color="#eb6834", lw=1.5, ls="--",
            label="tuned stage-threshold trigger")
    for s in crop.stage_starts[1:]:
        ax.axvline(s, color="white", lw=1)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 5, 11.25, 18.75])
    cb.ax.set_yticklabels(["0 mm", "5 mm", "10 mm", "20 mm"])
    ax.set_xlabel("Day of season")
    ax.set_ylabel("Soil water, mm")
    ax.set_title("CVaR-PPO (floor 0.90) policy: forecast 0 mm, 5 days since rain", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "policy_map.png")
    plt.close(fig)

    save_json({"year": year, "fold": j + 1, "season_rain": float(ws.rain[i].sum()),
               "controllers": {k: {"water": float(r.water[0]), "yield": float(r.yield_[0]),
                                   "label": pols[k].label()} for k, r in runs.items()},
               "sequence": seq,
               "policy_map_share": {f"{a:.0f}mm": float((grid == a).mean()) for a in ACTIONS}},
              "case_study.json")


if __name__ == "__main__":
    main()
