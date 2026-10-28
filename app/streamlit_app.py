"""Streamlit demo: a season playing day by day, four fields on identical
weather, the water-vs-tail frontier and the agent's policy map.

    pip install -r requirements.txt
    python scripts/fetch_noaa.py
    python experiments/run_agents.py && python experiments/run_baseline_sweep.py
    streamlit run app/streamlit_app.py

(The static site in docs/index.html shows the same views without a server.)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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

RESULTS = ROOT / "results"
st.set_page_config(page_title="Tailwater", layout="wide")


@st.cache_resource
def load():
    ws = belle_glade_seasons()
    folds = make_folds(ws.n)
    fold_of = np.zeros(ws.n, dtype=int)
    for j, (_, te) in enumerate(folds):
        fold_of[te] = j
    return ws, folds, fold_of, eval_forecast(ws), ws.forecasts(FORECAST_NOISE, TUNE_FORECAST_SEED)


@st.cache_resource
def tuned(family, floor):
    ws, folds, _, _, fc_tune = load()
    return [select(sweep(family, ws.subset(tr), fc_tune[tr]), "cvar5_yield", floor)["policy"]
            for tr, _ in folds]


@st.cache_resource
def agent_models(name):
    path = RESULTS / "policies" / f"{name}.json"
    models = json.loads((path if path.exists() else RESULTS / "models" / f"{name}.json").read_text())
    return [PPOPolicy(MLP.from_state(models[f"0-{j}"]), name) for j in range(4)]


ws, folds, fold_of, fc_eval, _ = load()
agents = json.loads((RESULTS / "agents.json").read_text())
cvar_names = [n for n in agents if n.startswith("cvar")]

st.title("Tailwater")
st.caption("Risk-constrained RL for irrigation scheduling · Belle Glade, FL (NOAA GHCN USC00080611) · "
           "spring maize · every controller scored on years it was not tuned or trained on")

tab_race, tab_frontier, tab_map = st.tabs(["Season race", "Frontier", "Policy map"])

with tab_race:
    c1, c2 = st.columns([2, 1])
    year = c1.selectbox("Season", ws.labels, index=int(np.argmin(ws.rain.sum(1))),
                        format_func=lambda y: f"{y} · {ws.rain[ws.labels.index(y)].sum():.0f} mm rain")
    agent = c2.selectbox("Agent", cvar_names, index=len(cvar_names) - 1)
    floor = agents[agent]["config"]["floor"]
    i = ws.labels.index(year)
    j = int(fold_of[i])
    one = ws.subset([i])
    lanes = {"Rain-fed": RainFed(), "ET controller": tuned("ET controller", floor)[j],
             "Stage threshold": tuned("stage threshold", floor)[j], "CVaR-PPO": agent_models(agent)[j]}
    runs = {k: simulate(p, one, fc_eval[[i]], record=True) for k, p in lanes.items()}
    day = st.slider("Day of season", 1, 120, 120)
    if st.button("Play season"):
        holder = st.empty()
        for d in range(1, 121, 2):
            holder.line_chart(pd.DataFrame({k: r.traj["moisture"][0, :d] for k, r in runs.items()}))
            time.sleep(0.03)
    st.line_chart(pd.DataFrame({k: r.traj["moisture"][0, :day] for k, r in runs.items()}),
                  y_label="Soil water available to the crop, mm", x_label="Day")
    st.bar_chart(pd.Series(ws.rain[i, :day], name="rain"), y_label="Rain, mm/day", x_label="Day")
    cols = st.columns(4)
    for col, (k, r) in zip(cols, runs.items()):
        col.metric(k, f"{r.traj['irrigation'][0, :day].sum():.0f} mm",
                   f"final yield {r.yield_[0]:.0%}", delta_color="off")
    st.caption(f"{year} is in fold {j + 1}: all controllers were tuned/trained on the other three folds. "
               f"Stage on day {day}: {STAGE_NAMES[CropParams().stage_of_day(day - 1)]}.")

with tab_frontier:
    base = json.loads((RESULTS / "baselines.json").read_text())
    rows = []
    for fam, by in base["tuned"].items():
        for f, v in by.items():
            rows.append({"controller": fam, "floor": float(f), **v["summary"]})
    for n, a in agents.items():
        rows.append({"controller": n, "floor": a["config"].get("floor"), **a["summary"]})
    df = pd.DataFrame(rows)
    metric = st.radio("Yield metric", ["cvar5_yield", "mean_yield"], horizontal=True)
    st.scatter_chart(df[df.water < 400], x="water", y=metric, color="controller")
    st.dataframe(df[["controller", "floor", "water", "mean_yield", "cvar5_yield", "worst_yield"]]
                 .sort_values(["controller", "floor"]), hide_index=True)
    if (RESULTS / "summary.md").exists():
        with st.expander("Full results summary"):
            st.markdown((RESULTS / "summary.md").read_text())

with tab_map:
    saved = {p.stem for d in ("policies", "models") for p in (RESULTS / d).glob("*.json")}
    name = st.selectbox("Policy", sorted(agents.keys() & saved))
    fc = st.slider("3-day rain forecast, mm", 0, 40, 0)
    dsr = st.slider("Days since rain", 0, 15, 5)
    net = agent_models(name)[0]
    crop = CropParams()
    m = np.arange(1, 100, 2.0)
    et0 = float(ws.et0.mean())
    grid = np.zeros((len(m), 120))
    for d in range(120):
        obs = {"moisture": m, "day": d, "stage": int(crop.stage_of_day(d)), "kc": float(crop.kc(d)),
               "forecast": np.full(len(m), fc), "days_since_rain": np.full(len(m), dsr),
               "et0_yesterday": np.full(len(m), et0), "etc_yesterday": np.full(len(m), et0 * crop.kc(max(d - 1, 0)))}
        grid[:, d] = ACTIONS[np.argmax(net.net(features(obs)), 1)]
    st.caption("Irrigation (mm) chosen at each soil-water level (rows, mm) and day (columns).")
    st.dataframe(pd.DataFrame(grid[::-1], index=m[::-1].astype(int)).style.background_gradient(cmap="Blues"),
                 height=520)
