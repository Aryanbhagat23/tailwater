import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def to_jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def save_json(obj, name):
    path = RESULTS / name
    path.write_text(json.dumps(to_jsonable(obj), indent=1))
    print(f"wrote {path}")


TRIMMED = ("cvar_f80", "cvar_f90", "cvar_f95", "ppo_k7", "ppo_k10")


def load_models(name):
    """PPO weights by '<seed>-<fold>' key: the full set written by run_agents.py
    (gitignored, 24 MB) if present, else the trimmed committed copy."""
    for d in ("policies", "models"):
        p = RESULTS / d / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"no saved weights for {name}; run experiments/run_agents.py")


def write_trimmed_models():
    """Seed 0 for every fold (demo) and fold 0 for every seed (robustness),
    rounded to 7 decimals."""
    import numpy as np
    out = RESULTS / "models"
    out.mkdir(exist_ok=True)
    for name in TRIMMED:
        full = json.loads((RESULTS / "policies" / f"{name}.json").read_text())
        keep = {k: {"W": [np.round(np.array(w), 7).tolist() for w in v["W"]],
                    "b": [np.round(np.array(b), 7).tolist() for b in v["b"]]}
                for k, v in full.items() if k.startswith("0-") or k.endswith("-0")}
        (out / f"{name}.json").write_text(json.dumps(keep, separators=(",", ":")))


if __name__ == "__main__":
    write_trimmed_models()
