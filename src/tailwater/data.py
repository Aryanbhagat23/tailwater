"""Project datasets: one place that defines which weather the experiments use."""
from __future__ import annotations

from pathlib import Path

from .env.weather import WeatherSet, load_ghcn_seasons

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

BELLE_GLADE = {"station": "USC00080611", "name": "Belle Glade, FL", "lat": 26.6928,
               "lon": -80.6711, "elev_m": 6.1}
SEASON_START = "03-01"   # spring maize: 1 March - 28 June, dry-to-wet season transition
SEASON_LENGTH = 120


def belle_glade_seasons() -> WeatherSet:
    path = RAW / f"{BELLE_GLADE['station']}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python scripts/fetch_noaa.py")
    return load_ghcn_seasons(path, BELLE_GLADE["lat"], SEASON_START, SEASON_LENGTH,
                             station=BELLE_GLADE["station"])
