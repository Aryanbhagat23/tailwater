# Data

Raw weather files live in `data/raw/` and are not committed (see `.gitignore`).

## NOAA GHCN-Daily, Belle Glade, FL

| Field | Value |
|---|---|
| Dataset | GHCN-Daily (`daily-summaries`), by-station CSV |
| Station | `USC00080611`, BELLE GLADE, FL (HCN), 26.6928 N, 80.6711 W, 6.1 m |
| Elements used | `PRCP`, `TMAX`, `TMIN` (plus `EVAP`, Class-A pan, for the ET0 cross-check) |
| Record | May 1924 to Oct 2005; 74 complete 1 March – 28 June seasons, 1925–2001 (later springs fail the completeness rule) |
| Source URL | `https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/USC00080611.csv` |
| Downloaded | 2026-10-27 (`scripts/fetch_noaa.py` writes the date next to the file) |

The URL is NOAA's official GHCN-Daily mirror on the AWS Open Data Registry
(`s3://noaa-ghcn-pds`). It serves the same by-station files as NCEI, whose own
endpoints (`ncei.noaa.gov`) were not reachable from the build sandbox.

```bash
python scripts/fetch_noaa.py              # Belle Glade
python scripts/fetch_noaa.py USC00084091  # Homestead Exp Stn, FL (alternative)
```

### Processing (`src/tailwater/env/weather.py`)

* Values carrying a GHCN quality flag are dropped. Units: tenths of mm / tenths of °C.
* One season per calendar year, 1 March + 120 days. Days are never moved between
  years, so wet and drought years stay intact.
* A year is kept only if each of PRCP, TMAX and TMIN has at most 6 missing days in
  the window. Missing PRCP is set to 0 (the count is stored in `meta`), and missing
  temperatures are interpolated linearly.
* ET0 is computed by Hargreaves & Samani (1985), with extraterrestrial radiation from
  FAO-56 eq. 21. At this station Hargreaves comes out at about 0.95× pan evaporation
  over the full record (about 0.92× in March–June). Reference ET is normally
  0.65–0.85× pan, so absolute ET0, and therefore absolute irrigation in mm, is
  probably overestimated. `load_ghcn_seasons(..., et0_scale=...)` is available for
  sensitivity runs.
