"""Download a NOAA GHCN-Daily station file into data/raw/.

Uses NOAA's official GHCN-Daily mirror on the AWS Open Data registry
(s3://noaa-ghcn-pds), which serves the same by-station CSV files as NCEI.

    python scripts/fetch_noaa.py                 # Belle Glade, FL (USC00080611)
    python scripts/fetch_noaa.py USC00084091     # Homestead Exp Stn, FL
"""
import sys
import urllib.request
from datetime import date
from pathlib import Path

URL = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{sid}.csv"


def main(sid: str = "USC00080611"):
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / f"{sid}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL.format(sid=sid)}")
    urllib.request.urlretrieve(URL.format(sid=sid), out)
    (out.parent / f"{sid}.downloaded").write_text(date.today().isoformat() + "\n")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
