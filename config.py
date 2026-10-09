"""Shared paths and constants for the daily Gljivar pipeline.

Two output grids are built by the same code, picked by the environment variable GLJIVAR_GRID:
  hrsi1  (default) Croatia + Slovenia, published under data/ for the 0.4.x apps
  eu9v1  nine countries, published under data/v2/ for 0.5.0+ (docs/PLAN-0.5.0.md)
The variable has to be set before this module is imported (run_daily.py --v2 does that).
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
BACKTEST = ROOT.parent / "backtest"
CACHE = ROOT / "cache"        # raw downloads, safe to delete
ARCHIVE = ROOT / "archive"    # one small file per past day; the job cannot rebuild these

STEP = 0.1
# Bumped whenever the box or the country set changes: it keys the ERA5-Land month cache and
# the per-day archive, so arrays of the old shape are never reused with the new grid.
GRID_TAG = os.environ.get("GLJIVAR_GRID", "hrsi1")

if GRID_TAG == "hrsi1":
    # Output grid = the ERA5-Land 0.1 degree grid (cell centres on x.x0), cropped to Croatia
    # and Slovenia. Slovenia reaches 46.88 N and 13.38 E, so the box grew north and west.
    LAT0, LAT1 = 42.3, 47.0
    LON0, LON1 = 13.3, 19.5
    COUNTRIES = ["HRV", "SVN"]   # Natural Earth ADM0_A3 codes that make up the grid
    # What the app calls them (Country.code in the Android app).
    COUNTRY_CODES = {"HRV": "hr", "SVN": "si"}
    OUT = ROOT / "out"            # what gets published
    GRID_FILE = "grid.npz"
    SCHEMA = 1
    USE_IMERG = True
    # How far back missing ECMWF 00z analyses are re-fetched (data.ecmwf.int keeps ~4 days)
    ECMWF_BACKFILL_DAYS = 4
elif GRID_TAG == "eu9v1":
    # Lampedusa 35.49 N .. Sylt 55.07 N, Germany 5.85 E .. Serbia 22.98 E.
    LAT0, LAT1 = 35.5, 55.1
    LON0, LON1 = 5.8, 23.0
    # Order = index in grid.json "country"; hr and si first, as in v1.
    COUNTRIES = ["HRV", "SVN", "ITA", "AUT", "DEU", "CHE", "SRB", "BIH", "HUN"]
    COUNTRY_CODES = {"HRV": "hr", "SVN": "si", "ITA": "it", "AUT": "at", "DEU": "de", "CHE": "ch",
                     "SRB": "rs", "BIH": "ba", "HUN": "hu"}
    OUT = ROOT / "out" / "v2"
    GRID_FILE = f"grid_{GRID_TAG}.npz"
    SCHEMA = 2
    # IMERG's monthly scale factors were fitted on Croatia only, and since ERA5 fills the
    # ERA5-Land gap (era5.py) IMERG only covers the last ~5 days. v2 uses the ECMWF 00z run of
    # each of those days instead (0-24 h forecast), v1's next fallback anyway.
    USE_IMERG = False
    # A new grid starts with an empty archive, so missing ECMWF analyses are fetched from the
    # AWS mirror of ECMWF open data, which keeps old runs.
    ECMWF_BACKFILL_DAYS = 60
else:
    raise SystemExit(f"unknown GLJIVAR_GRID {GRID_TAG!r}")

V2 = SCHEMA >= 2
RAIN_PERIODS = [7, 14]        # rain layer (v2): sums over the last 7 / 14 complete days

GROUPS = ["vrganj", "lisicarka", "suncanica", "smrcak"]
RR_LAGS, TG_LAGS = 61, 31     # rain lags 0..60, temperature lags 0..30 (see backtest/model2.py)
FORECAST_DAYS = 8             # today + 7

EDL_TOKEN = Path.home() / ".gljivar_edl_token"


def cds_area(margin):
    """CDS "area" [N, W, S, E] around the box, with a jitter of up to 0.019 deg that changes daily.

    CDS answers a request identical to an earlier one from its result cache. A month that
    was still incomplete when first requested therefore came back with the same missing
    days on every later run: the v1 job's ERA5-Land September file stayed at 2026-09-12 for
    four weeks while a fresh request returned data to 2026-10-03. The jitter is smaller than
    the margin to the next grid line (margins here are 0.05 and 0.3), so the returned grid
    points do not change - only the request does.
    """
    import pandas as pd
    j = (pd.Timestamp.now("UTC").dayofyear % 19 + 1) * 0.001
    return [round(LAT1 + margin + j, 4), round(LON0 - margin - j, 4),
            round(LAT0 - margin - j, 4), round(LON1 + margin + j, 4)]


def lats():
    import numpy as np
    return np.round(np.arange(LAT0, LAT1 + STEP / 2, STEP), 2)


def lons():
    import numpy as np
    return np.round(np.arange(LON0, LON1 + STEP / 2, STEP), 2)
