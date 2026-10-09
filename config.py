"""Shared paths and constants for the daily Gljivar pipeline."""
from pathlib import Path

ROOT = Path(__file__).parent
BACKTEST = ROOT.parent / "backtest"
CACHE = ROOT / "cache"        # raw downloads, safe to delete
ARCHIVE = ROOT / "archive"    # one small file per past day; the job cannot rebuild these
OUT = ROOT / "out"            # what gets published

# Output grid = the ERA5-Land 0.1 degree grid (cell centres on x.x0), cropped to Croatia
# and Slovenia. Slovenia reaches 46.88 N and 13.38 E, so the box grew north and west.
LAT0, LAT1 = 42.3, 47.0
LON0, LON1 = 13.3, 19.5
STEP = 0.1
COUNTRIES = ["HRV", "SVN"]   # Natural Earth ADM0_A3 codes that make up the grid
# What the app calls them (Country.code in the Android app).
COUNTRY_CODES = {"HRV": "hr", "SVN": "si"}

# Bumped whenever the box or the country set changes: it keys the ERA5-Land month cache and
# the per-day archive, so arrays of the old shape are never reused with the new grid.
GRID_TAG = "hrsi1"

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
