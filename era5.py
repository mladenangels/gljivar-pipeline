"""ERA5 (single levels, 0.25 degree) rain for days ERA5-Land does not cover yet.

ERA5-Land's rain is ERA5's rain interpolated to 0.1 degree, so ERA5 is the closest stand-in
when ERA5-Land falls behind. It normally does not: ERA5-Land runs ~6 days late. But from
2026-09-12 CDS stopped extending ERA5-Land while ERA5 kept going (~5 days late), and four
weeks of the gap were filled with IMERG, which on Pohorje read 4.7 mm for a 20 mm day.

One request per calendar month (hourly tp), cached like era5land.py; interpolated to the
output grid. Rain only: temperature keeps its own sources (ECMWF analysis).
"""
import time

import numpy as np
import pandas as pd
import xarray as xr

import config as C


def _month_file(y, m):
    return C.CACHE / f"era5_{C.GRID_TAG}_{y}{m:02d}.nc"


def _download(y, m, target):
    import cdsapi
    last = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
    req = {"product_type": ["reanalysis"], "variable": ["total_precipitation"],
           "year": str(y), "month": f"{m:02d}", "day": [f"{d:02d}" for d in range(1, last.day + 1)],
           "time": [f"{h:02d}:00" for h in range(24)],
           "area": [C.LAT1 + 0.3, C.LON0 - 0.3, C.LAT0 - 0.3, C.LON1 + 0.3],
           "data_format": "netcdf", "download_format": "unarchived"}
    tmp = target.with_suffix(".part")
    for k in range(4):
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-single-levels", req, str(tmp))
            tmp.rename(target)
            return
        except Exception as e:  # CDS queues fail now and then
            err = e
            time.sleep(30 * (k + 1))
    raise RuntimeError(f"ERA5 {y}-{m:02d} failed: {err}")


def _load(path):
    d = xr.open_dataset(path)
    if "valid_time" in d.dims:
        d = d.rename({"valid_time": "time"})
    return d.sortby("latitude")


def fetch(days, today):
    """{day: rain[y, x] mm} for those of `days` ERA5 already covers completely."""
    days = pd.DatetimeIndex(days)
    # CDS publishes ERA5 ~5 days late: asking for anything newer just re-downloads the month
    newest = min(days[-1] + pd.Timedelta(days=1), pd.Timestamp(today) - pd.Timedelta(days=6)) if len(days) else None
    if not len(days):
        return {}
    C.CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in pd.period_range(days[0].to_period("M"), days[-1].to_period("M"), freq="M"):
        if p.to_timestamp() > newest:
            continue  # nothing published for this month yet
        f = _month_file(p.year, p.month)
        if f.exists():
            with _load(f) as d:
                ok = np.isfinite(d["tp"].values).reshape(d.sizes["time"], -1).any(1)
                have = pd.DatetimeIndex(d.time.values)[ok].max() if ok.any() else pd.Timestamp(0)
            final = have >= (p + 1).to_timestamp()
            # mtime is no use here: actions/cache restores files with a fresh one
            if not final and have < newest:
                f.unlink()
        if not f.exists():
            _download(p.year, p.month, f)
        frames.append(_load(f)["tp"].load())
    if not frames:
        return {}
    tp = xr.concat(frames, dim="time").sortby("time")
    tp = tp.interp(latitude=C.lats(), longitude=C.lons())
    t = pd.DatetimeIndex(tp.time.values)
    vals = tp.values
    ok = np.isfinite(vals).reshape(len(t), -1).any(1)
    out = {}
    for day in days:
        # hourly accumulations valid 01 UTC .. 00 UTC of the next day = the same day as ERA5-Land
        hrs = pd.date_range(day + pd.Timedelta(hours=1), day + pd.Timedelta(days=1), freq="h")
        idx = t.get_indexer(hrs)
        if (idx < 0).any() or not ok[idx].all():
            continue
        out[day] = (np.clip(vals[idx], 0, None).sum(0) * 1000.0).astype("float32")
    return out
