"""Recent ERA5-Land from the Copernicus CDS, on the output grid.

One request per calendar month (00/06/12/18 UTC, tp + t2m), cached. The current and the
previous month are re-requested when the cache does not yet reach the newest data CDS
could have (ERA5-Land runs ~6 days behind). Definitions match backtest/build_era5land.py.
"""
import time

import numpy as np
import pandas as pd
import xarray as xr

import config as C


def _month_file(y, m):
    return C.CACHE / f"era5land_{y}{m:02d}.nc"


def _download(y, m, target):
    import cdsapi
    last = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
    days = [f"{d:02d}" for d in range(1, last.day + 1)]
    req = {"variable": ["total_precipitation", "2m_temperature"], "year": str(y), "month": f"{m:02d}",
           "day": days, "time": ["00:00", "06:00", "12:00", "18:00"],
           "area": [C.LAT1 + 0.05, C.LON0 - 0.05, C.LAT0 - 0.05, C.LON1 + 0.05],
           "data_format": "netcdf", "download_format": "unarchived"}
    tmp = target.with_suffix(".part")
    for k in range(4):
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-land", req, str(tmp))
            tmp.rename(target)
            return
        except Exception as e:  # CDS queues fail now and then
            err = e
            time.sleep(30 * (k + 1))
    raise RuntimeError(f"ERA5-Land {y}-{m:02d} failed: {err}")


def _load(path):
    d = xr.open_dataset(path)
    if "valid_time" in d.dims:
        d = d.rename({"valid_time": "time"})
    d = d.sortby("latitude").sel(latitude=C.lats(), longitude=C.lons(), method="nearest")
    return d


def _last_hour(path):
    with _load(path) as d:
        ok = np.isfinite(d["t2m"].values).any((1, 2))
        t = pd.DatetimeIndex(d.time.values)
        return t[ok].max() if ok.any() else None


def fetch(start, today):
    """(days, rain[days,y,x], temp[days,y,x]) for complete days from `start` onward."""
    start, today = pd.Timestamp(start), pd.Timestamp(today)
    months = pd.period_range(start.to_period("M"), today.to_period("M"), freq="M")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in months:
        f = _month_file(p.year, p.month)
        if p.to_timestamp() > today - pd.Timedelta(days=5):
            continue  # nothing published for this month yet
        next_month = (p + 1).to_timestamp()
        final = f.exists() and (_last_hour(f) or pd.Timestamp(0)) >= next_month - pd.Timedelta(hours=6)
        stale = f.exists() and time.time() - f.stat().st_mtime > 12 * 3600
        if not f.exists() or (not final and stale):
            f.unlink(missing_ok=True)
            _download(p.year, p.month, f)
        if f.exists():
            frames.append(_load(f)[["tp", "t2m"]].load())
    d = xr.concat(frames, dim="time").sortby("time")
    t = pd.DatetimeIndex(d.time.values)
    tp, t2m = d["tp"].values, d["t2m"].values
    ok = np.isfinite(t2m).reshape(len(t), -1).any(1)

    days, rain, temp = [], [], []
    for day in pd.date_range(start, today):
        hrs = [day + pd.Timedelta(hours=h) for h in (0, 6, 12, 18)]
        nxt = day + pd.Timedelta(days=1)
        idx = [t.get_indexer([h])[0] for h in hrs + [nxt]]
        if min(idx) < 0 or not all(ok[i] for i in idx):
            break  # first incomplete day ends the continuous run
        days.append(day)
        rain.append(tp[idx[-1]] * 1000.0)          # 00 UTC of the next day closes this day
        temp.append(t2m[idx[:4]].mean(0) - 273.15)
    return pd.DatetimeIndex(days), np.array(rain, "float32"), np.array(temp, "float32")
