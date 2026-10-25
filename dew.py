"""Night dewpoint depression (fog/dew) for days ERA5-Land does not cover.

dd = t2m - d2m (deg C), mean of 00 and 06 UTC: small at night and dawn = saturated air,
fog or dew. The model was fitted on ERA5-Land dd (backtest/build_dew.py). ERA5-Land has
stalled since 2026-09-12, so past days come from ERA5 (0.25 deg, ~5 days late), and the
last days and the forecast from ECMWF IFS (forecast.ecmwf_days). Depression, unlike the
dewpoint itself, needs no elevation correction: t2m and d2m shift together.
"""
import time

import numpy as np
import pandas as pd
import xarray as xr

import config as C


def _month_file(y, m):
    return C.CACHE / f"era5_dd_{C.GRID_TAG}_{y}{m:02d}.nc"


def _download(y, m, target):
    import cdsapi
    last = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
    req = {"product_type": ["reanalysis"], "variable": ["2m_temperature", "2m_dewpoint_temperature"],
           "year": str(y), "month": f"{m:02d}", "day": [f"{d:02d}" for d in range(1, last.day + 1)],
           "time": ["00:00", "06:00"],
           "area": C.cds_area(0.3),
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
    raise RuntimeError(f"ERA5 dew {y}-{m:02d} failed: {err}")


def _load(path):
    d = xr.open_dataset(path)
    if "valid_time" in d.dims:
        d = d.rename({"valid_time": "time"})
    return d.sortby("latitude")


def era5(days, today):
    """{day: dd[y, x]} for those of `days` ERA5 already covers (both 00 and 06 UTC)."""
    days = pd.DatetimeIndex(days)
    if not len(days):
        return {}
    newest = min(days[-1] + pd.Timedelta(hours=6), pd.Timestamp(today) - pd.Timedelta(days=6))
    if days[0] > newest:
        return {}  # nothing ERA5 can have yet; don't sit in the CDS queue for it (see era5.py)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in pd.period_range(days[0].to_period("M"), days[-1].to_period("M"), freq="M"):
        if p.to_timestamp() > newest:
            continue
        f = _month_file(p.year, p.month)
        if f.exists():
            with _load(f) as d:
                ok = np.isfinite(d["d2m"].values).reshape(d.sizes["time"], -1).any(1)
                have = pd.DatetimeIndex(d.time.values)[ok].max() if ok.any() else pd.Timestamp(0)
            final = have >= (p + 1).to_timestamp() - pd.Timedelta(hours=18)
            if not final and have < newest:
                f.unlink()
        if not f.exists():
            _download(p.year, p.month, f)
        with _load(f) as d:
            frames.append((d["t2m"] - d["d2m"]).load())
    if not frames:
        return {}
    dd = xr.concat(frames, dim="time").sortby("time").interp(latitude=C.lats(), longitude=C.lons())
    t = pd.DatetimeIndex(dd.time.values)
    vals = dd.values
    out = {}
    for day in days:
        idx = t.get_indexer([day, day + pd.Timedelta(hours=6)])
        if (idx < 0).any() or not np.isfinite(vals[idx]).any(axis=(1, 2)).all():
            continue
        out[day] = vals[idx].mean(0).astype("float32")
    return out
