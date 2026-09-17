"""Daily job: stitch observed + forecast weather, score every grid cell, write out/.

Weather per day, best source first:
  rain  ERA5-Land > IMERG Late > IMERG Early (both scaled by month to ERA5-Land) > ECMWF run of that day > archived forecast
  temp  ERA5-Land > ECMWF run of that day > archived forecast > linear fill (at most 2 days)
  D..D+4 ICON-EU 00z (fallback ECMWF), D+5..D+7 ECMWF.

Everything fetched for a day that ERA5-Land does not cover yet is kept in archive/<day>/,
because ECMWF keeps only 4 days and DWD only the latest run. The archive must persist
between runs (commit it, or keep it on the server).

Usage: run_daily.py [YYYY-MM-DD]   (default: today, UTC)
"""
import json
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as C
import era5land
import forecast as F
import grid as G
import imerg

SCHEMA = 1


# ---------------------------------------------------------------- archive

def _arch(day, source):
    return C.ARCHIVE / C.GRID_TAG / f"{day:%Y-%m-%d}" / f"{source}.npz"


def save(day, source, rain=None, temp=None):
    p = _arch(day, source)
    p.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(C.lats()), len(C.lons()))
    nan = np.full(shape, np.nan, "float32")
    np.savez_compressed(p, rain=nan if rain is None else rain, temp=nan if temp is None else temp)


def load(day, source):
    p = _arch(day, source)
    if not p.exists():
        return None, None
    z = np.load(p)
    r, t = z["rain"], z["temp"]
    return (None if np.isnan(r).all() else r), (None if np.isnan(t).all() else t)


# ---------------------------------------------------------------- helpers

def imerg_factors():
    p = C.ROOT / "static" / "imerg_factors.json"
    return json.loads(p.read_text()) if p.exists() else None


def log(*a):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}]", *a, flush=True)


def collect_gap(today, gap_days):
    """Fetch and archive IMERG + ECMWF for days ERA5-Land does not cover yet."""
    for day in gap_days:
        if not _arch(day, "imerg_late").exists():
            f = imerg.fetch_day(day, "late")
            if f is not None:
                save(day, "imerg_late", rain=f)
            elif not _arch(day, "imerg_early").exists():
                f = imerg.fetch_day(day, "early")
                if f is not None:
                    save(day, "imerg_early", rain=f)
        if not _arch(day, "ecmwf_an").exists() and day >= today - pd.Timedelta(days=4):
            try:
                (rain, temp), = F.ecmwf_days(day, [day]).values()
                save(day, "ecmwf_an", rain=rain, temp=temp)
            except Exception as e:
                log(f"ECMWF run {day:%Y-%m-%d} 00z not available: {str(e)[:120]}")


def pick_rain(day, factors):
    month = day.month - 1
    for src in ("imerg_late", "imerg_early"):
        r, _ = load(day, src)
        if r is not None:
            k = factors[src.split("_")[1]][month] if factors else 1.0
            return r * k, src
    for src in ("ecmwf_an", "icon_fc", "ecmwf_fc"):
        r, _ = load(day, src)
        if r is not None:
            return r, src
    return None, None


def pick_temp(day):
    for src in ("ecmwf_an", "icon_fc", "ecmwf_fc"):
        _, t = load(day, src)
        if t is not None:
            return t, src
    return None, None


# ---------------------------------------------------------------- model

def score_cube(model, season, rain, temp, days, t_index, src, med):
    """Score 0-100 per group, target day, cell. `src` = (iy, ix) of each cell's weather cell."""
    from model import score
    iy, ix = src
    out = np.zeros((len(C.GROUPS), len(t_index), len(iy)), "uint8")
    for j, t in enumerate(t_index):
        R = rain[t - np.arange(C.RR_LAGS)][:, iy, ix].T.astype("float64")
        T = temp[t - np.arange(C.TG_LAGS)][:, iy, ix].T.astype("float64")
        s = score(model, season, R, T, days[t], med)
        for g, name in enumerate(C.GROUPS):
            out[g, j] = s[name]
    return out


# ---------------------------------------------------------------- main

def main(today):
    today = pd.Timestamp(today).normalize()
    grid = G.load()
    model = json.loads((C.OUT / "model.json").read_text())
    first = today - pd.Timedelta(days=C.RR_LAGS - 1)
    last = today + pd.Timedelta(days=C.FORECAST_DAYS - 1)
    days = pd.date_range(first, last)
    shape = (len(days), len(C.lats()), len(C.lons()))
    rain, temp = np.full(shape, np.nan, "float32"), np.full(shape, np.nan, "float32")
    src_rain, src_temp = [None] * len(days), [None] * len(days)

    log("ERA5-Land")
    e_days, e_rain, e_temp = era5land.fetch(first, today)
    for k, d in enumerate(e_days):
        i = days.get_loc(d)
        rain[i], temp[i], src_rain[i], src_temp[i] = e_rain[k], e_temp[k], "era5land", "era5land"
    last_obs = e_days[-1] if len(e_days) else first - pd.Timedelta(days=1)
    gap = pd.date_range(last_obs + pd.Timedelta(days=1), today - pd.Timedelta(days=1))
    log(f"ERA5-Land to {last_obs:%Y-%m-%d}; gap {len(gap)} days")

    log("IMERG + ECMWF for the gap")
    collect_gap(today, gap)

    log("ICON-EU")
    try:
        for d, (r, t) in F.icon_eu(today).items():
            save(d, "icon_fc", rain=r, temp=t)
    except Exception as e:
        log(f"ICON-EU failed, ECMWF only: {str(e)[:160]}")

    log("ECMWF forecast")
    run = F.ecmwf_latest_run()
    fc_days = pd.date_range(today, last)
    for d, (r, t) in F.ecmwf_days(run, fc_days).items():
        save(d, "ecmwf_fc", rain=r, temp=t)
    log(f"ECMWF run {run:%Y-%m-%d %H}z")

    factors = imerg_factors()
    for i, d in enumerate(days):
        if src_rain[i] == "era5land":
            continue
        if d < today:
            r, sr = pick_rain(d, factors)
            t, st = pick_temp(d)
        else:
            r, sr, t, st = None, None, None, None
            for src in ("icon_fc", "ecmwf_fc"):
                rr_, tt_ = load(d, src)
                if rr_ is not None and r is None:
                    r, sr = rr_, src
                if tt_ is not None and t is None:
                    t, st = tt_, src
        if r is not None:
            rain[i], src_rain[i] = r, sr
        if t is not None:
            temp[i], src_temp[i] = t, st

    # short temperature holes (bootstrap: ECMWF keeps 4 days, ERA5-Land lags 6)
    for i in range(len(days)):
        if src_temp[i] is None:
            a = max((k for k in range(i) if src_temp[k] not in (None, "filled")), default=None)
            b = min((k for k in range(i + 1, len(days)) if src_temp[k] not in (None, "filled")), default=None)
            if a is None or b is None or b - a > 3:
                raise SystemExit(f"no temperature for {days[i]:%Y-%m-%d}")
            w = (i - a) / (b - a)
            temp[i], src_temp[i] = (1 - w) * temp[a] + w * temp[b], "filled"
    missing = [f"{d:%Y-%m-%d}" for d, s in zip(days, src_rain) if s is None]
    if missing:
        raise SystemExit(f"no rain for {missing}")

    hr = grid["hr"]
    iy, ix = np.where(hr)
    siy, six = grid["src_iy"][iy, ix], grid["src_ix"][iy, ix]
    if not (np.isfinite(rain[:, siy, six]).all() and np.isfinite(temp[:, siy, six]).all()):
        raise SystemExit("NaN in the weather cells the grid uses")

    season = json.loads((C.ROOT / "static" / "season.json").read_text())
    med = grid["med"][iy, ix]
    t_index = [days.get_loc(today + pd.Timedelta(days=k)) for k in range(C.FORECAST_DAYS)]
    cube = score_cube(model, season, rain, temp, days, t_index, (siy, six), med)

    C.OUT.mkdir(exist_ok=True)
    name = f"forecast-{today:%Y-%m-%d}.bin"
    (C.OUT / name).write_bytes(cube.tobytes())
    (C.OUT / "grid.json").write_text(json.dumps({
        "schemaVersion": SCHEMA, "lat0": C.LAT0, "lon0": C.LON0, "step": C.STEP,
        "ny": int(hr.shape[0]), "nx": int(hr.shape[1]),
        "cells": [[int(a), int(b)] for a, b in zip(iy, ix)],
        "elev": [int(round(float(v))) for v in grid["elev"][siy, six]],
        "forestPct": [int(round(float(v) * 100)) for v in grid["tree"][iy, ix]],
        "mediterranean": [int(v) for v in med],
        "countries": [C.COUNTRY_CODES[str(c)] for c in grid["countries"]],
        "country": [int(v) for v in grid["country"][iy, ix]]}))
    meta = {
        "schemaVersion": SCHEMA,
        "dataVersion": int(f"{today:%Y%m%d}"),
        "date": f"{today:%Y-%m-%d}",
        "generatedUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file": name,
        "layout": "uint8 score 0-100 (percentile x season weight), [group][day][cell], cells as in grid.json",
        "groups": C.GROUPS,
        "days": [f"{today + pd.Timedelta(days=k):%Y-%m-%d}" for k in range(C.FORECAST_DAYS)],
        "ecmwfRun": f"{run:%Y-%m-%dT%HZ}",
        "sources": {f"{d:%Y-%m-%d}": [src_rain[i], src_temp[i]] for i, d in enumerate(days)
                    if d > last_obs - pd.Timedelta(days=1)},
        "imergScaled": factors is not None,
        "attribution": "Contains modified Copernicus Climate Change Service information (ERA5-Land); "
                       "NASA GPM IMERG; DWD ICON-EU (CC BY 4.0); ECMWF open data (CC BY 4.0).",
    }
    (C.OUT / "latest.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {name}: {cube.nbytes} bytes; grid mean per group/day:")
    for g, n in enumerate(C.GROUPS):
        log(f"  {n:10s}", " ".join(f"{cube[g, j].mean():5.1f}" for j in range(C.FORECAST_DAYS)))
    return meta


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        main(arg)
    except BaseException:
        traceback.print_exc()
        raise
