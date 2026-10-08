"""Daily job: stitch observed + forecast weather, score every grid cell, write out/.

Weather per day, best source first:
  rain  ERA5-Land > ERA5 > IMERG Late > IMERG Early (both scaled by month to ERA5-Land) > ECMWF run of that day > archived forecast
  temp  ERA5-Land > ECMWF run of that day > archived forecast > linear fill (at most 2 days)
  D..D+4 the newest complete ICON-EU 00z/12z run (fallback ECMWF), the rest ECMWF.
  dd    (night dewpoint depression, fog/dew) ERA5 > ECMWF run of that day > ECMWF forecast >
        linear fill (at most 3 days); still missing = the model treats the dew features as average.

Everything fetched for a day that ERA5-Land does not cover yet is kept in archive/<day>/,
because ECMWF keeps only 4 days and DWD only the latest run. The archive must persist
between runs (commit it, or keep it on the server).

Usage: run_daily.py [YYYY-MM-DD]   (default: today in Croatia, see TZ below)
"""
import json
import sys
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as C
import dew
import era5
import era5land
import forecast as F
import grid as G
import imerg

SCHEMA = 1

# "Today" is the day the app's users are in, not the UTC day: a job that runs at 23z has
# to forecast the date Croatia has already turned to, or a picker leaving at 4 am gets a
# forecast that starts yesterday.
TZ = ZoneInfo("Europe/Zagreb")


# ---------------------------------------------------------------- archive

def _arch(day, source):
    return C.ARCHIVE / C.GRID_TAG / f"{day:%Y-%m-%d}" / f"{source}.npz"


def save(day, source, rain=None, temp=None, dd=None):
    p = _arch(day, source)
    p.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(C.lats()), len(C.lons()))
    nan = np.full(shape, np.nan, "float32")
    arr = {"rain": rain, "temp": temp, "dd": dd}
    np.savez_compressed(p, **{k: nan if v is None else v for k, v in arr.items()})


def load(day, source):
    p = _arch(day, source)
    if not p.exists():
        return None, None
    z = np.load(p)
    r, t = z["rain"], z["temp"]
    return (None if np.isnan(r).all() else r), (None if np.isnan(t).all() else t)


def load_dd(day, source):
    p = _arch(day, source)
    if not p.exists():
        return None
    z = np.load(p)
    # archives written before 2026-10-09 have no dd
    return None if "dd" not in z.files or np.isnan(z["dd"]).all() else z["dd"]


# ---------------------------------------------------------------- helpers

def imerg_factors():
    p = C.ROOT / "static" / "imerg_factors.json"
    return json.loads(p.read_text()) if p.exists() else None


def log(*a):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}]", *a, flush=True)


def collect_gap(today, gap_days):
    """Fetch and archive IMERG + ECMWF for days ERA5-Land does not cover yet."""
    imerg_up = True
    for day in gap_days:
        if imerg_up and not _arch(day, "imerg_late").exists():
            try:
                f = imerg.fetch_day(day, "late")
                if f is not None:
                    save(day, "imerg_late", rain=f)
                elif not _arch(day, "imerg_early").exists():
                    f = imerg.fetch_day(day, "early")
                    if f is not None:
                        save(day, "imerg_early", rain=f)
            except imerg.Unavailable as e:
                # Every gap day would hit the same dead server, so stop asking. pick_rain
                # falls back to ECMWF/ICON and latest.json["sources"] records which it used.
                log(f"IMERG unavailable, falling back to ECMWF/ICON rain: {str(e)[:160]}")
                imerg_up = False
        # also re-fetch days archived before the dew fields existed, while ECMWF still has them
        if load_dd(day, "ecmwf_an") is None and day >= today - pd.Timedelta(days=4):
            try:
                (rain, temp, dd), = F.ecmwf_days(day, [day]).values()
                save(day, "ecmwf_an", rain=rain, temp=temp, dd=dd)
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


def fill_dd(days, today):
    """Night dewpoint depression for `days`: ERA5, then the archived ECMWF fields, then short fills."""
    from model import DD_LAGS
    shape = (len(days), len(C.lats()), len(C.lons()))
    dd, src = np.full(shape, np.nan, "float32"), [None] * len(days)
    need = [d for d in days if d >= today - pd.Timedelta(days=DD_LAGS + 1)]
    try:
        e5 = dew.era5([d for d in need if d < today], today)
    except Exception as e:
        e5 = {}
        log(f"ERA5 dew failed, ECMWF only: {str(e)[:160]}")
    for d in need:
        i = days.get_loc(d)
        if d in e5:
            dd[i], src[i] = e5[d], "era5"
            continue
        for s in (("ecmwf_an", "ecmwf_fc") if d < today else ("ecmwf_fc",)):
            x = load_dd(d, s)
            if x is not None:
                dd[i], src[i] = x, s
                break
    for i in range(len(days)):
        if src[i] is None and days[i] in need:
            a = max((k for k in range(i) if src[k] not in (None, "filled")), default=None)
            b = min((k for k in range(i + 1, len(days)) if src[k] not in (None, "filled")), default=None)
            if a is not None and b is not None and b - a <= 4:
                w = (i - a) / (b - a)
                dd[i], src[i] = (1 - w) * dd[a] + w * dd[b], "filled"
    log("dew sources: " + ", ".join(f"{d:%m-%d} {s}" for d, s in zip(days, src) if d in need))
    return dd, src


# ---------------------------------------------------------------- model

def score_cube(model, season, rain, temp, dd, days, t_index, src, med):
    """Score 0-100 per group, target day, cell. `src` = (iy, ix) of each cell's weather cell."""
    from model import DD_LAGS, score
    iy, ix = src
    out = np.zeros((len(C.GROUPS), len(t_index), len(iy)), "uint8")
    for j, t in enumerate(t_index):
        R = rain[t - np.arange(C.RR_LAGS)][:, iy, ix].T.astype("float64")
        T = temp[t - np.arange(C.TG_LAGS)][:, iy, ix].T.astype("float64")
        DD = dd[t - np.arange(DD_LAGS)][:, iy, ix].T.astype("float64")
        s = score(model, season, R, T, days[t], med, DD)
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
    if len(gap) in (11, 25):
        # ERA5-Land normally lags ~6 days. From 2026-09-12 CDS stopped extending it for weeks
        # and nobody noticed; ERA5 below covers the rain, but someone should look.
        try:
            import notify
            notify.send(f"Gljivar: ERA5-Land kasni {len(gap)} dana (zadnji dan {last_obs:%Y-%m-%d}). "
                        "Kiša ide iz ERA5, provjeri CDS.")
        except Exception as e:
            log(f"notify failed: {e}")

    log("ERA5 rain for the gap")
    try:
        e5_rain = era5.fetch(gap, today)
        log(f"ERA5 covers {len(e5_rain)} of {len(gap)} gap days")
    except Exception as e:
        e5_rain = {}
        log(f"ERA5 failed, IMERG/ECMWF rain for the gap: {str(e)[:160]}")

    log("IMERG + ECMWF for the gap")
    collect_gap(today, gap)

    log("ICON-EU")
    icon_run = None
    try:
        # The newest complete run, not today's 00z: before ~03z DWD has only yesterday's 12z,
        # and the early-morning job must not wait for a run that is still being computed.
        icon_run = F.icon_latest_run()
        for d, (r, t) in F.icon_eu(icon_run).items():
            save(d, "icon_fc", rain=r, temp=t)
        log(f"ICON-EU run {icon_run:%Y-%m-%d %H}z")
    except Exception as e:
        log(f"ICON-EU failed, ECMWF only: {str(e)[:160]}")

    log("ECMWF forecast")
    run = F.ecmwf_latest_run()
    # A late-evening run for today's date finds today's 12z run already out, and that one
    # cannot give today's 00-24 total. Today's 00z run still reaches the last day.
    run = min(run, today)
    fc_days = pd.date_range(today, last)
    for d, (r, t, x) in F.ecmwf_days(run, fc_days).items():
        save(d, "ecmwf_fc", rain=r, temp=t, dd=x)
    log(f"ECMWF run {run:%Y-%m-%d %H}z")

    factors = imerg_factors()
    for i, d in enumerate(days):
        if src_rain[i] == "era5land":
            continue
        if d < today:
            r, sr = (e5_rain[d], "era5") if d in e5_rain else pick_rain(d, factors)
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
    dd, src_dd = fill_dd(days, today)
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
    cube = score_cube(model, season, rain, temp, dd, days, t_index, (siy, six), med)

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
    now = datetime.now(timezone.utc)
    meta = {
        "schemaVersion": SCHEMA,
        # Generation hour, not the forecast date: the job runs twice a day, so two runs for
        # the same date must still compare as newer and older in the app.
        "dataVersion": int(f"{now:%Y%m%d%H}"),
        "date": f"{today:%Y-%m-%d}",
        "generatedUtc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "file": name,
        "layout": "uint8 score 0-100 (percentile x season weight), [group][day][cell], cells as in grid.json",
        "groups": C.GROUPS,
        "days": [f"{today + pd.Timedelta(days=k):%Y-%m-%d}" for k in range(C.FORECAST_DAYS)],
        "ecmwfRun": f"{run:%Y-%m-%dT%HZ}",
        "iconRun": None if icon_run is None else f"{icon_run:%Y-%m-%dT%HZ}",
        "sources": {f"{d:%Y-%m-%d}": [src_rain[i], src_temp[i], src_dd[i]] for i, d in enumerate(days)
                    if d > last_obs - pd.Timedelta(days=1)},
        "imergScaled": factors is not None,
        "era5landLagDays": len(gap),
        "attribution": "Contains modified Copernicus Climate Change Service information (ERA5-Land); "
                       "NASA GPM IMERG; DWD ICON-EU (CC BY 4.0); ECMWF open data (CC BY 4.0).",
    }
    (C.OUT / "latest.json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {name}: {cube.nbytes} bytes; grid mean per group/day:")
    for g, n in enumerate(C.GROUPS):
        log(f"  {n:10s}", " ".join(f"{cube[g, j].mean():5.1f}" for j in range(C.FORECAST_DAYS)))
    return meta


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else datetime.now(TZ).strftime("%Y-%m-%d")
    try:
        main(arg)
    except BaseException:
        traceback.print_exc()
        raise
