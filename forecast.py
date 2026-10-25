"""Forecast and short-range analysis fields on the output grid.

ICON-EU (DWD, CC BY 4.0): 0.0625 deg, any 00z/12z run, every day the run covers whole
(it stops at +120 h, so a 00z run gives D..D+4 and a 12z run D+1..D+4).
ECMWF IFS open data (CC BY 4.0): 0.25 deg, any 00z/12z run, used for D+5..D+7 and, from the
run started at 00z of a past day, as that day's temperature (the "analysis" the job archives).

Every field matches ERA5-Land's definitions:
  rain  = total 00-24 UTC (mm)
  temp  = mean of 00/06/12/18 UTC (deg C), shifted to ERA5-Land elevation at 6.5 K/km
  dd    = night dewpoint depression, t2m - d2m at 00 and 06 UTC (deg C), ECMWF only (see dew.py)
"""
import bz2

import numpy as np
import pandas as pd
import requests
import xarray as xr

import config as C
import grid as G

LAPSE = -0.0065
ICON = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
ICON_MAX_STEP = 120


def _open(path, **filt):
    kw = {"indexpath": ""}
    if filt:
        kw["filter_by_keys"] = filt
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=kw)


def _regrid(da):
    da = da.sortby("latitude")
    if float(da.longitude.max()) > 180:
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
    box = da.sel(latitude=slice(C.LAT0 - 0.5, C.LAT1 + 0.5), longitude=slice(C.LON0 - 0.5, C.LON1 + 0.5))
    return box.interp(latitude=C.lats(), longitude=C.lons(), method="linear").values.astype("float32")


# ---------------------------------------------------------------- ICON-EU

def _icon_name(var, run, step, invariant=False):
    kind = "time-invariant" if invariant else "single-level"
    stamp = f"{run:%Y%m%d%H}"
    tail = f"{stamp}_{var.upper()}" if invariant else f"{stamp}_{step:03d}_{var.upper()}"
    return f"icon-eu_europe_regular-lat-lon_{kind}_{tail}.grib2"


def _icon_file(var, run, step, invariant=False):
    run = pd.Timestamp(run)
    name = _icon_name(var, run, step, invariant)
    target = C.CACHE / name
    if not target.exists():
        r = requests.get(f"{ICON}/{run:%H}/{var}/{name}.bz2", timeout=120)
        if r.status_code == 404:
            raise FileNotFoundError(name)
        r.raise_for_status()
        C.CACHE.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bz2.decompress(r.content))
    return target


def icon_latest_run(now=None):
    """Newest ICON-EU 00z/12z run whose +120 h fields are already published.

    DWD needs about three hours to finish a run, so before 03z the newest complete one is
    yesterday's 12z. Only the last runs stay on the open-data server, so we look back a day.
    """
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow().tz_localize(None)
    base = pd.Timestamp(now).normalize()
    cand = [base + pd.Timedelta(hours=h) for h in (12, 0)] + \
           [base - pd.Timedelta(days=1) + pd.Timedelta(hours=h) for h in (12, 0)]
    for run in cand:
        if run > now:
            continue
        name = _icon_name("tot_prec", run, ICON_MAX_STEP)
        if requests.head(f"{ICON}/{run:%H}/tot_prec/{name}.bz2", timeout=60).status_code == 200:
            return run
    raise FileNotFoundError("no complete ICON-EU run on opendata.dwd.de")


def icon_eu(run):
    """{day: (rain, temp)} for every whole day the given ICON-EU 00z/12z run covers."""
    run = pd.Timestamp(run)
    elev = G.load()["elev"]
    with _open(_icon_file("hsurf", run, 0, invariant=True)) as d:
        z_model = _regrid(d[list(d.data_vars)[0]])

    def tot_prec(step):
        # tot_prec accumulates from the run start; step 0 is the zero it starts from.
        if step == 0:
            return 0.0
        with _open(_icon_file("tot_prec", run, step)) as d:
            return _regrid(d["tp"])

    out = {}
    for day in pd.date_range(run.normalize(), run.normalize() + pd.Timedelta(days=5)):
        h0 = int((day - run) / pd.Timedelta(hours=1))
        if h0 < 0 or h0 + 24 > ICON_MAX_STEP:
            continue
        rain = np.maximum(tot_prec(h0 + 24) - tot_prec(h0), 0.0)
        temps = []
        for h in (0, 6, 12, 18):
            with _open(_icon_file("t_2m", run, h0 + h)) as d:
                temps.append(_regrid(d["t2m"]))
        temp = np.mean(temps, 0) - 273.15 + LAPSE * (elev - z_model)
        out[day] = (rain.astype("float32"), temp.astype("float32"))
    return out


# ---------------------------------------------------------------- ECMWF IFS

def _ecmwf(run, steps, param, target):
    """data.ecmwf.int keeps only ~4 days of runs; the AWS mirror keeps them for years. A named
    run goes to AWS first: since 2026-10-20 data.ecmwf.int answers "503 Slow Down" and the
    client retries every 120 s up to 500 times, which ran every job into its timeout.
    data.ecmwf.int stays the fallback for a run AWS has not mirrored yet."""
    from ecmwf.opendata import Client
    kw = dict(type="fc", step=steps, param=[param], target=str(target))
    if run is None:
        return Client(source="ecmwf", model="ifs", resol="0p25").retrieve(**kw)
    kw.update(date=run.strftime("%Y%m%d"), time=run.hour)
    try:
        return Client(source="aws", model="ifs", resol="0p25").retrieve(**kw)
    except Exception:
        return Client(source="ecmwf", model="ifs", resol="0p25").retrieve(**kw)


def ecmwf_latest_run():
    """Newest 00z/12z run that reaches +192 h (06z/18z runs stop at +144 h)."""
    from ecmwf.opendata import Client
    c = Client(source="ecmwf", model="ifs", resol="0p25")
    t = pd.Timestamp(c.latest(type="fc", step=192, param="tp"))
    return t


def ecmwf_days(run, days):
    """{day: (rain, temp, dd)} for the given days from one ECMWF run (00z or 12z)."""
    run = pd.Timestamp(run)
    elev = G.load()["elev"]
    tag = f"{run:%Y%m%d%H}"
    zf = C.CACHE / f"ecmwf_{tag}_z.grib2"
    if not zf.exists():
        _ecmwf(run, [0], "z", zf)
    with _open(zf, typeOfLevel="surface") as d:
        z_model = _regrid(d["z"]) / 9.80665

    days = [pd.Timestamp(x).normalize() for x in days]
    hours = lambda t: int((t - run) / pd.Timedelta(hours=1))  # noqa: E731
    tp_steps = sorted({hours(x + pd.Timedelta(days=j)) for x in days for j in (0, 1)})
    t_steps = sorted({hours(x + pd.Timedelta(hours=h)) for x in days for h in (0, 6, 12, 18)})
    if min(tp_steps + t_steps) < 0:
        raise ValueError(f"run {run} starts after the first requested day")

    span = f"{min(t_steps)}-{max(tp_steps)}"
    tpf, tf = C.CACHE / f"ecmwf_{tag}_{span}_tp.grib2", C.CACHE / f"ecmwf_{tag}_{span}_2t.grib2"
    df = C.CACHE / f"ecmwf_{tag}_{span}_2d.grib2"
    d_steps = sorted({hours(x + pd.Timedelta(hours=h)) for x in days for h in (0, 6)})
    if not tpf.exists():
        _ecmwf(run, tp_steps, "tp", tpf)
    if not tf.exists():
        _ecmwf(run, t_steps, "2t", tf)
    if not df.exists():
        _ecmwf(run, d_steps, "2d", df)

    with _open(tpf) as d:
        tp = d["tp"]
        tp = tp if "step" in tp.dims else tp.expand_dims("step")
        tp_by_h = {int(s / np.timedelta64(1, "h")): _regrid(tp.sel(step=s)) * 1000.0 for s in tp.step.values}
    with _open(tf) as d:
        t2 = d["t2m"]
        t2 = t2 if "step" in t2.dims else t2.expand_dims("step")
        t_by_h = {int(s / np.timedelta64(1, "h")): _regrid(t2.sel(step=s)) for s in t2.step.values}
    with _open(df) as d:
        d2 = d["d2m"]
        d2 = d2 if "step" in d2.dims else d2.expand_dims("step")
        d_by_h = {int(s / np.timedelta64(1, "h")): _regrid(d2.sel(step=s)) for s in d2.step.values}

    out = {}
    for x in days:
        a, b = hours(x), hours(x + pd.Timedelta(days=1))
        rain = np.maximum(tp_by_h[b] - (tp_by_h[a] if a > 0 else 0.0), 0.0)
        temp = np.mean([t_by_h[hours(x + pd.Timedelta(hours=h))] for h in (0, 6, 12, 18)], 0)
        temp = temp - 273.15 + LAPSE * (elev - z_model)
        dd = np.mean([t_by_h[hours(x + pd.Timedelta(hours=h))] - d_by_h[hours(x + pd.Timedelta(hours=h))]
                      for h in (0, 6)], 0)
        out[x] = (rain.astype("float32"), temp.astype("float32"), dd.astype("float32"))
    return out
