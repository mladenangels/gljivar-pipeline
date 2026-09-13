"""Forecast and short-range analysis fields on the output grid.

ICON-EU (DWD, CC BY 4.0): 0.0625 deg, 00z run, days D..D+4 (the run stops at +120 h).
ECMWF IFS open data (CC BY 4.0): 0.25 deg, any 00z/12z run, used for D+5..D+7 and, from the
run started at 00z of a past day, as that day's temperature (the "analysis" the job archives).

Every field matches ERA5-Land's definitions:
  rain  = total 00-24 UTC (mm)
  temp  = mean of 00/06/12/18 UTC (deg C), shifted to ERA5-Land elevation at 6.5 K/km
"""
import bz2

import numpy as np
import pandas as pd
import requests
import xarray as xr

import config as C
import grid as G

LAPSE = -0.0065
ICON = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00"


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

def _icon_file(var, run, step, invariant=False):
    kind = "time-invariant" if invariant else "single-level"
    tail = f"{run:%Y%m%d}00_{var.upper()}" if invariant else f"{run:%Y%m%d}00_{step:03d}_{var.upper()}"
    name = f"icon-eu_europe_regular-lat-lon_{kind}_{tail}.grib2"
    target = C.CACHE / name
    if not target.exists():
        r = requests.get(f"{ICON}/{var}/{name}.bz2", timeout=120)
        if r.status_code == 404:
            raise FileNotFoundError(name)
        r.raise_for_status()
        C.CACHE.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bz2.decompress(r.content))
    return target


def icon_eu(run):
    """{day: (rain, temp)} for run day D..D+4 from the 00z ICON-EU run of `run`."""
    run = pd.Timestamp(run).normalize()
    elev = G.load()["elev"]
    with _open(_icon_file("hsurf", run, 0, invariant=True)) as d:
        z_model = _regrid(d[list(d.data_vars)[0]])
    out = {}
    tp_prev = 0.0
    for k in range(5):
        with _open(_icon_file("tot_prec", run, 24 * (k + 1))) as d:
            tp = _regrid(d["tp"])
        rain = np.maximum(tp - tp_prev, 0.0)
        tp_prev = tp
        temps = []
        for h in (0, 6, 12, 18):
            with _open(_icon_file("t_2m", run, 24 * k + h)) as d:
                temps.append(_regrid(d["t2m"]))
        temp = np.mean(temps, 0) - 273.15 + LAPSE * (elev - z_model)
        out[run + pd.Timedelta(days=k)] = (rain.astype("float32"), temp.astype("float32"))
    return out


# ---------------------------------------------------------------- ECMWF IFS

def _ecmwf(run, steps, param, target):
    from ecmwf.opendata import Client
    c = Client(source="ecmwf", model="ifs", resol="0p25")
    kw = dict(type="fc", step=steps, param=[param], target=str(target))
    if run is not None:
        kw.update(date=run.strftime("%Y%m%d"), time=run.hour)
    return c.retrieve(**kw)


def ecmwf_latest_run():
    """Newest 00z/12z run that reaches +192 h (06z/18z runs stop at +144 h)."""
    from ecmwf.opendata import Client
    c = Client(source="ecmwf", model="ifs", resol="0p25")
    t = pd.Timestamp(c.latest(type="fc", step=192, param="tp"))
    return t


def ecmwf_days(run, days):
    """{day: (rain, temp)} for the given days from one ECMWF run (00z or 12z)."""
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
    if not tpf.exists():
        _ecmwf(run, tp_steps, "tp", tpf)
    if not tf.exists():
        _ecmwf(run, t_steps, "2t", tf)

    with _open(tpf) as d:
        tp = d["tp"]
        tp = tp if "step" in tp.dims else tp.expand_dims("step")
        tp_by_h = {int(s / np.timedelta64(1, "h")): _regrid(tp.sel(step=s)) * 1000.0 for s in tp.step.values}
    with _open(tf) as d:
        t2 = d["t2m"]
        t2 = t2 if "step" in t2.dims else t2.expand_dims("step")
        t_by_h = {int(s / np.timedelta64(1, "h")): _regrid(t2.sel(step=s)) for s in t2.step.values}

    out = {}
    for x in days:
        a, b = hours(x), hours(x + pd.Timedelta(days=1))
        rain = np.maximum(tp_by_h[b] - (tp_by_h[a] if a > 0 else 0.0), 0.0)
        temp = np.mean([t_by_h[hours(x + pd.Timedelta(hours=h))] for h in (0, 6, 12, 18)], 0)
        temp = temp - 273.15 + LAPSE * (elev - z_model)
        out[x] = (rain.astype("float32"), temp.astype("float32"))
    return out
