"""Static grid: which output cells are Croatian land, and their ERA5-Land elevation.

Writes static/grid.npz. Run once (or when the region changes).
"""
import json

import numpy as np
import xarray as xr
from matplotlib.path import Path as MPath

import config as C

STATIC = C.ROOT / "static"


def build():
    lats, lons = C.lats(), C.lons()
    z = xr.open_dataset(STATIC / "era5land_z.nc")["z"].isel(time=0)
    z = z.sortby("latitude").sel(latitude=lats, longitude=lons, method="nearest")
    elev = (z.values / 9.80665).astype("float32")

    rr = xr.open_dataset(C.BACKTEST / "data" / "era5land_rr.nc")["rr"]
    land = np.isfinite(rr.isel(time=slice(0, 10)).sel(latitude=lats, longitude=lons, method="nearest")
                       .values).all(0)

    geo = json.loads((C.BACKTEST / "data" / "ne_50m_admin0.geojson").read_text())
    hrv = next(f for f in geo["features"] if f["properties"]["ADM0_A3"] == "HRV")["geometry"]
    polys = hrv["coordinates"] if hrv["type"] == "MultiPolygon" else [hrv["coordinates"]]
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    inside = np.zeros(len(pts), bool)
    for p in polys:
        inside |= MPath(np.array(p[0])).contains_points(pts)
    hr = inside.reshape(LO.shape) & land

    np.savez_compressed(STATIC / "grid.npz", lats=lats, lons=lons, elev=elev, land=land, hr=hr)
    print(f"grid {len(lats)} x {len(lons)}, land {land.sum()}, Croatia {hr.sum()}, "
          f"elev {elev[hr].min():.0f}-{elev[hr].max():.0f} m")


def load():
    return dict(np.load(STATIC / "grid.npz"))


if __name__ == "__main__":
    build()
