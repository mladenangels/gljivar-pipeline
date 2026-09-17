"""Static grid: Croatian and Slovenian cells, where their weather comes from, forest and zone.

A cell belongs to the grid if any part of it is land of one of config.COUNTRIES (Natural
Earth 10m), so coast and islands are in. ERA5-Land has no data for cells that are mostly
sea; those cells take their weather from the nearest ERA5-Land land cell (src_iy, src_ix).

Forest: share of ESA WorldCover 2021 "tree cover" (class 10) in the cell, CC BY 4.0.
Zone: "med" where the source cell's mean January temperature (ERA5-Land 2011-2025) is
>= 4 deg C - the warm coast and Istria, where the Austrian-trained model is least reliable.

Writes static/grid.npz. Build-time only (needs ../backtest data and rasterio/shapely/pyshp).
"""
import numpy as np
import pandas as pd
import xarray as xr

import config as C

STATIC = C.ROOT / "static"
SUB = 10          # sub-samples per cell side for the land share
MED_JAN_C = 4.0
MAX_BORROW = 3    # cells
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")


def land_share(lats, lons):
    """Share of each cell inside each country of config.COUNTRIES, as {code: array}."""
    import shapefile
    import shapely
    from shapely.geometry import shape
    r = shapefile.Reader(str(C.CACHE / "ne10" / "ne_10m_admin_0_countries.shp"))
    recs = r.records()
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    out = {}
    for code in C.COUNTRIES:
        i = next(k for k, rec in enumerate(recs) if rec["ADM0_A3"] == code)
        geom = shape(r.shape(i).__geo_interface__)
        shapely.prepare(geom)
        share = np.zeros((len(lats), len(lons)), "float32")
        for y, la in enumerate(lats):
            for x, lo in enumerate(lons):
                px, py = np.meshgrid(lo + off * C.STEP, la + off * C.STEP)
                share[y, x] = shapely.contains_xy(geom, px.ravel(), py.ravel()).mean()
        out[code] = share
        print(f"land share {code}: {(share > 0).sum()} cells", flush=True)
    return out


def tree_share(lats, lons):
    import rasterio
    from rasterio.enums import Resampling
    dec = 12                                   # 10 m pixels -> ~120 m
    px_per_deg = 12000 // dec
    tree = np.zeros((len(lats), len(lons)), "float32")
    half = C.STEP / 2
    tiles = {(la, lo) for la in range(int(np.floor((C.LAT0 - half) / 3) * 3), int(np.ceil(C.LAT1 + half)), 3)
             for lo in range(int(np.floor((C.LON0 - half) / 3) * 3), int(np.ceil(C.LON1 + half)), 3)}
    for la0, lo0 in sorted(tiles):
        name = f"N{la0:02d}E{lo0:03d}"
        with rasterio.open(WORLDCOVER.format(tile=name)) as s:
            a = s.read(1, out_shape=(3 * px_per_deg, 3 * px_per_deg), resampling=Resampling.nearest)
        print("WorldCover", name, flush=True)
        for y, la in enumerate(lats):
            r0 = int(round((la0 + 3 - (la + C.STEP / 2)) * px_per_deg))
            r1 = r0 + int(round(C.STEP * px_per_deg))
            if r1 <= 0 or r0 >= a.shape[0]:
                continue
            for x, lo in enumerate(lons):
                c0 = int(round(((lo - C.STEP / 2) - lo0) * px_per_deg))
                c1 = c0 + int(round(C.STEP * px_per_deg))
                if c1 <= 0 or c0 >= a.shape[1]:
                    continue
                block = a[max(r0, 0):min(r1, a.shape[0]), max(c0, 0):min(c1, a.shape[1])]
                # cells split across tiles: accumulate counts weighted by the part's size
                tree[y, x] += (block == 10).sum() / (C.STEP * px_per_deg) ** 2
    return tree


def cached_tree_share(lats, lons):
    """tree_share() is ~1 GB of WorldCover reads, so keep the result per grid tag."""
    p = STATIC / f"tree_{C.GRID_TAG}.npz"
    if p.exists():
        z = np.load(p)
        if z["tree"].shape == (len(lats), len(lons)):
            print(f"tree share from {p.name}", flush=True)
            return z["tree"]
    tree = tree_share(lats, lons)
    np.savez_compressed(p, tree=tree)
    return tree


def build():
    lats, lons = C.lats(), C.lons()
    z = xr.open_dataset(STATIC / "era5land_z.nc")["z"].isel(time=0)
    z = z.sortby("latitude").sel(latitude=lats, longitude=lons, method="nearest")
    elev = (z.values / 9.80665).astype("float32")

    tg = xr.open_dataset(C.BACKTEST / "data" / "era5land_tg.nc")["tg"]
    tg = tg.sel(latitude=lats, longitude=lons, method="nearest")
    land = np.isfinite(tg.isel(time=slice(0, 10)).values).all(0)
    t = pd.DatetimeIndex(tg.time.values)
    jan = tg.values[t.month == 1].mean(0)

    shares = land_share(lats, lons)
    hr = np.zeros((len(lats), len(lons)), bool)
    for a in shares.values():
        hr |= a > 0
    # Each cell is labelled with the country that covers most of it (index into C.COUNTRIES).
    stack = np.stack([shares[c] for c in C.COUNTRIES])
    country = np.where(hr, stack.argmax(0).astype("int8"), np.int8(-1))

    ly, lx = np.where(land)
    src_iy = np.full(hr.shape, -1, "int16")
    src_ix = np.full(hr.shape, -1, "int16")
    for y, x in zip(*np.where(hr)):
        if land[y, x]:
            src_iy[y, x], src_ix[y, x] = y, x
            continue
        d = np.hypot(ly - y, (lx - x) * np.cos(np.radians(lats[y])))
        k = d.argmin()
        if d[k] <= MAX_BORROW:
            src_iy[y, x], src_ix[y, x] = ly[k], lx[k]
    dropped = hr & (src_iy < 0)
    hr &= src_iy >= 0

    med = np.zeros(hr.shape, bool)
    med[hr] = jan[src_iy[hr], src_ix[hr]] >= MED_JAN_C
    tree = cached_tree_share(lats, lons)

    country = np.where(hr, country, np.int8(-1)).astype("int8")
    np.savez_compressed(STATIC / "grid.npz", lats=lats, lons=lons, elev=elev, land=land, hr=hr,
                        src_iy=src_iy, src_ix=src_ix, med=med, tree=tree, country=country,
                        countries=np.array(C.COUNTRIES),
                        **{f"share_{c}": shares[c] for c in C.COUNTRIES})
    print(f"grid {len(lats)}x{len(lons)}: cells {hr.sum()} "
          f"(own ERA5-Land {(hr & land).sum()}, borrowed {(hr & ~land).sum()}, dropped {dropped.sum()}), "
          f"med {med.sum()}, tree share mean {tree[hr].mean():.2f}, <10% forest {(tree[hr] < 0.1).sum()}")
    for i, c in enumerate(C.COUNTRIES):
        print(f"  {c}: {(hr & (country == i)).sum()} cells")


def load():
    return dict(np.load(STATIC / "grid.npz"))


if __name__ == "__main__":
    build()
