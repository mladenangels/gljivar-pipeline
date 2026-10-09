"""Static grid: the cells of config.COUNTRIES, where their weather comes from, forest and zone.

A cell belongs to the grid if any part of it is land of one of config.COUNTRIES (Natural
Earth 10m), so coast and islands are in. ERA5-Land has no data for cells that are mostly
sea; those cells take their weather from the nearest ERA5-Land land cell (src_iy, src_ix).

Forest: share of ESA WorldCover 2021 "tree cover" (class 10) in the cell, CC BY 4.0.
Zone: "med" where the source cell's mean January temperature (ERA5-Land 2011-2025) is
>= 4 deg C - the warm coast and Istria, where the Austrian-trained model is least reliable.

Writes static/<config.GRID_FILE>. Build-time only (needs ../backtest data and rasterio/shapely/pyshp).
The eu9v1 grid also needs cache/era5land_jan_t2m.nc (ERA5-Land monthly means, January 2m
temperature 2011-2025 over the box, see jan_mean) because the backtest data stops at 50 N / 20 E.
"""
import numpy as np
import pandas as pd
import xarray as xr

import config as C

STATIC = C.ROOT / "static"
SUB = 10          # sub-samples per cell side for the land share
MED_JAN_C = 4.0
MAX_BORROW = 3    # cells
Z_FILE = "era5land_z.nc" if C.GRID_TAG == "hrsi1" else f"era5land_z_{C.GRID_TAG}.nc"
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
        x0, y0, x1, y1 = geom.bounds
        ys = np.where((lats + C.STEP / 2 >= y0) & (lats - C.STEP / 2 <= y1))[0]
        xs = np.where((lons + C.STEP / 2 >= x0) & (lons - C.STEP / 2 <= x1))[0]
        # all sub-samples of the country's bounding box in one call: [y, x, sub_y, sub_x]
        py = lats[ys][:, None, None, None] + off[None, None, :, None] * C.STEP
        px = lons[xs][None, :, None, None] + off[None, None, None, :] * C.STEP
        py, px = np.broadcast_arrays(py, px)
        inside = shapely.contains_xy(geom, px.ravel(), py.ravel()).reshape(py.shape)
        share[np.ix_(ys, xs)] = inside.mean((2, 3))
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
        try:
            with rasterio.open(WORLDCOVER.format(tile=name)) as s:
                a = s.read(1, out_shape=(3 * px_per_deg, 3 * px_per_deg), resampling=Resampling.nearest)
        except rasterio.errors.RasterioIOError:
            print("WorldCover", name, "missing (sea only)", flush=True)
            continue
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


def jan_mean(lats, lons):
    """(land mask, mean January temperature) from ERA5-Land 2011-2025.

    The backtest's daily ERA5-Land (mean of 00/06/12/18 UTC) covers 42-50 N, 5.5-20 E. The
    eu9v1 box is bigger, so outside it the January mean comes from the CDS monthly means
    (24-hour mean), shifted by their mean difference to the daily data where both exist,
    and the land mask is the monthly means' (the same ERA5-Land land-sea mask).
    """
    tg = xr.open_dataset(C.BACKTEST / "data" / "era5land_tg.nc")["tg"]
    inside = (lats >= float(tg.latitude.min()))[:, None] & (lats <= float(tg.latitude.max()))[:, None] & \
             (lons >= float(tg.longitude.min()))[None, :] & (lons <= float(tg.longitude.max()))[None, :]
    # load only what is used before the (fancy-indexed) nearest selection; on the 15-year file
    # on disk that selection takes many minutes
    t = pd.DatetimeIndex(tg.time.values)
    first = tg.isel(time=slice(0, 10)).load().sel(latitude=lats, longitude=lons, method="nearest")
    land = np.isfinite(first.values).all(0)
    janv = tg.isel(time=np.where(t.month == 1)[0]).load().sel(latitude=lats, longitude=lons, method="nearest")
    jan = janv.values.mean(0)
    if inside.all():
        return land, jan
    m = xr.open_dataset(C.CACHE / "era5land_jan_t2m.nc")["t2m"]
    m = m.rename({"valid_time": "time"}) if "valid_time" in m.dims else m
    m = m.sortby("latitude").sel(latitude=lats, longitude=lons, method="nearest")
    land_m = np.isfinite(m.values).all(0)
    jan_m = m.values.mean(0) - 273.15
    both = inside & land & land_m
    bias = float(np.mean(jan[both] - jan_m[both]))
    print(f"January mean: daily - monthly means = {bias:+.2f} C over {both.sum()} cells, "
          f"MED flag agreement {np.mean((jan[both] >= MED_JAN_C) == (jan_m[both] + bias >= MED_JAN_C)):.3f}")
    land = np.where(inside, land, land_m)
    jan = np.where(inside, jan, jan_m + bias)
    return land, jan


def build():
    lats, lons = C.lats(), C.lons()
    z = xr.open_dataset(STATIC / Z_FILE)["z"].isel(time=0)
    z = z.sortby("latitude").sel(latitude=lats, longitude=lons, method="nearest")
    elev = (z.values / 9.80665).astype("float32")
    land, jan = jan_mean(lats, lons)

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
    np.savez_compressed(STATIC / C.GRID_FILE, lats=lats, lons=lons, elev=elev, land=land, hr=hr,
                        src_iy=src_iy, src_ix=src_ix, med=med, tree=tree, country=country,
                        countries=np.array(C.COUNTRIES),
                        **{f"share_{c}": shares[c] for c in C.COUNTRIES})
    print(f"grid {len(lats)}x{len(lons)}: cells {hr.sum()} "
          f"(own ERA5-Land {(hr & land).sum()}, borrowed {(hr & ~land).sum()}, dropped {dropped.sum()}), "
          f"med {med.sum()}, tree share mean {tree[hr].mean():.2f}, <10% forest {(tree[hr] < 0.1).sum()}")
    for i, c in enumerate(C.COUNTRIES):
        print(f"  {c}: {(hr & (country == i)).sum()} cells")
    if C.GRID_TAG != "hrsi1":
        write_borders()


NEIGHBOURS = ["FRA", "BEL", "NLD", "LUX", "DNK", "POL", "CZE", "SVK", "UKR", "ROU", "BGR", "MKD", "MNE",
              "ALB", "GRC", "KOS", "LIE", "SMR", "VAT", "MCO", "TUN", "MLT"]


def write_borders():
    """Simplified country outlines for preview.py (the 9 countries + neighbours, ~0.01 deg)."""
    import json

    import shapefile
    from shapely.geometry import mapping, shape
    r = shapefile.Reader(str(C.CACHE / "ne10" / "ne_10m_admin_0_countries.shp"))
    feats = []
    for k, rec in enumerate(r.records()):
        if rec["ADM0_A3"] in C.COUNTRIES or rec["ADM0_A3"] in NEIGHBOURS:
            g = shape(r.shape(k).__geo_interface__).simplify(0.01)
            geo = json.loads(json.dumps(mapping(g)))
            feats.append({"type": "Feature", "properties": {"ADM0_A3": rec["ADM0_A3"]}, "geometry": geo})
    target = STATIC / f"borders_{C.GRID_TAG}.geojson"
    target.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    print(target, target.stat().st_size, "bytes")


def load():
    return dict(np.load(STATIC / C.GRID_FILE))


if __name__ == "__main__":
    build()
