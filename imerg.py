"""NASA GPM IMERG daily precipitation, subset to the pipeline grid over OPeNDAP.

Late is preferred (more input data); Early arrives about a day sooner. File names carry a
version tag that changes over time (V07B, V07C), so they are read from the directory listing.
IMERG cell centres sit on x.x5, the output grid on x.x0: each output cell is the mean of the
four IMERG cells around it.
"""
import functools
import re
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

import config as C

BASE = "https://gpm1.gesdisc.eosdis.nasa.gov"
PRODUCTS = {"late": "GPM_3IMERGDL.07", "early": "GPM_3IMERGDE.07"}


class Unavailable(RuntimeError):
    """GES DISC did not answer. IMERG is one rain source of several, so the caller falls
    back to ECMWF/ICON rather than failing the day. A rejected login raises SystemExit
    instead - that one never fixes itself."""


@functools.lru_cache(maxsize=1)
def _session():
    """Bearer token if ~/.gljivar_edl_token exists (local), otherwise Earthdata login from
    ~/.netrc (machine urs.earthdata.nasa.gov), which requests re-applies on the redirect."""
    s = requests.Session()
    if C.EDL_TOKEN.exists():
        s.headers["Authorization"] = "Bearer " + C.EDL_TOKEN.read_text().strip()
    return s


@functools.lru_cache(maxsize=64)
def _listing(product, year, month):
    url = f"{BASE}/data/GPM_L3/{product}/{year}/{month:02d}/"
    r = _get(url)
    # A failed login ends on the Earthdata login page with HTTP 200, which would otherwise
    # read as "no files yet" and silently drop IMERG. (An empty listing is normal on the 1st.)
    if "urs.earthdata.nasa.gov" in r.url:
        raise SystemExit(f"IMERG listing not readable (Earthdata login failed?): {r.url}")
    return sorted(set(re.findall(r'3B-DAY-[EL]\.MS\.MRG\.3IMERG\.\d{8}-S000000-E235959\.V\w+\.nc4(?=")', r.text)))


def _get(url, tries=4):
    """Budget about two and a half minutes: long enough to ride out a blip, short enough
    that a real GES DISC outage does not eat the run before ICON and ECMWF are fetched."""
    for k in range(tries):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403):
                raise SystemExit(f"IMERG auth failed ({r.status_code}) - Earthdata token expired?")
        except requests.RequestException:
            pass
        if k < tries - 1:
            time.sleep(5 * (k + 1))
    raise Unavailable(f"IMERG download failed: {url}")


def _index_ranges():
    # IMERG: lon -179.95 + 0.1 i, lat -89.95 + 0.1 j. Take one extra cell each side.
    i0 = int(round((C.LON0 - 0.05 + 179.95) / 0.1))
    i1 = int(round((C.LON1 + 0.05 + 179.95) / 0.1))
    j0 = int(round((C.LAT0 - 0.05 + 89.95) / 0.1))
    j1 = int(round((C.LAT1 + 0.05 + 89.95) / 0.1))
    return i0, i1, j0, j1


def fetch_day(day, kind="late"):
    """Daily total (mm) on the output grid, or None if that product is not out yet."""
    day = pd.Timestamp(day)
    product = PRODUCTS[kind]
    tag = day.strftime("%Y%m%d")
    target = C.CACHE / f"imerg_{kind}_{tag}.nc4"
    if not target.exists():
        names = [n for n in _listing(product, day.year, day.month) if f".{tag}-" in n]
        if not names:
            return None
        i0, i1, j0, j1 = _index_ranges()
        ce = (f"/precipitation%5B0:1:0%5D%5B{i0}:1:{i1}%5D%5B{j0}:1:{j1}%5D;"
              f"/lat%5B{j0}:1:{j1}%5D;/lon%5B{i0}:1:{i1}%5D")
        url = f"{BASE}/opendap/GPM_L3/{product}/{day.year}/{day.month:02d}/{names[-1]}.dap.nc4?dap4.ce={ce}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(_get(url).content)
        tmp.rename(target)
    with xr.open_dataset(target) as d:
        p = d["precipitation"].isel(time=0).transpose("lat", "lon").load()
    # mean of the 4 surrounding cells = linear interpolation exactly halfway
    out = p.interp(lat=C.lats(), lon=C.lons(), method="linear")
    return out.values.astype("float32")


def best_day(day):
    """(field, 'late'|'early') or (None, None)."""
    for kind in ("late", "early"):
        f = fetch_day(day, kind)
        if f is not None:
            return f, kind
    return None, None


def check_login():
    """Preflight for the daily job: prove the Earthdata credentials still work. A server
    that is merely down is not a reason to stop - the run degrades to ECMWF/ICON rain."""
    try:
        f = fetch_day("2025-10-05", "early")
    except Unavailable as e:
        print(f"Earthdata login NOT CHECKED - GES DISC unreachable: {e}")
        return
    if f is None or f.shape != (len(C.lats()), len(C.lons())):
        raise SystemExit(f"IMERG preflight returned {None if f is None else f.shape}")
    print("Earthdata login OK", f.shape)
