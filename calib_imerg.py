"""One-off: how does IMERG daily rain compare with ERA5-Land over Croatia?

The model was trained on ERA5-Land, but the last ~5 days of rain in production come from
IMERG. This measures the bias (and whether a per-cell scale factor is needed) per year.
Writes cache/calib_imerg_<kind>_<year>.npz with both stacks.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

import config as C
import imerg


def main(kind, year):
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31")

    def one(d):
        try:
            return imerg.fetch_day(d, kind)
        except Exception as e:  # keep going, report at the end
            print(d.date(), e, flush=True)
            return None

    with ThreadPoolExecutor(4) as ex:
        fields = list(ex.map(one, days))
    got = np.array([f is not None for f in fields])
    I = np.stack([f if f is not None else np.full((len(C.lats()), len(C.lons())), np.nan, "float32")
                  for f in fields])

    rr = xr.open_dataset(C.BACKTEST / "data" / "era5land_rr.nc")["rr"]
    E = rr.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).sel(latitude=C.lats(), longitude=C.lons(),
                                                          method="nearest").values
    np.savez_compressed(C.CACHE / f"calib_imerg_{kind}_{year}.npz", imerg=I, era5=E, got=got)
    print(f"{kind}: {got.sum()} / {len(days)} days downloaded")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
