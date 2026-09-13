"""Score past days from the ERA5-Land archive (perfect-forecast replay), for validation.

Local only: reads ../backtest/data/era5land_{rr,tg}.nc (2011-2025).

  replay.py monthly                     HR mean score by month, 2016-2025
  replay.py map 2025-10-05 [...]        writes out/replay-<dates>.png (vrganj, one panel per date)
  replay.py point 45.40 14.80 2025      daily scores for one place and year
"""
import json
import sys

import numpy as np
import pandas as pd
import xarray as xr

import config as C
import grid as G
import model as M


def load():
    sel = dict(latitude=C.lats(), longitude=C.lons(), method="nearest")
    rr = xr.open_dataset(C.BACKTEST / "data" / "era5land_rr.nc")["rr"].sel(**sel)
    tg = xr.open_dataset(C.BACKTEST / "data" / "era5land_tg.nc")["tg"].sel(**sel)
    return pd.DatetimeIndex(rr.time.values), rr.values, tg.values


def scores(time, rr, tg, day, iy, ix, med, model, season):
    t = time.get_loc(pd.Timestamp(day))
    R = rr[t - np.arange(C.RR_LAGS)][:, iy, ix].T.astype("float64")
    T = tg[t - np.arange(C.TG_LAGS)][:, iy, ix].T.astype("float64")
    return M.score(model, season, R, T, day, med)


def main(cmd, *args):
    g = G.load()
    model = json.loads((C.OUT / "model.json").read_text())
    season = json.loads((C.ROOT / "static" / "season.json").read_text())
    time, rr, tg = load()
    cy, cx = np.where(g["hr"])
    iy, ix, med = g["src_iy"][cy, cx], g["src_ix"][cy, cx], g["med"][cy, cx]

    if cmd == "monthly":
        days = pd.date_range("2016-01-01", "2025-12-31", freq="3D")
        rows = []
        for d in days:
            s = scores(time, rr, tg, d, iy, ix, med, model, season)
            rows.append({"date": d, **{f"{k}_cont": v[~med].mean() for k, v in s.items()},
                         **{f"{k}_med": v[med].mean() for k, v in s.items()},
                         **{f"{k}_80": (v >= 80).mean() * 100 for k, v in s.items()}})
        df = pd.DataFrame(rows).set_index("date")
        print(df.groupby(df.index.month).mean().round(0).T.to_string())

    elif cmd == "map":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from preview import CMAP, borders
        bd = borders()
        fig, axes = plt.subplots(1, len(args), figsize=(3.2 * len(args), 3), squeeze=False)
        for ax, d in zip(axes[0], args):
            s = scores(time, rr, tg, d, iy, ix, med, model, season)["vrganj"]
            m = np.full(g["hr"].shape, np.nan)
            m[cy, cx] = s
            ax.pcolormesh(C.lons(), C.lats(), m, cmap=CMAP, vmin=0, vmax=100, shading="nearest")
            for is_hr, p in bd:
                ax.plot(p[:, 0], p[:, 1], color="#333" if is_hr else "#aaa", lw=0.5 if is_hr else 0.3)
            ax.set_xlim(13.4, 19.5)
            ax.set_ylim(42.3, 46.6)
            ax.set_aspect(1 / np.cos(np.radians(44.5)))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"vrganji {d}", fontsize=9)
        target = C.OUT / f"replay-{'_'.join(args)}.png"
        fig.savefig(target, dpi=110, bbox_inches="tight")
        print(target)

    elif cmd == "point":
        la, lo, year = float(args[0]), float(args[1]), int(args[2])
        y, x = np.abs(C.lats() - la).argmin(), np.abs(C.lons() - lo).argmin()
        k = np.flatnonzero((cy == y) & (cx == x))
        if not len(k):
            raise SystemExit("not a Croatian cell")
        k = k[0]
        for d in pd.date_range(f"{year}-03-01", f"{year}-11-30", freq="7D"):
            s = scores(time, rr, tg, d, iy[k:k + 1], ix[k:k + 1], med[k:k + 1], model, season)
            print(d.date(), " ".join(f"{n} {int(v[0]):3d}" for n, v in s.items()))


if __name__ == "__main__":
    main(*sys.argv[1:])
