"""Monthly IMERG -> ERA5-Land rain scale factors over Croatia, from calib_imerg.py output.

factor[kind][month] = sum(ERA5-Land) / sum(IMERG) over Croatian cells and all calibrated
years, shrunk toward the annual ratio when a month is noisy between years.
Writes static/imerg_factors.json.
"""
import json

import numpy as np
import pandas as pd

import config as C
import grid as G


def main():
    hr = G.load()["hr"]
    out = {"years": [], "note": "ERA5-Land / IMERG, Croatian cells, per calendar month"}
    for kind in ("late", "early"):
        rows = []
        for f in sorted(C.CACHE.glob(f"calib_imerg_{kind}_*.npz")):
            year = int(f.stem.split("_")[-1])
            z = np.load(f)
            days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
            ok = z["got"]
            rows.append(pd.DataFrame({"i": z["imerg"][ok][:, hr].mean(1), "e": z["era5"][ok][:, hr].mean(1)},
                                     index=days[ok]))
            out["years"] = sorted(set(out["years"]) | {year})
        df = pd.concat(rows)
        annual = df.e.sum() / df.i.sum()
        by_year = df.groupby([df.index.year, df.index.month]).sum()
        ratio_y = (by_year.e / by_year.i).unstack(0)
        month = df.groupby(df.index.month).sum()
        raw = month.e / month.i
        spread = ratio_y.std(1)
        n = ratio_y.notna().sum(1)
        # shrink: weight of the monthly estimate falls as its between-year spread grows
        w = 1 / (1 + (spread / np.sqrt(n)) ** 2 / 0.05)
        fac = (w * raw + (1 - w) * annual).clip(0.5, 2.0)
        out[kind] = [round(float(x), 3) for x in fac.values]
        print(f"{kind}: annual {annual:.2f}")
        print(pd.DataFrame({"raw": raw, "per-year min": ratio_y.min(1), "per-year max": ratio_y.max(1),
                            "used": fac}).round(2).T.to_string())
    (C.ROOT / "static" / "imerg_factors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
