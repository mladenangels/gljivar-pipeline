"""Render out/forecast-<date>.bin as small-multiple maps (for eyeballing and Telegram).

Usage: preview.py [--v2] [YYYY-MM-DD]          all groups x days (default: date in latest.json)
       preview.py [--v2] --map vrganj [DATE]    one big map of today for one group
       preview.py --v2 --rain 7|14 [DATE]       one big map of the rain layer
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

if "--v2" in sys.argv:
    os.environ["GLJIVAR_GRID"] = "eu9v1"
import config as C  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("gljive", ["#efe9e1", "#d9c7a7", "#a5b46a", "#4f8a3a", "#1f5a2a"])
LABEL = {"vrganj": "vrganji", "lisicarka": "lisičarke", "suncanica": "sunčanica", "smrcak": "smrčci"}


RAIN_CMAP = LinearSegmentedColormap.from_list("rain", ["#f4f1ea", "#c6dbef", "#6baed6", "#2171b5", "#08306b", "#54278f"])
RAIN_BOUNDS = [0, 1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]


def borders():
    if C.GRID_TAG == "hrsi1":
        geo = json.loads((C.ROOT / "static" / "borders10.geojson").read_text())
        keep, strong = ("HRV", "SVN", "BIH", "SRB", "MNE", "HUN", "ITA"), ("HRV",)
    else:
        geo = json.loads((C.ROOT / "static" / f"borders_{C.GRID_TAG}.geojson").read_text())
        keep, strong = None, tuple(C.COUNTRIES)
    out = []
    for f in geo["features"]:
        if keep and f["properties"]["ADM0_A3"] not in keep:
            continue
        g = f["geometry"]
        for p in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            out.append((f["properties"]["ADM0_A3"] in strong, np.array(p[0])))
    return out


def extent(grid):
    if C.GRID_TAG == "hrsi1":
        return (13.4, 19.5), (42.3, 46.6)
    return (grid["lon0"] - 0.1, grid["lon0"] + grid["step"] * grid["nx"]), \
           (grid["lat0"] - 0.1, grid["lat0"] + grid["step"] * grid["ny"])


def big_map(grid, values, cmap, norm, title, label, target):
    """One map of per-cell values over the whole grid."""
    cells = np.array(grid["cells"])
    lats = grid["lat0"] + grid["step"] * np.arange(grid["ny"])
    lons = grid["lon0"] + grid["step"] * np.arange(grid["nx"])
    m = np.full((grid["ny"], grid["nx"]), np.nan)
    m[cells[:, 0], cells[:, 1]] = values
    (x0, x1), (y0, y1) = extent(grid)
    k = 1 / np.cos(np.radians((y0 + y1) / 2))
    fig, ax = plt.subplots(figsize=(10, 10 * (y1 - y0) * k / (x1 - x0) + 0.6))
    ax.pcolormesh(lons, lats, m, cmap=cmap, norm=norm, shading="nearest")
    for strong, p in borders():
        ax.plot(p[:, 0], p[:, 1], color="#333" if strong else "#aaa", lw=0.6 if strong else 0.4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(k)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.6, label=label)
    ax.set_title(title, fontsize=11)
    fig.savefig(target, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(target)


def main_map(group, date=None):
    meta = json.loads((C.OUT / "latest.json").read_text())
    date = date or meta["date"]
    grid = json.loads((C.OUT / "grid.json").read_text())
    n = len(grid["cells"])
    raw = np.frombuffer((C.OUT / f"forecast-{date}.bin").read_bytes(), "uint8")
    cube = raw.reshape(len(meta["groups"]), len(meta["days"]), n)
    g = meta["groups"].index(group)
    big_map(grid, cube[g, 0], CMAP, plt.Normalize(0, 100),
            f"Gljivar — {LABEL.get(group, group)} {date} (ECMWF {meta['ecmwfRun']})", "indeks (percentil)",
            C.OUT / f"preview-{group}-{date}.png")


def main_rain(period, date=None):
    meta = json.loads((C.OUT / "latest.json").read_text())
    date = date or meta["date"]
    grid = json.loads((C.OUT / "grid.json").read_text())
    rain = meta["rain"]
    raw = np.frombuffer((C.OUT / f"rain-{date}.bin").read_bytes(), "<u2")
    cube = raw.reshape(len(rain["periods"]), len(grid["cells"])) / 10.0
    k = rain["periods"].index(period)
    big_map(grid, cube[k], RAIN_CMAP, BoundaryNorm(RAIN_BOUNDS, RAIN_CMAP.N, extend="max"),
            f"Kiša zadnjih {period} dana do {rain['endDay']} (procjena od {rain['estimatedFrom']})", "mm",
            C.OUT / f"preview-rain{period}-{date}.png")


def main(date=None):
    meta = json.loads((C.OUT / "latest.json").read_text())
    date = date or meta["date"]
    grid = json.loads((C.OUT / "grid.json").read_text())
    cells = np.array(grid["cells"])
    raw = np.frombuffer((C.OUT / f"forecast-{date}.bin").read_bytes(), "uint8")
    cube = raw.reshape(len(meta["groups"]), len(meta["days"]), len(cells))
    lats = grid["lat0"] + grid["step"] * np.arange(grid["ny"])
    lons = grid["lon0"] + grid["step"] * np.arange(grid["nx"])
    bd = borders()
    (x0, x1), (y0, y1) = extent(grid)

    tall = 1.75 if C.GRID_TAG == "hrsi1" else 2.6
    fig, axes = plt.subplots(len(meta["groups"]), len(meta["days"]),
                             figsize=(2.1 * len(meta["days"]), tall * len(meta["groups"])))
    for g, name in enumerate(meta["groups"]):
        for j, day in enumerate(meta["days"]):
            ax = axes[g, j]
            m = np.full((grid["ny"], grid["nx"]), np.nan)
            m[cells[:, 0], cells[:, 1]] = cube[g, j]
            ax.pcolormesh(lons, lats, m, cmap=CMAP, vmin=0, vmax=100, shading="nearest")
            for is_hr, p in bd:
                ax.plot(p[:, 0], p[:, 1], color="#333" if is_hr else "#aaa", lw=0.5 if is_hr else 0.3)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect(1 / np.cos(np.radians(44.5 if C.GRID_TAG == "hrsi1" else (y0 + y1) / 2)))
            ax.set_xticks([])
            ax.set_yticks([])
            if g == 0:
                ax.set_title(f"{day[8:10]}.{day[5:7]}.", fontsize=9)
            if j == 0:
                ax.set_ylabel(LABEL.get(name, name), fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 100))
    fig.colorbar(sm, ax=axes, shrink=0.5, label="indeks (percentil)")
    fig.suptitle(f"Gljivar — izračun {date} (ECMWF {meta['ecmwfRun']})", fontsize=11)
    target = C.OUT / f"preview-{date}.png"
    fig.savefig(target, dpi=110, bbox_inches="tight")
    print(target)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--v2"]
    if args and args[0] == "--map":
        main_map(args[1], args[2] if len(args) > 2 else None)
    elif args and args[0] == "--rain":
        main_rain(int(args[1]), args[2] if len(args) > 2 else None)
    else:
        main(args[0] if args else None)
