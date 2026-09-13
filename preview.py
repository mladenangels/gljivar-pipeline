"""Render out/forecast-<date>.bin as small-multiple maps (for eyeballing and Telegram).

Usage: preview.py [YYYY-MM-DD]   (default: date in out/latest.json)
"""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import config as C  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("gljive", ["#efe9e1", "#d9c7a7", "#a5b46a", "#4f8a3a", "#1f5a2a"])
LABEL = {"vrganj": "vrganji", "lisicarka": "lisičarke", "suncanica": "sunčanica", "smrcak": "smrčci"}


def borders():
    geo = json.loads((C.ROOT / "static" / "borders10.geojson").read_text())
    out = []
    for f in geo["features"]:
        if f["properties"]["ADM0_A3"] not in ("HRV", "SVN", "BIH", "SRB", "MNE", "HUN", "ITA"):
            continue
        g = f["geometry"]
        for p in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            out.append((f["properties"]["ADM0_A3"] == "HRV", np.array(p[0])))
    return out


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

    fig, axes = plt.subplots(len(meta["groups"]), len(meta["days"]),
                             figsize=(2.1 * len(meta["days"]), 1.75 * len(meta["groups"])))
    for g, name in enumerate(meta["groups"]):
        for j, day in enumerate(meta["days"]):
            ax = axes[g, j]
            m = np.full((grid["ny"], grid["nx"]), np.nan)
            m[cells[:, 0], cells[:, 1]] = cube[g, j]
            ax.pcolormesh(lons, lats, m, cmap=CMAP, vmin=0, vmax=100, shading="nearest")
            for is_hr, p in bd:
                ax.plot(p[:, 0], p[:, 1], color="#333" if is_hr else "#aaa", lw=0.5 if is_hr else 0.3)
            ax.set_xlim(13.4, 19.5)
            ax.set_ylim(42.3, 46.6)
            ax.set_aspect(1 / np.cos(np.radians(44.5)))
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
    main(sys.argv[1] if len(sys.argv) > 1 else None)
