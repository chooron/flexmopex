#!/usr/bin/env python3
"""Figure 7 - standalone module.

Drawing logic lives here; shared fonts, colours and geometry helpers come from `_common.py`
(which is `figures.py` minus the per-figure functions). Reads the archived figure inputs and
writes Figure 7 into the archive's `figures/main`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
sys.path.insert(0, str(REPO / "plots/main"))
import paths  # noqa: E402
import figstyle as S  # noqa: E402
import geo  # noqa: E402
from _common import MAP_GAP_X, MAP_MARGIN, MAP_TITLE_H, NAMES, W, _map_axes, _map_size, _read, set_numbers

set_numbers(range(3, 11))


def fig6(data_dir, out_dir, layers):
    regions, states, outline = layers
    cat = _read(data_dir, "catchments.csv")
    cat["x"], cat["y"] = geo.project_points(cat.lon, cat.lat)
    ev = _read(data_dir, "fig6_evidence.csv").merge(cat[["gauge_id", "x", "y"]], on="gauge_id")
    ext, mw, mh = _map_size(outline)
    bar_gap, bar_h, row_gap, leg_h = 0.03, 0.19, 0.10, 0.27
    panel_h = MAP_TITLE_H + mh + bar_gap + bar_h
    H = 2 * panel_h + row_gap + leg_h + 2 * MAP_MARGIN
    fig = plt.figure(figsize=(W, H))
    for i, p in enumerate(S.PROC):
        r, c = divmod(i, 2)
        x0 = MAP_MARGIN + c * (mw + MAP_GAP_X)
        top = H - MAP_MARGIN - r * (panel_h + row_gap)
        y_map = top - MAP_TITLE_H - mh
        ax = _map_axes(fig, (x0, y_map, mw, mh), ext, W, H)
        geo.draw_base(ax, regions, states, outline, region_lw=0.8, extent=ext)
        d = ev[ev.process == p]
        unc, off, on = d[d.cls == "UNC"], d[d.cls == "OFF"], d[d.cls == "ON"]
        ax.scatter(unc.x, unc.y, s=7, facecolor="white", edgecolor=S.GRAY_EDGE, lw=0.5, zorder=4)
        ax.scatter(off.x, off.y, s=7, color=S.LIGHT[p], edgecolor="white", lw=0.3, zorder=5)
        ax.scatter(on.x, on.y, s=7, color=S.DARK[p], edgecolor="white", lw=0.3, zorder=6)
        S.panel_title(ax, "abcd"[i], S.PROC_LABEL[p], x=0.0, y=1.01, fontsize=10.0)

        bx = _map_axes(fig, (x0, y_map - bar_gap - bar_h, mw, bar_h), None, W, H)
        counts = [len(on), len(off), len(unc)]
        tot = sum(counts)
        left = 0.0
        for cnt, fc, ec, tc in zip(counts, [S.DARK[p], S.LIGHT[p], "white"],
                                   [S.DARK[p], S.LIGHT[p], S.GRAY_EDGE], ["white", "#222222", "#444444"]):
            wdt = cnt / tot
            bx.add_patch(Rectangle((left, 0.0), wdt, 1.0, facecolor=fc, edgecolor=ec, lw=0.6))
            if wdt < 0.055:
                tx, ha, tc = left + wdt + 0.035, "left", S.INK
                bx.plot([left + wdt / 2, tx - 0.008], [0.45, 0.75],
                        color=S.INK, lw=0.75, zorder=4, clip_on=False)
                bx.plot([left + wdt / 2], [0.45], marker="o", ms=1.8,
                        color=S.INK, zorder=5, clip_on=False)
                bx.text(tx, 0.75, f"{cnt}", ha=ha, va="center",
                        fontsize=11, color=tc)
            else:
                tx, ha = left + wdt / 2, "center"
                bx.text(tx, 0.5, f"{cnt}", ha=ha, va="center",
                        fontsize=11, color=tc)
            left += wdt
        bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.set_axis_off()

    h = [Line2D([], [], marker="o", ls="", color=S.DARK["snow"], mec="white", ms=4.5, label="Decisive ON"),
         Line2D([], [], marker="o", ls="", color=S.LIGHT["snow"], mec="white", ms=4.5, label="Decisive OFF"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=S.GRAY_EDGE, ms=4.5, label="Uncertain")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=11.0,
               bbox_to_anchor=(0.5, 0.0025))
    S.save(fig, out_dir, NAMES["evidence"])

def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 7")
    ap.add_argument("--data", type=Path, default=paths.FIG_INPUTS)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "main")
    ap.add_argument("--regions", type=Path, default=paths.FIG_INPUTS / "regions.gpkg")
    ap.add_argument("--states", type=Path, default=paths.FIG_INPUTS / "states.gpkg")
    ap.add_argument("--field", default="loro_regio")

    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = None
    layers = geo.load_layers(str(a.regions), str(a.states), region_field=a.field)
    fig6(a.data, a.out, layers)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
