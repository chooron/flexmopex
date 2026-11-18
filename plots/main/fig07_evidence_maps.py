#!/usr/bin/env python3
"""Figure 7 - counterfactual evidence maps with a shared composition panel (e)."""
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
    """Figure 7 - counterfactual evidence maps with a shared composition panel (e)."""
    regions, states, outline = layers
    cat = _read(data_dir, "catchments.csv")
    cat["x"], cat["y"] = geo.project_points(cat.lon, cat.lat)
    ev = _read(data_dir, "fig6_evidence.csv").merge(cat[["gauge_id", "x", "y"]], on="gauge_id")

    # one semantic palette for the whole figure (deep blue / soft coral / light grey)
    CLS = {"ON": "#2166AC", "OFF": "#F4A582", "UNC": "#D9D9D9"}
    LABEL = {"ON": "Decisive ON", "OFF": "Decisive OFF", "UNC": "Uncertain"}
    EXPECT = {"snow": (347, 257, 67), "sub": (509, 47, 115),
              "phen": (205, 257, 209), "int": (20, 581, 70)}
    for p, (on_n, off_n, unc_n) in EXPECT.items():
        d = ev[ev.process == p]
        got = (int((d.cls == "ON").sum()), int((d.cls == "OFF").sum()),
               int((d.cls == "UNC").sum()))
        assert got == (on_n, off_n, unc_n), (p, got)

    ext, mw, mh = _map_size(outline)
    # trim the vertical whitespace above/below the conterminous US so the map block is compact
    trim = 0.025 * (ext[3] - ext[2])
    ext = (ext[0], ext[1], ext[2] + trim, ext[3] - trim)
    mh = mw * (ext[3] - ext[2]) / (ext[1] - ext[0])
    row_gap = 0.02          # (a)(b) / (c)(d) gap: title band only, tight grid
    band = 0.44             # blank band between (c)(d) and panel (e)
    bar_h = 1.32            # panel (e) height: +29 % vs the previous 1.02 in
    xlab_h = 0.40           # tick labels + axis title below the bars
    H = 2 * (MAP_TITLE_H + mh) + row_gap + band + bar_h + xlab_h + 2 * MAP_MARGIN
    fig = plt.figure(figsize=(W, H))

    row0_map_y = H - MAP_MARGIN - MAP_TITLE_H - mh
    row1_map_y = row0_map_y - mh - MAP_TITLE_H - row_gap
    for i, p in enumerate(S.PROC):
        r, c = divmod(i, 2)
        x0 = MAP_MARGIN + c * (mw + MAP_GAP_X)
        y_map = row0_map_y if r == 0 else row1_map_y
        ax = _map_axes(fig, (x0, y_map, mw, mh), ext, W, H)
        geo.draw_base(ax, regions, states, outline, region_lw=0.45, extent=ext,
                      region_color="#8C8C8C")
        d = ev[ev.process == p]
        # Uncertain at the bottom; the minority decisive class is drawn last so it stays visible
        ax.scatter(d[d.cls == "UNC"].x, d[d.cls == "UNC"].y, s=11, c=CLS["UNC"],
                   edgecolors="none", alpha=0.90, zorder=2)
        majority, minority = ("ON", "OFF") if (d.cls == "ON").sum() >= (d.cls == "OFF").sum() \
            else ("OFF", "ON")
        for k, z in ((majority, 3), (minority, 4)):
            dd = d[d.cls == k]
            ax.scatter(dd.x, dd.y, s=11, c=CLS[k], edgecolors="none", alpha=0.90, zorder=z)
        S.panel_title(ax, "abcd"[i], S.PROC_LABEL[p], x=0.0, y=1.01, fontsize=10.0)

    # ------------------------------------------------ panel (e): one shared composition bar
    left_in, right_in = 1.58, 0.26
    bx = fig.add_axes([left_in / W, (MAP_MARGIN + xlab_h) / H,
                       (W - left_in - right_in) / W, bar_h / H])
    procs = list(S.PROC)[::-1]                       # Snow on top, Interception at the bottom
    for y, p in enumerate(procs):
        d = ev[ev.process == p]
        counts = {"ON": int((d.cls == "ON").sum()), "OFF": int((d.cls == "OFF").sum()),
                  "UNC": int((d.cls == "UNC").sum())}
        total = sum(counts.values())
        left = 0.0
        for k in ("ON", "OFF", "UNC"):
            wdt = counts[k] / total
            bx.add_patch(Rectangle((left, y - 0.37), wdt, 0.74, facecolor=CLS[k],
                                   edgecolor="white", lw=0.8, zorder=2))
            if wdt >= 0.05:
                bx.text(left + wdt / 2, y, f"{counts[k]}", ha="center", va="center",
                        fontsize=9.6, color="white" if k == "ON" else "#1A1A1A", zorder=3)
            elif wdt > 0.0:
                # too narrow for an inline count: place it just outside and point at the block
                bx.plot([left + wdt + 0.006, left + wdt + 0.028], [y, y], ls=(0, (1.0, 1.8)),
                        color="#4A4A4A", lw=1.0, zorder=4, solid_capstyle="butt")
                bx.text(left + wdt + 0.034, y, f"{counts[k]}", ha="left", va="center",
                        fontsize=9.6, color="#1A1A1A", zorder=4)
            left += wdt
    bx.set_xlim(0, 1)
    bx.set_ylim(-0.55, len(procs) - 0.45)
    bx.set_yticks(range(len(procs)))
    bx.set_yticklabels([S.PROC_LABEL[p] for p in procs], fontsize=10.0)
    bx.tick_params(axis="y", length=0, pad=4.5)
    bx.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bx.tick_params(axis="x", labelsize=9.0)
    bx.set_xlabel("Fraction of basins", fontsize=10.5, labelpad=3.0)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)

    # headline row: panel title on the left, legend flush with the bar end on the right
    y_head = (MAP_MARGIN + xlab_h + bar_h + 0.11) / H
    fig.text((MAP_MARGIN + 0.32) / W, y_head,
             "(e) Counterfactual evidence-class composition", fontsize=10.0,
             fontweight="bold", ha="left", va="center")
    fig.legend(handles=[Patch(facecolor=CLS[k], label=LABEL[k]) for k in ("ON", "OFF", "UNC")],
               loc="center right", ncol=3, frameon=False, fontsize=9.0,
               bbox_to_anchor=((W - right_in) / W, y_head), handlelength=1.25,
               columnspacing=1.5, borderaxespad=0.0)

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
    layers = geo.load_layers(str(a.regions), str(a.states), region_field=a.field)
    fig6(a.data, a.out, layers)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
