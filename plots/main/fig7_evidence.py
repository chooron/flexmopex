"""Real-catchment evidence classes, one map per process with a composition bar."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import figstyle as S
import geo
from data_io import read_csv
from map_frame import map_axes

SHORT = "evidence_maps"

MAP_GAP_X = 0.10      # in, between map columns
MAP_TITLE_H = 0.20    # in, panel title band above each map
MAP_MARGIN = 0.03     # in, outer margin


def map_size(outline, width, ncols=2):
    ext = geo.map_extent(outline)
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    mw = (width - 2 * MAP_MARGIN - (ncols - 1) * MAP_GAP_X) / ncols
    return ext, mw, mw * aspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--region-field", default="region")
    parser.add_argument("--number", type=int, default=7)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    regions, states, outline = geo.load_layers(
        args.regions, args.states, region_field=args.region_field
    )
    cat = read_csv(args.data, "catchments.csv")
    cat["x"], cat["y"] = geo.project_points(cat.lon, cat.lat)
    ev = read_csv(args.data, "fig6_evidence.csv").merge(cat[["gauge_id", "x", "y"]], on="gauge_id")
    ext, mw, mh = map_size(outline, width)
    bar_gap, bar_h, row_gap, leg_h = 0.03, 0.15, 0.10, 0.27
    panel_h = MAP_TITLE_H + mh + bar_gap + bar_h
    height = 2 * panel_h + row_gap + leg_h + 2 * MAP_MARGIN
    fig = plt.figure(figsize=(width, height))
    for i, p in enumerate(S.PROC):
        r, c = divmod(i, 2)
        x0 = MAP_MARGIN + c * (mw + MAP_GAP_X)
        top = height - MAP_MARGIN - r * (panel_h + row_gap)
        y_map = top - MAP_TITLE_H - mh
        ax = map_axes(fig, (x0, y_map, mw, mh), height, width)
        geo.draw_base(ax, regions, states, outline, region_lw=0.8, extent=ext)
        d = ev[ev.process == p]
        unc, off, on = d[d.cls == "UNC"], d[d.cls == "OFF"], d[d.cls == "ON"]
        ax.scatter(unc.x, unc.y, s=7, facecolor="white", edgecolor=S.GRAY_EDGE, lw=0.5, zorder=4)
        ax.scatter(off.x, off.y, s=7, color=S.LIGHT[p], edgecolor="white", lw=0.3, zorder=5)
        ax.scatter(on.x, on.y, s=7, color=S.DARK[p], edgecolor="white", lw=0.3, zorder=6)
        S.panel_title(ax, "abcd"[i], S.PROC_LABEL[p], x=0.0, y=1.01)

        bx = map_axes(fig, (x0, y_map - bar_gap - bar_h, mw, bar_h), height, width)
        counts = [len(on), len(off), len(unc)]
        tot = sum(counts)
        left = 0.0
        for cnt, fc, ec, tc in zip(counts, [S.DARK[p], S.LIGHT[p], "white"],
                                   [S.DARK[p], S.LIGHT[p], S.GRAY_EDGE], ["white", "#222222", "#444444"]):
            wdt = cnt / tot
            bx.add_patch(Rectangle((left, 0.0), wdt, 1.0, facecolor=fc, edgecolor=ec, lw=0.6))
            bx.text(left + wdt / 2, 0.5, f"{cnt}", ha="center", va="center",
                    fontsize=S.FS_SMALL, color=tc)
            left += wdt
        bx.set_xlim(0, 1)
        bx.set_ylim(0, 1)
        bx.set_axis_off()

    h = [Line2D([], [], marker="o", ls="", color=S.DARK["snow"], mec="white", ms=4.5, label="Decisive ON"),
         Line2D([], [], marker="o", ls="", color=S.LIGHT["snow"], mec="white", ms=4.5, label="Decisive OFF"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=S.GRAY_EDGE, ms=4.5, label="Uncertain")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=S.FS_LABEL,
               bbox_to_anchor=(0.5, MAP_MARGIN / height))
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
