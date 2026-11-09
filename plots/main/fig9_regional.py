"""Regional exclusion: evidence agreement and direction AUC per region."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figstyle as S
import geo
from data_io import read_csv

SHORT = "regional_exclusion"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--number", type=int, default=9)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    d = read_csv(args.data, "fig7_regional.csv")
    rows = geo.REGION_ORDER + ["Pooled"]
    cat = read_csv(args.data, "catchments.csv")
    n_region = cat.region.value_counts().to_dict()
    n_region["Pooled"] = len(cat)

    cw, gap_m, gap_p = 1.0, 0.16, 0.42      # cell width, gap between metrics, gap between processes
    rh, bh = 1.0, 0.64                      # row pitch, bar height
    ypos = {r: -(i + (0.45 if r == "Pooled" else 0)) * rh for i, r in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(width, 2.75))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    x = 0.0
    col_x = {}
    for p in S.PROC:
        for m in ("BA", "AUC"):
            col_x[(p, m)] = x
            x += cw + gap_m
        x += gap_p - gap_m
    xmax = x - gap_p

    for (p, m), x0 in col_x.items():
        dark = S.DARK[p] if m == "BA" else S.LIGHT[p]
        for r in rows:
            row = d[(d.process == p) & (d.metric == m) & (d.region == r)].iloc[0]
            y = ypos[r] - bh / 2
            v, small = row.value, bool(row.small)
            if np.isnan(v):
                ax.add_patch(Rectangle((x0, y), cw, bh, facecolor="white", edgecolor="#BDBDBD",
                                       lw=0.6, hatch="////", zorder=2))
                ax.text(x0 + cw / 2, ypos[r], "n/a", ha="center", va="center", fontsize=S.FS_SMALL,
                        color="#8A8A8A", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
                continue
            ax.add_patch(Rectangle((x0, y), cw, bh, facecolor="#EFEFEF", lw=0, zorder=1))
            ax.plot([x0 + 0.5 * cw] * 2, [y, y + bh], color="#6E6E6E", lw=0.6, ls=(0, (1.5, 1.2)), zorder=1.5)
            if small:
                ax.add_patch(Rectangle((x0, y), v * cw, bh, facecolor="white",
                                       edgecolor=S.DARK[p], lw=0.9, zorder=2))
                tcol = "#4A4A4A"
            else:
                ax.add_patch(Rectangle((x0, y), v * cw, bh, facecolor=dark, lw=0, zorder=2))
                tcol = "white" if m == "BA" else "#1A1A1A"
            if v >= 0.34:
                ax.text(x0 + v * cw - 0.04, ypos[r], f"{v:.2f}", ha="right", va="center",
                        fontsize=S.FS_TICK, color=tcol, zorder=4,
                        fontweight="bold" if r == "Pooled" else "normal")
            else:
                ax.text(x0 + v * cw + 0.04, ypos[r], f"{v:.2f}", ha="left", va="center",
                        fontsize=S.FS_TICK, color="#4A4A4A", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.08", fc="#EFEFEF", ec="none"))
    ytop = ypos["R1"] + bh / 2
    for p in S.PROC:
        xa = col_x[(p, "BA")]
        xb = col_x[(p, "AUC")] + cw
        ax.plot([xa, xb], [ytop + 0.95, ytop + 0.95], color=S.DARK[p], lw=2.2, solid_capstyle="butt")
        ax.text((xa + xb) / 2, ytop + 1.35, S.PROC_LABEL[p], ha="center", va="bottom",
                fontsize=S.FS_TITLE, fontweight="bold", color="#1A1A1A")
        ax.text(xa + cw / 2, ytop + 0.18, "Evidence\nagreement", ha="center", va="bottom",
                fontsize=S.FS_SMALL, color="#4A4A4A", linespacing=0.95)
        ax.text(col_x[(p, "AUC")] + cw / 2, ytop + 0.18, "Direction\nAUC", ha="center", va="bottom",
                fontsize=S.FS_SMALL, color="#4A4A4A", linespacing=0.95)
    for r in rows:
        ax.text(-0.12, ypos[r], r, ha="right", va="center", fontsize=S.FS_LABEL,
                fontweight="bold" if r == "Pooled" else "normal")
        ax.text(xmax + 0.12, ypos[r], f"{n_region.get(r, '')}", ha="left", va="center",
                fontsize=S.FS_TICK, color="#4A4A4A")
    ax.text(xmax + 0.12, ytop + 0.18, "n", ha="left", va="bottom", fontsize=S.FS_SMALL, color="#4A4A4A")
    ysep = (ypos["R7"] + ypos["Pooled"]) / 2
    ax.plot([-0.05, xmax], [ysep, ysep], color="#9A9A9A", lw=0.6)
    ax.set_xlim(-0.80, xmax + 0.45)
    ax.set_ylim(ypos["Pooled"] - 0.45, ytop + 1.85)
    ax.set_axis_off()
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
