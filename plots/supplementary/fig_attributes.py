"""Supplementary attribute matrix: gate and counterfactual-target correlations."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

import figstyle as S
from data_io import FMT, read_csv

SHORT = "attribute_heatmap"
CMAP_DIV = LinearSegmentedColormap.from_list(
    "rdbu_nc", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None, help="output basename (default: attribute_heatmap)")
    parser.add_argument("--vmax", type=float, default=0.8)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    vmax = args.vmax
    ac = read_csv(args.data, "si_attr_corr.csv")
    procs = [p for p in S.PROC if p in set(ac.process)]
    attrs = ac.drop_duplicates("attribute")[["attribute", "label", "category"]].reset_index(drop=True)
    cats = list(dict.fromkeys(attrs.category))
    ch, pgap, cat_gap = 0.142, 0.14, 0.07
    lab_cat, lab_att = 0.92, 1.50
    x_l = lab_cat + lab_att
    cw = (width - x_l - 0.04 - (len(procs) - 1) * pgap) / (2 * len(procs))
    head_h, cb_h = 0.50, 0.68
    n = len(attrs)
    grid_w = len(procs) * 2 * cw + (len(procs) - 1) * pgap
    grid_h = n * ch + (len(cats) - 1) * cat_gap
    height = head_h + grid_h + cb_h
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    norm = Normalize(-vmax, vmax)
    x_off = x_l
    y = height - head_h
    ypos = {}
    for c in cats:
        sub = attrs[attrs.category == c]
        for k, (_, r) in enumerate(sub.iterrows()):
            ypos[r.attribute] = y - ch
            if k == 0:
                ax.text(0.02, y - ch / 2, c, ha="left", va="center", fontsize=S.FS_TICK, fontweight="bold",
                        color="#1A1A1A")
            ax.text(x_off - 0.06, y - ch / 2, r.label, ha="right", va="center", fontsize=S.FS_TICK)
            y -= ch
        if c != cats[-1]:
            ax.plot([0.02, x_off + grid_w], [y - cat_gap / 2] * 2, color="#D0D0D0", lw=0.5)
            y -= cat_gap
    for i, p in enumerate(procs):
        g = ac[ac.process == p].set_index("attribute")
        x0 = x_off + i * (2 * cw + pgap)
        for j, col in enumerate(("rho_w", "rho_q")):
            for a_, yy in ypos.items():
                v = g.loc[a_, col]
                ax.add_patch(Rectangle((x0 + j * cw + 0.008, yy + 0.008), cw - 0.016, ch - 0.016,
                                       facecolor=CMAP_DIV(norm(v)), lw=0))
            ax.text(x0 + (j + 0.5) * cw, height - head_h + 0.04, ["$w_p$", "$q_p$"][j], ha="center", va="bottom",
                    fontsize=S.FS_TICK, color="#4A4A4A")
        ax.plot([x0 + 0.01, x0 + 2 * cw - 0.01], [height - head_h + 0.26] * 2, color=S.DARK[p], lw=2.2,
                solid_capstyle="butt")
        ax.text(x0 + cw, height - head_h + 0.31, S.PROC_SHORT[p], ha="center", va="bottom",
                fontsize=S.FS_LABEL, fontweight="bold", color="#1A1A1A")
    cbw = min(2.6, grid_w * 0.75)
    cax = fig.add_axes([(x_off + (grid_w - cbw) / 2) / width, 0.42 / height, cbw / width, 0.09 / height])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DIV)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.ax.xaxis.set_major_formatter(FMT)
    cb.ax.tick_params(length=2, labelsize=S.FS_TICK)
    cb.set_label("Spearman $\\rho$ with attribute", fontsize=S.FS_LABEL, labelpad=2)
    S.save(fig, str(args.out), args.name or SHORT)


if __name__ == "__main__":
    main()
