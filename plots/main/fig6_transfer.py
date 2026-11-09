"""Recovery and transfer: ON recall and structural AUC across scenarios."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import figstyle as S
from data_io import read_csv

SHORT = "transfer"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--number", type=int, default=6)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    d = read_csv(args.data, "fig5_recovery.csv")
    fig, axs = plt.subplots(2, 2, figsize=(width, 3.5))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.105, top=0.875, wspace=0.24, hspace=0.50)
    sc_order = ["A", "B", "C", "D"]
    bh = 0.36
    for ax, p, letter in zip(axs.flat, S.PROC, "abcd"):
        dd = d[d.process == p]
        ylabels = []
        for i, s in enumerate(sc_order):
            y = len(sc_order) - 1 - i
            row = dd[dd.scenario == s]
            typ = row.assignment.iloc[0]
            if typ == "random":
                ax.axhspan(y - 0.5, y + 0.5, color=S.BAND_NEG, lw=0, zorder=0)
            for split, off, col in (("train", +bh / 2, S.DARK[p]), ("heldout", -bh / 2, S.LIGHT[p])):
                r = row[row.split == split].iloc[0]
                ax.barh(y + off, r["mean"], bh, color=col, lw=0.6 if split == "heldout" else 0,
                        edgecolor=S.DARK[p] if split == "heldout" else "none", zorder=2)
                ax.plot([r.lo, r.hi], [y + off, y + off], color="black", lw=0.8, zorder=3)
            ylabels.append(f"{s}  {typ}" if typ != "fixed" else s)
        ax.set_yticks(range(len(sc_order))[::-1])
        ax.set_yticklabels(ylabels)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.55, len(sc_order) - 0.45)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
        S.vgrid(ax)
        if p == "snow":
            ax.set_xlabel("ON recall ($w$ > 0.5)")
        else:
            ax.axvline(0.5, color=S.REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.set_xlabel("Structural discrimination (AUC)")
        S.panel_title(ax, letter, S.PROC_LABEL[p], y=1.03)
    h = [Patch(color="#4A4A4A", label="Training (536)"),
         Patch(facecolor="#BDBDBD", edgecolor="#4A4A4A", lw=0.6, label="Held-out (135)"),
         Line2D([], [], color="black", lw=0.8, label="Seed range"),
         Patch(color=S.BAND_NEG, label="Choice randomized within strata")]
    xc5 = (axs[0, 0].get_position().x0 + axs[0, 1].get_position().x1) / 2
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=S.FS_TICK, bbox_to_anchor=(xc5, 1.0),
               borderaxespad=0.3, columnspacing=1.6)
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
