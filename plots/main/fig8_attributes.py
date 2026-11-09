"""Attribute relationships of learned gates and counterfactual targets."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import figstyle as S
from data_io import FMT, read_csv

SHORT = "attribute_relationships"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--number", type=int, default=8)
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    ntop = args.top
    ac = read_csv(args.data, "attr_corr.csv")
    procs = [p for p in S.PROC if p in set(ac.process)]
    nat = ac.attribute.nunique()
    lab_w, gap_c = 1.48, 0.10
    ncol = 2
    axw = (width - ncol * lab_w - 0.12 - (ncol - 1) * gap_c) / ncol
    row_h, top_m, row_gap, bot_m = 0.165, 0.40, 0.46, 0.42
    ph = (ntop + 1.3) * row_h
    nrow = int(np.ceil(len(procs) / ncol))
    height = top_m + nrow * ph + (nrow - 1) * row_gap + bot_m
    fig = plt.figure(figsize=(width, height))
    for i, p in enumerate(procs):
        r, c = divmod(i, ncol)
        x0 = lab_w + c * (axw + lab_w + gap_c)
        y0 = height - top_m - (r + 1) * ph - r * row_gap
        ax = fig.add_axes([x0 / width, y0 / height, axw / width, ph / height])
        g = ac[ac.process == p].copy()
        g["a"] = g.rho_w.abs()
        top = g.sort_values("a", ascending=False).head(ntop)
        ystrip = ntop + 0.15
        ax.add_patch(Rectangle((-1, ystrip - 0.42), 2, 0.84, color="#F2F2F2", lw=0, zorder=0))
        others = g[~g.attribute.isin(top.attribute)]
        ax.vlines(others.rho_w, ystrip - 0.3, ystrip + 0.3, color="#B5B5B5", lw=0.9, zorder=1)
        ax.vlines(top.rho_w, ystrip - 0.3, ystrip + 0.3, color=S.DARK[p], lw=1.4, zorder=2)
        ys = np.arange(ntop)[::-1]
        ax.barh(ys, top.rho_w, 0.64, color=S.DARK[p], lw=0, zorder=2)
        ax.plot(top.rho_q, ys, "D", ms=4.6, mfc="white", mec="#1A1A1A", mew=0.9, zorder=3)
        ax.axvline(0, color=S.INK, lw=0.6, zorder=1)
        ax.set_yticks(list(ys) + [ystrip])
        ax.set_yticklabels(list(top.label) + [f"All {nat}"])
        ax.get_yticklabels()[-1].set_color("#7A7A7A")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, ystrip + 0.45)
        ax.set_xlim(-0.95, 0.95)
        ax.xaxis.set_major_locator(FixedLocator([-0.8, -0.4, 0, 0.4, 0.8]))
        ax.xaxis.set_major_formatter(FMT)
        S.vgrid(ax)
        if r == nrow - 1:
            ax.set_xlabel("Spearman $\\rho$ with attribute")
        S.panel_title(ax, "abcdefgh"[i], S.PROC_LABEL[p], x=-0.02, y=1.0)
    h = [Patch(color="#555555", label=f"$\\rho(w_p, x)$, top {ntop} by $|\\rho|$"),
         Line2D([], [], color="#B5B5B5", lw=1.2, label="Other attributes"),
         Line2D([], [], marker="D", ls="", mfc="white", mec="#1A1A1A", ms=4.6,
                label="$\\rho(q_p, x)$, counterfactual target")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=S.FS_TICK, bbox_to_anchor=(0.5, 1.0),
               borderaxespad=0.25, columnspacing=1.6)
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
