#!/usr/bin/env python3
"""Figure 8 - standalone module.

Drawing logic lives here; shared fonts, colours and geometry helpers come from `_common.py`
(which is `figures.py` minus the per-figure functions). Reads the archived figure inputs and
writes Figure 8 into the archive's `figures/main`.
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
from _common import NAMES, W, _FMT, _read, set_numbers

set_numbers(range(3, 11))


def fig_attr(data_dir, out_dir, ntop=6):
    ac = _read(data_dir, "attr_corr.csv")
    procs = [p for p in S.PROC if p in set(ac.process)]
    nat = ac.attribute.nunique()
    lab_w, gap_c, axw = 1.48, 0.02, None
    ncol = 2
    axw = (W - ncol * lab_w - 0.12 - (ncol - 1) * gap_c) / ncol
    row_h, top_m, row_gap, bot_m = 0.165, 0.40, 0.30, 0.42
    ph = (ntop + 1.3) * row_h
    nrow = int(np.ceil(len(procs) / ncol))
    H = top_m + nrow * ph + (nrow - 1) * row_gap + bot_m
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i, p in enumerate(procs):
        r, c = divmod(i, ncol)
        x0 = lab_w + c * (axw + lab_w + gap_c)
        y0 = H - top_m - (r + 1) * ph - r * row_gap
        ax = fig.add_axes([x0 / W, y0 / H, axw / W, ph / H])
        axes.append(ax)
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
        ax.xaxis.set_major_formatter(_FMT)
        S.vgrid(ax)
        if r == nrow - 1:
            ax.set_xlabel("Spearman $\\rho$ with attribute")
        S.panel_title(ax, "abcdefgh"[i], S.PROC_LABEL[p], x=-0.02, y=1.0)
    h = [Patch(color="#555555", label=rf"$\rho(\bar{{w}}_p, a)$, top {ntop} by $|\rho|$"),
         Line2D([], [], color="#B5B5B5", lw=1.2, label="Other attributes"),
         Line2D([], [], marker="D", ls="", mfc="white", mec="#1A1A1A", ms=4.6,
                label=r"$\rho(\mathrm{median}\,\Delta J_p, a)$, median contrast")]
    # centre the legend over the panel block (label column included), not over the whole canvas
    panel_cx = (lab_w + (W - 0.12)) / 2.0 / W
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=S.FS_TICK,
               bbox_to_anchor=(panel_cx, 0.985),
               borderpad=0.15, borderaxespad=0.0, columnspacing=1.6)
    S.save(fig, out_dir, NAMES["attr"])

def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 8")
    ap.add_argument("--data", type=Path, default=paths.FIG_INPUTS)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "main")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = None

    fig_attr(a.data, a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
