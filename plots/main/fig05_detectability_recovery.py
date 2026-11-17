#!/usr/bin/env python3
"""Figure 5 - standalone module.

Drawing logic lives here; shared fonts, colours and geometry helpers come from `_common.py`
(which is `figures.py` minus the per-figure functions). Reads the archived figure inputs and
writes Figure 5 into the archive's `figures/main`.
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
from _common import NAMES, W, _json, _read, _symlog_x, set_numbers

set_numbers(range(3, 11))


def fig4(data_dir, out_dir):
    sc = _read(data_dir, "fig4_scatter.csv")
    rc = _read(data_dir, "fig4_recall.csv")
    P = _json(data_dir, "fig4_meta.json")["P_p"]
    fig = plt.figure(figsize=(W, 4.15))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.82], hspace=0.58, wspace=0.10,
                          left=0.08, right=0.995, top=0.94, bottom=0.10)
    for i, p in enumerate(S.PROC):
        ax = fig.add_subplot(gs[0, i])
        d = sc[sc.process == p]
        ax.axvspan(-1, 0, color=S.BAND_NEG, lw=0, zorder=0)
        ax.axvspan(0, P[p], color=S.BAND[p], lw=0, zorder=0)
        ax.axvline(0, color=S.INK, lw=0.6, zorder=1)
        ax.axvline(P[p], color=S.DARK[p], lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0.5, color=S.REF, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        off = d[d.truth == 0]; on = d[d.truth == 1]
        ax.scatter(off.D_J, off.w, s=7, facecolor="none", edgecolor=S.GRAY_EDGE, lw=0.45, zorder=2)
        ax.scatter(on.D_J, on.w, s=7, color=S.DARK[p], alpha=0.85, lw=0, zorder=3)
        _symlog_x(ax)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if i == 0:
            ax.set_ylabel(r"Learned gate $\bar{w}_{b,p}$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$D_{J,b,p}$", labelpad=1)
        ax.text(P[p], 0.985, r" $\lambda k_p$", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=S.FS_SMALL, color=S.DARK[p])
        S.panel_title(ax, "abcd"[i], S.PROC_LABEL[p], y=1.03)

    top4 = fig.axes[:4]
    ax = fig.add_subplot(gs[1, :])
    order = ["high", "weak", "negative"]
    bw, gap = 0.26, 0.2
    xt, xl = [], []
    for i, p in enumerate(S.PROC):
        for j, s in enumerate(order):
            r = rc[(rc.process == p) & (rc.stratum == s)].iloc[0]
            x = i * (3 * bw + gap) + j * bw
            col = S.DARK[p] if s == "high" else S.LIGHT[p]
            if r.n < 10:
                ax.bar(x, r["mean"], bw * 0.92, color=S.LIGHT[p], alpha=0.35, edgecolor=S.DARK[p],
                       lw=0.7, ls=(0, (2, 1.2)), zorder=2)
            else:
                ax.bar(x, r["mean"], bw * 0.92, color=col, lw=0, zorder=2,
                       hatch=None)
                if s == "negative":
                    ax.bar(x, r["mean"], bw * 0.92, color="none", edgecolor="white", hatch="////",
                           lw=0, zorder=3)
                if r.hi > r.lo:
                    ax.plot([x, x], [r.lo, r.hi], color="black", lw=0.8, zorder=4)
            ax.text(x, -0.035, f"{int(r.n)}", ha="center", va="top", fontsize=S.FS_SMALL,
                    color="#555555" if r.n >= 10 else "#9A9A9A", transform=ax.transData)
        xt.append(i * (3 * bw + gap) + bw)
        xl.append(S.PROC_LABEL[p])
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.25, 3 * (3 * bw + gap) + 2 * bw + 0.2)
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r"ON recall ($\pi_{b,p}$ > 0.5)")
    S.hgrid(ax)
    ax.text(-0.25, -0.035, "n", ha="left", va="top", fontsize=S.FS_SMALL, color="#555555")
    S.panel_title(ax, "e", "Prescribed-ON recall by detectability stratum", y=1.03)
    h = [Patch(color="#4A4A4A", label="High"), Patch(color="#BDBDBD", label="Weak"),
         Patch(facecolor="#BDBDBD", edgecolor="white", hatch="////", label="Negative"),
         Line2D([], [], color="black", lw=0.8, label="Seed range")]
    ax.legend(handles=h, loc="upper right", ncol=4, fontsize=S.FS_TICK, handlelength=1.2,
              bbox_to_anchor=(1.0, 1.02))

    items = [
        (Line2D([], [], marker="o", ls="", color=S.DARK["snow"], ms=4.5), "Prescribed ON (process color)"),
        (Line2D([], [], marker="o", ls="", mfc="none", mec=S.GRAY_EDGE, ms=4.5), "Prescribed OFF"),
        (Patch(color=S.BAND["snow"]), r"Weak: 0 ≤ $D_{J,b,p}$ ≤ $\lambda k_p$"),
        (Patch(color=S.BAND_NEG), r"Negative: $D_{J,b,p}$ < 0"),
    ]
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    ybot = min(t.xaxis.label.get_window_extent(rr).y0 for t in top4) / fig.bbox.height
    fig.legend([a for a, _ in items], [b for _, b in items], loc="upper center", ncol=4,
               fontsize=S.FS_TICK, handlelength=1.3, columnspacing=1.4, borderaxespad=0,
               bbox_to_anchor=((top4[0].get_position().x0 + top4[-1].get_position().x1) / 2, ybot - 0.012))
    S.save(fig, out_dir, NAMES["detect"])

def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 5")
    ap.add_argument("--data", type=Path, default=paths.FIG_INPUTS)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "main")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = None

    fig4(a.data, a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
