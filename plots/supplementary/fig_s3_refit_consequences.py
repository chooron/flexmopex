#!/usr/bin/env python3
"""Figure S3 - consequences of refitting after process removal.

(a) across-catchment distribution of NSE(LOPO) - NSE(fixed-off), symmetric-log axis
(b) fraction of decisive-OFF catchments whose Full-minus-LOPO loss exceeds tau, on a
    log-spaced threshold axis

Input   manuscripts/tables/FigureS3_perbasin_refit_diff.csv
        manuscripts/tables/FigureS3_refit_exceedance_by_threshold.csv
Output  manuscripts/figures/supplementary/FigureS3_refit_consequences.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
import paths  # noqa: E402
import wrrstyle as S  # noqa: E402

S.apply()

TICKS = [-10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10]


def _fmt(v, _):
    return f"\u2212{abs(v):g}" if v < 0 else f"{v:g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.TABLES / "figure_inputs")
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "supplementary")
    a = ap.parse_args()

    df = pd.read_csv(a.data / "FigureS3_perbasin_refit_diff.csv", dtype={"gauge_id": str})
    ex = pd.read_csv(a.data / "FigureS3_refit_exceedance_by_threshold.csv")
    assert len(df) == 2684

    fig = plt.figure(figsize=(S.FULL_WIDTH, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.30,
                          left=0.108, right=0.985, top=0.855, bottom=0.195)

    # ---------------------------------------------------------------- (a) symlog
    ax = fig.add_subplot(gs[0])
    for i, p in enumerate(S.PROCS):                      # snow on top
        y = len(S.PROCS) - 1 - i
        v = df[df.process == p]["diff"].to_numpy()
        q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        ax.add_patch(Rectangle((q25, y - 0.27), q75 - q25, 0.54, facecolor=S.TINT[p],
                               edgecolor=S.PROC[p], lw=0.8, zorder=2))
        ax.plot([q5, q25], [y, y], color=S.PROC[p], lw=1.1, zorder=2)
        ax.plot([q75, q95], [y, y], color=S.PROC[p], lw=1.1, zorder=2)
        ax.plot([q50, q50], [y - 0.27, y + 0.27], color="#111111", lw=1.7, zorder=3)
        o = v[(v < q5) | (v > q95)]
        ax.scatter(o, np.full(len(o), y), s=5, facecolor="none", edgecolor=S.PROC[p],
                   lw=0.5, alpha=0.65, zorder=1)
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.8)
    ax.set_xlim(-14, 14)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axvline(0, color=S.ZERO, lw=0.9, ls="--", zorder=1)
    ax.set_yticks(range(len(S.PROCS))[::-1])
    ax.set_yticklabels([S.NAME[p] for p in S.PROCS])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.62, len(S.PROCS) - 0.38)
    S.despine(ax)
    S.hgrid(ax, "x")
    ax.set_xlabel("NSE$_{\\mathrm{LOPO}}$ \u2212 NSE$_{\\mathrm{fixed\\text{-}off}}$")
    S.panel_title(ax, "a", "Change after refitting")

    # ------------------------------------------------------------- (b) exceedance
    bx = fig.add_subplot(gs[1])
    taus = sorted(ex.tau.unique())
    for p in S.PROCS:
        g = ex[ex.process == p].set_index("tau").loc[taus]
        bx.plot(taus, g.frac, "-", color=S.PROC[p], lw=1.3, zorder=2)
        bx.plot(taus, g.frac, marker="o", ls="", color=S.PROC[p], ms=4.0, zorder=3)
    bx.set_xscale("log")
    bx.set_xlim(0.0088, 0.058)
    bx.set_xticks(taus)
    bx.set_xticklabels([f"{t:g}" for t in taus])
    bx.xaxis.set_minor_formatter(NullFormatter())
    bx.set_ylim(0, 0.64)
    bx.set_yticks([0, 0.2, 0.4, 0.6])
    S.despine(bx)
    S.hgrid(bx)
    bx.set_xlabel("Loss threshold $\\tau$ (\u0394NSE)")
    bx.set_ylabel("Fraction with loss > $\\tau$")
    hl = [Line2D([], [], color=S.PROC[p], marker="o", lw=1.3, ms=4.0,
                 label=f"{S.NAME[p]} ($n$ = {int(ex[ex.process == p].n.iloc[0])})")
          for p in S.PROCS]
    bx.legend(handles=hl, loc="upper right", fontsize=S.FS_TICK, frameon=False,
              handlelength=1.3, labelspacing=0.28, borderaxespad=0.2)
    S.panel_title(bx, "b", "Decisive-OFF exceedance")

    S.save(fig, a.out, "FigureS3_refit_consequences")
    print("wrote", a.out / "FigureS3_refit_consequences.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
