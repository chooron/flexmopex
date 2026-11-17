#!/usr/bin/env python3
"""Figure S5 - metric sensitivity of the process-removal losses.

Row 1 (a): median and interquartile range of the seed-mean Full-minus-LOPO loss over 671
catchments for four evaluation scales, coloured by process.
Row 2 (b): within-catchment process-pair ranking reversals relative to NSE (4,026 pairs),
including the seasonal-normalised loss, which is rank-preserving by construction.

Input   manuscripts/tables/TableS9a_metric_sensitivity_all671.csv   (six evaluation scales)
        manuscripts/tables/TableS9b_metric_rank_reversals.csv       (pairwise reversals)
Output  manuscripts/figures/supplementary/FigureS5_metric_sensitivity.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
import paths  # noqa: E402
import wrrstyle as S  # noqa: E402

S.apply()

BAR = "#3b6fb6"
N_PAIRS = 4026
RANK_ROWS = [("ds_clim", "Seasonal-norm. $\\Delta S^{\\mathrm{clim}}$"),
             ("kge", "KGE"),
             ("log1", "logNSE (10%)"),
             ("log01", "logNSE (1%)"),
             ("log001", "logNSE (0.1%)")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.TABLES / "figure_inputs")
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "supplementary")
    a = ap.parse_args()

    tab = pd.read_csv(a.data / "TableS9a_metric_sensitivity_all671.csv")
    rnk = pd.read_csv(a.data / "TableS9b_metric_rank_reversals.csv").set_index("metric")

    facets = []
    for metric, title in (("nse", "NSE"),
                          ("ds_clim", "Seasonal-norm. $\\Delta S^{\\mathrm{clim}}$"),
                          ("kge", "KGE"), ("log01", "logNSE (1%)")):
        d = tab[tab.metric == metric].set_index("process").reindex(S.PROCS)
        facets.append(dict(title=title, med=d["median"], lo=d["p25"], hi=d["p75"]))

    fig = plt.figure(figsize=(S.FULL_WIDTH, 4.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 2.15], hspace=1.62, wspace=0.22,
                          left=0.158, right=0.952, top=0.838, bottom=0.132)

    for i, f in enumerate(facets):
        ax = fig.add_subplot(gs[0, i])
        for j, p in enumerate(S.PROCS):
            ax.plot([f["lo"][p], f["hi"][p]], [j, j], color=S.PROC[p], lw=2.0,
                    solid_capstyle="butt", zorder=3)
            ax.plot(f["med"][p], j, "o", color=S.PROC[p], ms=4.0, zorder=4)
        ax.axvline(0, color=S.ZERO, lw=0.9, zorder=1)
        lo = min(-0.02, 1.07 * float(f["lo"].min()))
        hi = 1.07 * max(float(f["hi"].max()), float(f["med"].max()))
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
        ax.set_ylim(len(S.PROCS) - 0.45, -0.55)
        ax.set_yticks(range(len(S.PROCS)))
        ax.set_yticklabels([S.NAME[p] for p in S.PROCS] if i == 0 else [])
        ax.tick_params(axis="y", length=0 if i else 3)
        S.despine(ax)
        S.hgrid(ax, "x")
        ax.set_title(f["title"], loc="center", fontsize=S.FS_TICK + 0.25, pad=4.5,
                     color="#1a1a1a")
        ax.set_xlabel("Full \u2212 LOPO metric", labelpad=1.5)

    fig.text(0.158, 0.947, "(a) Process-wise LOPO removal losses across evaluation scales",
             fontsize=S.FS_TITLE, fontweight="bold", color="#1a1a1a", ha="left", va="top")

    # ------------------------------------------------------------- (b) reversals
    axb = fig.add_subplot(gs[1, :])
    vals = [0] + [int(rnk.loc[k, "pair_reversals"]) for k, _ in RANK_ROWS[1:]]
    ys = range(len(RANK_ROWS))
    axb.barh(list(ys), vals, height=0.62, color=BAR, zorder=3)
    axb.set_yticks(list(ys))
    axb.set_yticklabels([lbl for _, lbl in RANK_ROWS])
    axb.invert_yaxis()
    axb.set_xlim(0, 1800)
    axb.set_ylim(len(RANK_ROWS) - 0.45, -0.55)
    axb.tick_params(axis="y", length=0)
    S.despine(axb)
    S.hgrid(axb, "x")
    axb.set_xlabel("Ranking reversals relative to NSE (pairs)")
    top = axb.secondary_xaxis("top", functions=(lambda v: v / N_PAIRS * 100,
                                               lambda p: p * N_PAIRS / 100))
    top.set_xlabel("Share of pairs (%)", fontsize=S.FS_TICK, labelpad=7)
    top.tick_params(labelsize=S.FS_TICK)

    pos = axb.get_position()
    fig.text(0.158, pos.y1 + 0.138,
             "(b) Within-catchment pairwise process-rank reversals relative to NSE (out of 4,026 pairs)",
             fontsize=S.FS_TITLE, fontweight="bold", color="#1a1a1a", ha="left", va="bottom")

    for y, v in zip(ys, vals):
        if v == 0:
            S.note(axb, 22, y, f"0 / {N_PAIRS:,} (0.0%) \u2014 invariant within catchment",
                   va="center", ha="left")
        else:
            S.note(axb, v + 22, y, f"{v:,} ({v / N_PAIRS * 100:.1f}%)", va="center", ha="left")

    S.save(fig, a.out, "FigureS5_metric_sensitivity")
    print("wrote", a.out / "FigureS5_metric_sensitivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
