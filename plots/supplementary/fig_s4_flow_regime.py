#!/usr/bin/env python3
"""Figure S4 - flow-regime contributions to the Full-minus-LOPO NSE difference.

Every process row carries its own canonical hue and the four tracks are separated by luminance
instead of by foreign hues: Total (charcoal reference line) is drawn above High flow (solid
process colour), Middle flow (55 % tint) and Low flow (22 % tint). The "high contribution
largest" callouts come from Table S8 dominance.

(a) all valid catchments (n = 670), (b) larger-loss crossover cases (mean loss > 0.05).
The four regime contributions share one full-period SST denominator, so they add within a
catchment; separate medians are not additive, which is why both are drawn as median and IQR.

Input   manuscripts/tables/TableS8_flow_regime_contributions.csv
Output  manuscripts/figures/supplementary/FigureS4_flow_regime.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
import paths  # noqa: E402
import wrrstyle as S  # noqa: E402

S.apply()

TOTAL_INK = "#2b2b2b"
SLATE = {"total": "#4d5a63", "high": "#5d6b74", "mid": "#9aa6ad", "low": "#d3dade"}
# track offset inside a process group: Total on top, then High, Middle, Low
TRACKS = [("loss", "total", -0.27, 0.00, "circle"),
          ("highflow", "high", -0.09, 0.00, "diamond"),     # solid process colour
          ("midflow", "mid", 0.09, 0.52, "triangle"),        # 48 % tint
          ("lowflow", "low", 0.27, 0.80, "square")]          # 20 % tint
FILL = {"total": None, "high": 0.0, "mid": 0.45, "low": 0.78}


def _darken(hex_colour: str, f: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(round(c * (1 - f))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.TABLES / "figure_inputs")
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "supplementary")
    a = ap.parse_args()

    b = pd.read_csv(a.data / "TableS8_flow_regime_contributions.csv")

    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_WIDTH, 4.1), sharey=True)
    fig.subplots_adjust(left=0.108, right=0.985, top=0.800, bottom=0.115, wspace=0.10)

    panels = [("all", "(a) All valid catchments ($n$ = 670)"),
              ("cross_loss_gt_005", "(b) Larger-loss crossover (loss > 0.05)")]
    for ax, (group, title) in zip(axes, panels):
        for j in range(1, len(S.PROCS)):
            ax.axhline(j - 0.5, color="#e6e6e6", lw=0.7, zorder=0)

        for j, p in enumerate(S.PROCS):
            for metric, key, dy, tint_f, marker in TRACKS:
                d = b[(b.group == group) & (b.process == p) & (b.metric == metric)]
                if d.empty:
                    continue
                d = d.iloc[0]
                y = j + dy
                if key == "total":
                    ax.plot([d.p25, d.p75], [y, y], color=TOTAL_INK, lw=1.0, zorder=4)
                    ax.plot([d.p25, d.p75], [y, y], "|", color=TOTAL_INK, ms=3.2, mew=0.8,
                            zorder=4)
                    ax.plot(d["median"], y, marker="o", ms=3.6, mfc="white",
                            mec=TOTAL_INK, mew=1.0, zorder=5)
                else:
                    ax.add_patch(plt.Rectangle((0, y - 0.075), d["median"], 0.15,
                                               facecolor=S.tint(p, tint_f),
                                               edgecolor=S.PROC[p], lw=0.7, zorder=3))
                    ax.plot([d.p25, d.p75], [y, y], color="white", lw=2.2, zorder=4)
                    ax.plot([d.p25, d.p75], [y, y], color=_darken(S.PROC[p], 0.35), lw=0.9,
                            zorder=5)
                    ax.plot([d.p25, d.p75], [y, y], "|", color=_darken(S.PROC[p], 0.35),
                            ms=3.2, mew=0.8, zorder=5)
            frac = b[(b.group == group) & (b.process == p)].iloc[0]
            txt = (f"high largest\n{int(frac.high_largest_n)}/{int(frac.n)} "
                   f"({100 * frac.high_largest_fraction:.0f}%)")
            ax.text(0.992, j + 0.02, txt, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=S.FS_ANNOT, color="#444444", linespacing=1.25)

        used = [m for m, _, _, _, _ in TRACKS]
        sub = b[(b.group == group) & (b.metric.isin(used))]
        hi = 1.58 * max(sub.p75.max(), sub["median"].max())
        lo = min(-0.02, 1.08 * sub.p25.min())
        ax.set_xlim(lo, hi)
        ax.axvline(0, color=S.ZERO, lw=0.9, zorder=1)
        ax.set_yticks(range(len(S.PROCS)))
        ax.set_yticklabels([S.NAME[p] for p in S.PROCS])
        ax.set_ylim(len(S.PROCS) - 0.42, -0.58)
        ax.tick_params(axis="y", length=0)
        S.despine(ax)
        S.hgrid(ax, "x")
        ax.set_xlabel("Contribution to Full \u2212 LOPO NSE")
        S.panel_title(ax, title[1], title[4:])

    hl = [Line2D([], [], color=SLATE["total"], marker="o", mfc="white", mew=1.0, ms=4.0,
                 lw=1.0, label="Total loss (median, IQR)"),
          Patch(facecolor=SLATE["high"], edgecolor=S.INK, lw=0.6,
                label="High flow ($Y_{b,t}$ > $P_{90}$),\n"
                      "above the 90th observed-flow percentile"),
          Patch(facecolor=SLATE["mid"], edgecolor=S.INK, lw=0.6,
                label="Middle flow ($P_{50}$ < $Y_{b,t}$ \u2264 $P_{90}$, medium shade)"),
          Patch(facecolor=SLATE["low"], edgecolor=S.INK, lw=0.6,
                label="Low flow ($Y_{b,t}$ \u2264 $P_{50}$, light shade)")]
    fig.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=2,
               fontsize=S.FS_TICK, handletextpad=0.5, columnspacing=1.6, borderpad=0.1,
               labelspacing=0.35)

    S.save(fig, a.out, "FigureS4_flow_regime")
    print("wrote", a.out / "FigureS4_flow_regime.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
