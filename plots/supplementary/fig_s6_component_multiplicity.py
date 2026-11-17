#!/usr/bin/env python3
"""Figure S6 - parallel-component multiplicity (n_mul), Flex-MOPEX only.

(a) test-period NSE versus the number of parallel components, median of basin three-seed
    means with basin IQR, Full-MOPEX m = 4 as an open diamond
(b) learned process gates versus m, median across basins with P25-P75 bars
(c) exact Eq. (S6) evidence-class composition, one mini-facet per process

Input   manuscripts/tables/TableS11a_nmul_checkpoints_and_skill.csv
        manuscripts/tables/TableS11b_nmul_gates_and_evidence.csv
Output  manuscripts/figures/supplementary/fig_task1_nmul.{png,pdf}
        manuscripts/figures/supplementary/FigureS6_component_multiplicity.{png,pdf}
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
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
import paths  # noqa: E402
import wrrstyle as S  # noqa: E402

S.apply()

OFF = "#e6e6e6"
DX = {"snow": 0.90, "sub": 0.97, "phen": 1.03, "int": 1.11}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.TABLES)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "supplementary")
    a = ap.parse_args()

    sk = pd.read_csv(a.data / "TableS11a_nmul_checkpoints_and_skill.csv").sort_values("m")
    gb = pd.read_csv(a.data / "TableS11b_nmul_gates_and_evidence.csv")
    gt, ev = gb, gb
    ms = [int(v) for v in sk.m]

    fig = plt.figure(figsize=(S.FULL_WIDTH, 2.8))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.9], wspace=0.46,
                             left=0.092, right=0.988, bottom=0.215, top=0.775)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[0, 2])
    ax_c.axis("off")
    facets = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0, 2], wspace=0.14)
    caxes = [fig.add_subplot(facets[0, i]) for i in range(4)]

    # ---------------------------------------------------------------- (a) NSE
    ax_a.axvline(4, color=S.REF, lw=0.9, ls=(0, (4, 2)), zorder=1.5)
    ax_a.errorbar(ms, sk.nse_median_basin_means,
                  yerr=[sk.nse_median_basin_means - sk.nse_iqr_lo,
                        sk.nse_iqr_hi - sk.nse_median_basin_means],
                  fmt="o-", color=S.PROC["snow"], ms=3.8, lw=1.3, capsize=2.2,
                  elinewidth=0.9, zorder=4, label="Flex-MOPEX")
    full = float(sk[sk.m == 4].full_nse_median_basin_means.iloc[0])
    ax_a.plot([4], [full], "D", mfc="white", mec=S.REF, ms=5.6, mew=0.9, zorder=6,
              label=f"Full, $m$ = 4 ({full:.4f})")
    ax_a.text(4.25, 0.79, "Ref. ($m$ = 4)", ha="left", va="center", fontsize=S.FS_ANNOT,
              color=S.REF)
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(ms)
    ax_a.set_xticklabels([str(v) for v in ms])
    ax_a.set_xlim(0.85, 40)
    ax_a.set_ylim(0.28, 0.82)
    ax_a.set_xlabel("Components $m$")
    ax_a.set_ylabel("Test-period NSE")
    S.despine(ax_a)
    S.hgrid(ax_a)
    ax_a.legend(loc="lower right", fontsize=6.3, handletextpad=0.4, borderpad=0.2,
                labelspacing=0.28)
    S.panel_title(ax_a, "a", "Test-period NSE", pad=5.0)

    # ---------------------------------------------------------------- (b) gates
    ax_b.axvline(4, color=S.REF, lw=0.9, ls=(0, (4, 2)), zorder=1.5)
    for p in S.PROCS:
        s = gt[gt.process == p].sort_values("m")
        x = np.array(s.m, float) * DX[p]
        ax_b.vlines(x, s.gate_iqr_lo, s.gate_iqr_hi, color=S.PROC[p], lw=0.9, alpha=0.45,
                    zorder=2)
        ax_b.plot(x, s.gate_iqr_lo, "_", color=S.PROC[p], ms=2.6, mew=0.7, alpha=0.8, zorder=2)
        ax_b.plot(x, s.gate_iqr_hi, "_", color=S.PROC[p], ms=2.6, mew=0.7, alpha=0.8, zorder=2)
        ax_b.plot(s.m, s.gate_median, "o-", color=S.PROC[p], ms=3.5, lw=1.3, zorder=3,
                  label=S.NAME[p])
    ax_b.text(4.25, 0.98, "Ref. ($m$ = 4)", ha="left", va="top", fontsize=S.FS_ANNOT,
              color=S.REF)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(ms)
    ax_b.set_xticklabels([str(v) for v in ms])
    ax_b.set_xlim(0.85, 40)
    ax_b.set_ylim(0.15, 1.05)
    ax_b.set_xlabel("Components $m$")
    ax_b.set_ylabel("Learned gate $\\bar w_p$")
    S.despine(ax_b)
    S.hgrid(ax_b)
    S.panel_title(ax_b, "b", "Learned process gates", pad=5.0)

    # ------------------------------------------------- (c) evidence mini-facets
    groups = np.arange(len(ms), dtype=float)
    for i, (ax, p) in enumerate(zip(caxes, S.PROCS)):
        S.despine(ax)
        S.hgrid(ax)
        e = ev[ev.process == p].set_index("m").reindex(ms)
        ax.bar(groups, e.decisive_ON, width=0.62, color=S.PROC[p], lw=0, zorder=3)
        ax.bar(groups, e.uncertain, width=0.62, bottom=e.decisive_ON, color=S.TINT[p], lw=0,
               zorder=3)
        ax.bar(groups, e.decisive_OFF, width=0.62, bottom=e.decisive_ON + e.uncertain,
               color=OFF, lw=0, zorder=3)
        # subtle outline around the reference bar
        ax.add_patch(Rectangle((2 - 0.36, 0), 0.72, 671, fill=False, lw=1.0,
                               edgecolor=S.REF, ls=(0, (2, 1.6)), zorder=5, clip_on=False))
        ax.set_xticks(groups)
        ax.set_xticklabels(["1", "", "4", "", "", "32"], fontsize=6.4)
        ax.set_xlim(-0.62, len(ms) - 0.38)
        ax.set_ylim(0, 690)
        ax.set_yticks([0, 200, 400, 671])
        ax.tick_params(axis="x", length=0, pad=1.5)
        if i == 0:
            ax.set_ylabel("Catchments")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_title(S.NAME[p], fontsize=6.6, pad=2.2, color=S.PROC[p], fontweight="bold")
        if p == "sub":
            ax.annotate("0", (0, 0), textcoords="offset points", xytext=(0, 2.5),
                        ha="center", va="bottom", fontsize=6.2, color=S.INK,
                        fontweight="bold", zorder=7)
    ax_c.text(0.5, -0.13, "Components $m$", transform=ax_c.transAxes, ha="center", va="top",
              fontsize=S.FS_LABEL)
    fig.text(ax_c.get_position().x0, 0.858, "(c) Process-evidence classes",
             fontsize=S.FS_TITLE, fontweight="bold", color="#1a1a1a", ha="left", va="bottom")

    # stacked-state key, above panel (c) and clear of the mini-facets
    key = [Patch(color=S.INK, label="Decisive ON"),
           Patch(color=S.TINT["snow"], label="Uncertain"),
           Patch(color=OFF, label="Decisive OFF")]
    fig.legend(handles=key, loc="upper center", bbox_to_anchor=(0.62, 0.952), ncol=3,
               fontsize=S.FS_TICK - 0.5, handletextpad=0.3, columnspacing=0.8,
               borderpad=0.1, handlelength=0.9, frameon=False)

    # one shared process legend for (b) and (c), above the panels
    hl = [Line2D([], [], color=S.PROC[p], marker="o", lw=1.3, ms=3.8, label=S.NAME[p])
          for p in S.PROCS]
    fig.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.70, 0.998), ncol=4,
               fontsize=S.FS_TICK - 0.25, handletextpad=0.35, columnspacing=1.0,
               borderpad=0.1)

    S.save(fig, a.out, "fig_task1_nmul", extra_names=("FigureS6_component_multiplicity",))
    print("wrote", a.out / "FigureS6_component_multiplicity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
