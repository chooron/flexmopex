#!/usr/bin/env python3
"""Figure S7 - complexity weight (lambda) sensitivity, 13 points x 3 seeds.

(a) learned process gates versus lambda, median across basins with P25-P75 bars at five
    grid points
(b) exact Eq. (S6) decisive-ON catchment counts versus lambda
(c) upper: unpenalized training error and the number of negligible-snow basins whose snow
    gate stays above 0.5; lower: post hoc paired dNSE against the three-seed Full-MOPEX
    reference with 95% bootstrap CI

Input   manuscripts/tables/TableS12a_lambda_checkpoints_and_skill.csv
        manuscripts/tables/TableS12b_lambda_sparsity_snowfree_evidence.csv
Output  manuscripts/figures/supplementary/fig_task2_lambda.{png,pdf}
        manuscripts/figures/supplementary/FigureS7_lambda_sensitivity.{png,pdf}
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

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
import paths  # noqa: E402
import wrrstyle as S  # noqa: E402

S.apply()

REF = 0.007
BAR_LAMS = (0.0005, 0.002, 0.007, 0.02, 0.1)
SLATE = "#2b5c8f"
GREY = "#555555"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.TABLES)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "supplementary")
    a = ap.parse_args()

    sk = pd.read_csv(a.data / "TableS12a_lambda_checkpoints_and_skill.csv").sort_values("lam")
    dec = pd.read_csv(a.data / "TableS12b_lambda_sparsity_snowfree_evidence.csv").sort_values("lam")
    lc, sf = dec, dec

    fig = plt.figure(figsize=(S.FULL_WIDTH, 2.9))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.42], wspace=0.50,
                             hspace=0.34, left=0.088, right=0.898, bottom=0.205,
                             top=0.815)
    ax_a = fig.add_subplot(outer[:, 0])
    ax_b = fig.add_subplot(outer[:, 1])
    ax_c1 = fig.add_subplot(outer[0, 2])
    ax_c2 = fig.add_subplot(outer[1, 2])

    # ---------------------------------------------------------------- (a) gates
    ax_a.axvline(REF, color=S.REF, lw=0.9, ls=(0, (4, 2)), zorder=1.5)
    for p in S.PROCS:
        s = dec[["lam", f"gate_{p}_median", f"gate_{p}_iqr_lo", f"gate_{p}_iqr_hi"]].rename(
            columns={f"gate_{p}_median": "gate_median", f"gate_{p}_iqr_lo": "gate_iqr_lo",
                     f"gate_{p}_iqr_hi": "gate_iqr_hi"}).sort_values("lam")
        sel = s[s.lam.round(6).isin([round(v, 6) for v in BAR_LAMS])]
        ax_a.vlines(sel.lam, sel.gate_iqr_lo, sel.gate_iqr_hi, color=S.PROC[p], lw=0.8,
                    alpha=0.45, zorder=2)
        ax_a.plot(sel.lam, sel.gate_iqr_lo, "_", color=S.PROC[p], ms=2.4, mew=0.6,
                  alpha=0.75, zorder=2)
        ax_a.plot(sel.lam, sel.gate_iqr_hi, "_", color=S.PROC[p], ms=2.4, mew=0.6,
                  alpha=0.75, zorder=2)
        ax_a.plot(s.lam, s.gate_median, "o-", color=S.PROC[p], ms=3.0, lw=1.3, zorder=3,
                  label=S.NAME[p])
    ax_a.annotate("Ref. ($\\lambda$ = 0.007)", xy=(REF * 1.18, 0.985),
                  xycoords=("data", "axes fraction"), ha="left", va="top",
                  fontsize=S.FS_ANNOT, color=S.REF, bbox=S.HALO)
    ax_a.set_xscale("log")
    ax_a.set_xlim(3e-4, 2e-1)
    ax_a.set_ylim(0.18, 1.05)
    ax_a.set_xlabel("Complexity weight $\\lambda$")
    ax_a.set_ylabel("Learned gate $\\bar w_p$")
    S.despine(ax_a)
    S.hgrid(ax_a)
    S.panel_title(ax_a, "a", "Learned process gates", pad=5.0)

    # ------------------------------------------------------- (b) decisive counts
    ax_b.axvline(REF, color=S.REF, lw=0.9, ls=(0, (4, 2)), zorder=1.5)
    for p in S.PROCS:
        ax_b.plot(dec.lam, dec[f"{p}_decisive_ON"], "o-", color=S.PROC[p], ms=3.0, lw=1.3,
                  zorder=3, label=S.NAME[p])
    ax_b.annotate("Ref. ($\\lambda$ = 0.007)", xy=(REF * 1.18, 0.985),
                  xycoords=("data", "axes fraction"), ha="left", va="top",
                  fontsize=S.FS_ANNOT, color=S.REF, bbox=S.HALO)
    ax_b.set_xscale("log")
    ax_b.set_xlim(3e-4, 2e-1)
    ax_b.set_ylim(0, 671)
    ax_b.set_yticks([0, 200, 400, 600])
    ax_b.set_xlabel("Complexity weight $\\lambda$")
    ax_b.set_ylabel("Decisive-ON catchments")
    S.despine(ax_b)
    S.hgrid(ax_b)
    S.panel_title(ax_b, "b", "Decisive-ON catchments", pad=5.0)

    # ---------------------------------------------------- (c) diagnostics + skill
    for ax in (ax_c1, ax_c2):
        ax.axvline(REF, color=S.REF, lw=0.9, ls=(0, (4, 2)), zorder=1.5)
        ax.set_xscale("log")
        ax.set_xlim(3e-4, 2e-1)
        S.despine(ax)
        S.hgrid(ax)

    ax_c1.plot(lc.lam, lc.fit_loss_cf_mean, "s--", color=GREY, ms=3.0, lw=1.0, zorder=4)
    ax_c1.set_ylim(0.314, 0.339)
    ax_c1.set_yticks([0.315, 0.320, 0.325, 0.330, 0.335])
    ax_c1.set_ylabel("Unpenalized $\\bar J_{fit}$", color=GREY, fontsize=8.0)
    ax_c1.tick_params(axis="y", colors=GREY, labelsize=S.FS_TICK)
    ax_c1.tick_params(axis="x", labelbottom=False)
    axr = ax_c1.twinx()
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_color(SLATE)
    axr.fill_between(sf.lam, 0, sf["n_fracsnow_lt_0.01_wgt05"], color=SLATE, alpha=0.18,
                     lw=0, zorder=2)
    axr.plot(sf.lam, sf["n_fracsnow_lt_0.01_wgt05"], "v-", color=SLATE, ms=3.4, lw=1.1,
             zorder=5)
    axr.set_ylim(-4, 68)
    axr.set_yticks([0, 25, 50])
    axr.set_ylabel("Snow-free basins\n($\\bar w_{snow}$ > 0.5, of 105)", color=SLATE,
                   fontsize=7.6)
    axr.tick_params(axis="y", colors=SLATE, labelsize=S.FS_TICK)
    axr.text(0.0005, 59, "55", fontsize=S.FS_ANNOT, color=SLATE, ha="center", va="center",
             bbox=S.HALO, zorder=7)
    axr.text(0.0105, 7, "0", fontsize=S.FS_ANNOT, color=SLATE, ha="left", va="center",
             bbox=S.HALO, zorder=7)
    S.panel_title(ax_c1, "c", "Training fit and skill", pad=5.0)

    ax_c2.errorbar(sk.lam, sk.paired_nse_median,
                   yerr=[sk.paired_nse_median - sk.paired_nse_ci_lo,
                         sk.paired_nse_ci_hi - sk.paired_nse_median],
                   fmt="o-", color=S.PROC["snow"], ms=3.0, lw=1.2, capsize=1.5,
                   elinewidth=0.7, zorder=4)
    ax_c2.axhline(0, color=S.REF, lw=0.8, ls=":", zorder=2)
    ax_c2.set_ylim(-0.028, 0.007)
    ax_c2.set_yticks([0.0, -0.01, -0.02])
    ax_c2.set_yticklabels(["0.00", "−0.01", "−0.02"])
    ax_c2.set_ylabel("Paired $\\Delta$NSE", color=S.PROC["snow"], fontsize=8.0)
    ax_c2.tick_params(axis="y", colors=S.PROC["snow"], labelsize=S.FS_TICK)
    ax_c2.set_xlabel("Complexity weight $\\lambda$")

    # one shared process legend across the top of (a) and (b)
    hl = [Line2D([], [], color=S.PROC[p], marker="o", lw=1.3, ms=3.6, label=S.NAME[p])
          for p in S.PROCS]
    fig.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.42, 0.998), ncol=4,
               fontsize=S.FS_TICK - 0.25, handletextpad=0.35, columnspacing=1.1,
               borderpad=0.1)

    S.save(fig, a.out, "fig_task2_lambda", extra_names=("FigureS7_lambda_sensitivity",))
    print("wrote", a.out / "FigureS7_lambda_sensitivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
