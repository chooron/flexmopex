#!/usr/bin/env python3
"""Figure S1 - synthetic assignment rules and the corresponding learned gates.

Four panels: subsurface slope rule (scenario A) and phenology vegetation-range rule
(scenario B), each against its random-assignment control (scenario D). One frozen
basin-level table supplies every point; no statistic is computed here beyond the
three-seed mean that is already stored.

Input   manuscripts/tables/FigureS1_synthetic_assignment_points.csv
Output  manuscripts/figures/supplementary/FigureS1_synthetic_assignments.{png,pdf}
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

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
import paths  # noqa: E402
import wrrstyle as S  # noqa: E402

S.apply()

PANELS = [
    ("A", "sub", "Mean slope (m km$^{-1}$)", 13.675850, "13.68",
     "(a) Subsurface: rule-based (A)"),
    ("D", "sub", "Mean slope (m km$^{-1}$)", None, None,
     "(b) Subsurface: random (D)"),
    ("B", "phen", "Green vegetation fraction range", 0.315747975, "0.316",
     "(c) Phenology: rule-based (B)"),
    ("D", "phen", "Green vegetation fraction range", None, None,
     "(d) Phenology: random (D)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.TABLES / "figure_inputs")
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "supplementary")
    a = ap.parse_args()

    data = pd.read_csv(a.data / "FigureS1_synthetic_assignment_points.csv",
                       dtype={"gauge_id": str})
    fig, axes = plt.subplots(2, 2, figsize=(S.FULL_WIDTH, 4.8), sharey=True)
    fig.subplots_adjust(left=0.098, right=0.988, top=0.878, bottom=0.093,
                        wspace=0.10, hspace=0.35)

    for ax, (scen, proc, xlabel, thr, thr_txt, title) in zip(axes.flat, PANELS):
        d = data[(data.scenario == scen) & (data.process == proc)]
        assert len(d) == 671 and d.gauge_id.nunique() == 671, (scen, proc, len(d))
        col = S.PROC[proc]
        for split, marker, size, alpha, lw in (("train", "o", 16, 0.65, 0.0),
                                               ("holdout", "^", 22, 0.85, 0.7)):
            for truth in (1, 0):
                v = d[(d.split == split) & (d.truth == truth)]
                ax.scatter(v.x, v.gate_mean, s=size, marker=marker,
                           facecolors=col if truth else "none", edgecolors=col,
                           linewidths=lw, alpha=alpha, zorder=3 if truth else 2)
        ax.axhline(0.5, color=S.REF, lw=0.9, ls="--", zorder=1)
        if thr is not None:
            ax.axvline(thr, color="#1a1a1a", lw=0.9, ls=":", zorder=1)
            S.note(ax, thr * 1.02, 0.965, thr_txt, va="top", ha="left",
                   transform=ax.get_xaxis_transform())
        S.despine(ax)
        S.hgrid(ax)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel(xlabel)
        S.panel_title(ax, title[1], title[4:], pad=5.0)

    for ax in axes[:, 0]:
        ax.set_ylabel("Three-seed mean learned gate")

    handles = [Line2D([], [], marker="o", ls="", ms=4.6, mfc="0.35", mec="0.35",
                      label="Train, true ON"),
               Line2D([], [], marker="o", ls="", ms=4.6, mfc="none", mec="0.35", mew=0.8,
                      label="Train, true OFF"),
               Line2D([], [], marker="^", ls="", ms=5.0, mfc="0.35", mec="0.35",
                      label="Held out, true ON"),
               Line2D([], [], marker="^", ls="", ms=5.0, mfc="none", mec="0.35", mew=0.8,
                      label="Held out, true OFF"),
               Line2D([], [], color="#1a1a1a", ls=":", lw=1.0, label="Rule threshold")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=5,
               fontsize=S.FS_TICK, handletextpad=0.35, columnspacing=1.1, borderpad=0.1)

    S.save(fig, a.out, "FigureS1_synthetic_assignments")
    print("wrote", a.out / "FigureS1_synthetic_assignments.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
