#!/usr/bin/env python3
"""Figure 9 - standalone module.

Drawing logic lives here; shared fonts, colours and geometry helpers come from `_common.py`
(which is `figures.py` minus the per-figure functions). Reads the archived figure inputs and
writes Figure 9 into the archive's `figures/main`.
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
from _common import NAMES, W, _read, set_numbers

set_numbers(range(3, 11))


def fig7(data_dir, out_dir, n_region=None, data_file="fig7_regional.csv", png_path=None):
    d = _read(data_dir, data_file)
    rows = geo.REGION_ORDER + ["Pooled"]
    if n_region is None:
        cat = _read(data_dir, "catchments.csv")
        n_region = cat.region.value_counts().to_dict()
        n_region["Pooled"] = len(cat)

    cw, gap_m, gap_p = 1.0, 0.12, 0.30      # cell width, gap between metrics, gap between processes
    rh, bh = 1.0, 0.64                      # row pitch, bar height
    ypos = {r: -(i + (0.45 if r == "Pooled" else 0)) * rh for i, r in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(W, 3.05))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.11)
    x = 0.0
    col_x = {}
    for p in S.PROC:
        for m in ("BA", "AUC"):
            col_x[(p, m)] = x
            x += cw + gap_m
        x += gap_p - gap_m
    xmax = x - gap_p

    for (p, m), x0 in col_x.items():
        dark = S.DARK[p] if m == "BA" else S.LIGHT[p]
        for r in rows:
            row = d[(d.process == p) & (d.metric == m) & (d.region == r)].iloc[0]
            y = ypos[r] - bh / 2
            v, small = row.value, bool(row.small)
            if np.isnan(v):
                ax.add_patch(Rectangle((x0, y), cw, bh, facecolor="#E8E8E8", edgecolor="none", zorder=2))
                ax.text(x0 + cw / 2, ypos[r], "—", ha="center", va="center", fontsize=S.FS_TICK,
                        color="#555555", zorder=4)
                continue
            ax.add_patch(Rectangle((x0, y), cw, bh, facecolor="#EFEFEF", lw=0, zorder=1))
            ax.plot([x0 + 0.5 * cw] * 2, [y, y + bh], color="#6E6E6E", lw=0.6, ls=(0, (1.5, 1.2)), zorder=3)
            if small:
                ax.add_patch(Rectangle((x0, y), v * cw, bh, facecolor="white",
                                       edgecolor=S.DARK[p], lw=0.9, zorder=2))
                tcol = "#4A4A4A"
            else:
                ax.add_patch(Rectangle((x0, y), v * cw, bh, facecolor=dark, lw=0, zorder=2))
                tcol = "white" if m == "BA" else "#1A1A1A"
            fmt_v = f"{v:.3f}" if r == "Pooled" else f"{v:.2f}"
            if v >= 0.34:
                ax.text(x0 + v * cw - 0.04, ypos[r], fmt_v, ha="right", va="center",
                        fontsize=S.FS_TICK, color=tcol, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white" if small else dark, ec="none"),
                        fontweight="bold" if r == "Pooled" else "normal")
            else:
                ax.text(x0 + v * cw + 0.04, ypos[r], fmt_v, ha="left", va="center",
                        fontsize=S.FS_TICK, color="#4A4A4A", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.08", fc="#EFEFEF", ec="none"))
    ytop = ypos["R1"] + bh / 2
    for p in S.PROC:
        xa = col_x[(p, "BA")]
        xb = col_x[(p, "AUC")] + cw
        ax.plot([xa, xb], [ytop + 0.95, ytop + 0.95], color=S.DARK[p], lw=2.2, solid_capstyle="butt")
        ax.text((xa + xb) / 2, ytop + 1.23, S.PROC_LABEL[p], ha="center", va="bottom",
                fontsize=S.FS_TITLE, fontweight="bold", color="#1A1A1A")
        ax.text(xa + cw / 2, ytop + 0.18, "Evidence\nagreement", ha="center", va="bottom",
                fontsize=S.FS_SMALL, color="#4A4A4A", linespacing=0.95)
        ax.text(col_x[(p, "AUC")] + cw / 2, ytop + 0.18, "Direction\nAUC", ha="center", va="bottom",
                fontsize=S.FS_SMALL, color="#4A4A4A", linespacing=0.95)
    for r in rows:
        ax.text(-0.12, ypos[r], r, ha="right", va="center", fontsize=S.FS_LABEL,
                fontweight="bold" if r == "Pooled" else "normal")
        ax.text(xmax + 0.12, ypos[r], f"{n_region.get(r, '')}", ha="left", va="center",
                fontsize=S.FS_TICK, color="#4A4A4A")
    ax.text(xmax + 0.12, ytop + 0.18, "n", ha="left", va="bottom", fontsize=S.FS_SMALL, color="#4A4A4A")
    ysep = (ypos["R7"] + ypos["Pooled"]) / 2
    ax.plot([-0.05, xmax], [ysep, ysep], color="#9A9A9A", lw=0.6)
    ax.set_xlim(-0.80, xmax + 0.45)
    ax.set_ylim(ypos["Pooled"] - 0.45, ytop + 1.85)
    ax.set_axis_off()
    legend = [Patch(facecolor="#777777", edgecolor="none", label="ON and OFF ≥10"),
              Patch(facecolor="white", edgecolor="#555555", label="ON or OFF <10"),
              Patch(facecolor="#E8E8E8", edgecolor="none", label="—  Not computable"),
              Line2D([], [], color="#6E6E6E", lw=0.8, ls=(0, (1.5, 1.2)), label="0.5 reference")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=4, fontsize=S.FS_SMALL, handlelength=1.2, columnspacing=1.0)
    if png_path is None:
        S.save(fig, out_dir, NAMES["regional"])
    else:
        # Candidate-only export: require a new filename and never generate PDF.
        with open(png_path, "xb") as f:
            fig.savefig(f, format="png", dpi=600)
        plt.close(fig)

def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 9")
    ap.add_argument("--data", type=Path, default=paths.FIG_INPUTS)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "main")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = None

    fig7(a.data, a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
