#!/usr/bin/env python3
"""Figure 4 - standalone module.

Drawing logic lives here; shared fonts, colours and geometry helpers come from `_common.py`
(which is `figures.py` minus the per-figure functions). Reads the archived figure inputs and
writes Figure 4 into the archive's `figures/main`.
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
from _common import NAMES, W, _json, _neg, _read, set_numbers

set_numbers(range(3, 11))


def fig3(data_dir, out_dir):
    d = _read(data_dir, "fig3_catchment_nse.csv")
    m = _json(data_dir, "fig3_summary.json")
    medians = {k: float(np.median(d[k].values)) for k in ("bench", "basic", "full", "flex")}
    lo, hi = m["nse_xlim"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(width_ratios=[1, 1.05]))
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.165, top=0.91, wspace=0.17)

    styles = [
        ("bench", "Seasonal climatology", dict(color="#A7B3C4", ls=(0, (1.2, 1.2)), lw=1.4)),
        ("basic", "Basic-MOPEX", dict(color="#7C8CA3", ls="-", lw=1.1)),
        ("full", "Full-MOPEX", dict(color="#6CB4E4", ls="-", lw=2.6)),
        ("flex", "Flex-MOPEX", dict(color="#1F4E8C", ls=(0, (4, 2)), lw=1.3)),
    ]
    for key, lab, st in styles:
        v = np.sort(d[key].values)
        y = np.arange(1, len(v) + 1) / len(v)
        nlow = int((v < lo).sum())
        mlab = _neg(f"{medians[key]:.4f}")
        a.step(v, y, where="post", label=f"{lab}  {mlab}  ({nlow})", **st)
        med = medians[key]
        a.plot(med, 0.5, "o", ms=3.6, mfc=st["color"] if key != "full" else "#1F4E8C",
               mec="white", mew=0.6, zorder=5)
    a.axhline(0.5, color=S.GRID, lw=0.6, zorder=0)
    a.set_xlim(lo, hi); a.set_ylim(0, 1.0)
    a.set_xticks([-0.5, 0, 0.5, 1.0]); a.set_xticklabels(["\u22120.5", "0", "0.5", "1"])
    a.set_xlabel("Test-period NSE"); a.set_ylabel("Cumulative fraction of basins")
    S.hgrid(a)
    leg = a.legend(loc="upper left", title=_neg(f"Median  (n < {lo:g})"), fontsize=S.FS_TICK,
                   title_fontsize=S.FS_TICK, handlelength=2.0, borderaxespad=0.1)
    leg._legend_box.align = "left"
    S.panel_title(a, "a", "Test-period skill")

    diff = (d["flex"] - d["full"]).values
    x0, x1 = m["diff_xlim"]
    nl, nr = int((diff < x0).sum()), int((diff > x1).sum())
    inside = diff[(diff >= x0) & (diff <= x1)]
    b.hist(inside, bins=np.linspace(x0, x1, 49), color="#86BEE6", edgecolor="white", lw=0.4, zorder=2)
    ci = m["paired_ci"]
    b.axvspan(ci[0], ci[1], facecolor="#9E9E9E", alpha=0.30, lw=0, zorder=4)
    for _cb in ci:
        b.axvline(_cb, color="#9E9E9E", lw=0.6, zorder=5)
    b.axvline(m["paired_median"], color="#1F4E8C", lw=1.2, ls=(0, (3, 1.5)), zorder=6)
    b.axvline(0, color=S.INK, lw=0.7, zorder=6)
    b.set_xlim(x0 - 0.004, x1 + 0.004)
    b.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    b.set_xticklabels(["\u22120.1", "\u22120.05", "0", "0.05", "0.1"])
    S.hgrid(b)
    b.set_xlabel("NSE difference (Flex − Full)"); b.set_ylabel("Basins")
    yt = b.get_ylim()[1]
    b.text(0.985, 0.97, _neg(f"Median = {m['paired_median']:.4f}\n95% CI [{ci[0]:.4f}, {ci[1]:.4f}]"),
           transform=b.transAxes, ha="right", va="top", fontsize=S.FS_TICK, color="#1F4E8C")
    for xx, n, mk, ha, dx in ((x0, nl, "<", "left", 0.006), (x1, nr, ">", "right", -0.006)):
        b.plot(xx, 0.06 * yt, marker=mk, color="#555555", ms=4, clip_on=False, zorder=4)
        b.text(xx + dx, 0.06 * yt, f"{n}", ha=ha, va="center", fontsize=S.FS_SMALL, color="#555555")
    S.panel_title(b, "b", "Basin-paired difference")
    S.save(fig, out_dir, NAMES["prediction"])

def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 4")
    ap.add_argument("--data", type=Path, default=paths.FIG_INPUTS)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "main")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = None

    fig3(a.data, a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
