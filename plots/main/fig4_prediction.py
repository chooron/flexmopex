"""Test-period skill: cumulative NSE curves and the catchment-paired difference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

import figstyle as S
from data_io import neg, read_csv, read_json

SHORT = "prediction"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--number", type=int, default=4)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    d = read_csv(args.data, "fig3_catchment_nse.csv")
    m = read_json(args.data, "fig3_summary.json")
    lo, hi = m["nse_xlim"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(width, 2.5),
                               gridspec_kw=dict(width_ratios=[1, 1.05]))
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.165, top=0.91, wspace=0.17)

    styles = [
        ("bench", "Seasonal benchmark", dict(color="#A7B3C4", ls=(0, (1.2, 1.2)), lw=1.4)),
        ("basic", "Basic-MOPEX", dict(color="#7C8CA3", ls="-", lw=1.1)),
        ("full", "Full-MOPEX", dict(color="#6CB4E4", ls="-", lw=2.6)),
        ("flex", "Flex-MOPEX", dict(color="#1F4E8C", ls=(0, (4, 2)), lw=1.3)),
    ]
    for key, lab, st in styles:
        v = np.sort(d[key].values)
        y = np.arange(1, len(v) + 1) / len(v)
        nlow = int((v < lo).sum())
        a.step(v, y, where="post", label=neg(f"{lab}  {m['median_summary'][key]:.2f}  ({nlow})"), **st)
        med = m["median_summary"][key]
        a.plot(med, 0.5, "o", ms=3.6, mfc=st["color"] if key != "full" else "#1F4E8C",
               mec="white", mew=0.6, zorder=5)
    a.axhline(0.5, color=S.GRID, lw=0.6, zorder=0)
    a.set_xlim(lo, hi)
    a.set_ylim(0, 1.0)
    a.set_xticks([-0.5, 0, 0.5, 1.0])
    a.set_xticklabels(["\u22120.5", "0", "0.5", "1"])
    a.set_xlabel("Test-period NSE")
    a.set_ylabel("Cumulative fraction of catchments")
    S.hgrid(a)
    leg = a.legend(loc="upper left", title=neg(f"Median  (n < {lo:g})"), fontsize=S.FS_TICK,
                   title_fontsize=S.FS_TICK, handlelength=2.0, borderaxespad=0.1)
    leg._legend_box.align = "left"
    S.panel_title(a, "a", "Test-period skill")

    diff = (d["flex"] - d["full"]).values
    x0, x1 = m["diff_xlim"]
    nl, nr = int((diff < x0).sum()), int((diff > x1).sum())
    inside = diff[(diff >= x0) & (diff <= x1)]
    b.hist(inside, bins=np.linspace(x0, x1, 49), color="#86BEE6", edgecolor="white", lw=0.4, zorder=2)
    ci = m["paired_ci"]
    b.axvspan(ci[0], ci[1], color="#1F4E8C", alpha=0.18, lw=0, zorder=1)
    b.axvline(m["paired_median"], color="#1F4E8C", lw=1.2, ls=(0, (3, 1.5)), zorder=3)
    b.axvline(0, color=S.INK, lw=0.7, zorder=3)
    b.set_xlim(x0 - 0.004, x1 + 0.004)
    b.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
    b.set_xticklabels(["\u22120.1", "\u22120.05", "0", "0.05", "0.1"])
    S.hgrid(b)
    b.set_xlabel("ΔNSE (Flex − Full)")
    b.set_ylabel("Catchments")
    yt = b.get_ylim()[1]
    b.text(0.985, 0.97, neg(f"Median {m['paired_median']:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"),
           transform=b.transAxes, ha="right", va="top", fontsize=S.FS_TICK, color="#1F4E8C")
    for xx, n, mk, ha, dx in ((x0, nl, "<", "left", 0.006), (x1, nr, ">", "right", -0.006)):
        b.plot(xx, 0.06 * yt, marker=mk, color="#555555", ms=4, clip_on=False, zorder=4)
        b.text(xx + dx, 0.06 * yt, f"{n}", ha=ha, va="center", fontsize=S.FS_SMALL, color="#555555")
    S.panel_title(b, "b", "Catchment-paired difference")
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
