"""Supplementary refit comparison: loss change and decisive-OFF exceedance."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import figstyle as S
from data_io import neg, read_csv

SHORT = "refit_vs_fixed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None, help="output basename (default: refit_vs_fixed)")
    parser.add_argument("--linthresh", type=float, default=0.01)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    lt = args.linthresh
    df = read_csv(args.data, "si_refit_diff.csv")
    ex = read_csv(args.data, "si_refit_exceed.csv")
    procs = [p for p in S.PROC if p in set(df.process)]
    fig = plt.figure(figsize=(width, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.26,
                          left=0.105, right=0.985, top=0.90, bottom=0.17)
    a = fig.add_subplot(gs[0])
    for i, p in enumerate(procs):
        y = len(procs) - 1 - i
        v = df[df.process == p]["diff"].values
        q5, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        a.add_patch(Rectangle((q25, y - 0.27), q75 - q25, 0.54, facecolor=S.LIGHT[p], edgecolor=S.DARK[p],
                              lw=0.7, zorder=2))
        a.plot([q5, q25], [y, y], color=S.DARK[p], lw=1.0, zorder=1)
        a.plot([q75, q95], [y, y], color=S.DARK[p], lw=1.0, zorder=1)
        a.plot([q50, q50], [y - 0.27, y + 0.27], color="black", lw=1.6, zorder=3)
        o = v[(v < q5) | (v > q95)]
        a.scatter(o, np.full(len(o), y), s=6, facecolor="none", edgecolor=S.DARK[p], lw=0.5, alpha=0.7, zorder=1)
    a.set_xscale("symlog", linthresh=lt, linscale=0.8)
    m = np.nanmax(np.abs(df["diff"].values))
    dec = int(np.ceil(np.log10(max(m, lt * 10))))
    a.set_xlim(-(10 ** dec) * 1.2, (10 ** dec) * 1.2)
    ticks = [-(10 ** k) for k in range(dec, int(np.log10(lt)) - 1, -1)] + [0] + \
            [10 ** k for k in range(int(np.log10(lt)), dec + 1)]
    a.xaxis.set_major_locator(FixedLocator(ticks))
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: neg(f"{v:g}")))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.axvline(0, color=S.INK, lw=0.6, zorder=0)
    a.set_yticks(range(len(procs))[::-1])
    a.set_yticklabels([S.PROC_SHORT[p] for p in procs])
    a.tick_params(axis="y", length=0)
    a.set_ylim(-0.6, len(procs) - 0.4)
    S.vgrid(a)
    a.set_xlabel("NSE$_{\\mathrm{LOPO}}$ − NSE$_{\\mathrm{fixed\\text{-}off}}$")
    S.panel_title(a, "a", "Change after refitting", y=1.03)

    b = fig.add_subplot(gs[1])
    taus = sorted(ex.tau.unique())
    xs = np.arange(len(taus))
    ends, labs, cols = [], [], []
    for p in [q for q in S.PROC if q in set(ex.process)]:
        g = ex[ex.process == p].set_index("tau").loc[taus]
        b.plot(xs, g.frac.values, "-o", color=S.DARK[p], lw=1.3, ms=4.2, zorder=3)
        ends.append(g.frac.values[-1])
        labs.append(f"{S.PROC_SHORT[p]} ({int(g.n.iloc[0])})")
        cols.append(S.DARK[p])
    b.set_ylim(0, 1.0)
    hl = [Line2D([], [], color=c, marker="o", lw=1.3, ms=4.2, label=t) for t, c in zip(labs, cols)]
    b.legend(handles=hl, loc="upper right", fontsize=S.FS_SMALL, handlelength=1.4, labelspacing=0.3,
             borderaxespad=0.3)
    b.set_xticks(xs)
    b.set_xticklabels([f"{t:g}" for t in taus])
    b.set_xlim(-0.2, xs[-1] + 0.2)
    S.hgrid(b)
    b.set_xlabel("Loss threshold $\\tau$ (ΔNSE)")
    b.set_ylabel("Fraction with loss > $\\tau$")
    S.panel_title(b, "b", "Decisive-OFF exceedance", y=1.03)
    S.save(fig, str(args.out), args.name or SHORT)


if __name__ == "__main__":
    main()
