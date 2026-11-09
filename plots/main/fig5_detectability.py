"""Detectability: gate against the counterfactual signal, plus true-ON recall."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

import figstyle as S
from data_io import read_csv, read_json

SHORT = "detectability_recovery"
LIN4 = 0.02


def symlog_x(ax, lo=-0.15, hi=2.0):
    ax.set_xscale("symlog", linthresh=LIN4, linscale=1.0)
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([-0.1, 0, 0.1, 1]))
    ax.set_xticklabels(["−0.1", "0", "0.1", "1"])
    ax.xaxis.set_minor_formatter(NullFormatter())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--number", type=int, default=5)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    sc = read_csv(args.data, "fig4_scatter.csv")
    rc = read_csv(args.data, "fig4_recall.csv")
    P = read_json(args.data, "fig4_meta.json")["P_p"]
    fig = plt.figure(figsize=(width, 4.15))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.82], hspace=0.58, wspace=0.10,
                          left=0.08, right=0.995, top=0.94, bottom=0.10)
    for i, p in enumerate(S.PROC):
        ax = fig.add_subplot(gs[0, i])
        d = sc[sc.process == p]
        ax.axvspan(-1, 0, color=S.BAND_NEG, lw=0, zorder=0)
        ax.axvspan(0, P[p], color=S.BAND[p], lw=0, zorder=0)
        ax.axvline(0, color=S.INK, lw=0.6, zorder=1)
        ax.axvline(P[p], color=S.DARK[p], lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0.5, color=S.REF, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        off = d[d.truth == 0]
        on = d[d.truth == 1]
        ax.scatter(off.D_J, off.w, s=7, facecolor="none", edgecolor=S.GRAY_EDGE, lw=0.45, zorder=2)
        ax.scatter(on.D_J, on.w, s=7, color=S.DARK[p], alpha=0.85, lw=0, zorder=3)
        symlog_x(ax)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if i == 0:
            ax.set_ylabel("Learned gate $w$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("$D_J$", labelpad=1)
        ax.text(P[p], 0.985, " $P_p$", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=S.FS_SMALL, color=S.DARK[p])
        S.panel_title(ax, "abcd"[i], S.PROC_LABEL[p], y=1.03)

    top4 = fig.axes[:4]
    ax = fig.add_subplot(gs[1, :])
    order = ["high", "weak", "negative"]
    bw, gap = 0.26, 0.2
    xt, xl = [], []
    for i, p in enumerate(S.PROC):
        for j, s in enumerate(order):
            r = rc[(rc.process == p) & (rc.stratum == s)].iloc[0]
            x = i * (3 * bw + gap) + j * bw
            col = S.DARK[p] if s == "high" else S.LIGHT[p]
            if r.n < 10:
                ax.bar(x, r["mean"], bw * 0.92, color=S.LIGHT[p], alpha=0.35, edgecolor=S.DARK[p],
                       lw=0.7, ls=(0, (2, 1.2)), zorder=2)
            else:
                ax.bar(x, r["mean"], bw * 0.92, color=col, lw=0, zorder=2, hatch=None)
                if s == "negative":
                    ax.bar(x, r["mean"], bw * 0.92, color="none", edgecolor="white", hatch="////",
                           lw=0, zorder=3)
                if r.hi > r.lo:
                    ax.plot([x, x], [r.lo, r.hi], color="black", lw=0.8, zorder=4)
            ax.text(x, -0.035, f"{int(r.n)}", ha="center", va="top", fontsize=S.FS_SMALL,
                    color="#555555" if r.n >= 10 else "#9A9A9A", transform=ax.transData)
        xt.append(i * (3 * bw + gap) + bw)
        xl.append(S.PROC_LABEL[p])
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.25, 3 * (3 * bw + gap) + 2 * bw + 0.2)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("ON recall ($w$ > 0.5)")
    S.hgrid(ax)
    ax.text(-0.25, -0.035, "n", ha="left", va="top", fontsize=S.FS_SMALL, color="#555555")
    S.panel_title(ax, "e", "True-ON recall by detectability stratum", y=1.03)
    h = [Patch(color="#4A4A4A", label="High"), Patch(color="#BDBDBD", label="Weak"),
         Patch(facecolor="#BDBDBD", edgecolor="white", hatch="////", label="Negative"),
         Line2D([], [], color="black", lw=0.8, label="Seed range")]
    ax.legend(handles=h, loc="upper right", ncol=4, fontsize=S.FS_TICK, handlelength=1.2,
              bbox_to_anchor=(1.0, 1.02))

    items = [
        (Line2D([], [], marker="o", ls="", color=S.DARK["snow"], ms=4.5), "True ON (process color)"),
        (Line2D([], [], marker="o", ls="", mfc="none", mec=S.GRAY_EDGE, ms=4.5), "True OFF"),
        (Patch(color=S.BAND["snow"]), "Weak: 0 ≤ $D_J$ ≤ $P_p$"),
        (Patch(color=S.BAND_NEG), "Negative: $D_J$ < 0"),
        (Patch(facecolor=S.LIGHT["snow"], alpha=0.35, edgecolor=S.DARK["snow"], ls="--"), "n < 10 (no range)"),
    ]
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    ybot = min(t.xaxis.label.get_window_extent(rr).y0 for t in top4) / fig.bbox.height
    fig.legend([a for a, _ in items], [b for _, b in items], loc="upper center", ncol=5,
               fontsize=S.FS_TICK, handlelength=1.3, columnspacing=1.4, borderaxespad=0,
               bbox_to_anchor=((top4[0].get_position().x0 + top4[-1].get_position().x1) / 2, ybot - 0.012))
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
