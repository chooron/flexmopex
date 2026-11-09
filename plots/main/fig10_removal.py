"""Process removal: omission loss per catchment and the ON − OFF contrast."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter, FuncFormatter

import figstyle as S
from data_io import neg, read_csv, read_json

SHORT = "process_removal"
GAP_FILL = "#E4E4E4"


def snap_x(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--number", type=int, default=10)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    sc = read_csv(args.data, "fig8_scatter.csv")
    con = read_csv(args.data, "fig8_contrast.csv")
    meta = read_json(args.data, "fig8_meta.json")
    lt = meta["loss_linthresh"]
    ylo, yhi = meta["loss_ylim"]

    fig = plt.figure(figsize=(width, 4.85))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.60,
                          left=0.085, right=0.98, top=0.945, bottom=0.135)
    gt = gs[0].subgridspec(1, 4, wspace=0.10)
    gb = gs[1].subgridspec(1, 2, wspace=0.07)

    # ---- (a-d) scatter
    top_axes = []
    for i, p in enumerate(S.PROC):
        ax = fig.add_subplot(gt[0, i])
        top_axes.append(ax)
        d = sc[sc.process == p]
        ax.axhline(0, color=S.REF, lw=0.6, zorder=1)
        ax.axvline(0.5, color=S.REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
        colmap = {"ON": S.DARK[p], "OFF": S.LIGHT[p], "UNC": S.GRAY_FILL}
        for cls in ("UNC", "OFF", "ON"):
            dd = d[(d.cls == cls) & (d.loss >= ylo) & (d.loss <= yhi)]
            filled = dd.above_rb == 1
            ax.scatter(dd.w_mean[filled], dd.loss[filled], s=7, color=colmap[cls], lw=0, alpha=0.9, zorder=3)
            ax.scatter(dd.w_mean[~filled], dd.loss[~filled], s=7, facecolor="none", edgecolor=colmap[cls],
                       lw=0.5, zorder=2)
            hi_ = d[(d.cls == cls) & (d.loss > yhi)]
            lo_ = d[(d.cls == cls) & (d.loss < ylo)]
            ax.scatter(hi_.w_mean, np.full(len(hi_), yhi), marker="^", s=13, color=colmap[cls],
                       lw=0, clip_on=False, zorder=5)
            ax.scatter(lo_.w_mean, np.full(len(lo_), ylo), marker="v", s=13, color=colmap[cls],
                       lw=0, clip_on=False, zorder=5)
        ax.set_yscale("symlog", linthresh=lt, linscale=1.0)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks([-0.1, 0, 0.1, 1])
        ax.set_yticklabels(["\u22120.1", "0", "0.1", "1"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        S.hgrid(ax)
        if i == 0:
            ax.set_ylabel("LOPO omission loss (ΔNSE)")
        else:
            ax.set_yticklabels([])
        S.panel_title(ax, "abcd"[i], S.PROC_LABEL[p], y=1.03)
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    xc = (top_axes[0].get_position().x0 + top_axes[-1].get_position().x1) / 2
    ytk = min(t.get_window_extent(rr).y0 for t in top_axes[0].get_xticklabels()) / fig.bbox.height
    xl = fig.text(xc, ytk - 0.008, "Learned gate $\\bar{w}$ (three-seed mean)", ha="center", va="top",
                  fontsize=S.FS_LABEL)
    fig.canvas.draw()
    yxl = xl.get_window_extent(rr).y0 / fig.bbox.height
    h = [Line2D([], [], marker="o", ls="", color=S.DARK["snow"], ms=4.5, label="Decisive ON"),
         Line2D([], [], marker="o", ls="", color=S.LIGHT["snow"], ms=4.5, label="Decisive OFF"),
         Line2D([], [], marker="o", ls="", color=S.GRAY_FILL, ms=4.5, label="Uncertain"),
         Line2D([], [], marker="o", ls="", color="#555555", ms=4.5, label="Loss > $R_b$"),
         Line2D([], [], marker="o", ls="", mfc="none", mec="#555555", ms=4.5, label="Loss ≤ $R_b$"),
         Line2D([], [], marker="^", ls="", color="#555555", ms=4.5, label="Off-scale")]
    fig.legend(handles=h, loc="upper center", ncol=6, fontsize=S.FS_TICK, bbox_to_anchor=(xc, yxl - 0.010),
               handletextpad=0.2, columnspacing=1.3, borderaxespad=0)

    ys = {p: len(S.PROC) - 1 - i for i, p in enumerate(S.PROC)}
    off = 0.17

    # ---- (e) ON vs OFF loss level (what the contrast is made of)
    e = fig.add_subplot(gb[0, 0])
    for p in S.PROC:
        y = ys[p]
        q = {}
        for cls in ("ON", "OFF"):
            v = sc[(sc.process == p) & (sc.cls == cls)].loss.values
            q[cls] = np.percentile(v, [25, 50, 75])
        a_, b_ = sorted([q["OFF"][1], q["ON"][1]])
        e.add_patch(Rectangle((a_, y - off - 0.13), b_ - a_, 2 * off + 0.26, facecolor=GAP_FILL,
                              lw=0, zorder=0.5))
        for cls, dy, col in (("ON", off, S.DARK[p]), ("OFF", -off, S.LIGHT[p])):
            e.plot([q[cls][0], q[cls][2]], [y + dy] * 2, color=col, lw=1.6, solid_capstyle="butt", zorder=2)
            e.plot(q[cls][1], y + dy, "o", ms=5.2, color=col, mec="white", mew=0.6, zorder=3)
    e.axvline(0, color=S.INK, lw=0.6, zorder=1)
    labels = []
    for p in S.PROC:
        r = con[con.process == p].iloc[0]
        labels.append(f"{S.PROC_SHORT[p]}\n{int(r.n_on)}/{int(r.n_off)}")
    e.set_yticks([ys[p] for p in S.PROC])
    e.set_yticklabels(labels, linespacing=1.0)
    e.tick_params(axis="y", length=0)
    e.set_ylim(-0.6, len(S.PROC) - 0.4)
    S.vgrid(e)
    e.set_xlabel("LOPO omission loss (ΔNSE)")
    e.text(-0.015, 1.0, "n ON/OFF", transform=e.transAxes, ha="right", va="bottom",
           fontsize=S.FS_SMALL, color="#666666")
    S.panel_title(e, "e", "Loss of decisive ON and OFF catchments", y=1.03)
    snap_x(e)
    e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: neg(f"{v:g}")))

    # ---- (f) the gap, parameters fixed versus refitted
    f = fig.add_subplot(gb[0, 1], sharey=e)
    for p in S.PROC:
        y = ys[p]
        fx = con[(con.process == p) & (con.estimand == "fixed")].iloc[0]
        lp = con[(con.process == p) & (con.estimand == "lopo")].iloc[0]
        f.barh(y, fx.est, 0.62, color=S.BAND[p], edgecolor=S.DARK[p], lw=0.7, zorder=2)
        f.barh(y, lp.est, 0.30, color=S.DARK[p], lw=0, zorder=3)
        f.plot([fx.lo, fx.hi], [y + 0.23] * 2, color="#555555", lw=0.8, zorder=4)
        f.plot([lp.lo, lp.hi], [y] * 2, color="black", lw=0.8, zorder=4)
    f.axvline(0, color=S.INK, lw=0.6, zorder=1)
    f.tick_params(axis="y", length=0, labelleft=False)
    S.vgrid(f)
    f.set_xlabel("ON − OFF gap in median loss (ΔNSE)")
    snap_x(f)
    f.xaxis.set_major_formatter(FuncFormatter(lambda v, _: neg(f"{v:g}")))
    S.panel_title(f, "f", "ON − OFF gap: parameters fixed vs refitted", y=1.03)
    hb = [Line2D([], [], marker="o", color="#555555", lw=1.6, ms=4.5, mec="white", label="Median and IQR"),
          Patch(color=GAP_FILL, label="ON − OFF gap"),
          Patch(facecolor="#EDEDED", edgecolor="#555555", lw=0.7, label="Parameters fixed (Full)"),
          Patch(color="#555555", label="Refitted (LOPO)"),
          Line2D([], [], color="black", lw=0.8, label="95% CI")]
    fig.canvas.draw()
    yb = min(e.xaxis.label.get_window_extent(rr).y0, f.xaxis.label.get_window_extent(rr).y0) / fig.bbox.height
    fig.legend(handles=hb, loc="upper center", ncol=5, fontsize=S.FS_TICK, handlelength=1.5,
               handletextpad=0.35, columnspacing=1.3, borderaxespad=0,
               bbox_to_anchor=(xc, yb - 0.012))
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
