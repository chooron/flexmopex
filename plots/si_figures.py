"""Flex-MOPEX Supporting Information figures (layout frozen; feed data only).

One function per figure. Each reads the CSV/JSON files named in README.md and
writes <name>.pdf and <name>.png. Style comes from figstyle.py (shared with the
main-text figures).
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

import figstyle as S

S.apply()
W = S.FULL_WIDTH
MINUS = "−"
PROC_ALL = S.PROC
METHOD_COL = {"canonical": "#1F4E8C", "plain_bce": "#86BEE6", "strict": "#6E6E6E",
              "matched_df": "#A8A8A8", "original_df": "#D0D0D0"}
DARK_TEXT = {"canonical", "strict"}  # white labels on these fills


def _read(d, n):
    return pd.read_csv(os.path.join(d, n), dtype={"gauge_id": str})


def _json(d, n):
    with open(os.path.join(d, n)) as f:
        return json.load(f)


def _neg(s):
    return str(s).replace("-", MINUS)


_FMT = FuncFormatter(lambda v, _: _neg(f"{v:g}"))


def _procs(df):
    return [p for p in PROC_ALL if p in set(df.process)]


def _top_legend(fig, handles, axes, y=1.0, ncol=None, **kw):
    x0 = min(a.get_position().x0 for a in axes)
    x1 = max(a.get_position().x1 for a in axes)
    opt = dict(borderaxespad=0.2, handletextpad=0.35, columnspacing=1.3)
    opt.update(kw)
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles), fontsize=S.FS_TICK,
               bbox_to_anchor=((x0 + x1) / 2, y), **opt)


def _offscale(ax, x, y, lo, hi, color, **kw):
    """Draw points beyond [lo, hi] as triangles on the axis edge; return in-range mask."""
    up, dn = y > hi, y < lo
    ax.scatter(x[up], np.full(up.sum(), hi), marker="^", s=14, color=color, lw=0, clip_on=False, zorder=6)
    ax.scatter(x[dn], np.full(dn.sum(), lo), marker="v", s=14, color=color, lw=0, clip_on=False, zorder=6)
    return ~(up | dn)


# =============================================================================
# lambda path
# =============================================================================
def fig_lambda(d, out, name):
    sk = _read(d, "si_lambda_skill.csv")
    gt = _read(d, "si_lambda_gates.csv")
    lam0 = _json(d, "si_lambda_meta.json")["declared_lambda"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.55))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.165, wspace=0.24)
    col = "#1F4E8C"
    for ax in (a, b):
        ax.axvline(lam0, color=S.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xscale("log")
        S.hgrid(ax)
        ax.set_xlabel("Regularization weight $\\lambda$")
    best = sk[sk.ckpt == "best"].sort_values("lam")
    last = sk[sk.ckpt == "last"].sort_values("lam")
    a.plot(best.lam, best.median_nse, "-", color=col, lw=0.9, alpha=0.6, zorder=2)
    a.plot(best.lam, best.median_nse, "o", color=col, ms=4.2, zorder=3)
    a.plot(last.lam, last.median_nse, color=col, lw=0.8, ls=(0, (3, 1.6)), alpha=0.6, zorder=2)
    a.plot(last.lam, last.median_nse, "s", mfc="white", mec=col, mew=0.9, ms=4.2, zorder=3)
    a.yaxis.set_major_locator(MaxNLocator(5))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    a.set_ylabel("Test median NSE")
    S.panel_title(a, "a", "Predictive skill")
    for p in _procs(gt):
        g = gt[gt.process == p]
        bb = g[g.ckpt == "best"].sort_values("lam")
        ll = g[g.ckpt == "last"].sort_values("lam")
        b.plot(bb.lam, bb.mean_w, "-", color=S.DARK[p], lw=0.9, alpha=0.6, zorder=2)
        b.plot(bb.lam, bb.mean_w, "o", color=S.DARK[p], ms=4.0, zorder=3)
        b.plot(ll.lam, ll.mean_w, color=S.DARK[p], lw=0.8, ls=(0, (3, 1.6)), alpha=0.6, zorder=2)
        b.plot(ll.lam, ll.mean_w, "s", mfc="white", mec=S.DARK[p], mew=0.9, ms=4.0, zorder=3)
    b.set_ylim(0, 1)
    b.set_ylabel("Mean learned gate")
    S.panel_title(b, "b", "Structural use")
    h = [Line2D([], [], marker="o", ls="", color=S.DARK[p], ms=4.5, label=S.PROC_SHORT[p]) for p in _procs(gt)]
    h += [Line2D([], [], marker="o", ls="-", lw=0.9, color="#555555", ms=4.5, label="Best checkpoint"),
          Line2D([], [], marker="s", ls=(0, (3, 1.6)), lw=0.9, mfc="white", mec="#555555", color="#555555",
                 ms=4.5, label="Last checkpoint"),
          Line2D([], [], color=S.REF, lw=0.7, ls=(0, (3, 2)), label=_neg(f"Declared $\\lambda$ = {lam0:g}"))]
    _top_legend(fig, h, [a, b], columnspacing=1.0)
    S.save(fig, out, name)


# =============================================================================
# learning variants: recovery and predictive skill
# =============================================================================
def _variant_meta(d):
    m = _json(d, "si_variant_meta.json")
    return m, [x["key"] for x in m["methods"]], {x["key"]: x["label"] for x in m["methods"]}


def fig_variant_recovery(d, out, name):
    meta, meths, mlab = _variant_meta(d)
    rc = _read(d, "si_variant_recovery.csv")
    sk = _read(d, "si_variant_skill.csv").reset_index(drop=True)
    procs = _procs(rc)
    neg = {tuple(x) for x in meta.get("negative_control", [])}
    nrow = len(meta["metrics"])
    ncol = len(procs) + 1
    fig = plt.figure(figsize=(W, 1.62 * nrow + 0.42))
    gs = fig.add_gridspec(nrow, ncol, width_ratios=[1] * len(procs) + [0.92], hspace=0.42, wspace=0.10,
                          left=0.075, right=0.995, top=0.925, bottom=0.085)
    bw = 0.25
    letter = iter("abcdefghijkl")
    for r, met in enumerate(meta["metrics"]):
        for c, p in enumerate(procs):
            ax = fig.add_subplot(gs[r, c])
            for k, sp in enumerate(["train", "holdout"]):
                if (p, sp) in neg:
                    ax.axvspan(k - 0.48, k + 0.48, color=S.BAND_NEG, lw=0, zorder=0)
                for j, m in enumerate(meths):
                    row = rc[(rc.process == p) & (rc.metric == met["key"]) & (rc.split == sp) & (rc.method == m)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    x = k + (j - (len(meths) - 1) / 2) * bw
                    ax.bar(x, row["mean"], bw * 0.9, color=METHOD_COL.get(m, "#999999"), lw=0, zorder=2)
                    if row.hi > row.lo:
                        ax.plot([x, x], [row.lo, row.hi], color="black", lw=0.8, zorder=3)
            ax.axhline(0.5, color=S.REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Train", "Holdout"] if r == nrow - 1 else [])
            ax.tick_params(axis="x", length=0)
            S.hgrid(ax)
            if c == 0:
                ax.set_ylabel(met["label"])
            else:
                ax.set_yticklabels([])
            S.panel_title(ax, next(letter), S.PROC_SHORT[p], y=1.03)
    # predictive skill: vertical bars, one per learning variant, top-right cell
    k_ax = fig.add_subplot(gs[0, ncol - 1])
    xs = np.arange(len(sk))
    for x, (_, r) in zip(xs, sk.iterrows()):
        k_ax.bar(x, r.median_nse, 0.62, color=METHOD_COL.get(r.variant, "#BBBBBB"), lw=0, zorder=2)
        k_ax.text(x, r.median_nse + 0.015, f"{r.median_nse:.3f}", ha="center", va="bottom",
                  fontsize=S.FS_SMALL, color="#1A1A1A", zorder=3)
    k_ax.set_ylim(0, 1.0)
    k_ax.set_yticks([0, 0.5, 1.0])
    k_ax.set_yticklabels([])
    k_ax.tick_params(axis="y", length=0)
    k_ax.spines["left"].set_visible(False)
    k_ax.set_xticks(xs)
    k_ax.set_xticklabels([])
    k_ax.tick_params(axis="x", length=0)
    k_ax.set_xlim(-0.6, len(sk) - 0.4)
    S.hgrid(k_ax)
    S.panel_title(k_ax, next(letter), "Holdout NSE", y=1.03)
    # legend in the free bottom-right cell
    lg = fig.add_subplot(gs[nrow - 1, ncol - 1])
    lg.set_axis_off()
    h = [Patch(color=METHOD_COL[m], label=mlab[m]) for m in meths]
    h += [Line2D([], [], color="black", lw=0.8, label="Seed range")]
    if neg:
        h += [Patch(color=S.BAND_NEG, label="Negative control")]
    lg.legend(handles=h, loc="center left", fontsize=S.FS_TICK, handlelength=1.3, labelspacing=0.7,
              borderaxespad=0, bbox_to_anchor=(0.08, 0.5))
    S.save(fig, out, name)


# =============================================================================
# learning variants: paired difference and holdout gate distributions
# =============================================================================
def fig_variant_paired(d, out, name):
    meta, meths, mlab = _variant_meta(d)
    pr = _read(d, "si_variant_paired.csv")
    gt = _read(d, "si_variant_gates.csv")
    procs = _procs(pr)
    mets = meta["metrics"]
    fig = plt.figure(figsize=(W, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.22,
                          left=0.115, right=0.99, top=0.83, bottom=0.165)
    a = fig.add_subplot(gs[0])
    marks = [("o", True), ("D", False), ("s", False)]
    off = np.linspace(0.16, -0.16, len(mets)) if len(mets) > 1 else [0.0]
    ys = {p: len(procs) - 1 - i for i, p in enumerate(procs)}
    lim = 0.0
    for p in procs:
        for (mk, filled), met, dy in zip(marks, mets, off):
            r = pr[(pr.process == p) & (pr.metric == met["key"])]
            if r.empty:
                continue
            r = r.iloc[0]
            y = ys[p] + dy
            if np.isfinite(r.lo) and np.isfinite(r.hi):
                a.plot([r.lo, r.hi], [y, y], color=S.DARK[p], lw=1.2, zorder=2)
                lim = max(lim, abs(r.lo), abs(r.hi))
            a.plot(r.est, y, mk, ms=5 if mk == "o" else 4.6, mfc=S.DARK[p] if filled else "white",
                   mec=S.DARK[p], mew=1.1, zorder=3)
            lim = max(lim, abs(r.est))
    lim = np.ceil(lim * 1.15 / 0.05) * 0.05
    a.set_xlim(-lim, lim)
    a.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    a.xaxis.set_major_formatter(_FMT)
    a.axvline(0, color=S.INK, lw=0.6, zorder=1)
    a.set_yticks([ys[p] for p in procs])
    a.set_yticklabels([S.PROC_SHORT[p] for p in procs])
    a.tick_params(axis="y", length=0)
    a.set_ylim(-0.6, len(procs) - 0.4)
    S.vgrid(a)
    a.set_xlabel(f"{mlab.get('strict', 'Strict')} − {mlab.get('canonical', 'Canonical')}")
    a.text(0.015, 0.985, "← Canonical higher", transform=a.transAxes, ha="left", va="top",
           fontsize=S.FS_SMALL, color="#7A7A7A")
    a.text(0.985, 0.985, "Strict higher →", transform=a.transAxes, ha="right", va="top",
           fontsize=S.FS_SMALL, color="#7A7A7A")
    S.panel_title(a, "a", "Paired holdout difference", y=1.03)

    # cumulative distributions of holdout gates, one small panel per process
    gprocs = _procs(gt)
    pair = [m for m in ("canonical", "strict") if m in set(gt.method)]
    gb = gs[1].subgridspec(1, len(gprocs), wspace=0.10)
    bx = []
    for i, p in enumerate(gprocs):
        ax = fig.add_subplot(gb[0, i], sharey=bx[0] if bx else None)
        bx.append(ax)
        for m in pair:
            v = np.sort(gt[(gt.process == p) & (gt.method == m)].w.values)
            yv = np.arange(1, len(v) + 1) / len(v)
            ax.step(np.r_[0, v, 1], np.r_[0, yv, 1], where="post", color=METHOD_COL[m],
                    lw=1.6 if m == "canonical" else 1.3, zorder=3 if m == "canonical" else 2)
        ax.axvline(0.5, color=S.REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        S.hgrid(ax)
        if i == 0:
            ax.set_ylabel("Cumulative fraction")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        S.panel_title(ax, "bcdefg"[i], S.PROC_SHORT[p], y=1.03)
    fig.canvas.draw()
    xc = (bx[0].get_position().x0 + bx[-1].get_position().x1) / 2
    yt = min(t.get_window_extent().y0 for t in bx[0].get_xticklabels()) / fig.bbox.height
    fig.text(xc, yt - 0.01, "Holdout learned gate $w$", ha="center", va="top", fontsize=S.FS_LABEL)

    h = [Line2D([], [], marker=mk, ls="-", lw=1.2, color="#555555", mfc="#555555" if f else "white",
                ms=4.5, label=f"Δ{met['short']}") for (mk, f), met in zip(marks, mets)]
    h += [Line2D([], [], color=METHOD_COL[m], lw=1.6, label=mlab[m]) for m in pair]
    _top_legend(fig, h, [a] + bx)
    S.save(fig, out, name)


# =============================================================================
# synthetic scenarios: learned gate against the assignment attribute
# =============================================================================
def fig_synth(d, out, name):
    sc = _read(d, "si_synth_scatter.csv")
    panels = _json(d, "si_synth_meta.json")["panels"]
    nr = int(np.ceil(len(panels) / 2))
    fig, axs = plt.subplots(nr, 2, figsize=(W, 2.0 * nr + 0.35), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.90, bottom=0.095, wspace=0.07, hspace=0.50)
    for ax, pn, letter in zip(axs.flat, panels, "abcdefgh"):
        p = pn["process"]
        dd = sc[(sc.process == p) & (sc.scenario == pn["scenario"])]
        for split, mk, s, al in (("train", "o", 9, 0.45), ("holdout", "^", 17, 0.95)):
            g = dd[dd.split == split]
            on, off = g[g.truth == 1], g[g.truth == 0]
            ax.scatter(off.x, off.w, s=s, marker=mk, facecolor="none", edgecolor=S.DARK[p],
                       lw=0.6, alpha=al, zorder=2 if split == "train" else 4)
            ax.scatter(on.x, on.w, s=s, marker=mk, color=S.DARK[p], lw=0, alpha=al,
                       zorder=3 if split == "train" else 5)
        ax.axhline(0.5, color=S.REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
        if pn.get("threshold") is not None:
            ax.axvline(pn["threshold"], color=S.INK, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(_FMT)
        S.hgrid(ax)
        ax.set_xlabel(pn["x_label"])
        S.panel_title(ax, letter, f"{S.PROC_LABEL[p]}, Scenario {pn['scenario']} ({pn['assignment']})", y=1.03)
    for ax in axs[:, 0]:
        ax.set_ylabel("Learned gate $\\bar{w}$")
    for ax in axs.flat[len(panels):]:
        ax.set_axis_off()
    # share x within each row when the attribute is the same
    for row in axs:
        if row[0].get_xlabel() == row[1].get_xlabel():
            lo = min(r.get_xlim()[0] for r in row)
            hi = max(r.get_xlim()[1] for r in row)
            for r in row:
                r.set_xlim(lo, hi)
    gray = "#555555"
    h = [Line2D([], [], marker="o", ls="", color=gray, alpha=0.6, ms=4.2, label="Train, true ON"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=gray, alpha=0.6, ms=4.2, label="Train, true OFF"),
         Line2D([], [], marker="^", ls="", color=gray, ms=5, label="Holdout, true ON"),
         Line2D([], [], marker="^", ls="", mfc="none", mec=gray, ms=5, label="Holdout, true OFF"),
         Line2D([], [], color=S.INK, lw=0.8, ls=(0, (1.5, 1.5)), label="Rule threshold")]
    _top_legend(fig, h, list(axs.flat))
    S.save(fig, out, name)


# =============================================================================
# attribute correlations: top six per process
# =============================================================================
def fig_attr_top(d, out, name, ntop=6):
    ac = _read(d, "si_attr_corr.csv")
    procs = _procs(ac)
    nat = ac.attribute.nunique()
    lab_w, gap_c, axw = 1.48, 0.10, None
    ncol = 2
    axw = (W - ncol * lab_w - 0.12 - (ncol - 1) * gap_c) / ncol
    row_h, top_m, row_gap, bot_m = 0.165, 0.40, 0.46, 0.42
    ph = (ntop + 1.3) * row_h
    nrow = int(np.ceil(len(procs) / ncol))
    H = top_m + nrow * ph + (nrow - 1) * row_gap + bot_m
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i, p in enumerate(procs):
        r, c = divmod(i, ncol)
        x0 = lab_w + c * (axw + lab_w + gap_c)
        y0 = H - top_m - (r + 1) * ph - r * row_gap
        ax = fig.add_axes([x0 / W, y0 / H, axw / W, ph / H])
        axes.append(ax)
        g = ac[ac.process == p].copy()
        g["a"] = g.rho_w.abs()
        top = g.sort_values("a", ascending=False).head(ntop)
        ystrip = ntop + 0.15
        ax.add_patch(Rectangle((-1, ystrip - 0.42), 2, 0.84, color="#F2F2F2", lw=0, zorder=0))
        others = g[~g.attribute.isin(top.attribute)]
        ax.vlines(others.rho_w, ystrip - 0.3, ystrip + 0.3, color="#B5B5B5", lw=0.9, zorder=1)
        ax.vlines(top.rho_w, ystrip - 0.3, ystrip + 0.3, color=S.DARK[p], lw=1.4, zorder=2)
        ys = np.arange(ntop)[::-1]
        ax.barh(ys, top.rho_w, 0.64, color=S.DARK[p], lw=0, zorder=2)
        ax.plot(top.rho_q, ys, "D", ms=4.6, mfc="white", mec="#1A1A1A", mew=0.9, zorder=3)
        ax.axvline(0, color=S.INK, lw=0.6, zorder=1)
        ax.set_yticks(list(ys) + [ystrip])
        ax.set_yticklabels(list(top.label) + [f"All {nat}"])
        ax.get_yticklabels()[-1].set_color("#7A7A7A")
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, ystrip + 0.45)
        ax.set_xlim(-0.95, 0.95)
        ax.xaxis.set_major_locator(FixedLocator([-0.8, -0.4, 0, 0.4, 0.8]))
        ax.xaxis.set_major_formatter(_FMT)
        S.vgrid(ax)
        if r == nrow - 1:
            ax.set_xlabel("Spearman $\\rho$ with attribute")
        S.panel_title(ax, "abcdefgh"[i], S.PROC_LABEL[p], x=-0.02, y=1.0, ha="right" if False else "left")
    h = [Patch(color="#555555", label=f"$\\rho(w_p, x)$, top {ntop} by $|\\rho|$"),
         Line2D([], [], color="#B5B5B5", lw=1.2, label="Other attributes"),
         Line2D([], [], marker="D", ls="", mfc="white", mec="#1A1A1A", ms=4.6,
                label=r"$\rho(q_p^{\mathrm{post}}, x)$, preference proxy")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=S.FS_TICK, bbox_to_anchor=(0.5, 1.0),
               borderaxespad=0.25, columnspacing=1.6)
    S.save(fig, out, name)


# =============================================================================
# attribute correlations: all attributes (heat map)
# =============================================================================
CMAP_DIV = LinearSegmentedColormap.from_list(
    "attr_div",
    ["#426982", "#F4F2EF", "#BC6850"],  # blue-grey / off-white / terracotta (low saturation)
)


def fig_attr_heat(d, out, name, vmax=0.8):
    ac = _read(d, "si_attr_corr.csv")
    procs = _procs(ac)
    attrs = ac.drop_duplicates("attribute")[["attribute", "label", "category"]].reset_index(drop=True)
    cats = list(dict.fromkeys(attrs.category))
    ch, pgap, cat_gap = 0.142, 0.14, 0.07
    lab_cat, lab_att = 0.92, 1.50
    x_l = lab_cat + lab_att
    cw = (W - x_l - 0.04 - (len(procs) - 1) * pgap) / (2 * len(procs))
    head_h, cb_h = 0.50, 0.68
    n = len(attrs)
    grid_w = len(procs) * 2 * cw + (len(procs) - 1) * pgap
    grid_h = n * ch + (len(cats) - 1) * cat_gap
    H = head_h + grid_h + cb_h
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    norm = Normalize(-vmax, vmax)
    x_off = x_l
    y = H - head_h
    ypos = {}
    for c in cats:
        sub = attrs[attrs.category == c]
        for k, (_, r) in enumerate(sub.iterrows()):
            ypos[r.attribute] = y - ch
            if k == 0:
                ax.text(0.02, y - ch / 2, c, ha="left", va="center", fontsize=S.FS_TICK, fontweight="bold",
                        color="#1A1A1A")
            ax.text(x_off - 0.06, y - ch / 2, r.label, ha="right", va="center", fontsize=S.FS_TICK)
            y -= ch
        if c != cats[-1]:
            ax.plot([0.02, x_off + grid_w], [y - cat_gap / 2] * 2, color="#D0D0D0", lw=0.5)
            y -= cat_gap
    for i, p in enumerate(procs):
        g = ac[ac.process == p].set_index("attribute")
        x0 = x_off + i * (2 * cw + pgap)
        for j, col in enumerate(("rho_w", "rho_q")):
            for a_, yy in ypos.items():
                v = g.loc[a_, col]
                ax.add_patch(Rectangle((x0 + j * cw + 0.008, yy + 0.008), cw - 0.016, ch - 0.016,
                                       facecolor=CMAP_DIV(norm(v)), lw=0))
            ax.text(x0 + (j + 0.5) * cw, H - head_h + 0.04, ["$w_p$", "$q_p^{\\mathrm{post}}$"][j], ha="center", va="bottom",
                    fontsize=S.FS_TICK, color="#4A4A4A")
        ax.plot([x0 + 0.01, x0 + 2 * cw - 0.01], [H - head_h + 0.26] * 2, color=S.DARK[p], lw=2.2,
                solid_capstyle="butt")
        ax.text(x0 + cw, H - head_h + 0.31, S.PROC_SHORT[p], ha="center", va="bottom",
                fontsize=S.FS_LABEL, fontweight="bold", color="#1A1A1A")
    # colour bar
    cbw = min(2.6, grid_w * 0.75)
    cax = fig.add_axes([(x_off + (grid_w - cbw) / 2) / W, 0.42 / H, cbw / W, 0.09 / H])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DIV)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.ax.xaxis.set_major_formatter(_FMT)
    cb.ax.tick_params(length=2, labelsize=S.FS_TICK)
    cb.set_label("Spearman $\\rho$ with attribute", fontsize=S.FS_LABEL, labelpad=2)
    S.save(fig, out, name)


# =============================================================================
# snow removal against the seasonal benchmark
# =============================================================================
def fig_snow_bench(d, out, name):
    sb = _read(d, "si_snow_bench.csv")
    cr = _read(d, "si_snow_bench_corr.csv")
    lo, hi = _json(d, "si_snow_bench_meta.json")["loss_ylim"]
    fig = plt.figure(figsize=(W, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.24,
                          left=0.085, right=0.985, top=0.83, bottom=0.17)
    a = fig.add_subplot(gs[0])
    cf, cl = S.LIGHT["snow"], S.DARK["snow"]
    x = sb.bench_nse.values
    yf, yl = sb.loss_fixed.values, sb.loss_lopo.values
    mf = _offscale(a, x, yf, lo, hi, cf)
    ml = _offscale(a, x, yl, lo, hi, cl)
    a.scatter(x[mf], yf[mf], s=8, color=cf, lw=0, alpha=0.85, zorder=2)
    a.scatter(x[ml], yl[ml], s=8, marker="s", facecolor="none", edgecolor=cl, lw=0.5, zorder=3)
    a.axhline(0, color=S.INK, lw=0.6, zorder=1)
    a.set_ylim(lo, hi)
    a.yaxis.set_major_formatter(_FMT)
    a.xaxis.set_major_formatter(_FMT)
    S.hgrid(a)
    a.set_xlabel("Seasonal-benchmark NSE")
    a.set_ylabel("Snow omission loss (ΔNSE)")
    S.panel_title(a, "a", "Snow removal and benchmark skill", y=1.03)

    b = fig.add_subplot(gs[1])
    lab = {"fixed": "Fixed-off", "lopo": "LOPO"}
    ests = [e for e in ("fixed", "lopo") if e in set(cr.estimand)]
    for i, e in enumerate(ests):
        yb = len(ests) - 1 - i
        rw = cr[(cr.estimand == e) & (cr.kind == "raw")].iloc[0]
        pt = cr[(cr.estimand == e) & (cr.kind == "partial")].iloc[0]
        b.barh(yb, rw.rho, 0.62, color=S.BAND["snow"], edgecolor=cl, lw=0.7, zorder=2)
        b.barh(yb, pt.rho, 0.30, color=cl, lw=0, zorder=3)
        b.plot([rw.lo, rw.hi], [yb + 0.22] * 2, color="#555555", lw=0.8, zorder=4)
        b.plot([pt.lo, pt.hi], [yb] * 2, color="black", lw=0.8, zorder=4)
    b.axvline(0, color=S.INK, lw=0.6, zorder=1)
    b.set_yticks(range(len(ests))[::-1])
    b.set_yticklabels([lab[e] for e in ests])
    b.tick_params(axis="y", length=0)
    b.set_ylim(-0.55, len(ests) - 0.45)
    b.set_xlim(left=min(0, b.get_xlim()[0]))
    b.xaxis.set_major_locator(MaxNLocator(5))
    b.xaxis.set_major_formatter(_FMT)
    S.vgrid(b)
    b.set_xlabel("Spearman $\\rho$ with benchmark NSE")
    S.panel_title(b, "b", "Raw and snow-adjusted \u03c1", y=1.03)

    h = [Line2D([], [], marker="o", ls="", color=cf, ms=4.5, label="Fixed-off"),
         Line2D([], [], marker="s", ls="", mfc="none", mec=cl, ms=4.5, label="LOPO"),
         Line2D([], [], marker="^", ls="", color="#555555", ms=4.5, label="Off-scale"),
         Patch(facecolor=S.BAND["snow"], edgecolor=cl, lw=0.7, label="Raw ρ"),
         Patch(color=cl, label="Partial ρ (snow fraction)"),
         Line2D([], [], color="black", lw=0.8, label="95% CI")]
    _top_legend(fig, h, [a, b])
    S.save(fig, out, name)


# =============================================================================
# refitting versus fixed-parameter removal
# =============================================================================
def _repel(vals, gap):
    order = np.argsort(vals)
    out = np.array(vals, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        if out[i] - out[j] < gap:
            out[i] = out[j] + gap
    return out


def fig_refit(d, out, name, lt=0.01):
    df = _read(d, "si_refit_diff.csv")
    ex = _read(d, "si_refit_exceed.csv")
    procs = _procs(df)
    fig = plt.figure(figsize=(W, 2.55))
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
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _neg(f"{v:g}")))
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
    for p in _procs(ex):
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
    S.save(fig, out, name)


FIGS = {
    "lambda": (fig_lambda, "lambda_path"),
    "recovery": (fig_variant_recovery, "learning_variants_recovery"),
    "paired": (fig_variant_paired, "learning_variants_gates"),
    "synth": (fig_synth, "synthetic_gate_attribute"),
    "attrheat": (fig_attr_heat, "attribute_heatmap"),
    "snowbench": (fig_snow_bench, "snow_benchmark"),
    "refit": (fig_refit, "refit_vs_fixed"),
}
