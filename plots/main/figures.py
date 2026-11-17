"""Flex-MOPEX manuscript Figures 2-8.

Each function reads only the CSV/JSON files documented in README.md from
`data_dir` and writes PDF + PNG to `out_dir`. Do not edit the plotting logic
when switching from demo to real data; supply real files with the same schema.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import figstyle as S
import geo

# Figure numbers. make_all.py overwrites this from --numbers.
ORDER = ["study", "prediction", "detect", "transfer", "evidence", "attr", "regional", "removal"]
SHORT = {"study": "study_area", "prediction": "prediction", "detect": "detectability_recovery",
         "transfer": "transfer", "evidence": "evidence_maps", "attr": "attribute_relationships",
         "regional": "regional_exclusion", "removal": "process_removal"}
NAMES = {}


def set_numbers(numbers):
    NAMES.clear()
    for k, n in zip(ORDER, numbers):
        NAMES[k] = f"Figure{n:02d}_{SHORT[k]}"


set_numbers(range(3, 11))

S.apply()
W = S.FULL_WIDTH


def _read(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), dtype={"gauge_id": str})


def _json(data_dir, name):
    return json.load(open(os.path.join(data_dir, name)))


def _neg(s):
    """Use the typographic minus sign in all printed numbers."""
    return s.replace("-", "\u2212")


def _map_extent(outline, padx=0.01, pady=0.02):
    xmin, ymin, xmax, ymax = outline.bounds
    dx, dy = (xmax - xmin) * padx, (ymax - ymin) * pady
    return (xmin - dx, xmax + dx, ymin - dy, ymax + dy)


def _map_axes(fig, rect_in, extent, fig_w, fig_h):
    """Create an axes at an exact position given in inches (x0, y0, w, h)."""
    x0, y0, w, h = rect_in
    ax = fig.add_axes([x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h])
    return ax


# ======================================================================
# Figure 2 - study catchments, regions, selected attributes
# Replicates the reference study-domain layout: bold centred title over each
# map, vertical colour bar on the right edge of each map, and an inset
# histogram + KDE of the attribute distribution in the lower-left corner.
# GIS layers / projection stay the package ones (geo.py, EPSG:5070).
# ======================================================================
CB_PAD = 0.06        # in, breathing room around the colour bar label
CB_LABEL_W = 0.30    # in, reserved width of the rotated colour-bar label
TITLE_H = 0.26       # in, band reserved above each map for its centred title
ROW_GAP = 0.10       # in, vertical gap between map rows
MARGIN = 0.03        # in, outer margin
BOT_MARGIN = 0.14     # in, bottom margin (clears the inset x tick labels)

INSET_POS = [0.03, 0.03, 0.20, 0.20]   # lower-left histogram, axes fractions
INSET_BINS = 30
ARIDITY_CBAR_CLIP_PCT = 98


def _fig2_geometry(outline):
    """Map width/height so every map plus its right-edge colour bar fits W."""
    ext = geo.map_extent(outline)
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    # column pitch = map + (colour-bar offset + label + pad); the right margin
    # reserves the same block for the last column.
    cb_block = 0.055  # colour-bar offset+width as a fraction of the map width
    usable = W - 2 * MARGIN - 2 * CB_LABEL_W - 2 * CB_PAD
    mw = usable / (2 + 2 * cb_block)
    mh = mw * aspect
    return ext, mw, mh


def _colorbar(fig, ax, sc, title, unit, fmt, vmax_override=None, vmax_label=""):
    """Vertical colour bar as a thin inset on the right edge of the map."""
    cax = ax.inset_axes([1.02, 0.125, 0.035, 0.75], transform=ax.transAxes)
    if vmax_override is not None:
        norm = plt.Normalize(sc.norm.vmin, vmax_override)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=sc.cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    else:
        cb = fig.colorbar(sc, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=S.FS_TICK, pad=1, length=2, color=S.INK)
    cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _neg(fmt.format(v))))
    cb.set_label(f"{title}{unit}{vmax_label}", fontsize=S.FS_LABEL, labelpad=2)
    return cb


def _inset_distribution(ax_map, values, cmap_name):
    """Inset histogram + KDE of the global distribution, lower-left of the map."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    ax_ins = ax_map.inset_axes(INSET_POS, transform=ax_map.transAxes)
    ax_ins.set_facecolor("white")
    ax_ins.patch.set_alpha(0.70)
    counts, bins, patches = ax_ins.hist(v, bins=INSET_BINS, density=True,
                                        alpha=0.75, edgecolor="none", zorder=1)
    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = plt.Normalize(v.min(), v.max())
    for patch, le, re_ in zip(patches, bins[:-1], bins[1:]):
        patch.set_facecolor(cmap(norm((le + re_) / 2)))
    try:
        kde_x = np.linspace(v.min(), v.max(), 200)
        ax_ins.plot(kde_x, gaussian_kde(v)(kde_x), color="#333333", lw=0.8, zorder=3)
    except Exception:
        pass  # KDE can fail on degenerate input; the histogram remains
    ax_ins.set_xlim(v.min(), v.max())
    # keep only ticks inside the data range (the locator may round past the edges)
    keep = [tv for tv in ax_ins.get_xticks() if v.min() <= tv <= v.max()]
    if keep:
        ax_ins.set_xticks(keep)
    ax_ins.set_yticks([])
    for side in ("top", "right", "left"):
        ax_ins.spines[side].set_visible(False)
    ax_ins.tick_params(axis="x", labelsize=S.FS_SMALL - 0.5, pad=1.5, length=2)
    ax_ins.tick_params(axis="y", length=0)
    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))


def fig2(data_dir, out_dir, layers):
    regions, states, outline = layers
    cat = _read(data_dir, "catchments.csv")
    cat["x"], cat["y"] = geo.project_points(cat.lon, cat.lat)
    n_reg = cat.region.value_counts().to_dict()
    ext, mw, mh = _fig2_geometry(outline)
    H = 2 * (TITLE_H + mh) + ROW_GAP + MARGIN + BOT_MARGIN
    fig = plt.figure(figsize=(W, H))
    col_pitch = mw + 0.055 * mw + CB_LABEL_W + CB_PAD
    cb_block = 0.055 * mw + CB_LABEL_W

    spec = [
        ("frac_snow", "Snow fraction", " (\u2013)", "Blues", "{:.1f}", None),
        ("aridity", "Aridity, PET/P", " (\u2013)", "YlOrBr", "{:.1f}", "  (>P98)"),
        ("elev_mean", "Elevation", " (m)", "YlGnBu", "{:.0f}", None),
        ("gvf_diff", "Green-veg. fraction range", " (–)", "Greens", "{:.1f}", None),
    ]
    for i, (col, title, unit, cmap, fmt, clip) in enumerate(spec):
        r, c = divmod(i, 2)
        x0 = MARGIN + c * col_pitch
        y0 = H - MARGIN - (r + 1) * (TITLE_H + mh) - r * ROW_GAP
        ax = fig.add_axes([x0 / W, y0 / H, mw / W, mh / H])
        geo.draw_base(ax, regions, states, outline, extent=ext)
        v = cat[col].to_numpy(dtype=float)
        sc = ax.scatter(cat.x.to_numpy(), cat.y.to_numpy(), c=v, cmap=cmap,
                        s=10, alpha=0.78, edgecolors="none", linewidths=0, zorder=3)
        vmax_override = np.nanpercentile(v, ARIDITY_CBAR_CLIP_PCT) if clip else None
        _colorbar(fig, ax, sc, title, unit, fmt,
                  vmax_override=vmax_override, vmax_label=clip or "")
        ax.set_title(f"({'abcd'[i]}) {title}", fontsize=S.FS_TITLE, fontweight="bold", pad=6)
        if i == 0:
            for rg, (px, py) in geo.region_label_points(regions).items():
                ax.text(px, py, f"{rg} (n={n_reg.get(rg, 0)})", ha="center", va="center",
                        fontsize=S.FS_SMALL, fontweight="bold", color="#1A1A1A", zorder=6,
                        linespacing=0.95,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
        _inset_distribution(ax, v, cmap)
    S.save(fig, out_dir, NAMES["study"])


# ======================================================================
# Figure 3 - predictive context
# ======================================================================
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


# ======================================================================
# Figure 4 - detectability and recovery (Scenario A, training catchments)
# ======================================================================
LIN4 = 0.02


def _symlog_x(ax, lo=-0.15, hi=2.0):
    ax.set_xscale("symlog", linthresh=LIN4, linscale=1.0)
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([-0.1, 0, 0.1, 1]))
    ax.set_xticklabels(["−0.1", "0", "0.1", "1"])
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig4(data_dir, out_dir):
    sc = _read(data_dir, "fig4_scatter.csv")
    rc = _read(data_dir, "fig4_recall.csv")
    P = _json(data_dir, "fig4_meta.json")["P_p"]
    fig = plt.figure(figsize=(W, 4.15))
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
        off = d[d.truth == 0]; on = d[d.truth == 1]
        ax.scatter(off.D_J, off.w, s=7, facecolor="none", edgecolor=S.GRAY_EDGE, lw=0.45, zorder=2)
        ax.scatter(on.D_J, on.w, s=7, color=S.DARK[p], alpha=0.85, lw=0, zorder=3)
        _symlog_x(ax)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if i == 0:
            ax.set_ylabel(r"Learned gate $\bar{w}_{b,p}$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$D_{J,b,p}$", labelpad=1)
        ax.text(P[p], 0.985, r" $\lambda k_p$", transform=ax.get_xaxis_transform(), ha="left", va="top",
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
                ax.bar(x, r["mean"], bw * 0.92, color=col, lw=0, zorder=2,
                       hatch=None)
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
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r"ON recall ($\pi_{b,p}$ > 0.5)")
    S.hgrid(ax)
    ax.text(-0.25, -0.035, "n", ha="left", va="top", fontsize=S.FS_SMALL, color="#555555")
    S.panel_title(ax, "e", "Prescribed-ON recall by detectability stratum", y=1.03)
    h = [Patch(color="#4A4A4A", label="High"), Patch(color="#BDBDBD", label="Weak"),
         Patch(facecolor="#BDBDBD", edgecolor="white", hatch="////", label="Negative"),
         Line2D([], [], color="black", lw=0.8, label="Seed range")]
    ax.legend(handles=h, loc="upper right", ncol=4, fontsize=S.FS_TICK, handlelength=1.2,
              bbox_to_anchor=(1.0, 1.02))

    items = [
        (Line2D([], [], marker="o", ls="", color=S.DARK["snow"], ms=4.5), "Prescribed ON (process color)"),
        (Line2D([], [], marker="o", ls="", mfc="none", mec=S.GRAY_EDGE, ms=4.5), "Prescribed OFF"),
        (Patch(color=S.BAND["snow"]), r"Weak: 0 ≤ $D_{J,b,p}$ ≤ $\lambda k_p$"),
        (Patch(color=S.BAND_NEG), r"Negative: $D_{J,b,p}$ < 0"),
    ]
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    ybot = min(t.xaxis.label.get_window_extent(rr).y0 for t in top4) / fig.bbox.height
    fig.legend([a for a, _ in items], [b for _, b in items], loc="upper center", ncol=4,
               fontsize=S.FS_TICK, handlelength=1.3, columnspacing=1.4, borderaxespad=0,
               bbox_to_anchor=((top4[0].get_position().x0 + top4[-1].get_position().x1) / 2, ybot - 0.012))
    S.save(fig, out_dir, NAMES["detect"])


# ======================================================================
# Figure 5 - recovery and transfer across scenarios
# ======================================================================
def fig5(data_dir, out_dir):
    d = _read(data_dir, "fig5_recovery.csv")
    fig, axs = plt.subplots(2, 2, figsize=(W, 3.5))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.105, top=0.875, wspace=0.24, hspace=0.50)
    sc_order = ["A", "B", "C", "D"]
    bh = 0.36
    for ax, p, letter in zip(axs.flat, S.PROC, "abcd"):
        dd = d[d.process == p]
        ylabels = []
        for i, s in enumerate(sc_order):
            y = len(sc_order) - 1 - i
            row = dd[dd.scenario == s]
            typ = row.assignment.iloc[0]
            if typ == "random":
                ax.axhspan(y - 0.5, y + 0.5, color=S.BAND_NEG, lw=0, zorder=0)
            for split, off, col in (("train", +bh / 2, S.DARK[p]), ("heldout", -bh / 2, S.LIGHT[p])):
                r = row[row.split == split].iloc[0]
                ax.barh(y + off, r["mean"], bh, color=col, lw=0.6 if split == "heldout" else 0,
                        edgecolor=S.DARK[p] if split == "heldout" else "none", zorder=2)
                ax.plot([r.lo, r.hi], [y + off, y + off], color="black", lw=0.8, zorder=3)
            ylabels.append(f"{s}  {typ}" if typ != "fixed" else s)
        ax.set_yticks(range(len(sc_order))[::-1])
        ax.set_yticklabels(ylabels)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.55, len(sc_order) - 0.45)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "", "0.5", "", "1"])
        S.vgrid(ax)
        if p == "snow":
            ax.set_xlabel("ON recall ($w$ > 0.5)")
        else:
            ax.axvline(0.5, color=S.REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.set_xlabel("Structural discrimination (AUC)")
        S.panel_title(ax, letter, S.PROC_LABEL[p], y=1.03)
    h = [Patch(color="#4A4A4A", label="Training (536)"),
         Patch(facecolor="#BDBDBD", edgecolor="#4A4A4A", lw=0.6, label="Held-out (135)"),
         Line2D([], [], color="black", lw=0.8, label="Seed range"),
         Patch(color=S.BAND_NEG, label="Choice randomized within strata")]
    xc5 = (axs[0, 0].get_position().x0 + axs[0, 1].get_position().x1) / 2
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=S.FS_TICK, bbox_to_anchor=(xc5, 1.0),
               borderaxespad=0.3, columnspacing=1.6)
    S.save(fig, out_dir, NAMES["transfer"])


MAP_GAP_X = 0.10      # in, between map columns
MAP_TITLE_H = 0.20    # in, panel title band above each map
MAP_MARGIN = 0.03     # in, outer margin


def _map_size(outline, ncols=2):
    ext = geo.map_extent(outline)
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    mw = (W - 2 * MAP_MARGIN - (ncols - 1) * MAP_GAP_X) / ncols
    return ext, mw, mw * aspect


# ======================================================================
# Figure 6 - real-catchment evidence classes (composition bar below each map)
# ======================================================================
def fig6(data_dir, out_dir, layers):
    regions, states, outline = layers
    cat = _read(data_dir, "catchments.csv")
    cat["x"], cat["y"] = geo.project_points(cat.lon, cat.lat)
    ev = _read(data_dir, "fig6_evidence.csv").merge(cat[["gauge_id", "x", "y"]], on="gauge_id")
    ext, mw, mh = _map_size(outline)
    bar_gap, bar_h, row_gap, leg_h = 0.03, 0.19, 0.10, 0.27
    panel_h = MAP_TITLE_H + mh + bar_gap + bar_h
    H = 2 * panel_h + row_gap + leg_h + 2 * MAP_MARGIN
    fig = plt.figure(figsize=(W, H))
    for i, p in enumerate(S.PROC):
        r, c = divmod(i, 2)
        x0 = MAP_MARGIN + c * (mw + MAP_GAP_X)
        top = H - MAP_MARGIN - r * (panel_h + row_gap)
        y_map = top - MAP_TITLE_H - mh
        ax = _map_axes(fig, (x0, y_map, mw, mh), ext, W, H)
        geo.draw_base(ax, regions, states, outline, region_lw=0.8, extent=ext)
        d = ev[ev.process == p]
        unc, off, on = d[d.cls == "UNC"], d[d.cls == "OFF"], d[d.cls == "ON"]
        ax.scatter(unc.x, unc.y, s=7, facecolor="white", edgecolor=S.GRAY_EDGE, lw=0.5, zorder=4)
        ax.scatter(off.x, off.y, s=7, color=S.LIGHT[p], edgecolor="white", lw=0.3, zorder=5)
        ax.scatter(on.x, on.y, s=7, color=S.DARK[p], edgecolor="white", lw=0.3, zorder=6)
        S.panel_title(ax, "abcd"[i], S.PROC_LABEL[p], x=0.0, y=1.01, fontsize=10.0)

        bx = _map_axes(fig, (x0, y_map - bar_gap - bar_h, mw, bar_h), None, W, H)
        counts = [len(on), len(off), len(unc)]
        tot = sum(counts)
        left = 0.0
        for cnt, fc, ec, tc in zip(counts, [S.DARK[p], S.LIGHT[p], "white"],
                                   [S.DARK[p], S.LIGHT[p], S.GRAY_EDGE], ["white", "#222222", "#444444"]):
            wdt = cnt / tot
            bx.add_patch(Rectangle((left, 0.0), wdt, 1.0, facecolor=fc, edgecolor=ec, lw=0.6))
            if wdt < 0.055:
                tx, ha, tc = left + wdt + 0.035, "left", S.INK
                bx.plot([left + wdt / 2, tx - 0.008], [0.45, 0.75],
                        color=S.INK, lw=0.75, zorder=4, clip_on=False)
                bx.plot([left + wdt / 2], [0.45], marker="o", ms=1.8,
                        color=S.INK, zorder=5, clip_on=False)
                bx.text(tx, 0.75, f"{cnt}", ha=ha, va="center",
                        fontsize=11, color=tc)
            else:
                tx, ha = left + wdt / 2, "center"
                bx.text(tx, 0.5, f"{cnt}", ha=ha, va="center",
                        fontsize=11, color=tc)
            left += wdt
        bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.set_axis_off()

    h = [Line2D([], [], marker="o", ls="", color=S.DARK["snow"], mec="white", ms=4.5, label="Decisive ON"),
         Line2D([], [], marker="o", ls="", color=S.LIGHT["snow"], mec="white", ms=4.5, label="Decisive OFF"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=S.GRAY_EDGE, ms=4.5, label="Uncertain")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=11.0,
               bbox_to_anchor=(0.5, 0.0025))
    S.save(fig, out_dir, NAMES["evidence"])


# ======================================================================
# Figure 7 - regional exclusion (bar-cell layout)
# ======================================================================
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


# ======================================================================
# Figure 8 - process removal
# ======================================================================
GAP_FILL = "#E4E4E4"


def _snap_x(ax, lo0=False):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))


def fig8(data_dir, out_dir):
    sc = _read(data_dir, "fig8_scatter.csv")
    con = _read(data_dir, "fig8_contrast.csv")
    meta = _json(data_dir, "fig8_meta.json")
    lt = meta["loss_linthresh"]
    ylo, yhi = meta["loss_ylim"]

    fig = plt.figure(figsize=(W, 4.85))
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
        ax.set_yticks([-0.1, 0, 0.1, 1]); ax.set_yticklabels(["\u22120.1", "0", "0.1", "1"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
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
    h = [Line2D([], [], marker="o", ls="", color=S.DARK["snow"], ms=4.5, label="Decisive ON (dark tone)"),
         Line2D([], [], marker="o", ls="", color=S.LIGHT["snow"], ms=4.5, label="Decisive OFF (light tone)"),
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
    e.set_yticks([ys[p] for p in S.PROC]); e.set_yticklabels(labels, linespacing=1.0)
    e.tick_params(axis="y", length=0)
    e.set_ylim(-0.6, len(S.PROC) - 0.4)
    S.vgrid(e)
    e.set_xlabel("LOPO omission loss (ΔNSE)")
    e.text(-0.015, 1.0, "n ON/OFF", transform=e.transAxes, ha="right", va="bottom",
           fontsize=S.FS_SMALL, color="#666666")
    S.panel_title(e, "e", "Loss of decisive ON and OFF catchments", y=1.03)
    _snap_x(e)
    e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _neg(f"{v:g}")))

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
    _snap_x(f, lo0=True)
    f.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _neg(f"{v:g}")))
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
    S.save(fig, out_dir, NAMES["removal"])


# ======================================================================
# Attribute relationships of learned gates and counterfactual targets
# ======================================================================
_FMT = FuncFormatter(lambda v, _: _neg(f"{v:g}"))


def fig_attr(data_dir, out_dir, ntop=6):
    ac = _read(data_dir, "attr_corr.csv")
    procs = [p for p in S.PROC if p in set(ac.process)]
    nat = ac.attribute.nunique()
    lab_w, gap_c, axw = 1.48, 0.10, None
    ncol = 2
    axw = (W - ncol * lab_w - 0.12 - (ncol - 1) * gap_c) / ncol
    row_h, top_m, row_gap, bot_m = 0.165, 0.40, 0.30, 0.42
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
        S.panel_title(ax, "abcdefgh"[i], S.PROC_LABEL[p], x=-0.02, y=1.0)
    h = [Patch(color="#555555", label=f"$\\rho(w_p, x)$, top {ntop} by $|\\rho|$"),
         Line2D([], [], color="#B5B5B5", lw=1.2, label="Other attributes"),
         Line2D([], [], marker="D", ls="", mfc="white", mec="#1A1A1A", ms=4.6,
                label=r"$\rho(q_p^{\mathrm{post}}, x)$, preference proxy")]
    # centre the legend over the panel block (label column included), not over the whole canvas
    panel_cx = (lab_w + (W - 0.12)) / 2.0 / W
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=S.FS_TICK,
               bbox_to_anchor=(panel_cx, 1.0),
               borderpad=0.15, borderaxespad=0.0, columnspacing=1.6)
    S.save(fig, out_dir, NAMES["attr"])
