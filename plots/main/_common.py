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




# ======================================================================
# Figure 3 - predictive context
# ======================================================================


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




# ======================================================================
# Figure 5 - recovery and transfer across scenarios
# ======================================================================


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


# ======================================================================
# Figure 7 - regional exclusion (bar-cell layout)
# ======================================================================


# ======================================================================
# Figure 8 - process removal
# ======================================================================
GAP_FILL = "#E4E4E4"


def _snap_x(ax, lo0=False):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))




# ======================================================================
# Attribute relationships of learned gates and counterfactual targets
# ======================================================================
_FMT = FuncFormatter(lambda v, _: _neg(f"{v:g}"))


