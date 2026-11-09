"""Study catchments: regions, snow fraction, aridity, elevation, GVF difference.

Bold centred title over each map, vertical colour bar on the right edge, and an
inset histogram + KDE of the attribute distribution in the lower-left corner.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

import figstyle as S
import geo
from data_io import neg, read_csv

SHORT = "study_area"

CB_PAD = 0.06        # in, breathing room around the colour bar label
CB_LABEL_W = 0.30    # in, reserved width of the rotated colour-bar label
TITLE_H = 0.26       # in, band reserved above each map for its centred title
ROW_GAP = 0.10       # in, vertical gap between map rows
MARGIN = 0.03        # in, outer margin
BOT_MARGIN = 0.14    # in, bottom margin (clears the inset x tick labels)

INSET_POS = [0.03, 0.03, 0.20, 0.20]   # lower-left histogram, axes fractions
INSET_BINS = 30
ARIDITY_CBAR_CLIP_PCT = 98


def geometry(outline, width):
    """Map width/height so every map plus its right-edge colour bar fits the page."""
    ext = geo.map_extent(outline)
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    cb_block = 0.055  # colour-bar offset+width as a fraction of the map width
    usable = width - 2 * MARGIN - 2 * CB_LABEL_W - 2 * CB_PAD
    mw = usable / (2 + 2 * cb_block)
    return ext, mw, mw * aspect


def colourbar(fig, ax, sc, title, unit, fmt, vmax_override=None, vmax_label=""):
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
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: neg(fmt.format(v))))
    cb.set_label(f"{title}{unit}{vmax_label}", fontsize=S.FS_LABEL, labelpad=2)
    return cb


def inset_distribution(ax_map, values, cmap_name):
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
    keep = [tv for tv in ax_ins.get_xticks() if v.min() <= tv <= v.max()]
    if keep:
        ax_ins.set_xticks(keep)
    ax_ins.set_yticks([])
    for side in ("top", "right", "left"):
        ax_ins.spines[side].set_visible(False)
    ax_ins.tick_params(axis="x", labelsize=S.FS_SMALL - 0.5, pad=1.5, length=2)
    ax_ins.tick_params(axis="y", length=0)
    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--region-field", default="region")
    parser.add_argument("--number", type=int, default=3)
    args = parser.parse_args()

    S.apply()
    width = S.FULL_WIDTH
    regions, states, outline = geo.load_layers(
        args.regions, args.states, region_field=args.region_field
    )
    cat = read_csv(args.data, "catchments.csv")
    cat["x"], cat["y"] = geo.project_points(cat.lon, cat.lat)
    n_reg = cat.region.value_counts().to_dict()
    ext, mw, mh = geometry(outline, width)
    height = 2 * (TITLE_H + mh) + ROW_GAP + MARGIN + BOT_MARGIN
    fig = plt.figure(figsize=(width, height))
    col_pitch = mw + 0.055 * mw + CB_LABEL_W + CB_PAD

    spec = [
        ("frac_snow", "Snow fraction", " (\u2013)", "Blues", "{:.1f}", None),
        ("aridity", "Aridity, PET/P", " (\u2013)", "YlOrBr", "{:.1f}", "  (>P98)"),
        ("elev_mean", "Elevation", " (m)", "YlGnBu", "{:.0f}", None),
        ("gvf_diff", "GVF difference", " (\u2013)", "Greens", "{:.1f}", None),
    ]
    for i, (col, title, unit, cmap, fmt, clip) in enumerate(spec):
        r, c = divmod(i, 2)
        x0 = MARGIN + c * col_pitch
        y0 = height - MARGIN - (r + 1) * (TITLE_H + mh) - r * ROW_GAP
        ax = fig.add_axes([x0 / width, y0 / height, mw / width, mh / height])
        geo.draw_base(ax, regions, states, outline, extent=ext)
        v = cat[col].to_numpy(dtype=float)
        sc = ax.scatter(cat.x.to_numpy(), cat.y.to_numpy(), c=v, cmap=cmap,
                        s=10, alpha=0.78, edgecolors="none", linewidths=0, zorder=3)
        vmax_override = np.nanpercentile(v, ARIDITY_CBAR_CLIP_PCT) if clip else None
        colourbar(fig, ax, sc, title, unit, fmt,
                  vmax_override=vmax_override, vmax_label=clip or "")
        ax.set_title(title, fontsize=S.FS_TITLE, fontweight="bold", pad=6)
        if i == 0:
            for rg, (px, py) in geo.region_label_points(regions).items():
                ax.text(px, py, f"{rg} (n={n_reg.get(rg, 0)})", ha="center", va="center",
                        fontsize=S.FS_SMALL, fontweight="bold", color="#1A1A1A", zorder=6,
                        linespacing=0.95,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
        inset_distribution(ax, v, cmap)
    S.save(fig, str(args.out), f"Figure{args.number}_{SHORT}")


if __name__ == "__main__":
    main()
