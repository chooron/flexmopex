#!/usr/bin/env python3
"""Figure 3 - standalone module.

Drawing logic lives here; shared fonts, colours and geometry helpers come from `_common.py`
(which is `figures.py` minus the per-figure functions). Reads the archived figure inputs and
writes Figure 3 into the archive's `figures/main`.
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
from _common import ARIDITY_CBAR_CLIP_PCT, BOT_MARGIN, CB_LABEL_W, CB_PAD, MARGIN, NAMES, ROW_GAP, TITLE_H, W, _colorbar, _fig2_geometry, _inset_distribution, _read, set_numbers

set_numbers(range(3, 11))


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

def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 3")
    ap.add_argument("--data", type=Path, default=paths.FIG_INPUTS)
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "main")
    ap.add_argument("--regions", type=Path, default=paths.FIG_INPUTS / "regions.gpkg")
    ap.add_argument("--states", type=Path, default=paths.FIG_INPUTS / "states.gpkg")
    ap.add_argument("--field", default="loro_regio")

    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = None
    layers = geo.load_layers(str(a.regions), str(a.states), region_field=a.field)
    fig2(a.data, a.out, layers)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
