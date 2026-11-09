"""Shared style for Flex-MOPEX manuscript figures (Figures 2-8).

All figures are drawn at final print size (full width 7.0 in), so the font
sizes below are the sizes that appear in print.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

FULL_WIDTH = 7.0  # inches, WRR full page width

# ---------------------------------------------------------------- fonts ----
_PREFERRED = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_available = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _PREFERRED if f in _available), "DejaVu Sans")

FS_TITLE = 9.0   # panel letter + panel title (bold)
FS_LABEL = 8.0   # axis titles, legends
FS_TICK = 7.5    # tick labels, in-plot numbers
FS_SMALL = 7.0   # counts, minor annotations

INK = "#333333"     # text and axes
GRID = "#E6E6E6"
REF = "#555555"     # reference lines
GRAY_FILL = "#A8A8A8"
GRAY_EDGE = "#8C8C8C"
BAND_NEG = "#F1F1F1"

# ---------------------------------------------------------- process colours
PROC = ["snow", "sub", "phen", "int"]
PROC_LABEL = {
    "snow": "Snow",
    "sub": "Subsurface fast flow",
    "phen": "Phenology",
    "int": "Interception",
}
PROC_SHORT = {"snow": "Snow", "sub": "Subsurface", "phen": "Phenology", "int": "Interception"}
DARK = {"snow": "#1F5FA0", "sub": "#139E8A", "phen": "#9C3A6E", "int": "#D2953B"}
LIGHT = {"snow": "#A3BEDB", "sub": "#97D5C9", "phen": "#D7A9C3", "int": "#EDD2A7"}
BAND = {"snow": "#E3EBF4", "sub": "#E0F3EF", "phen": "#F3E6EE", "int": "#FAF0E1"}


def apply():
    mpl.rcParams.update({
        "font.family": FONT,
        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LABEL,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.labelpad": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.2,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_title(ax, letter, text, x=0.0, y=1.02, ha="left", **kw):
    """Bold '(a) Title' placed above the axes, left-aligned with the axes."""
    ax.text(x, y, f"({letter}) {text}" if text else f"({letter})",
            transform=ax.transAxes, ha=ha, va="bottom",
            fontsize=FS_TITLE, fontweight="bold", color="#1A1A1A", **kw)


def hgrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def save(fig, out_dir, name, png_dpi=600):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=png_dpi)
    plt.close(fig)
