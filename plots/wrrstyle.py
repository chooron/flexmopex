"""Shared visual identity for the Flex-MOPEX Supporting Information figures.

One style sheet for every SI figure that carries the four-process comparison (S1, S3-S7).
`FigureS2_attribute_matrix` keeps its own diverging palette and does not import this module.

Design rules (enforced here so the individual scripts stay short):
  * page width 7.0 in, 600 dpi, PNG + vector PDF, no "Figure Sx" text in the canvas
  * panel titles 9.5 pt bold, left aligned, pad 6
  * axis labels 8.75 pt, tick labels and legends 7.75 pt, annotations 7.5 pt
  * canonical process colours and a fixed top-to-bottom order: snow, subsurface, phenology,
    interception
  * despined top/right, #eaeaea horizontal grid, #555555 reference lines
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

FULL_WIDTH = 7.0

# ------------------------------------------------------------------ typography
_PREFERRED = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_available = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in _PREFERRED if f in _available), "DejaVu Sans")

FS_TITLE = 9.5      # panel titles, bold
FS_LABEL = 8.75     # axis labels
FS_TICK = 7.75      # tick labels, legends
FS_ANNOT = 7.5      # in-plot annotations

INK = "#333333"
GRID = "#eaeaea"
REF = "#555555"
ZERO = "#444444"

# ------------------------------------------------- canonical process palette
PROCS = ["snow", "sub", "phen", "int"]
PROC = {
    "snow": "#1f5a99",   # deep blue
    "sub": "#18987b",    # teal green
    "phen": "#993366",   # plum
    "int": "#d4923a",    # ochre
}
NAME = {
    "snow": "Snow",
    "sub": "Subsurface",
    "phen": "Phenology",
    "int": "Interception",
}


def tint(p: str, f: float = 0.62) -> str:
    """Light tint of a process colour (blend towards white by factor f)."""
    r, g, b = (int(PROC[p][i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(round(c + (255 - c) * f)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


TINT = {p: tint(p) for p in PROCS}
PALE = {p: tint(p, 0.78) for p in PROCS}

# regime palette for the flow-regime figure (deliberately not process colours)
REGIME = {
    "total": ("#222222", "o"),
    "high": ("#c44e52", "D"),
    "mid": ("#e68613", "^"),
    "low": ("#56B4E9", "s"),
}

HALO = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)


def apply() -> None:
    mpl.rcParams.update({
        "font.family": FONT,
        "font.size": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "axes.labelpad": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.1,
        "legend.borderaxespad": 0.25,
        # mathtext must match the text font, not the matplotlib default
        "mathtext.fontset": "dejavusans" if FONT == "DejaVu Sans" else "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def hgrid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, letter: str, text: str, y: float = 1.0, pad: float = 6.0,
                x: float = 0.0, **kw):
    """Bold, left-aligned '(x) Title' above the axes (optionally x-shifted)."""
    t = ax.set_title(f"({letter}) {text}", loc="left", pad=pad, fontsize=FS_TITLE,
                     fontweight="bold", color="#1a1a1a", y=y, **kw)
    if x:
        t.set_x(x)
    return t


def note(ax, x, y, text, **kw):
    """In-plot annotation with a white halo so it never collides with ink."""
    kw.setdefault("fontsize", FS_ANNOT)
    kw.setdefault("color", INK)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    kw.setdefault("zorder", 6)
    return ax.text(x, y, text, bbox=HALO, **kw)


def save(fig, out_dir, stem: str, extra_names=()) -> None:
    """Write <stem>.png and <stem>.pdf, plus byte-identical copies under extra names."""
    import os
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"CreationDate": None, "ModDate": None}
    for name in (stem,) + tuple(extra_names):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=600, metadata=meta,
                        facecolor="white")
    plt.close(fig)
