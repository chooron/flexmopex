#!/usr/bin/env python3
"""Figure S2 - descriptive attribute-correlation matrix (heat map).

Both columns come straight from the canonical table
`analysis/tables/Figure8_and_S2_attribute_spearman_correlations.csv` (32 continuous attributes x
4 processes, rho_w and rho_q with finite-pair counts), so the figure and the table cannot diverge.
Panel geometry, fonts and the diverging palette are inherited from `si_figures.fig_attr_heat`.

Output  <FLEXMOPEX_DATA_ROOT>/figures/supplementary/FigureS2_attribute_matrix.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
import paths  # noqa: E402
import si_figures  # noqa: E402

DPI = 600


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=paths.TABLES / "figure_inputs")
    ap.add_argument("--out", type=Path, default=paths.FIGURES / "supplementary")
    a = ap.parse_args()

    corr = pd.read_csv(a.data / "Figure8_and_S2_attribute_spearman_correlations.csv")
    corr = corr.rename(columns={"spearman_rho_w": "rho_w", "spearman_rho_q": "rho_q"})
    corr = corr[["process", "attribute", "label", "category", "rho_w", "rho_q"]]
    assert len(corr) == 128 and corr.attribute.nunique() == 32


    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / "FigureS2_attribute_matrix.png"
    si_figures._read = lambda _d, _n: corr.copy()

    def save(fig, _out, _name):
        fig.savefig(target, dpi=DPI, facecolor="white")
        plt.close(fig)

    si_figures.S.save = save
    si_figures.fig_attr_heat(None, None, None)
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
