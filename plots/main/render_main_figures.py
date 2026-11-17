#!/usr/bin/env python3
"""
Render all Flex-MOPEX main-text figures (Figures 3-10) using verified frozen data.
Strictly adheres to project instructions: PNG only, 600 DPI, full WRR width (7.0 in).
"""
import argparse
import os
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
sys.path.insert(0, str(REPO / "plots/main"))
import paths  # noqa: E402
import figures as F  # noqa: E402
import geo  # noqa: E402
import figstyle as S  # noqa: E402

DATA_DIR = paths.FIG_INPUTS
OUT_DIR = paths.FIGURES / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROUTING = {name: DATA_DIR / name for name in (
    "catchments.csv", "fig3_catchment_nse.csv", "fig3_summary.json", "fig4_scatter.csv",
    "fig4_recall.csv", "fig4_meta.json", "fig5_recovery.csv", "fig6_evidence.csv",
    "fig7_regional.csv", "fig8_scatter.csv", "fig8_contrast.csv", "fig8_meta.json",
    "attr_corr.csv")}

def custom_read(_d, name):
    if name in DATA_ROUTING:
        p = DATA_ROUTING[name]
        return pd.read_csv(p, dtype={"gauge_id": str})
    raise FileNotFoundError(f"Unknown data file requested: {name}")

def custom_json(_d, name):
    if name in DATA_ROUTING:
        p = DATA_ROUTING[name]
        with open(p, "r", encoding="utf-8") as fp:
            return json.load(fp)
    raise FileNotFoundError(f"Unknown json file requested: {name}")

F._read = custom_read
F._json = custom_json

# Configure figure numbers: 3 to 10
F.set_numbers(range(3, 11))

JOBS = {
    "study": ("Figure 3 (Study Area)", "study", F.fig2, True),
    "prediction": ("Figure 4 (Predictive Performance)", "prediction", F.fig3, False),
    "detect": ("Figure 5 (Detectability & Recovery)", "detect", F.fig4, False),
    "transfer": ("Figure 6 (Scenario Recovery & Transfer)", "transfer", F.fig5, False),
    "evidence": ("Figure 7 (Counterfactual Evidence Maps)", "evidence", F.fig6, True),
    "attr": ("Figure 8 (Attribute Associations)", "attr", F.fig_attr, False),
    "regional": ("Figure 9 (Regional Exclusion LORO)", "regional", F.fig7, False),
    "removal": ("Figure 10 (Process Removal Effects)", "removal", F.fig8, False),
}


def main():
    ap = argparse.ArgumentParser(
        description="Render Flex-MOPEX main-text figures from verified frozen data (PNG only)."
    )
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated subset of: " + ",".join(JOBS) + " (default: all)",
    )
    args = ap.parse_args()
    wanted = [k.strip() for k in args.only.split(",") if k.strip()] or list(JOBS)
    unknown = [k for k in wanted if k not in JOBS]
    if unknown:
        ap.error(f"unknown figure key(s): {', '.join(unknown)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    layers = None
    for key in wanted:
        title, name_key, fn, needs_layers = JOBS[key]
        if needs_layers and layers is None:
            regions_path = DATA_DIR / "regions.gpkg"
            states_path = DATA_DIR / "states.gpkg"
            print("Loading GIS layers...")
            layers = geo.load_layers(str(regions_path), str(states_path), region_field="loro_regio")
        print(f"\n--- Rendering {title} ---")
        if needs_layers:
            fn(None, str(OUT_DIR), layers)
        else:
            fn(None, str(OUT_DIR))
        print(f"Rendered: {OUT_DIR / F.NAMES[name_key]}.png")

    print("\nRendered successfully to:", OUT_DIR)

if __name__ == "__main__":
    main()
