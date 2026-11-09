"""Validate a data folder against the fixed schema before plotting.

Usage: python check_data.py data_real
Exits with a list of problems if any file, column or category is wrong.
Do not edit the plotting code to make data fit; fix the data files instead.
"""
import json
import os
import sys

import pandas as pd

PROC = {"snow", "sub", "phen", "int"}
REG = {f"R{i}" for i in range(1, 8)}

CSV = {
    "catchments.csv": ["gauge_id", "lon", "lat", "region", "elev_mean", "frac_snow", "aridity", "gvf_diff"],
    "fig3_catchment_nse.csv": ["gauge_id", "basic", "full", "flex", "bench"],
    "fig4_scatter.csv": ["process", "gauge_id", "D_J", "w", "truth"],
    "fig4_recall.csv": ["process", "stratum", "n", "mean", "lo", "hi"],
    "fig5_recovery.csv": ["process", "scenario", "split", "mean", "lo", "hi", "assignment", "metric"],
    "fig6_evidence.csv": ["gauge_id", "process", "cls"],
    "fig7_regional.csv": ["process", "region", "metric", "value", "small"],
    "fig8_scatter.csv": ["process", "gauge_id", "w_mean", "loss", "cls", "above_rb"],
    "fig8_contrast.csv": ["process", "estimand", "est", "lo", "hi", "n_on", "n_off"],
    "attr_corr.csv": ["process", "attribute", "label", "category", "rho_w", "rho_q"],
}
JSON = {
    "fig3_summary.json": ["median_summary", "paired_median", "paired_ci", "diff_xlim", "nse_xlim"],
    "fig4_meta.json": ["P_p"],
    "fig8_meta.json": ["loss_linthresh", "loss_ylim"],
}
CATS = [
    ("fig4_scatter.csv", "process", PROC), ("fig4_scatter.csv", "truth", {0, 1}),
    ("fig4_recall.csv", "stratum", {"high", "weak", "negative"}),
    ("fig5_recovery.csv", "scenario", {"A", "B", "C", "D"}),
    ("fig5_recovery.csv", "split", {"train", "heldout"}),
    ("fig5_recovery.csv", "assignment", {"rule", "random", "fixed"}),
    ("fig5_recovery.csv", "metric", {"recall", "auc"}),
    ("fig6_evidence.csv", "cls", {"ON", "OFF", "UNC"}),
    ("fig7_regional.csv", "region", REG | {"Pooled"}),
    ("fig7_regional.csv", "metric", {"BA", "AUC"}),
    ("fig7_regional.csv", "small", {0, 1}),
    ("fig8_scatter.csv", "cls", {"ON", "OFF", "UNC"}),
    ("fig8_scatter.csv", "above_rb", {0, 1}),
    ("fig8_contrast.csv", "estimand", {"fixed", "lopo"}),
    ("catchments.csv", "region", REG),
]


def main(d):
    err = []
    dfs = {}
    for f, cols in CSV.items():
        p = os.path.join(d, f)
        if not os.path.exists(p):
            err.append(f"missing file: {f}")
            continue
        df = pd.read_csv(p, dtype={"gauge_id": str})
        if "gauge_id" in df and df.gauge_id.notna().all() and df.gauge_id.str.fullmatch(r"\d{1,8}").all():
            df["gauge_id"] = df.gauge_id.str.zfill(8)
        miss = [c for c in cols if c not in df.columns]
        if miss:
            err.append(f"{f}: missing columns {miss}")
        if "process" in df.columns:
            bad = set(df.process.unique()) - PROC
            if bad:
                err.append(f"{f}: unknown process codes {sorted(bad)} (use snow/sub/phen/int)")
        if "gauge_id" in df.columns and df.gauge_id.str.len().ne(8).any():
            err.append(f"{f}: gauge_id must be 8-character strings with leading zeros")
        dfs[f] = df
    for f, keys in JSON.items():
        p = os.path.join(d, f)
        if not os.path.exists(p):
            err.append(f"missing file: {f}")
            continue
        js = json.load(open(p))
        miss = [k for k in keys if k not in js]
        if miss:
            err.append(f"{f}: missing keys {miss}")
    for f, c, allowed in CATS:
        if f in dfs and c in dfs[f].columns:
            vals = dfs[f][c].dropna().unique()
            bad = {v for v in vals if v not in allowed}
            if bad:
                err.append(f"{f}: column {c} has values {sorted(map(str, bad))}; allowed {sorted(map(str, allowed))}")
    if "catchments.csv" in dfs:
        n = len(dfs["catchments.csv"])
        if n != 671:
            err.append(f"catchments.csv: {n} rows (expected 671)")
    if err:
        print("DATA CHECK FAILED:")
        for e in err:
            print("  -", e)
        sys.exit(1)
    print("data check passed:", d)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_real")
