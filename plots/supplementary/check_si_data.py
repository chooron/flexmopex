"""Validate the SI data folder against the fixed schema. Fix the data, never the code.

Usage: python check_si_data.py data_real
"""
import json
import os
import sys

import pandas as pd

PROC = {"snow", "sub", "phen", "int"}
CSV = {
    "si_lambda_skill.csv": ["lam", "ckpt", "median_nse"],
    "si_lambda_gates.csv": ["lam", "ckpt", "process", "mean_w"],
    "si_variant_recovery.csv": ["process", "metric", "split", "method", "mean", "lo", "hi"],
    "si_variant_skill.csv": ["variant", "label", "median_nse"],
    "si_variant_paired.csv": ["process", "metric", "est", "lo", "hi"],
    "si_variant_gates.csv": ["method", "process", "seed", "gauge_id", "w"],
    "si_synth_scatter.csv": ["process", "scenario", "split", "gauge_id", "x", "w", "truth"],
    "si_attr_corr.csv": ["process", "attribute", "label", "category", "rho_w", "rho_q"],
    "si_snow_bench.csv": ["gauge_id", "bench_nse", "loss_fixed", "loss_lopo"],
    "si_snow_bench_corr.csv": ["estimand", "kind", "rho", "lo", "hi"],
    "si_refit_diff.csv": ["process", "gauge_id", "diff"],
    "si_refit_exceed.csv": ["process", "tau", "frac", "n"],
}
JSON = {
    "si_lambda_meta.json": ["declared_lambda"],
    "si_variant_meta.json": ["metrics", "methods"],
    "si_synth_meta.json": ["panels"],
    "si_snow_bench_meta.json": ["loss_ylim"],
}
CATS = [
    ("si_lambda_skill.csv", "ckpt", {"best", "last"}),
    ("si_lambda_gates.csv", "ckpt", {"best", "last"}),
    ("si_variant_recovery.csv", "split", {"train", "holdout"}),
    ("si_variant_gates.csv", "method", {"canonical", "strict"}),
    ("si_synth_scatter.csv", "split", {"train", "holdout"}),
    ("si_synth_scatter.csv", "truth", {0, 1}),
    ("si_snow_bench_corr.csv", "estimand", {"fixed", "lopo"}),
    ("si_snow_bench_corr.csv", "kind", {"raw", "partial"}),
]


def main(d):
    err, dfs = [], {}
    for f, cols in CSV.items():
        p = os.path.join(d, f)
        if not os.path.exists(p):
            err.append(f"missing file: {f}")
            continue
        df = pd.read_csv(p, dtype={"gauge_id": str})
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
    js = {}
    for f, keys in JSON.items():
        p = os.path.join(d, f)
        if not os.path.exists(p):
            err.append(f"missing file: {f}")
            continue
        js[f] = json.load(open(p))
        miss = [k for k in keys if k not in js[f]]
        if miss:
            err.append(f"{f}: missing keys {miss}")
    for f, c, allowed in CATS:
        if f in dfs and c in dfs[f].columns:
            bad = {v for v in dfs[f][c].dropna().unique() if v not in allowed}
            if bad:
                err.append(f"{f}: column {c} has values {sorted(map(str, bad))}; allowed {sorted(map(str, allowed))}")
    if "si_variant_meta.json" in js:
        keys = {m["key"] for m in js["si_variant_meta.json"].get("metrics", [])}
        meths = {m["key"] for m in js["si_variant_meta.json"].get("methods", [])}
        for f in ("si_variant_recovery.csv", "si_variant_paired.csv"):
            if f in dfs:
                bad = set(dfs[f].metric) - keys
                if bad:
                    err.append(f"{f}: metric keys {sorted(bad)} not declared in si_variant_meta.json")
        if "si_variant_recovery.csv" in dfs:
            bad = set(dfs["si_variant_recovery.csv"].method) - meths
            if bad:
                err.append(f"si_variant_recovery.csv: methods {sorted(bad)} not declared in si_variant_meta.json")
    if "si_attr_corr.csv" in dfs:
        cnt = dfs["si_attr_corr.csv"].groupby("process").attribute.nunique()
        if cnt.nunique() != 1:
            err.append(f"si_attr_corr.csv: processes have different attribute counts {cnt.to_dict()}")
    if "si_synth_meta.json" in js and "si_synth_scatter.csv" in dfs:
        sc = dfs["si_synth_scatter.csv"]
        for pn in js["si_synth_meta.json"]["panels"]:
            if sc[(sc.process == pn["process"]) & (sc.scenario == pn["scenario"])].empty:
                err.append(f"si_synth_scatter.csv: no rows for panel {pn['process']}/{pn['scenario']}")
    if err:
        print("SI DATA CHECK FAILED:")
        for e in err:
            print("  -", e)
        sys.exit(1)
    print("SI data check passed:", d)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_real")
