#!/usr/bin/env python3
"""Results 3.1: Full/Flex test-period skill aggregation and paired differences.

Reads only the frozen ``metrics.json`` of each run; no model forward, no scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
ARMS = ("full", "flex")


def metrics(results, arm, seed):
    run = results / f"{arm}/ref/seed_{seed}"
    hits = sorted(run.glob("test1995-2010_Ep*/metrics.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one scored test window under {run}, found {hits}")
    m = json.loads(hits[0].read_text())
    m = json.loads(m) if isinstance(m, str) else m
    return m, hits[0].name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "inputs/trained/normal")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    per = {}
    window = {}
    for arm in ARMS:
        per[arm] = {}
        for seed in SEEDS:
            m, win = metrics(args.results, arm, seed)
            per[arm][seed] = {k: np.asarray(m[k], float) for k in ("nse", "kge")}
            assert per[arm][seed]["nse"].size == 671, (arm, seed)
            window[arm] = win
        for metric in ("nse", "kge"):
            seed_med = {s: float(np.median(per[arm][s][metric])) for s in SEEDS}
            basin_mean = np.mean([per[arm][s][metric] for s in SEEDS], axis=0)
            rows.append(dict(
                model=arm, metric=metric,
                A_mean_of_seed_medians=round(float(np.mean(list(seed_med.values()))), 4),
                A_seed_values=";".join(f"{s}:{v:.4f}" for s, v in seed_med.items()),
                B_median_of_basin_means=round(float(np.median(basin_mean)), 4),
                C_pooled_median=round(float(np.median(np.concatenate(
                    [per[arm][s][metric] for s in SEEDS]))), 4),
                D_mean_of_basin_means=round(float(np.mean(basin_mean)), 4),
                n_basins=671, n_seeds=3, window=window[arm]))

    rng = np.random.default_rng(20261106)
    paired = []
    for metric in ("nse", "kge"):
        delta = np.mean([per["flex"][s][metric] - per["full"][s][metric] for s in SEEDS], axis=0)
        boot = np.median(delta[rng.integers(0, delta.size, size=(2000, delta.size))], axis=1)
        paired.append(dict(metric=metric, scope="seed_mean", n=671,
                           median=float(np.median(delta)), mean=float(delta.mean()),
                           ci_lo=float(np.quantile(boot, .025)),
                           ci_hi=float(np.quantile(boot, .975)),
                           frac_flex_worse=float((delta < 0).mean()),
                           aggregation="per-seed Flex-Full, then basin mean over 3 seeds, then cross-basin median",
                           bootstrap="2000 basin resamples, seed 20261106"))

    out = pd.concat([pd.DataFrame(rows),
                     pd.DataFrame(paired).assign(model="flex_minus_full")], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        out.to_csv(handle, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
