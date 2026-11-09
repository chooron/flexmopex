#!/usr/bin/env python3
"""Results 3.2 and 3.3: gate detectability and Table 2 synthetic main effects.

Uses the saved learner gates and the released evaluator archives; no model
forward and no NSE cache.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ("phen", "int", "snow", "sub")
SEEDS = (42, 43, 44)
SCEN = ("A", "B", "C", "D")


def truth(s2_dir, split, scen):
    arc = np.load(s2_dir / f"scenario_{scen}/evaluator_archive.npz")
    ids = arc["basin_ids"].astype(np.int64)
    holdout = set(json.loads(split.read_text())["ordered_holdout_ids"])
    train = np.array([i for i, g in enumerate(ids) if int(g) not in holdout])
    return arc["T"][train], arc["D_J_train"][train], arc["P_matrix"][train], ids[train]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, default=ROOT / "inputs/trained/synthetic/synthetic_e1")
    parser.add_argument("--holdout-gates", type=Path, default=ROOT / "inputs/trained/synthetic_eval")
    parser.add_argument("--s2", type=Path, default=ROOT / "inputs/synthetic")
    parser.add_argument("--split", type=Path, default=ROOT / "train/synthetic/split.json")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 3.2 detectability + recovery, training scope, Scenario A
    T, DJ, PP, tr_ids = truth(args.s2, args.split, "A")
    rows = []
    for seed in SEEDS:
        g = np.load(args.gates / f"scenA/seed_{seed}/p_struct_best_train.npy").astype(np.float64)
        assert g.shape == (536, 4)
        for j, p in enumerate(PROC):
            on = T[:, j] == 1
            hi = on & (DJ[:, j] > PP[:, j])
            wk = on & (DJ[:, j] >= 0) & (DJ[:, j] <= PP[:, j])
            ng = on & (DJ[:, j] < 0)
            rows.append(dict(process=p, seed=seed, n_train=int(T.shape[0]), n_on=int(on.sum()),
                             n_high=int(hi.sum()), n_weak=int(wk.sum()), n_negative=int(ng.sum()),
                             recall_high=float((g[hi, j] > .5).mean()) if hi.any() else np.nan,
                             recall_weak=float((g[wk, j] > .5).mean()) if wk.any() else np.nan,
                             recall_negative=float((g[ng, j] > .5).mean()) if ng.any() else np.nan,
                             gate_mean_on=float(g[on, j].mean())))
    det = pd.DataFrame(rows)
    with (args.out / "results_3_2_detectability_recall.csv").open("x") as handle:
        det.to_csv(handle, index=False)
    print(det.groupby("process")[["n_on", "recall_high"]].describe().to_string())

    # ---- 3.3 Table 2 main effects from the recorded holdout gates
    holdout = np.array(json.loads(args.split.read_text())["ordered_holdout_ids"])
    gates, truth_ho = {}, {}
    for scen in SCEN:
        with np.load(args.s2 / f"scenario_{scen}/evaluator_archive.npz") as arc:
            T_all = arc["T"]
            arc_ids = arc["basin_ids"].astype(np.int64)
        pos = {int(g): i for i, g in enumerate(arc_ids)}
        ho_idx = np.array([pos[int(g)] for g in holdout])
        assert len(ho_idx) == 135
        truth_ho[scen] = T_all[ho_idx]
        for seed in SEEDS:
            f = args.holdout_gates / f"rfs0.synthetic_e1.E1.scen{scen}.nmul4.s{seed}/p_struct_best_holdout.npy"
            gates[(scen, seed)] = np.load(f).astype(float)
            assert gates[(scen, seed)].shape == (135, 4)

    rng = np.random.default_rng(20261106)
    NB = 2000
    allidx = np.arange(135)

    def metric(scen, seed, p, idx, kind):
        j = PROC.index(p)
        t = truth_ho[scen][idx, j]
        if len(np.unique(t)) < 2:
            return np.nan
        g = gates[(scen, seed)][idx, j]
        return float(roc_auc_score(t.astype(int), g)) if kind == "auc" else \
            float(balanced_accuracy_score(t.astype(int), (g > .5).astype(int)))

    out = []
    for p, rule, rand in (("sub", ("A", "C"), ("B", "D")), ("phen", ("B", "C"), ("A", "D"))):
        for kind in ("ba", "auc"):
            per_seed = []
            for seed in SEEDS:
                r = np.nanmean([metric(s, seed, p, allidx, kind) for s in rule])
                q = np.nanmean([metric(s, seed, p, allidx, kind) for s in rand])
                per_seed.append(r - q)
            draws = []
            for _ in range(NB):
                idx = rng.integers(0, 135, 135)
                d = [np.nanmean([metric(s, seed, p, idx, kind) for s in rule])
                     - np.nanmean([metric(s, seed, p, idx, kind) for s in rand]) for seed in SEEDS]
                draws.append(np.nanmean(d))
            draws = np.asarray(draws)
            out.append(dict(process=p, statistic=kind, rule="+".join(rule), random="+".join(rand),
                            effect_pp=round(float(np.nanmean(per_seed)) * 100, 2),
                            ci_lo_pp=round(float(np.nanquantile(draws, .025)) * 100, 2),
                            ci_hi_pp=round(float(np.nanquantile(draws, .975)) * 100, 2),
                            n_boot=int(np.isfinite(draws).sum()), n_holdout=135, n_seeds=3,
                            bootstrap="2000 basin resamples, seed 20261106"))
    tab = pd.DataFrame(out)
    with (args.out / "results_3_3_table2_main_effects.csv").open("x") as handle:
        tab.to_csv(handle, index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
