#!/usr/bin/env python3
"""Results 3.4: attribute direction AUC from held-out regional gates.

The Full-data evidence class is the reference label; the held-out (LORO) gate is
the predictor. A single-reference-class region yields NaN, not a substituted
value.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
PROCS = ("w_phen", "w_int", "w_snow", "w_sub")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(y, score, region, rng):
    """Paired basin resampling stratified by region x decisive class (frozen P5 rule)."""
    groups = [np.flatnonzero((region == r) & (y == c)) for r in np.unique(region) for c in (0, 1)]
    groups = [x for x in groups if len(x)]
    nboot = 1000
    picks = np.concatenate([x[rng.integers(len(x), size=(nboot, len(x)))] for x in groups], axis=1)
    labs = y[picks]
    ranks = rankdata(score[picks], axis=1, method="average")
    n1 = labs.sum(axis=1)
    n0 = labs.shape[1] - n1
    vals = ((ranks * labs).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.quantile(vals, [.025, .975]), int(np.isfinite(vals).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "inputs/trained/loro/flex")
    parser.add_argument("--classes", type=Path,
                        default=ROOT / "inputs/evidence/rf_evidence_classes_comparison.csv")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ids = np.load(args.data / "gage_id.npy").astype(int).reshape(-1)
    assert len(ids) == 671 and len(set(ids)) == 671
    cls = pd.read_csv(args.classes)
    assert len(cls) == 2684 and not cls.duplicated(["gauge_id", "process"]).any()

    parts, membership, checks = [], [], []
    for r in range(7):
        group = np.load(args.data / f"basin_groups/group_{11 + r}.npy").astype(int).reshape(-1)
        ix = np.flatnonzero(np.isin(ids, group))
        assert len(ix) == len(group) and len(set(group)) == len(group)
        membership.extend(ids[ix].tolist())
        for seed in SEEDS:
            run = args.runs / f"region{r}/seed_{seed}"
            m = json.loads((run / "manifest.json").read_text())
            cfg = json.loads((run / "config_snapshot.json").read_text())
            assert m["family"] == "loro_flex"
            assert int(m["config"]["loro_region"]) == r
            assert int(m["training"]["seed"]) == seed
            assert cfg["test"]["test_group_id"] == 11 + r
            assert cfg["model"]["nn"]["name"] == "LearnedStructureNetPureAttrEncoder"
            assert len(cfg["model"]["nn"]["attributes"]) == 35
            checkpoint = run / "model/best_checkpoint.pt"
            assert sha256(checkpoint) == m["checkpoints"]["best"]["sha256"]
            epoch = int(m["selection"]["selected_epoch"])
            diag = np.load(run / f"diagnostics/p_struct_epoch_{epoch:03d}.npy")
            assert diag.shape == (671, 4)
            for j, p in enumerate(PROCS):
                v = np.load(run / "sim" / f"{p}.npy", mmap_mode="r")
                assert v.shape == (5113, len(ix), 1)
                first = np.asarray(v[0, :, 0], float)
                middle = np.asarray(v[len(v) // 2, :, 0], float)
                last = np.asarray(v[-1, :, 0], float)
                temporal = float(max(np.max(abs(first - middle)), np.max(abs(first - last))))
                err = float(np.max(abs(first - diag[ix, j])))
                assert temporal < 1e-7 and err < 1e-5, (r, seed, p, temporal, err)
                assert np.isfinite(first).all() and (first >= 0).all() and (first <= 1).all()
                checks.append(dict(region=r + 1, seed=seed, process=p, n=len(ix), selected_epoch=epoch,
                                   max_abs_vs_selected_diagnostic=err, max_temporal_sample_diff=temporal,
                                   checkpoint_sha256=m["checkpoints"]["best"]["sha256"]))
                parts.append(pd.DataFrame({"gauge_id": ids[ix], "region": r + 1, "seed": seed,
                                           "process": p, "gate": first}))
    assert sorted(membership) == sorted(ids.tolist())

    gates = pd.concat(parts, ignore_index=True).merge(
        cls[["gauge_id", "process", "evidence_class_v2"]],
        on=["gauge_id", "process"], validate="many_to_one")
    assert len(gates) == 8052 and not gates.duplicated(["gauge_id", "process", "seed"]).any()
    agg = gates.groupby(["gauge_id", "region", "process", "evidence_class_v2"], as_index=False).gate.median()
    agg["seed"] = "median_gate"
    allg = pd.concat([gates, agg], ignore_index=True)

    rng = np.random.default_rng(20260521)
    rows = []
    for proc in PROCS:
        for seed in [42, 43, 44, "median_gate"]:
            for region in range(8):
                d = allg[(allg.process == proc) & (allg.seed == seed)]
                d = d if region == 0 else d[d.region == region]
                elig = d[d.evidence_class_v2.isin(["decisive_open", "decisive_closed"])]
                y = (elig.evidence_class_v2 == "decisive_open").to_numpy(int)
                v = elig.gate.to_numpy(float)
                rg = elig.region.to_numpy(int)
                n1, n0 = int(y.sum()), int(len(y) - y.sum())
                metric = np.nan
                lo = hi = np.nan
                nvalid = 0
                tpr = float(np.mean(v[y == 1] > .5)) if n1 else np.nan
                tnr = float(np.mean(v[y == 0] <= .5)) if n0 else np.nan
                if n1 and n0 and len(y) >= 3:
                    metric = float(roc_auc_score(y, v))
                    ranks = rankdata(v, method="average")
                    other = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
                    assert abs(metric - other) < 1e-12
                    (lo, hi), nvalid = interval(y, v, rg, rng)
                rows.append(dict(process=proc, seed=seed, scope="pooled" if region == 0 else f"R{region}",
                                 n_total=len(d), n_decisive=len(y), n_on=n1, n_off=n0,
                                 coverage=len(y) / len(d), auc=metric, ci_lo=lo, ci_hi=hi,
                                 n_boot_valid=nvalid, tpr=tpr, tnr=tnr,
                                 balanced_accuracy=(tpr + tnr) / 2 if n1 and n0 else np.nan,
                                 bootstrap="1000 basin bootstrap stratified region x full-data class; seed=20260521; sequential RNG",
                                 score="seedwise gate" if seed != "median_gate" else "per-basin median across 3 seed gates"))

    with (args.out / "attribute_direction_gate_seedwise.csv").open("x") as handle:
        gates.to_csv(handle, index=False)
    out = pd.DataFrame(rows)
    with (args.out / "attribute_direction_auc.csv").open("x") as handle:
        out.to_csv(handle, index=False)
    with (args.out / "attribute_direction_source_checks.csv").open("x") as handle:
        pd.DataFrame(checks).to_csv(handle, index=False)
    print(out[(out.scope == "pooled") & (out.seed == "median_gate")].to_string(index=False))


if __name__ == "__main__":
    main()
