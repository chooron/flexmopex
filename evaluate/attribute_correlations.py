#!/usr/bin/env python3
"""Results 3.4 attribute associations: gate correlation and counterfactual rank proxy.

``rho_w`` is Spearman(three-seed mean gate, attribute) and uses the prepared
Caravan attribute table. ``rho_q`` is Spearman(sigmoid(median across seeds of the
selected-checkpoint DeltaJ), raw CAMELS attribute); it is a rank proxy for the
counterfactual preference, not the training BCE target. The two use different
attribute tables and masks, so their sample sizes differ.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
PROC = {"snow": "w_snow", "subsurface": "w_sub", "phenology": "w_phen",
        "interception": "w_int", "sub": "w_sub", "phen": "w_phen", "int": "w_int"}
CATEGORICAL = {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}
ALIAS = {
    "area_gages2": "area", "elev_mean": "elev", "slope_mean": "slope",
    "frac_snow": "f_snow", "aridity": "arid", "p_seasonality": "p_seas",
    "high_prec_freq": "hi_p_freq", "high_prec_dur": "hi_p_dur",
    "low_prec_freq": "lo_p_freq", "low_prec_dur": "lo_p_dur",
    "frac_forest": "f_forest", "dom_land_cover_frac": "lc_dom_frac",
    "root_depth_50": "root_d50", "root_depth_99": "root_d50",
    "soil_depth_pelletier": "s_depth_p", "soil_depth_statsgo": "s_depth_s",
    "soil_porosity": "s_poros", "soil_conductivity": "s_cond",
    "max_water_content": "w_max", "sand_frac": "sand", "silt_frac": "silt",
    "clay_frac": "clay", "glim_1st_class_frac": "g_1st_frac",
    "glim_2nd_class_frac": "g_2nd_frac", "carbonate_rocks_frac": "carb_frac",
    "geol_porosity": "g_poros", "geol_permeability": "g_perm",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "inputs/evidence/v6_weights_seedwise.csv")
    parser.add_argument("--delta-dir", type=Path, default=ROOT / "inputs/evidence")
    parser.add_argument("--caravan", type=Path, default=ROOT / "inputs/caravan_671_attributes.csv")
    parser.add_argument("--camels", type=Path, default=ROOT / "inputs/camels_attributes.npy",
                        help="(671, 35) attribute matrix; a pickled camels_dataset also works")
    parser.add_argument("--ids", type=Path, default=ROOT / "data/gage_id.npy")
    parser.add_argument("--names", type=Path, default=ROOT / "configs/flex.yaml")
    parser.add_argument("--base", type=Path, default=ROOT / "inputs/figures/attr_corr.csv",
                        help="verified table supplying the 4x32 attribute list and labels")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ids = np.load(args.ids).astype(int).reshape(-1)
    assert len(ids) == 671 and len(set(ids)) == 671
    names = yaml.safe_load(args.names.read_text())["delta_model"]["nn_model"]["attributes"]
    keep = [i for i, n in enumerate(names) if n not in CATEGORICAL]
    assert len(names) == 35 and len(keep) == 32

    if args.camels.suffix == ".npy":
        camels = np.load(args.camels)
    else:
        with args.camels.open("rb") as handle:
            _, _, camels = pickle.load(handle)
    camels = np.asarray(camels, dtype=np.float64)
    assert camels.shape == (671, 35)

    weights = pd.read_csv(args.weights)
    weights["gauge_id"] = weights.gage_id.astype(int).map(lambda g: str(int(g)).zfill(8))
    w_mean = weights.groupby("gauge_id").mean(numeric_only=True)
    assert len(w_mean) == 671

    delta = pd.concat([pd.read_csv(args.delta_dir / f"rf_counterfactual_training_s{s}.csv")
                       for s in SEEDS], ignore_index=True)
    assert len(delta) == 8052 and not delta.duplicated(["seed", "gauge_id", "process"]).any()

    caravan = pd.read_csv(args.caravan, dtype={"gauge_id": str})
    caravan["gauge_id"] = caravan.gauge_id.str.zfill(8)
    base = pd.read_csv(args.base)
    assert len(base) == 128 and not base.duplicated(["process", "attribute"]).any()

    rows = []
    for _, r in base.iterrows():
        p = PROC[r.process]
        alias = ALIAS.get(r.attribute, r.attribute)
        if alias in caravan.columns:
            rho_w = float(stats.spearmanr(w_mean.loc[caravan.gauge_id, p].to_numpy(float),
                                          caravan[alias].to_numpy(float),
                                          nan_policy="omit")[0])
        else:
            rho_w = np.nan
        block = delta[delta.process == p].pivot(index="gauge_id", columns="seed", values="delta_J")
        block = block.reindex(index=ids, columns=SEEDS).to_numpy(float)
        assert block.shape == (671, 3)
        med = np.median(block, axis=1)
        q = 1.0 / (1.0 + np.exp(-med))
        x = camels[:, names.index(r.attribute)]
        gate = w_mean.loc[[str(i).zfill(8) for i in ids], p].to_numpy(float)
        mask = np.isfinite(gate) & np.isfinite(med) & np.isfinite(x)
        assert mask.sum() >= 30 and np.std(x[mask]) > 0
        rho_q = float(stats.spearmanr(q[mask], x[mask]).statistic)
        rows.append(dict(process=r.process, attribute=r.attribute, label=r.label,
                         category=r.category, n_rho_w=int(len(caravan)),
                         n_rho_q=int(mask.sum()), rho_w=rho_w, rho_q=rho_q,
                         rho_delta=float(stats.spearmanr(med[mask], x[mask]).statistic)))

    out = pd.DataFrame(rows)
    assert len(out) == 128
    with (args.out / "attr_corr.csv").open("x") as handle:
        out[["process", "attribute", "label", "category", "rho_w", "rho_q"]].to_csv(handle, index=False)
    with (args.out / "attr_corr_detail.csv").open("x") as handle:
        out.to_csv(handle, index=False)
    merged = base.merge(out[["process", "attribute", "rho_w", "rho_q"]],
                        on=["process", "attribute"], suffixes=("_verified", "_new"))
    print("max |rho_w - verified|", float(np.max(np.abs(merged.rho_w_verified - merged.rho_w_new))))
    print("max |rho_q - verified|", float(np.max(np.abs(merged.rho_q_verified - merged.rho_q_new))))
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
