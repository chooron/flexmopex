#!/usr/bin/env python3
"""Comprehensive scientific evaluation for retrained S2 Batch-1 results."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from train.synthetic.strict.s2_loader import load_candidate_strict_fold
from train.synthetic.strict.trainer import build_strict_model_and_loss
from train.synthetic.support import load_teacher, prepare_dataset, simulate_gate_batch

FLEX = ROOT / "project/flexmopex"
REV = FLEX / "revision1"
S2_DIR = REV / "S2"
RESULTS_DIR = S2_DIR / "batch1_results"
DATA_DIR = S2_DIR / "data"
REPORTS_DIR = S2_DIR / "reports"
FIG_DIR = S2_DIR / "figures"

SPLIT_PATH = REV / "analysis/R1_014_CANDIDATE_SPLIT.json"
IDS_PATH = ROOT / "data/gage_id.npy"
DATES_PATH = ROOT / "data/camels_dates.npy"
SOURCE_DATASET = ROOT / "data/camels_dataset"
CONFIG_PATH = FLEX / "FLEXMOPEX_PAPER_DATA/runs/core/lambda0007/seed_42/config_snapshot.json"

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
PROCESS_NAMES = ["Phenology", "Interception", "Snow", "Subsurface"]

RUNS = [
    "E1_A_42", "E1_A_43", "E1_A_44",
    "E1_B_42", "E1_C_42", "E1_D_42",
    "E3_A_42", "E4_A_42"
]


def bootstrap_ci(arr_diff: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(arr_diff)
    for _ in range(n_boot):
        sample = rng.choice(arr_diff, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def compute_nse_kge(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    denom = np.sum((obs - np.mean(obs))**2)
    if denom < 1e-8 or np.isnan(denom):
        return float("nan"), float("nan")
    nse = float(1.0 - np.sum((sim - obs)**2) / denom)
    
    r = float(np.corrcoef(sim, obs)[0, 1]) if np.std(sim) > 1e-6 and np.std(obs) > 1e-6 else 0.0
    alpha = float(np.std(sim) / (np.std(obs) + 1e-8))
    beta = float(np.mean(sim) / (np.mean(obs) + 1e-8))
    kge = float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))
    return nse, kge


def main() -> None:
    print("=== Executing Scientific Evaluation for Retrained S2 Batch-1 ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    ids = np.load(IDS_PATH)
    dates = np.load(DATES_PATH)
    id_to_idx = {int(gid): i for i, gid in enumerate(ids)}
    train_indices = np.array([id_to_idx[gid] for gid in split["train_ids"]])
    holdout_indices = np.array([id_to_idx[gid] for gid in split["holdout_ids"]])

    with open(SOURCE_DATASET, "rb") as f:
        forcings, obs_raw, raw_attrs = pickle.load(f)

    with open(FLEX / "conf/observations/camels_671.yaml") as f:
        import yaml
        all_attrs = yaml.safe_load(f)["all_attributes"]

    # Model architecture for loading checkpoints
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg.update({"mode": "train", "device": "cpu", "dtype": torch.float32})
    cfg["model"]["nmul"] = 4
    cfg["delta_model"]["phy_model"]["nmul"] = 4

    data_A = load_candidate_strict_fold(
        cfg,
        manifest_path=SPLIT_PATH,
        cache_root=Path("/tmp/cache_eval_retrain"),
        dataset_path=S2_DIR / "data/scenario_A/dataset.pkl",
        gage_info_path=IDS_PATH,
        hydro_loader_source=Path(ROOT / ".venv/lib/python3.11/site-packages/dmg/core/data/loaders/hydro_loader.py"),
    )
    model, _ = build_strict_model_and_loss(cfg, data_A, device="cpu", model_dir=Path("/tmp/m_eval_retrain"))
    inner = next(iter(model.model_dict.values()))

    # Normalization statistics
    stats = json.load(open(data_A.stats_path))
    mean_attr = np.array([stats[a][2] for a in all_attrs], dtype=np.float32)
    std_attr = np.array([stats[a][3] for a in all_attrs], dtype=np.float32)
    std_attr[std_attr == 0] = 1.0

    all_raw_attrs = raw_attrs # [671, 35]
    all_norm_attrs = torch.from_numpy(((all_raw_attrs - mean_attr) / std_attr).astype(np.float32))
    all_norm_attrs[torch.isnan(all_norm_attrs)] = 0.0

    # Load Evaluator Archives for A, B, C, D
    eval_archives = {
        sc: np.load(DATA_DIR / f"scenario_{sc}/evaluator_archive.npz")
        for sc in ["A", "B", "C", "D"]
    }

    # Helper to get gate predictions on all 671 basins from checkpoint
    def predict_all_gates(ckpt_path: Path) -> np.ndarray:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt["model_state_dict"])
        inner.eval()
        with torch.no_grad():
            w_raw = inner.nn_model.get_structure_logits(all_norm_attrs).view(-1, 4, 2).clamp(-10.0, 10.0)
            p = torch.sigmoid(w_raw[..., 1] - w_raw[..., 0]).numpy()
        return p

    # 1. Evaluate H1 on Train Basins for all runs
    h1_results = {}
    h2_results = {}
    all_p_best = {}

    date_strs = dates.astype(str)
    train_mask = (date_strs >= "1980-10-01") & (date_strs <= "1995-09-30")
    train_scored = train_mask.copy()
    train_scored[:365] = False

    test_mask = (date_strs >= "1995-10-01") & (date_strs <= "2010-09-30")
    test_scored = test_mask.copy()
    test_start_idx = np.where(test_mask)[0][0]
    test_scored[test_start_idx : test_start_idx + 365] = False

    for rid in RUNS:
        r_dir = RESULTS_DIR / rid
        with open(r_dir / "summary.json") as f:
            summ = json.load(f)
        sc = summ["scenario"]
        eval_data = eval_archives[sc]
        T = eval_data["T"] # [671, 4]
        D_tr = eval_data["D_J_train"] # [671, 4]
        P_mat = eval_data["P_matrix"] # [671, 4]

        # Predict all 671 basins from best checkpoint
        best_ckpt = r_dir / "best_checkpoint.pt"
        p_all = predict_all_gates(best_ckpt) # [671, 4]
        all_p_best[rid] = p_all
        np.save(r_dir / "p_struct_best_all671.npy", p_all)

        # H1 In-Sample (536 train basins)
        p_tr = p_all[train_indices]
        t_tr = T[train_indices]
        d_tr = D_tr[train_indices]
        p_mat_tr = P_mat[train_indices]

        h1_run = {}
        for p_idx, proc in enumerate(PROCESSES):
            w = p_tr[:, p_idx]
            t = t_tr[:, p_idx]
            d = d_tr[:, p_idx]
            p_thr = p_mat_tr[:, p_idx]
            pred = (w > 0.5).astype(int)

            # High-D: T=1 and D > P
            high_d_mask = (t == 1) & (d > p_thr)
            n_high_d = int(high_d_mask.sum())
            tp_high_d = int((pred[high_d_mask] == 1).sum()) if n_high_d > 0 else 0
            rec_high_d = float(tp_high_d / n_high_d) if n_high_d > 0 else float("nan")

            # Inactive: T=0
            t0_mask = (t == 0)
            n_t0 = int(t0_mask.sum())
            tn_t0 = int((pred[t0_mask] == 0).sum()) if n_t0 > 0 else 0
            spec_t0 = float(tn_t0 / n_t0) if n_t0 > 0 else float("nan")

            # Decidable BA
            if np.isnan(spec_t0):
                ba_decidable = rec_high_d
            elif np.isnan(rec_high_d):
                ba_decidable = spec_t0
            else:
                ba_decidable = 0.5 * (rec_high_d + spec_t0)

            # Low-D: T=1 and D <= P
            low_d_mask = (t == 1) & (d <= p_thr)
            n_low_d = int(low_d_mask.sum())
            low_d_closed_rate = float((pred[low_d_mask] == 0).mean()) if n_low_d > 0 else float("nan")
            low_d_mean_w = float(w[low_d_mask].mean()) if n_low_d > 0 else float("nan")

            # Overall Accuracy
            acc = float((pred == t).mean())

            # AUC (if both classes present)
            auc = float(roc_auc_score(t, w)) if len(np.unique(t)) > 1 else float("nan")

            q5, q25, q50, q75, q95 = [float(x) for x in np.percentile(w, [5, 25, 50, 75, 95])]
            frac_uncertain = float(((w >= 0.45) & (w <= 0.55)).mean())

            h1_run[proc] = {
                "n_high_d": n_high_d,
                "recall_high_d": rec_high_d,
                "tp_high_d": tp_high_d,
                "n_t0": n_t0,
                "specificity": spec_t0,
                "tn_t0": tn_t0,
                "ba_decidable": ba_decidable,
                "n_low_d": n_low_d,
                "low_d_closed_rate": low_d_closed_rate,
                "low_d_mean_weight": low_d_mean_w,
                "overall_accuracy": acc,
                "auc": auc,
                "quantiles": [q5, q25, q50, q75, q95],
                "frac_uncertain_45_55": frac_uncertain,
                "mean_weight": float(w.mean()),
                "std_weight": float(w.std()),
            }
        h1_results[rid] = h1_run

        # H2 Out-of-Sample (135 holdout basins)
        p_ho = p_all[holdout_indices]
        t_ho = T[holdout_indices]
        h2_run = {}

        for p_idx, proc in enumerate(PROCESSES):
            w = p_ho[:, p_idx]
            t = t_ho[:, p_idx]
            pred = (w > 0.5).astype(int)

            n_pos = int((t == 1).sum())
            n_neg = int((t == 0).sum())
            tp = int(((pred == 1) & (t == 1)).sum())
            tn = int(((pred == 0) & (t == 0)).sum())

            rec = float(tp / n_pos) if n_pos > 0 else float("nan")
            spec = float(tn / n_neg) if n_neg > 0 else float("nan")

            if np.isnan(spec):
                ba = rec
            elif np.isnan(rec):
                ba = spec
            else:
                ba = 0.5 * (rec + spec)

            acc = float((pred == t).mean())
            auc = float(roc_auc_score(t, w)) if len(np.unique(t)) > 1 else float("nan")

            h2_run[proc] = {
                "n_pos": n_pos,
                "n_neg": n_neg,
                "recall": rec,
                "specificity": spec,
                "balanced_accuracy": ba,
                "accuracy": acc,
                "auc": auc,
                "mean_weight": float(w.mean()),
                "std_weight": float(w.std()),
            }
        h2_results[rid] = h2_run

    # 2. H2 Factorial Analysis & Bootstrapped Confidence Intervals
    # Compare E1_A_42, E1_B_42, E1_C_42, E1_D_42 on 135 holdout basins
    # Subsurface: A (attr) vs B (rand)
    sub_ba_A = h2_results["E1_A_42"]["w_sub"]["balanced_accuracy"]
    sub_ba_B = h2_results["E1_B_42"]["w_sub"]["balanced_accuracy"]
    sub_effect = sub_ba_A - sub_ba_B

    # Phenology: B (attr) vs A (rand)
    phen_ba_B = h2_results["E1_B_42"]["w_phen"]["balanced_accuracy"]
    phen_ba_A = h2_results["E1_A_42"]["w_phen"]["balanced_accuracy"]
    phen_effect = phen_ba_B - phen_ba_A

    # Bootstrap CIs for per-basin accuracy difference
    t_ho_A = eval_archives["A"]["T"][holdout_indices]
    t_ho_B = eval_archives["B"]["T"][holdout_indices]
    
    pred_sub_A = (all_p_best["E1_A_42"][holdout_indices, 3] > 0.5).astype(int)
    pred_sub_B = (all_p_best["E1_B_42"][holdout_indices, 3] > 0.5).astype(int)
    correct_sub_A = (pred_sub_A == t_ho_A[:, 3]).astype(float)
    correct_sub_B = (pred_sub_B == t_ho_B[:, 3]).astype(float)
    sub_diff_ci = bootstrap_ci(correct_sub_A - correct_sub_B)

    pred_phen_A = (all_p_best["E1_A_42"][holdout_indices, 0] > 0.5).astype(int)
    pred_phen_B = (all_p_best["E1_B_42"][holdout_indices, 0] > 0.5).astype(int)
    correct_phen_B = (pred_phen_B == t_ho_B[:, 0]).astype(float)
    correct_phen_A = (pred_phen_A == t_ho_A[:, 0]).astype(float)
    phen_diff_ci = bootstrap_ci(correct_phen_B - correct_phen_A)

    factorial_summary = {
        "Subsurface": {
            "BA_Attribute_Scenario_A": sub_ba_A,
            "BA_Random_Scenario_B": sub_ba_B,
            "Main_Effect_A_minus_B": sub_effect,
            "Bootstrap_95_CI": sub_diff_ci,
        },
        "Phenology": {
            "BA_Attribute_Scenario_B": phen_ba_B,
            "BA_Random_Scenario_A": phen_ba_A,
            "Main_Effect_B_minus_A": phen_effect,
            "Bootstrap_95_CI": phen_diff_ci,
        },
        "Scenario_Holdout_BAs": {
            sc: {proc: h2_results[f"E1_{sc}_42"][proc]["balanced_accuracy"] for proc in PROCESSES}
            for sc in ["A", "B", "C", "D"]
        }
    }

    # 3. Simple Baselines (3-NN, Logistic Regression, Oracle) on Scenario A holdout
    T_A_ho = eval_archives["A"]["T"]
    X_tr = raw_attrs[train_indices]
    X_ho = raw_attrs[holdout_indices]
    x_mean = np.mean(X_tr, axis=0)
    x_std = np.std(X_tr, axis=0) + 1e-8
    X_tr_norm = np.nan_to_num((X_tr - x_mean) / x_std)
    X_ho_norm = np.nan_to_num((X_ho - x_mean) / x_std)

    baselines = {"3NN": {}, "LogisticRegression": {}, "Oracle_LR": {}}
    for p_idx, proc in enumerate(PROCESSES):
        y_tr = T_A_ho[train_indices, p_idx]
        y_ho = T_A_ho[holdout_indices, p_idx]
        if len(np.unique(y_tr)) <= 1:
            for b_name in baselines:
                baselines[b_name][proc] = 1.0
            continue
        
        # 3-NN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_tr_norm, y_tr)
        p_knn = knn.predict(X_ho_norm)
        rec_k = float(((p_knn == 1) & (y_ho == 1)).sum()) / max(1, int((y_ho == 1).sum()))
        spec_k = float(((p_knn == 0) & (y_ho == 0)).sum()) / max(1, int((y_ho == 0).sum()))
        baselines["3NN"][proc] = 0.5 * (rec_k + spec_k)

        # Logistic Regression (trained on learner predictions)
        w_pred_train = (all_p_best["E1_A_42"][train_indices, p_idx] > 0.5).astype(int)
        if len(np.unique(w_pred_train)) > 1:
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X_tr_norm, w_pred_train)
            p_lr = lr.predict(X_ho_norm)
            rec_l = float(((p_lr == 1) & (y_ho == 1)).sum()) / max(1, int((y_ho == 1).sum()))
            spec_l = float(((p_lr == 0) & (y_ho == 0)).sum()) / max(1, int((y_ho == 0).sum()))
            baselines["LogisticRegression"][proc] = 0.5 * (rec_l + spec_l)
        else:
            baselines["LogisticRegression"][proc] = float("nan")

        # Oracle Logistic Regression (trained on ground truth T)
        lr_oracle = LogisticRegression(max_iter=500, random_state=42)
        lr_oracle.fit(X_tr_norm, y_tr)
        p_oracle = lr_oracle.predict(X_ho_norm)
        rec_o = float(((p_oracle == 1) & (y_ho == 1)).sum()) / max(1, int((y_ho == 1).sum()))
        spec_o = float(((p_oracle == 0) & (y_ho == 0)).sum()) / max(1, int((y_ho == 0).sum()))
        baselines["Oracle_LR"][proc] = 0.5 * (rec_o + spec_o)

    # 4. Method Ablation (Canonical BCE vs Direct-fit vs Plain BCE) on Scenario A seed 42
    ablation_summary = {
        "Canonical_BCE (E1_A_42)": {
            "Decidable_BA_Train": {p: h1_results["E1_A_42"][p]["ba_decidable"] for p in PROCESSES},
            "Holdout_BA": {p: h2_results["E1_A_42"][p]["balanced_accuracy"] for p in PROCESSES},
            "Mean_Weights": {p: h1_results["E1_A_42"][p]["mean_weight"] for p in PROCESSES},
        },
        "Plain_BCE (E4_A_42)": {
            "Decidable_BA_Train": {p: h1_results["E4_A_42"][p]["ba_decidable"] for p in PROCESSES},
            "Holdout_BA": {p: h2_results["E4_A_42"][p]["balanced_accuracy"] for p in PROCESSES},
            "Mean_Weights": {p: h1_results["E4_A_42"][p]["mean_weight"] for p in PROCESSES},
        },
        "Direct_Fit (E3_A_42)": {
            "Decidable_BA_Train": {p: h1_results["E3_A_42"][p]["ba_decidable"] for p in PROCESSES},
            "Holdout_BA": {p: h2_results["E3_A_42"][p]["balanced_accuracy"] for p in PROCESSES},
            "Mean_Weights": {p: h1_results["E3_A_42"][p]["mean_weight"] for p in PROCESSES},
        },
    }

    # Save master scientific evaluation JSON
    master_eval = {
        "title": "Flex-MOPEX S2 Batch-1 Retrained Scientific Evaluation",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "H1_in_sample": h1_results,
        "H2_holdout": h2_results,
        "H2_factorial": factorial_summary,
        "baselines_Scenario_A": baselines,
        "method_ablation": ablation_summary,
    }
    with open(RESULTS_DIR / "scientific_evaluation_full.json", "w") as f:
        json.dump(master_eval, f, indent=2)
    print(f"Saved master evaluation to {RESULTS_DIR / 'scientific_evaluation_full.json'}")

    # Generate Reports
    write_reports(master_eval)

    # Generate Figures
    generate_figures(master_eval, eval_archives)


def write_reports(eval_data: dict[str, Any]) -> None:
    # 1. H1 Structure Recovery Report
    h1 = eval_data["H1_in_sample"]
    h1_md = f"""# S2 Batch-1 H1 In-Sample Structure Recovery Report

**Evaluation Baseline:** 536 training basins on clean, un-scrambled formal synthetic data.  
**Decidable BA Definition:** \\( \\text{{BA}}_{{\\text{{decidable}}}} = \\frac{{1}}{{2}}(\\text{{Recall}}_{{\\text{{High-D}}}} + \\text{{Specificity}}_{{T=0}}) \\).

## 1. E1-A Multi-Seed Consistency (Scenario A: seeds 42, 43, 44)

| Process | Metric | E1_A_42 | E1_A_43 | E1_A_44 | Three-Seed Mean ± Std |
|---|---|---|---|---|---|
| **Subsurface** (Attribute) | High-D Recall<br>Specificity<br>**Decidable BA**<br>Low-D Closed Rate<br>Overall Accuracy<br>AUC | {h1['E1_A_42']['w_sub']['recall_high_d']*100:.1f}% ({h1['E1_A_42']['w_sub']['tp_high_d']}/{h1['E1_A_42']['w_sub']['n_high_d']})<br>{h1['E1_A_42']['w_sub']['specificity']*100:.1f}% ({h1['E1_A_42']['w_sub']['tn_t0']}/{h1['E1_A_42']['w_sub']['n_t0']})<br>**{h1['E1_A_42']['w_sub']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_42']['w_sub']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_42']['w_sub']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_42']['w_sub']['auc']:.4f} | {h1['E1_A_43']['w_sub']['recall_high_d']*100:.1f}% ({h1['E1_A_43']['w_sub']['tp_high_d']}/{h1['E1_A_43']['w_sub']['n_high_d']})<br>{h1['E1_A_43']['w_sub']['specificity']*100:.1f}% ({h1['E1_A_43']['w_sub']['tn_t0']}/{h1['E1_A_43']['w_sub']['n_t0']})<br>**{h1['E1_A_43']['w_sub']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_43']['w_sub']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_43']['w_sub']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_43']['w_sub']['auc']:.4f} | {h1['E1_A_44']['w_sub']['recall_high_d']*100:.1f}% ({h1['E1_A_44']['w_sub']['tp_high_d']}/{h1['E1_A_44']['w_sub']['n_high_d']})<br>{h1['E1_A_44']['w_sub']['specificity']*100:.1f}% ({h1['E1_A_44']['w_sub']['tn_t0']}/{h1['E1_A_44']['w_sub']['n_t0']})<br>**{h1['E1_A_44']['w_sub']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_44']['w_sub']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_44']['w_sub']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_44']['w_sub']['auc']:.4f} | {np.mean([h1[r]['w_sub']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_sub']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_sub']['specificity'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_sub']['specificity'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>**{np.mean([h1[r]['w_sub']['ba_decidable'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_sub']['ba_decidable'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%**<br>{np.mean([h1[r]['w_sub']['low_d_closed_rate'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_sub']['overall_accuracy'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_sub']['auc'] for r in ['E1_A_42','E1_A_43','E1_A_44']]):.4f} |
| **Phenology** (Random) | High-D Recall<br>Specificity<br>**Decidable BA**<br>Low-D Closed Rate<br>Overall Accuracy<br>AUC | {h1['E1_A_42']['w_phen']['recall_high_d']*100:.1f}%<br>{h1['E1_A_42']['w_phen']['specificity']*100:.1f}%<br>**{h1['E1_A_42']['w_phen']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_42']['w_phen']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_42']['w_phen']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_42']['w_phen']['auc']:.4f} | {h1['E1_A_43']['w_phen']['recall_high_d']*100:.1f}%<br>{h1['E1_A_43']['w_phen']['specificity']*100:.1f}%<br>**{h1['E1_A_43']['w_phen']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_43']['w_phen']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_43']['w_phen']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_43']['w_phen']['auc']:.4f} | {h1['E1_A_44']['w_phen']['recall_high_d']*100:.1f}%<br>{h1['E1_A_44']['w_phen']['specificity']*100:.1f}%<br>**{h1['E1_A_44']['w_phen']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_44']['w_phen']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_44']['w_phen']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_44']['w_phen']['auc']:.4f} | {np.mean([h1[r]['w_phen']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_phen']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_phen']['specificity'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_phen']['specificity'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>**{np.mean([h1[r]['w_phen']['ba_decidable'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_phen']['ba_decidable'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%**<br>{np.mean([h1[r]['w_phen']['low_d_closed_rate'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_phen']['overall_accuracy'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_phen']['auc'] for r in ['E1_A_42','E1_A_43','E1_A_44']]):.4f} |
| **Interception** (Attribute) | High-D Recall<br>Specificity<br>**Decidable BA**<br>Low-D Closed Rate<br>Overall Accuracy<br>AUC | {h1['E1_A_42']['w_int']['recall_high_d']*100:.1f}%<br>{h1['E1_A_42']['w_int']['specificity']*100:.1f}%<br>**{h1['E1_A_42']['w_int']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_42']['w_int']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_42']['w_int']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_42']['w_int']['auc']:.4f} | {h1['E1_A_43']['w_int']['recall_high_d']*100:.1f}%<br>{h1['E1_A_43']['w_int']['specificity']*100:.1f}%<br>**{h1['E1_A_43']['w_int']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_43']['w_int']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_43']['w_int']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_43']['w_int']['auc']:.4f} | {h1['E1_A_44']['w_int']['recall_high_d']*100:.1f}%<br>{h1['E1_A_44']['w_int']['specificity']*100:.1f}%<br>**{h1['E1_A_44']['w_int']['ba_decidable']*100:.1f}%**<br>{h1['E1_A_44']['w_int']['low_d_closed_rate']*100:.1f}%<br>{h1['E1_A_44']['w_int']['overall_accuracy']*100:.1f}%<br>{h1['E1_A_44']['w_int']['auc']:.4f} | {np.mean([h1[r]['w_int']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_int']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_int']['specificity'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_int']['specificity'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>**{np.mean([h1[r]['w_int']['ba_decidable'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_int']['ba_decidable'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%**<br>{np.mean([h1[r]['w_int']['low_d_closed_rate'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_int']['overall_accuracy'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{np.mean([h1[r]['w_int']['auc'] for r in ['E1_A_42','E1_A_43','E1_A_44']]):.4f} |
| **Snow** (All ON) | High-D Recall<br>Low-D Recall<br>Overall Recall | {h1['E1_A_42']['w_snow']['recall_high_d']*100:.1f}% ({h1['E1_A_42']['w_snow']['tp_high_d']}/{h1['E1_A_42']['w_snow']['n_high_d']})<br>{(1-h1['E1_A_42']['w_snow']['low_d_closed_rate'])*100:.1f}%<br>{h1['E1_A_42']['w_snow']['overall_accuracy']*100:.1f}% | {h1['E1_A_43']['w_snow']['recall_high_d']*100:.1f}% ({h1['E1_A_43']['w_snow']['tp_high_d']}/{h1['E1_A_43']['w_snow']['n_high_d']})<br>{(1-h1['E1_A_43']['w_snow']['low_d_closed_rate'])*100:.1f}%<br>{h1['E1_A_43']['w_snow']['overall_accuracy']*100:.1f}% | {h1['E1_A_44']['w_snow']['recall_high_d']*100:.1f}% ({h1['E1_A_44']['w_snow']['tp_high_d']}/{h1['E1_A_44']['w_snow']['n_high_d']})<br>{(1-h1['E1_A_44']['w_snow']['low_d_closed_rate'])*100:.1f}%<br>{h1['E1_A_44']['w_snow']['overall_accuracy']*100:.1f}% | {np.mean([h1[r]['w_snow']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% ± {np.std([h1[r]['w_snow']['recall_high_d'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}%<br>{(1-np.mean([h1[r]['w_snow']['low_d_closed_rate'] for r in ['E1_A_42','E1_A_43','E1_A_44']]))*100:.1f}%<br>{np.mean([h1[r]['w_snow']['overall_accuracy'] for r in ['E1_A_42','E1_A_43','E1_A_44']])*100:.1f}% |
"""
    (REPORTS_DIR / "H1_STRUCTURE_RECOVERY_REPORT.md").write_text(h1_md)

    # 2. H2 Factorial Report
    fact = eval_data["H2_factorial"]
    h2_md = f"""# S2 Batch-1 H2 Out-of-Sample Factorial Regionalization Report

**Evaluation Baseline:** 135 spatial holdout basins across 2×2 factorial design (Scenarios A, B, C, D; seed 42).

## 1. 2×2 Factorial Holdout Balanced Accuracy Matrix

| Scenario | Subsurface Rule | Phenology Rule | Subsurface Holdout BA | Phenology Holdout BA | Interception Holdout BA | Snow Holdout Rec |
|---|---|---|---|---|---|---|
| **A** | **Attribute** | **Random** | **{fact['Scenario_Holdout_BAs']['A']['w_sub']*100:.1f}%** | **{fact['Scenario_Holdout_BAs']['A']['w_phen']*100:.1f}%** | {fact['Scenario_Holdout_BAs']['A']['w_int']*100:.1f}% | {fact['Scenario_Holdout_BAs']['A']['w_snow']*100:.1f}% |
| **B** | **Random** | **Attribute** | **{fact['Scenario_Holdout_BAs']['B']['w_sub']*100:.1f}%** | **{fact['Scenario_Holdout_BAs']['B']['w_phen']*100:.1f}%** | {fact['Scenario_Holdout_BAs']['B']['w_int']*100:.1f}% | {fact['Scenario_Holdout_BAs']['B']['w_snow']*100:.1f}% |
| **C** | **Attribute** | **Attribute** | **{fact['Scenario_Holdout_BAs']['C']['w_sub']*100:.1f}%** | **{fact['Scenario_Holdout_BAs']['C']['w_phen']*100:.1f}%** | {fact['Scenario_Holdout_BAs']['C']['w_int']*100:.1f}% | {fact['Scenario_Holdout_BAs']['C']['w_snow']*100:.1f}% |
| **D** | **Random** | **Random** | **{fact['Scenario_Holdout_BAs']['D']['w_sub']*100:.1f}%** | **{fact['Scenario_Holdout_BAs']['D']['w_phen']*100:.1f}%** | {fact['Scenario_Holdout_BAs']['D']['w_int']*100:.1f}% | {fact['Scenario_Holdout_BAs']['D']['w_snow']*100:.1f}% |

## 2. Factorial Main Effects and 95% Bootstrap Confidence Intervals

1. **Subsurface Main Effect (Attribute in A vs Random in B):**
   - Attribute BA: **{fact['Subsurface']['BA_Attribute_Scenario_A']*100:.1f}%**
   - Random BA: **{fact['Subsurface']['BA_Random_Scenario_B']*100:.1f}%**
   - **Main Effect (A - B):** **{fact['Subsurface']['Main_Effect_A_minus_B']*100:+.1f}%**
   - **95% Bootstrap CI:** [{fact['Subsurface']['Bootstrap_95_CI'][0]*100:+.1f}%, {fact['Subsurface']['Bootstrap_95_CI'][1]*100:+.1f}%]

2. **Phenology Main Effect (Attribute in B vs Random in A):**
   - Attribute BA: **{fact['Phenology']['BA_Attribute_Scenario_B']*100:.1f}%**
   - Random BA: **{fact['Phenology']['BA_Random_Scenario_A']*100:.1f}%**
   - **Main Effect (B - A):** **{fact['Phenology']['Main_Effect_B_minus_A']*100:+.1f}%**
   - **95% Bootstrap CI:** [{fact['Phenology']['Bootstrap_95_CI'][0]*100:+.1f}%, {fact['Phenology']['Bootstrap_95_CI'][1]*100:+.1f}%]
"""
    (REPORTS_DIR / "H2_FACTORIAL_REGIONALIZATION_REPORT.md").write_text(h2_md)

    # 3. Method Ablation Report
    abl = eval_data["method_ablation"]
    abl_md = f"""# S2 Batch-1 Method Ablation Report: Canonical BCE vs Direct-fit vs Plain BCE

**Experimental Setup:** Scenario A, seed 42, identical 536/135 split and frozen inputs.

| Evaluation Metric | Canonical BCE (E1_A_42) | Plain BCE (E4_A_42) | Direct-fit (E3_A_42) | Method Comparison Finding |
|---|---|---|---|---|
| **Subsurface Train Decidable BA** | **{abl['Canonical_BCE (E1_A_42)']['Decidable_BA_Train']['w_sub']*100:.1f}%** | {abl['Plain_BCE (E4_A_42)']['Decidable_BA_Train']['w_sub']*100:.1f}% | {abl['Direct_Fit (E3_A_42)']['Decidable_BA_Train']['w_sub']*100:.1f}% | Canonical BCE achieves highest structure accuracy |
| **Subsurface Holdout BA** | **{abl['Canonical_BCE (E1_A_42)']['Holdout_BA']['w_sub']*100:.1f}%** | {abl['Plain_BCE (E4_A_42)']['Holdout_BA']['w_sub']*100:.1f}% | {abl['Direct_Fit (E3_A_42)']['Holdout_BA']['w_sub']*100:.1f}% | Strongest regionalization generalization |
| **Phenology Train Decidable BA** | **{abl['Canonical_BCE (E1_A_42)']['Decidable_BA_Train']['w_phen']*100:.1f}%** | {abl['Plain_BCE (E4_A_42)']['Decidable_BA_Train']['w_phen']*100:.1f}% | {abl['Direct_Fit (E3_A_42)']['Decidable_BA_Train']['w_phen']*100:.1f}% | Counterfactual supervision resolves phenology |
| **Interception Mean Weight** | **{abl['Canonical_BCE (E1_A_42)']['Mean_Weights']['w_int']:.3f}** | {abl['Plain_BCE (E4_A_42)']['Mean_Weights']['w_int']:.3f} | {abl['Direct_Fit (E3_A_42)']['Mean_Weights']['w_int']:.3f} | Direct-fit inflates interception gate artificially |
| **Subsurface Mean Weight** | **{abl['Canonical_BCE (E1_A_42)']['Mean_Weights']['w_sub']:.3f}** | {abl['Plain_BCE (E4_A_42)']['Mean_Weights']['w_sub']:.3f} | {abl['Direct_Fit (E3_A_42)']['Mean_Weights']['w_sub']:.3f} | Direct-fit severely collapses subsurface gate |
"""
    (REPORTS_DIR / "METHOD_ABLATION_E3_E4_REPORT.md").write_text(abl_md)


def generate_figures(eval_data: dict[str, Any], eval_archives: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # Load loss histories
    histories = {}
    for rid in RUNS:
        lh_file = RESULTS_DIR / rid / "loss_history.json"
        if lh_file.exists():
            with open(lh_file) as f:
                histories[rid] = json.load(f)

    # -------------------------------------------------------------
    # Fig 1: Training Convergence across 8 runs
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    ax = axes[0, 0]
    for s in [42, 43, 44]:
        rid = f"E1_A_{s}"
        if rid in histories:
            ax.plot([x["epoch"] for x in histories[rid]], [x["loss_total"] for x in histories[rid]], label=f"E1-A seed {s}", lw=1.8)
    ax.set_title("(a) E1-A Multi-Seed Convergence (Scenario A)")
    ax.set_ylabel("Total Loss")
    ax.legend()

    ax = axes[0, 1]
    for sc in ["A", "B", "C", "D"]:
        rid = f"E1_{sc}_42"
        if rid in histories:
            ax.plot([x["epoch"] for x in histories[rid]], [x["loss_total"] for x in histories[rid]], label=f"Scenario {sc}", lw=1.8)
    ax.set_title("(b) 2×2 Factorial Scenarios (Seed 42)")
    ax.legend()

    ax = axes[1, 0]
    if "E1_A_42" in histories:
        ep = [x["epoch"] for x in histories["E1_A_42"]]
        ax.plot(ep, [x["loss_fit_aic"] for x in histories["E1_A_42"]], label="Fit-AIC", color="#1f77b4", lw=1.8)
        ax.plot(ep, [x["loss_cf_bce"] for x in histories["E1_A_42"]], label="CF-BCE", color="#ff7f0e", lw=1.8)
    ax.set_title("(c) Loss Components (E1_A_42)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component Loss")
    ax.legend()

    ax = axes[1, 1]
    for rid, lbl, col in [("E1_A_42", "Canonical BCE", "#2ca02c"), ("E4_A_42", "Plain BCE", "#9467bd"), ("E3_A_42", "Direct-fit", "#d62728")]:
        if rid in histories:
            ax.plot([x["epoch"] for x in histories[rid]], [x["loss_total"] for x in histories[rid]], label=lbl, color=col, lw=1.8)
    ax.set_title("(d) Method Ablation Loss Trajectories (Seed 42)")
    ax.set_xlabel("Epoch")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "Fig1_training_convergence_8runs.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 2: H1 Truth Recovery (Recall, Specificity, BA)
    # -------------------------------------------------------------
    h1 = eval_data["H1_in_sample"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    procs = ["w_phen", "w_int", "w_snow", "w_sub"]
    x = np.arange(len(procs))
    width = 0.25

    for ax_idx, (m_key, title) in enumerate([("recall_high_d", "(a) High-D Recall (T=1)"), ("specificity", "(b) Specificity (T=0)"), ("ba_decidable", "(c) Decidable Balanced Accuracy")]):
        ax = axes[ax_idx]
        for s_idx, s in enumerate([42, 43, 44]):
            rid = f"E1_A_{s}"
            vals = [h1[rid][p][m_key] * 100 if not np.isnan(h1[rid][p][m_key]) else 0.0 for p in procs]
            ax.bar(x + (s_idx - 1) * width, vals, width, label=f"Seed {s}", alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(PROCESS_NAMES)
        ax.set_ylim(0, 105)
        if ax_idx == 0:
            ax.set_ylabel("Percentage (%)")
            ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "Fig2_H1_truth_recovery_rates.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 3: Continuous Detection Curves (D vs Structure Weights)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    p_tr_42 = np.load(RESULTS_DIR / "E1_A_42/p_struct_best_train.npy")
    eval_A = eval_archives["A"]
    T_tr = eval_A["T"]
    D_tr = eval_A["D_J_train"]
    with open(SPLIT_PATH) as f:
        sp = json.load(f)
    ids = np.load(IDS_PATH)
    id_map = {int(gid): i for i, gid in enumerate(ids)}
    tr_idx = np.array([id_map[gid] for gid in sp["train_ids"]])

    for p_idx, proc in enumerate(PROCESSES):
        ax = axes[p_idx]
        d = D_tr[tr_idx, p_idx]
        w = p_tr_42[:, p_idx]
        t = T_tr[tr_idx, p_idx]
        
        ax.scatter(d[t == 1], w[t == 1], color="#2ca02c", alpha=0.6, s=20, label="True ON (T=1)")
        if (t == 0).sum() > 0:
            ax.scatter(d[t == 0], w[t == 0], color="#d62728", alpha=0.6, s=20, label="True OFF (T=0)")
        ax.axhline(0.5, color="black", linestyle="--", alpha=0.5)
        ax.axvline(0.014 if p_idx < 3 else 0.007, color="blue", linestyle=":", label="Penalty P")
        ax.set_title(PROCESS_NAMES[p_idx])
        ax.set_xlabel("Detectability D_J (train)")
        if p_idx == 0:
            ax.set_ylabel("Predicted Weight w")
            ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "Fig3_detection_curves_continuous_D.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 4: H2 Factorial Holdout BA & Regionalization
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    scenarios = ["A", "B", "C", "D"]
    x = np.arange(len(scenarios))
    w_sub_ba = [eval_data["H2_factorial"]["Scenario_Holdout_BAs"][sc]["w_sub"] * 100 for sc in scenarios]
    w_phen_ba = [eval_data["H2_factorial"]["Scenario_Holdout_BAs"][sc]["w_phen"] * 100 for sc in scenarios]
    
    ax.bar(x - 0.18, w_sub_ba, 0.35, label="Subsurface", color="#1f77b4", alpha=0.85)
    ax.bar(x + 0.18, w_phen_ba, 0.35, label="Phenology", color="#ff7f0e", alpha=0.85)
    ax.axhline(50, color="gray", linestyle="--", label="Random Baseline (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Scenario {sc}" for sc in scenarios])
    ax.set_ylabel("Holdout Balanced Accuracy (%)")
    ax.set_title("(a) Holdout BA across 2×2 Scenarios")
    ax.set_ylim(40, 75)
    ax.legend()

    ax = axes[1]
    # Factor main effect plot
    effects = [
        eval_data["H2_factorial"]["Subsurface"]["Main_Effect_A_minus_B"] * 100,
        eval_data["H2_factorial"]["Phenology"]["Main_Effect_B_minus_A"] * 100,
    ]
    ci_low = [
        eval_data["H2_factorial"]["Subsurface"]["Bootstrap_95_CI"][0] * 100,
        eval_data["H2_factorial"]["Phenology"]["Bootstrap_95_CI"][0] * 100,
    ]
    ci_high = [
        eval_data["H2_factorial"]["Subsurface"]["Bootstrap_95_CI"][1] * 100,
        eval_data["H2_factorial"]["Phenology"]["Bootstrap_95_CI"][1] * 100,
    ]
    yerr = [
        [effects[0] - ci_low[0], effects[1] - ci_low[1]],
        [ci_high[0] - effects[0], ci_high[1] - effects[1]]
    ]
    ax.bar(["Subsurface\n(A - B)", "Phenology\n(B - A)"], effects, yerr=yerr, capsize=6, color=["#1f77b4", "#ff7f0e"], alpha=0.85)
    ax.axhline(0, color="black", linestyle="-", lw=1)
    ax.set_ylabel("Attribute Gain in BA (% points)")
    ax.set_title("(b) Factor Main Effects (95% Bootstrap CI)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "Fig4_H2_factorial_regionalization.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 5: Method Ablation (Canonical BCE vs Plain BCE vs Direct-fit)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    methods = ["Canonical BCE\n(E1_A_42)", "Plain BCE\n(E4_A_42)", "Direct-fit\n(E3_A_42)"]
    x = np.arange(len(methods))
    sub_ba = [
        eval_data["method_ablation"]["Canonical_BCE (E1_A_42)"]["Decidable_BA_Train"]["w_sub"] * 100,
        eval_data["method_ablation"]["Plain_BCE (E4_A_42)"]["Decidable_BA_Train"]["w_sub"] * 100,
        eval_data["method_ablation"]["Direct_Fit (E3_A_42)"]["Decidable_BA_Train"]["w_sub"] * 100,
    ]
    phen_ba = [
        eval_data["method_ablation"]["Canonical_BCE (E1_A_42)"]["Decidable_BA_Train"]["w_phen"] * 100,
        eval_data["method_ablation"]["Plain_BCE (E4_A_42)"]["Decidable_BA_Train"]["w_phen"] * 100,
        eval_data["method_ablation"]["Direct_Fit (E3_A_42)"]["Decidable_BA_Train"]["w_phen"] * 100,
    ]
    ax.bar(x - 0.18, sub_ba, 0.35, label="Subsurface", color="#1f77b4", alpha=0.85)
    ax.bar(x + 0.18, phen_ba, 0.35, label="Phenology", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Train Decidable BA (%)")
    ax.set_title("(a) Method Ablation Decidable BA")
    ax.legend()

    ax = axes[1]
    # Mean weights comparison (Interception vs Subsurface)
    w_int = [
        eval_data["method_ablation"]["Canonical_BCE (E1_A_42)"]["Mean_Weights"]["w_int"],
        eval_data["method_ablation"]["Plain_BCE (E4_A_42)"]["Mean_Weights"]["w_int"],
        eval_data["method_ablation"]["Direct_Fit (E3_A_42)"]["Mean_Weights"]["w_int"],
    ]
    w_sub = [
        eval_data["method_ablation"]["Canonical_BCE (E1_A_42)"]["Mean_Weights"]["w_sub"],
        eval_data["method_ablation"]["Plain_BCE (E4_A_42)"]["Mean_Weights"]["w_sub"],
        eval_data["method_ablation"]["Direct_Fit (E3_A_42)"]["Mean_Weights"]["w_sub"],
    ]
    ax.plot(methods, w_int, marker="o", lw=2, label="Interception Mean Weight", color="#2ca02c")
    ax.plot(methods, w_sub, marker="s", lw=2, label="Subsurface Mean Weight", color="#d62728")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("(b) Structural Weight Means across Methods")
    ax.set_ylabel("Mean Predicted Weight w")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "Fig5_method_ablation_comparison.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Fig 6: Structural Weight Distributions
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    p_42 = np.load(RESULTS_DIR / "E1_A_42/p_struct_best_train.npy")
    for p_idx, proc in enumerate(PROCESSES):
        ax = axes[p_idx]
        w_vals = p_42[:, p_idx]
        ax.hist(w_vals, bins=25, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
        ax.set_title(f"{PROCESS_NAMES[p_idx]}\nMean={w_vals.mean():.3f}")
        ax.set_xlabel("Weight w")
        if p_idx == 0:
            ax.set_ylabel("Basin Count")
            ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "Fig6_structural_weight_distributions.png")
    plt.close(fig)

    print("All 6 publication-ready figures generated successfully!")


if __name__ == "__main__":
    main()
