#!/usr/bin/env python3
"""Scenario-correct synthetic learner NSE/KGE rescoring (local, no training).

What this does, and why
-----------------------
`revision2/run_nse_v2.py` scored every learner against Scenario A truth because
`r1_019.nse_stage` builds one Scenario-A fold and one Scenario-A truth set outside the run
loop.  This script keeps the *same* frozen scoring path (same loader, same model builder,
same windows, same `_per_basin_nse_kge`) but rebuilds the fold, the inputs and the truth
per scenario, so each learner is scored against the teacher archive it was fit to.

Everything that defines the estimand is imported from the frozen scorer, not re-implemented:
`load_canonical_axis`, `load_scenario_truth`, `build_test_window_sample`,
`validate_constructed_sample`, `_per_basin_nse_kge`, `_recompute_gates`.
The only code repeated here is the ~10-line config override block and the loader/builder
call of `build_strict_context`, parameterised by scenario (that function hard-codes
`scenario_A/dataset.pkl`).

Writes only into a new output directory; never touches formal run directories or old caches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
V2 = REPO
S2DATA = REPO / "inputs/synthetic"
RESULTS = REPO / "inputs/trained"
CKPT_ROOT = Path(os.environ.get("FLEXMOPEX_SYNTHETIC_CKPTS",
                                str(REPO / "inputs/synthetic_checkpoints")))
SCORER = REPO / "evaluate/scorer.py"
sys.path.insert(0, str(REPO))
FLEX_CFG = RESULTS / "normal/flex/ref/seed_42/config_snapshot.json"
SPLIT = REPO / "train/synthetic/split.json"
OLD_NSE = RESULTS / "synthetic_nse"

FAMILIES = ("synthetic", "synthetic_e2", "synthetic_e3", "synthetic_e4",
            "synthetic_e5", "synthetic_sdf", "synthetic_mdf")
# formal family -> scenario that defines the teacher truth (run_learner.LEARNER_SPEC)
FAMILY_FIXED_SCENARIO = {"synthetic_e2": "A_E2", "synthetic_e3": "A",
                         "synthetic_e4": "A", "synthetic_sdf": "A", "synthetic_mdf": "A"}


def _install_legacy_aliases() -> None:
    """Archived learner checkpoints pickle ``StrictTargets`` from the old module path."""
    import importlib
    import types

    legacy = {
        "project.flexmopex.revision1.strict.trainer": "train.synthetic.strict.trainer",
        "project.flexmopex.revision1.strict.fold": "train.synthetic.strict.fold",
    }
    for old, new in legacy.items():
        target = importlib.import_module(new)
        parts = old.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                stub = types.ModuleType(parent)
                stub.__path__ = []
                sys.modules[parent] = stub
        sys.modules[old] = target


_install_legacy_aliases()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scorer():
    spec = importlib.util.spec_from_file_location("r1_019_stat_analysis", SCORER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_module(m, out_dir: Path) -> None:
    """Redirect the frozen scorer's module-level inputs (same redirections as run_nse_v2)."""
    m.CONFIG_PATH = FLEX_CFG
    m.EVALUATOR_DIR = S2DATA
    m.SPLIT_PATH = SPLIT
    m.GAGE_ID_PATH = V2 / "data/gage_id.npy"
    m.DATES_PATH = V2 / "data/camels_dates.npy"
    m.HYDRO_LOADER_SOURCE = Path(__import__("dmg.core.data.loaders.hydro_loader",
                                            fromlist=["x"]).__file__).resolve()
    m.STAT_DIR = out_dir
    # window semantics actually used for the published V2 caches (run_nse_v2.py)
    m.TRAIN_SCORED = slice(365, 5478)      # 5113 days
    m.TEST_WINDOW = slice(5478, 10957)     # 5479 days incl. 365-day warmup
    m.TEST_SCORED = slice(5843, 10957)     # 5114 scored days


def strict_config(m) -> dict:
    """The build_strict_context() config, verbatim except for the dataset path."""
    cfg = json.loads(m.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg.update({"mode": "train", "device": "cpu", "dtype": torch.float32})
    # archived snapshots name the pre-rename class; map it to the released model
    cfg["model"]["phy"]["name"] = ["Mopex"]
    cfg["model"]["phy"]["model"] = ["Mopex"]
    cfg["delta_model"]["phy_model"]["model"] = ["Mopex"]
    cfg["model"]["rho"] = 365
    cfg["model"]["warmup"] = 365
    cfg["model"]["nmul"] = 4
    cfg["delta_model"]["phy_model"]["nmul"] = 4
    cfg["confidence_weighted_cf_loss"] = True
    cfg["structure_optimizer"] = "none"
    cfg["train"]["batch_size"] = 100
    cfg["train"]["optimizer"] = "Adadelta"
    cfg["train"]["learning_rate"] = 1.0
    cfg["train"]["start_epoch"] = 0
    cfg["loss_function"]["aic_alpha"] = 0.007
    return cfg


def build_context(m, scenario: str, cache_root: Path):
    from train.synthetic.strict.s2_loader import load_candidate_strict_fold
    from train.synthetic.strict.trainer import build_strict_model_and_loss

    cfg = strict_config(m)
    dataset_path = S2DATA / f"scenario_{scenario}/dataset.pkl"
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    fold_data = load_candidate_strict_fold(
        cfg,
        manifest_path=m.SPLIT_PATH,
        cache_root=cache_root,
        dataset_path=dataset_path,
        gage_info_path=m.GAGE_ID_PATH,
        hydro_loader_source=m.HYDRO_LOADER_SOURCE,
    )
    model, _ = build_strict_model_and_loss(
        cfg, fold_data, device="cpu", model_dir=cache_root / "_model_build")
    return cfg, fold_data, model, dataset_path


def discover_runs() -> list[dict]:
    runs = []
    for family in FAMILIES:
        for summary in sorted((CKPT_ROOT / family).rglob("summary.json")):
            d = json.loads(summary.read_text())
            run_id = d["run_id"]
            scenario = d["scenario"]
            expected = FAMILY_FIXED_SCENARIO.get(family)
            if expected is not None:
                if scenario != expected:
                    raise RuntimeError(f"{run_id}: summary scenario {scenario} != formal {expected}")
            elif family == "synthetic_e1":
                if scenario not in ("A", "B", "C", "D"):
                    raise RuntimeError(f"{run_id}: unexpected E1 scenario {scenario}")
            elif family == "synthetic_e5":
                if scenario not in ("A_low", "A_high"):
                    raise RuntimeError(f"{run_id}: unexpected E5 scenario {scenario}")
            runs.append(dict(family=family, run_id=run_id, scenario=scenario,
                             seed=int(d["seed"]), method=d["method"],
                             run_dir=summary.parent, summary=d,
                             ckpt=summary.parent / "best_checkpoint.pt"))
    if not runs:
        raise RuntimeError(f"no learner summary.json under {CKPT_ROOT}")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="new output directory")
    ap.add_argument("--only", default=None, help="substring filter on run_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenario", default=None, help="one scenario per process keeps peak RAM low")
    ap.add_argument("--resume", action="store_true", help="reuse completed per-run arrays in --out")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to reuse non-empty {args.out}")
    (args.out / "per_run").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    m = load_scorer()
    configure_module(m, args.out)
    axis = m.load_canonical_axis()
    print(f"axis: {axis.n_train} train / {axis.n_holdout} holdout", flush=True)

    runs = discover_runs()
    if not (args.only or args.scenario or args.limit) and len(runs) != 33:
        raise RuntimeError(f"expected 33 learners, found {len(runs)}")
    if args.only:
        runs = [r for r in runs if args.only in r["run_id"]]
    if args.scenario:
        runs = [r for r in runs if r["scenario"] == args.scenario]
    if args.limit:
        runs = runs[: args.limit]
    print(f"runs selected: {len(runs)}; scenarios: {sorted({r['scenario'] for r in runs})}", flush=True)

    rows, checks = [], []
    for scenario in sorted({r["scenario"] for r in runs}):
        group = [r for r in runs if r["scenario"] == scenario]
        cache_root = Path(f"/tmp/rescore_fold_{scenario}")
        t0 = time.perf_counter()
        cfg, fold_data, model, dataset_path = build_context(m, scenario, cache_root)
        m.SOURCE_DATASET_PATH = dataset_path
        truth = m.load_scenario_truth(scenario, axis)
        validation = m.validate_constructed_sample(cfg, fold_data, axis)
        if validation["attribute_normalization_max_abs_diff"] > 1e-4 or validation["x_phy_max_abs_diff"] > 1e-4:
            raise RuntimeError(f"{scenario}: constructed sample does not reproduce the loader: {validation}")
        train_sample = dict(fold_data.train_dataset)
        test_sample = m.build_test_window_sample(cfg, fold_data, axis)
        # static gates: normalize the 35 attributes once with this fold's statistics, then the
        # per-run gate check is a pure logits evaluation (same ops as the frozen _recompute_gates)
        stats = json.loads(Path(fold_data.stats_path).read_text(encoding="utf-8"))
        attrs_raw = m._load_x35()
        attr_names = list(cfg["observations"]["all_attributes"])
        a_mean = np.array([stats[n][2] for n in attr_names], dtype=np.float32)
        a_std = np.array([stats[n][3] for n in attr_names], dtype=np.float32)
        a_std[a_std == 0] = 1.0
        attrs_norm = torch.from_numpy(np.nan_to_num((attrs_raw - a_mean) / a_std).astype(np.float32))
        del attrs_raw
        obs_train = truth.y_mm_d[axis.train_idx][:, m.TRAIN_SCORED].T
        obs_test = truth.y_mm_d[axis.holdout_idx][:, m.TEST_SCORED].T
        ds_sha = sha256_file(dataset_path)
        truth_sha = sha256_file(truth.path)
        truth_path = truth.path
        del truth  # the archive is ~300 MB; obs arrays above are all we need
        gc.collect()
        inner = next(iter(model.model_dict.values()))
        print(f"[{scenario}] fold ready in {time.perf_counter()-t0:.1f}s | dataset sha {ds_sha[:12]} "
              f"| obs {obs_train.shape} / {obs_test.shape}", flush=True)
        checks.append(dict(scenario=scenario, dataset_path=str(dataset_path),
                           dataset_sha256=ds_sha,
                           truth_archive=str(truth_path), truth_archive_sha256=truth_sha,
                           train_obs_shape=str(obs_train.shape), holdout_obs_shape=str(obs_test.shape),
                           attr_norm_max_abs_diff=validation["attribute_normalization_max_abs_diff"],
                           x_phy_max_abs_diff=validation["x_phy_max_abs_diff"],
                           doy_max_abs_diff=validation["doy_max_abs_diff"]))

        for run in group:
            cached = args.out / "per_run" / f"nse_kge_per_basin_{run['run_id']}.npz"
            if cached.is_file():  # resume: reuse a completed per-run array set from this run
                with np.load(cached) as z:
                    tr_nse, tr_kge = z["train_nse"], z["train_kge"]
                    te_nse, te_kge = z["holdout_nse"], z["holdout_kge"]
                    row = dict(run_id=run["run_id"], family=run["family"], scenario=scenario,
                               seed=run["seed"], method=run["method"],
                               best_epoch=run["summary"]["best_epoch"],
                               dataset_sha256=str(z["dataset_sha256"]),
                               checkpoint_sha256=str(z["checkpoint_sha256"]),
                               train_median_nse=float(np.nanmedian(tr_nse)),
                               train_median_kge=float(np.nanmedian(tr_kge)),
                               holdout_median_nse=float(np.nanmedian(te_nse)),
                               holdout_median_kge=float(np.nanmedian(te_kge)),
                               train_n=int(np.isfinite(tr_nse).sum()),
                               holdout_n=int(np.isfinite(te_nse).sum()),
                               train_score_days=int(z["train_score_days"][0]),
                               holdout_score_days=int(z["holdout_score_days"][0]),
                               gate_max_abs_diff_vs_saved=float(z["gate_max_abs_diff_vs_saved"]),
                               prediction_finite=True, seconds=0.0, resumed_from_cache=True)
                old = OLD_NSE / f"nse_kge_per_basin_{run['run_id']}.npz"
                with np.load(old) as z:
                    row.update(old_train_median_nse=float(np.nanmedian(z["train_nse"])),
                               old_holdout_median_nse=float(np.nanmedian(z["holdout_nse"])),
                               old_train_median_kge=float(np.nanmedian(z["train_kge"])),
                               old_holdout_median_kge=float(np.nanmedian(z["holdout_kge"])),
                               max_abs_per_basin_nse_change_holdout=float(
                                   np.nanmax(np.abs(te_nse - z["holdout_nse"]))))
                rows.append(row)
                print(f"  [cached] {run['run_id']}: holdout median NSE {row['holdout_median_nse']:+.4f}", flush=True)
                continue
            t1 = time.perf_counter()
            digest = sha256_file(run["ckpt"])
            if digest != run["summary"]["checkpoints"]["best"]["sha256"]:
                raise RuntimeError(f"{run['run_id']}: checkpoint hash != summary")
            checkpoint = torch.load(run["ckpt"], map_location="cpu", weights_only=False)
            inner.load_state_dict(checkpoint["model_state_dict"])
            inner.eval()
            with torch.no_grad():
                sim_train = model(train_sample, eval=True)[next(iter(model.model_dict))]["streamflow"].numpy()[:, :, 0]
                sim_test = model(test_sample, eval=True)[next(iter(model.model_dict))]["streamflow"].numpy()[:, :, 0]
            with torch.no_grad():
                lg = inner.nn_model.get_structure_logits(attrs_norm).view(-1, 4, 2).clamp(-10.0, 10.0)
                gates = torch.sigmoid(lg[..., 1] - lg[..., 0]).numpy()
            saved_gates = OLD_NSE / f"gates_all671_{run['run_id']}.npy"
            gate_diff = float(np.max(np.abs(np.load(saved_gates) - gates))) if saved_gates.is_file() else float("nan")
            train_nse, train_kge = m._per_basin_nse_kge(sim_train, obs_train, m.load_retrain_helpers())
            test_nse, test_kge = m._per_basin_nse_kge(sim_test, obs_test, m.load_retrain_helpers())
            np.savez(args.out / "per_run" / f"nse_kge_per_basin_{run['run_id']}.npz",
                     train_basin_ids=axis.ids[axis.train_idx],
                     holdout_basin_ids=axis.ids[axis.holdout_idx],
                     train_nse=train_nse, train_kge=train_kge,
                     holdout_nse=test_nse, holdout_kge=test_kge,
                     train_score_days=np.full(axis.n_train, sim_train.shape[0]),
                     holdout_score_days=np.full(axis.n_holdout, sim_test.shape[0]),
                     checkpoint_sha256=digest, scenario=scenario,
                     dataset_sha256=ds_sha,
                     truth_archive_sha256=truth_sha,
                     gate_max_abs_diff_vs_saved=gate_diff)
            row = dict(run_id=run["run_id"], family=run["family"], scenario=scenario,
                       seed=run["seed"], method=run["method"], best_epoch=run["summary"]["best_epoch"],
                       dataset_sha256=ds_sha, checkpoint_sha256=digest,
                       train_median_nse=float(np.nanmedian(train_nse)),
                       train_median_kge=float(np.nanmedian(train_kge)),
                       holdout_median_nse=float(np.nanmedian(test_nse)),
                       holdout_median_kge=float(np.nanmedian(test_kge)),
                       train_n=int(np.isfinite(train_nse).sum()),
                       holdout_n=int(np.isfinite(test_nse).sum()),
                       train_score_days=int(sim_train.shape[0]),
                       holdout_score_days=int(sim_test.shape[0]),
                       gate_max_abs_diff_vs_saved=gate_diff,
                       prediction_finite=bool(np.isfinite(sim_train).all() and np.isfinite(sim_test).all()),
                       seconds=round(time.perf_counter() - t1, 2), resumed_from_cache=False)
            old = OLD_NSE / f"nse_kge_per_basin_{run['run_id']}.npz"
            if old.is_file():
                with np.load(old) as z:
                    row["old_train_median_nse"] = float(np.nanmedian(z["train_nse"]))
                    row["old_holdout_median_nse"] = float(np.nanmedian(z["holdout_nse"]))
                    row["old_train_median_kge"] = float(np.nanmedian(z["train_kge"]))
                    row["old_holdout_median_kge"] = float(np.nanmedian(z["holdout_kge"]))
                    row["max_abs_per_basin_nse_change_holdout"] = float(np.nanmax(np.abs(test_nse - z["holdout_nse"])))
            rows.append(row)
            print(f"  {run['run_id']}: holdout median NSE {row['holdout_median_nse']:+.4f} "
                  f"(old {row.get('old_holdout_median_nse', float('nan')):+.4f}) gate|d| {gate_diff:.2e}", flush=True)
        del cfg, fold_data, model, obs_train, obs_test, train_sample, test_sample, inner, attrs_norm
        gc.collect()

        import csv
        with (args.out / "rescore_by_run.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with (args.out / "scenario_data_identity.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
            w.writeheader(); w.writerows(checks)

    print(f"WROTE {args.out} | {len(rows)} runs", flush=True)


if __name__ == "__main__":
    main()
