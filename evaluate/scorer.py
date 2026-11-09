#!/usr/bin/env python3
"""R1-019 independent statistical re-analysis of the frozen S2 v1.1 batch-1 experiment.

Every reported number is recomputed from raw per-basin arrays:

* ``p_struct_*_train.npy`` / ``p_struct_best_all671.npy``  (learned gate probabilities)
* ``data/scenario_<S>/evaluator_archive.npz``              (T, D_J_train, D_J_test, P_matrix, Y_mm_d)
* ``data/camels_dataset`` + ``data/gage_id.npy``            (X35 attributes, canonical basin axis)
* ``best_checkpoint.pt`` (forward simulation for NSE/KGE; gate re-derivation cross-check)

Nothing is copied from earlier report prose.  Metric definitions reuse
``S2/evaluate_batch1_results.py`` (H1/H2 definitions) and
``revision1/scripts/evaluate_s2_batch1_retrain.py`` (``compute_nse_kge``,
``bootstrap_ci``, checkpoint->gate prediction path).

Stages (rerunnable, run directories auto-discovered in batch1_results/ and batch2_results/):

    python r1_019_stat_analysis.py --stage all      # metrics + NSE cache + report (default)
    python r1_019_stat_analysis.py --stage metrics  # gate/truth metrics only (seconds)
    python r1_019_stat_analysis.py --stage nse      # forward-simulation NSE/KGE cache (minutes)
    python r1_019_stat_analysis.py --stage report   # regenerate report from cached artifacts
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # evaluate/<file> -> package root
FLEX = ROOT / "train"
REV = FLEX / "revision1"
S2_DIR = ROOT / "inputs" / "synthetic"
STAT_DIR = ROOT / "outputs" / "scoring"
ANALYSIS_DIR = REV / "analysis"

RUN_BATCH_DIRS = (S2_DIR / "batch1_results", S2_DIR / "batch2_results")
EVALUATOR_DIR = S2_DIR / "data"
SPLIT_PATH = ANALYSIS_DIR / "R1_014_CANDIDATE_SPLIT.json"
TRUTH_SPEC_PATH = S2_DIR / "truth_specification.json"
S2_MANIFEST_PATH = S2_DIR / "S2_DATA_MANIFEST.json"
GAGE_ID_PATH = ROOT / "data/gage_id.npy"
DATES_PATH = ROOT / "data/camels_dates.npy"
SOURCE_DATASET_PATH = ROOT / "data/camels_dataset"
CONFIG_PATH = ROOT / "inputs/trained/normal/flex/ref/seed_42/config_snapshot.json"
HYDRO_LOADER_SOURCE = __import__("dmg.core.data.loaders.hydro_loader", fromlist=["x"]).__file__
NSE_CACHE_ROOT = Path("/tmp/r1_019_stat_fold_cache")  # in-repo caches are refused by fold.py

SUMMARY_PATH = ANALYSIS_DIR / "R1_019_STAT_SUMMARY.json"
REPORT_PATH = ANALYSIS_DIR / "R1_019_STAT_FINAL_REPORT.md"

# --------------------------------------------------------------------------------------
# Frozen constants
# --------------------------------------------------------------------------------------
# revision1/strict/trainer.py: PROCESS_ORDER = ("w_phen","w_int","w_snow","w_sub")
# project/flexmopex/models/base_mopex.py: WEIGHT_NAMES = ["w_phen","w_int","w_snow","w_sub"]
PROCESSES = ("w_phen", "w_int", "w_snow", "w_sub")
PROCESS_LABELS = {"w_phen": "Phenology", "w_int": "Interception", "w_snow": "Snow", "w_sub": "Subsurface"}
SCENARIOS = ("A", "B", "C", "D")
# revision1/strict/trainer.py: PROCESS_COSTS = {w_phen: 2.0, w_int: 2.0, w_snow: 2.0, w_sub: 1.0}
PROCESS_COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
METHOD_BY_EXPERIMENT = {
    "E1": "Canonical_BCE",
    "E2": "Old_Interception",
    "E3": "Direct_Fit",
    "E4": "Plain_BCE",
    "E5": "Noise_Sensitivity",
}
RUN_ID_RE = re.compile(r"^E(\d+)_([ABCD])_(\d+)$")
GATE_THRESHOLD = 0.5
LAMBDA = 0.007
UNIFORM_P = 0.007  # protocol document (S2_PROTOCOL_v1.1_FROZEN.md 4.2/text) simplified value
N_BOOT = 10000
BOOT_SEED = 20261019
TRAIN_SCORED = slice(365, 5478)  # 1981-10-01..1995-09-30 scored training period
TEST_WINDOW = slice(5478, 10956)  # 1995-10-01..2010-09-30
TEST_SCORED = slice(5843, 10956)  # after 365-day test warmup
MIN_VALID_DAYS = 30

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator: float, denominator: float) -> float:
    if denominator is None or denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def pct_ci(samples: np.ndarray) -> tuple[float, float]:
    """Percentile 95% CI (2.5/97.5), same convention as evaluate_s2_batch1_retrain.bootstrap_ci."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in names})


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100:.{digits}f}%"


def fmt_rate_ci(point: float, low: float, high: float, digits: int = 1) -> str:
    return f"{fmt_pct(point, digits)} [{fmt_pct(low, digits)}, {fmt_pct(high, digits)}]"


def fmt_signed_rate_ci(point: float, low: float, high: float, digits: int = 1) -> str:
    return f"{point * 100:+.{digits}f} [{low * 100:+.{digits}f}, {high * 100:+.{digits}f}]"


def mark_seed_pending(seeds: Sequence[int], expected: int = 3) -> str:
    return "SEED-PENDING" if len(seeds) < expected else "complete"


# --------------------------------------------------------------------------------------
# Run discovery (auto-discovers batch-2 if present)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunInfo:
    run_id: str
    experiment: str
    scenario: str
    seed: int
    method: str
    batch_dir: Path
    run_dir: Path
    summary: Mapping[str, Any]

    @property
    def batch_name(self) -> str:
        return self.batch_dir.name

    @property
    def best_checkpoint(self) -> Path:
        return self.run_dir / "best_checkpoint.pt"

    @property
    def gates_train_best(self) -> Path:
        return self.run_dir / "p_struct_best_train.npy"

    @property
    def gates_train_final(self) -> Path:
        return self.run_dir / "p_struct_final_train.npy"

    @property
    def gates_all671(self) -> Path:
        return self.run_dir / "p_struct_best_all671.npy"

    @property
    def cached_gates_all671(self) -> Path:
        return STAT_DIR / f"gates_all671_{self.run_id}.npy"


def discover_runs() -> list[RunInfo]:
    runs: list[RunInfo] = []
    for batch_dir in RUN_BATCH_DIRS:
        if not batch_dir.is_dir():
            continue
        for child in sorted(batch_dir.iterdir()):
            if not child.is_dir():
                continue
            match = RUN_ID_RE.match(child.name)
            if match is None:
                continue
            summary_path = child / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            experiment = f"E{match.group(1)}"
            runs.append(
                RunInfo(
                    run_id=child.name,
                    experiment=experiment,
                    scenario=match.group(2),
                    seed=int(match.group(3)),
                    method=summary.get("method", METHOD_BY_EXPERIMENT.get(experiment, experiment)),
                    batch_dir=batch_dir,
                    run_dir=child,
                    summary=summary,
                )
            )
    return runs


def has_all671_gates(run: RunInfo) -> bool:
    return run.gates_all671.is_file() or run.cached_gates_all671.is_file()


def load_all671_gates(run: RunInfo) -> tuple[np.ndarray, str]:
    if run.gates_all671.is_file():
        return np.load(run.gates_all671), str(run.gates_all671)
    return np.load(run.cached_gates_all671), str(run.cached_gates_all671)


# --------------------------------------------------------------------------------------
# Frozen inputs
# --------------------------------------------------------------------------------------
@dataclass
class CanonicalAxis:
    ids: np.ndarray
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    split: Mapping[str, Any]

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_holdout(self) -> int:
        return int(self.holdout_idx.size)


@dataclass
class ScenarioTruth:
    scenario: str
    path: Path
    T: np.ndarray
    d_train: np.ndarray
    d_test: np.ndarray
    p_matrix: np.ndarray
    y_mm_d: np.ndarray
    basin_ids: np.ndarray

    def active_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for index, process in enumerate(PROCESSES):
            t = self.T[:, index]
            d = self.d_train[:, index]
            p = self.p_matrix[:, index]
            active = t == 1
            out[process] = {
                "train_on": int(active[self.axis_train].sum()),
                "holdout_on": int(active[self.axis_holdout].sum()),
                "total_on": int(active.sum()),
                "high_d": int(((d > p) & active).sum()),
                "low_d": int(((d >= 0) & (d <= p) & active).sum()),
                "neg_d": int(((d < 0) & active).sum()),
            }
        return out

    axis_train: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    axis_holdout: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def load_canonical_axis() -> CanonicalAxis:
    ids = np.load(GAGE_ID_PATH).astype(np.int64)
    if not np.all(np.diff(ids) > 0):
        raise RuntimeError("gage_id.npy is not ascending; the frozen fold assumes a sorted canonical axis")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    index_of = {int(gid): position for position, gid in enumerate(ids)}
    train_idx = np.array([index_of[int(g)] for g in split["train_ids"]], dtype=int)
    holdout_idx = np.array([index_of[int(g)] for g in split["holdout_ids"]], dtype=int)
    if train_idx.size != 536 or holdout_idx.size != 135:
        raise RuntimeError(f"unexpected frozen split sizes: {train_idx.size}/{holdout_idx.size}")
    if not np.all(np.diff(train_idx) > 0) or not np.all(np.diff(holdout_idx) > 0):
        raise RuntimeError("frozen split IDs are not ascending; per-basin arrays would misalign")
    return CanonicalAxis(ids=ids, train_idx=train_idx, holdout_idx=holdout_idx, split=split)


def load_scenario_truth(scenario: str, axis: CanonicalAxis) -> ScenarioTruth:
    path = EVALUATOR_DIR / f"scenario_{scenario}/evaluator_archive.npz"
    with np.load(path) as archive:
        payload = {name: archive[name] for name in archive.files}
    truth = ScenarioTruth(
        scenario=scenario,
        path=path,
        T=np.asarray(payload["T"], dtype=np.float64),
        d_train=np.asarray(payload["D_J_train"], dtype=np.float64),
        d_test=np.asarray(payload["D_J_test"], dtype=np.float64),
        p_matrix=np.asarray(payload["P_matrix"], dtype=np.float64),
        y_mm_d=np.asarray(payload["Y_mm_d"], dtype=np.float64),
        basin_ids=np.asarray(payload["basin_ids"]).astype(np.int64),
    )
    if not np.array_equal(truth.basin_ids, axis.ids):
        raise RuntimeError(f"scenario {scenario} evaluator archive basin axis differs from gage_id.npy")
    truth.axis_train = axis.train_idx
    truth.axis_holdout = axis.holdout_idx
    return truth


def verify_process_column_order(axis: CanonicalAxis, truths: Mapping[str, ScenarioTruth]) -> dict[str, Any]:
    """Validate the 4-column process order of p_struct arrays against frozen truth counts.

    Two independent checks:
    1. the trainer's PROCESS_ORDER / PROCESS_COSTS constants (revision1/strict/trainer.py);
    2. the frozen truth_specification.json per-scenario active counts per process column.
    """
    evidence: dict[str, Any] = {}
    try:
        sys.path.insert(0, str(ROOT))
        from train.synthetic.strict.trainer import (  # noqa: WPS433 (lazy import)
            PROCESS_COSTS as TRAINER_COSTS,
            PROCESS_ORDER as TRAINER_ORDER,
        )

        if tuple(TRAINER_ORDER) != PROCESSES:
            raise RuntimeError(f"trainer PROCESS_ORDER {TRAINER_ORDER} != p_struct column order {PROCESSES}")
        if {k: float(v) for k, v in TRAINER_COSTS.items()} != PROCESS_COSTS:
            raise RuntimeError("trainer PROCESS_COSTS differ from the frozen cost map used here")
        evidence["trainer_process_order"] = list(TRAINER_ORDER)
        evidence["trainer_process_costs"] = {k: float(v) for k, v in TRAINER_COSTS.items()}
    except ImportError as exc:  # pragma: no cover - only when torch/dmg are unavailable
        evidence["trainer_process_order"] = f"UNVERIFIED (import failed: {exc})"

    spec = json.loads(TRUTH_SPEC_PATH.read_text(encoding="utf-8"))
    checks = []
    for scenario, truth in truths.items():
        expected = spec["counts_summary"][scenario]
        recomputed = truth.active_counts()
        for process in PROCESSES:
            checks.append(
                {
                    "scenario": scenario,
                    "process": process,
                    "truth_spec_train_on": int(expected[process]["train_on"]),
                    "recomputed_train_on": recomputed[process]["train_on"],
                    "truth_spec_holdout_on": int(expected[process]["holdout_on"]),
                    "recomputed_holdout_on": recomputed[process]["holdout_on"],
                    "match": bool(
                        int(expected[process]["train_on"]) == recomputed[process]["train_on"]
                        and int(expected[process]["holdout_on"]) == recomputed[process]["holdout_on"]
                    ),
                }
            )
    if not all(item["match"] for item in checks):
        bad = [item for item in checks if not item["match"]]
        raise RuntimeError(f"process column mapping disagrees with frozen truth specification: {bad[:3]}")
    evidence["truth_spec_column_checks"] = checks
    evidence["truth_spec_all_match"] = True
    return evidence


# --------------------------------------------------------------------------------------
# Core metric definitions (reuse the definitions of S2/evaluate_batch1_results.py)
# --------------------------------------------------------------------------------------
def stratified_metrics(
    weights: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    p_threshold: np.ndarray,
    uniform_threshold: bool = False,
) -> dict[str, Any]:
    """H1/H2 unified metrics for one process on one basin subset.

    ``uniform_threshold=True`` recomputes High-D/Low-D with a flat P=0.007 for every
    process (sensitivity variant of the protocol-text simplification).
    """
    threshold = np.full_like(d, UNIFORM_P) if uniform_threshold else p_threshold
    pred = weights > GATE_THRESHOLD
    active = t == 1
    inactive = t == 0
    high = active & (d > threshold)
    low = active & (d >= 0) & (d <= threshold)
    neg = active & (d < 0)

    recall_high = ratio(int(pred[high].sum()), int(high.sum()))
    specificity = ratio(int((~pred[inactive]).sum()), int(inactive.sum()))
    if math.isnan(specificity):
        ba_decidable = recall_high
    elif math.isnan(recall_high):
        ba_decidable = specificity
    else:
        ba_decidable = 0.5 * (recall_high + specificity)
    classes = np.unique(t)
    metrics = {
        "n_t1": int(active.sum()),
        "n_t0": int(inactive.sum()),
        "n_high_d": int(high.sum()),
        "tp_high_d": int(pred[high].sum()),
        "recall_high_d": recall_high,
        "n_t0_correct": int((~pred[inactive]).sum()),
        "specificity": specificity,
        "ba_decidable": ba_decidable,
        "n_low_d": int(low.sum()),
        "shutdown_rate_low_d": ratio(int((~pred[low]).sum()), int(low.sum())),
        "open_rate_low_d": ratio(int(pred[low].sum()), int(low.sum())),
        "mean_weight_low_d": float(weights[low].mean()) if low.any() else float("nan"),
        "n_neg_d": int(neg.sum()),
        "open_rate_neg_d": ratio(int(pred[neg].sum()), int(neg.sum())),
        "mean_weight_neg_d": float(weights[neg].mean()) if neg.any() else float("nan"),
        "n_t1_correct": int(pred[active].sum()),
        "full_truth_recovery_t1": ratio(int(pred[active].sum()), int(active.sum())),
        "overall_accuracy": float((pred == t).mean()),
        "auc": float(roc_auc_score(t, weights)) if classes.size > 1 else float("nan"),
        "mean_weight": float(weights.mean()),
        "std_weight": float(weights.std()),
        "open_rate_all": float(pred.mean()),
    }
    quantiles = np.percentile(weights, [5, 25, 50, 75, 95])
    metrics.update({f"w_q{name}": float(value) for name, value in zip(("05", "25", "50", "75", "95"), quantiles)})
    metrics["frac_weight_gt_half"] = float(pred.mean())
    metrics["frac_weight_0.45_0.55"] = float(((weights >= 0.45) & (weights <= 0.55)).mean())
    return metrics


def detection_relationship(
    weights: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    n_quantiles: int = 4,
) -> dict[str, Any]:
    """Continuous D_J -> predicted-weight relationship.

    Sign convention (S2_D_METRIC_VALIDATION.md 2): Delta J_open = (2T-1) D_J - P, so for
    T=1 an opening benefit is D_J - P, while for T=0 D_J is the loss of opening the
    (physically absent) process and is NOT an opening benefit.
    """
    active = t == 1
    inactive = t == 0
    out: dict[str, Any] = {
        "n_t1": int(active.sum()),
        "n_t0": int(inactive.sum()),
        "pearson_d_w_t1": pearson(d[active], weights[active]),
        "spearman_d_w_t1": spearman(d[active], weights[active]),
        "mean_d_t1": float(d[active].mean()) if active.any() else float("nan"),
        "median_d_t1": float(np.median(d[active])) if active.any() else float("nan"),
        "mean_weight_t1": float(weights[active].mean()) if active.any() else float("nan"),
    }
    if inactive.any():
        out.update(
            {
                "pearson_d_w_t0": pearson(d[inactive], weights[inactive]),
                "spearman_d_w_t0": spearman(d[inactive], weights[inactive]),
                "mean_d_t0": float(d[inactive].mean()),
                "mean_weight_t0": float(weights[inactive].mean()),
            }
        )
    if int(active.sum()) >= n_quantiles * 4:
        edges = np.quantile(d[active], np.linspace(0.0, 1.0, n_quantiles + 1))
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        means = []
        opens = []
        counts = []
        for position in range(n_quantiles):
            mask = active & (d > edges[position]) & (d <= edges[position + 1])
            means.append(float(weights[mask].mean()) if mask.any() else float("nan"))
            opens.append(ratio(int((weights[mask] > GATE_THRESHOLD).sum()), int(mask.sum())))
            counts.append(int(mask.sum()))
        out["d_quantile_edges"] = [float(e) for e in edges]
        out["mean_weight_by_d_quantile"] = means
        out["open_rate_by_d_quantile"] = opens
        out["n_by_d_quantile"] = counts
    return out


# --------------------------------------------------------------------------------------
# Bootstrap machinery (paired basin resampling)
# --------------------------------------------------------------------------------------
def paired_index_matrix(n_basins: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_basins, size=(n_boot, n_basins))


def ba_bootstrap(t_bool: np.ndarray, pred_bool: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Balanced accuracy on every resample (same NaN fallback convention as the frozen H1/H2 code)."""
    sampled_t = t_bool[idx]
    sampled_p = pred_bool[idx]
    n_pos = sampled_t.sum(axis=1).astype(float)
    n_neg = sampled_t.shape[1] - n_pos
    tp = np.logical_and(sampled_t, sampled_p).sum(axis=1).astype(float)
    tn = np.logical_and(~sampled_t, ~sampled_p).sum(axis=1).astype(float)
    recall = np.where(n_pos > 0, tp / np.maximum(n_pos, 1.0), np.nan)
    specificity = np.where(n_neg > 0, tn / np.maximum(n_neg, 1.0), np.nan)
    return np.where(np.isnan(specificity), recall, np.where(np.isnan(recall), specificity, 0.5 * (recall + specificity)))


def mean_bootstrap(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return values[idx].mean(axis=1)


def paired_mean_difference(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return mean_bootstrap(a, idx) - mean_bootstrap(b, idx)


def effect_of(ba_by_scenario: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """2x2 factorial effects of a per-scenario BA vector (scenarios A,B,C,D)."""
    a, b, c, d = (ba_by_scenario[name] for name in SCENARIOS)
    return {
        "me_attribute": 0.5 * (a + c) - 0.5 * (b + d),
        "me_phenology": 0.5 * (b + c) - 0.5 * (a + d),
        "interaction": c - a - b + d,
    }


# --------------------------------------------------------------------------------------
# Stage 1: gate/truth metrics
# --------------------------------------------------------------------------------------
def h1_block(
    weights_train: np.ndarray,
    truth: ScenarioTruth,
    stratum: str,
) -> dict[str, Any]:
    """Per-process H1-style metrics on the 536 training basins (primary + uniform-P sensitivity).

    ``weights_train`` is the run's ``p_struct_*_train.npy`` array, whose rows follow the frozen
    ascending train-ID axis (verified in ``load_canonical_axis``); no re-indexing is applied.
    """
    n_train = weights_train.shape[0]
    if n_train != truth.axis_train.size:
        raise RuntimeError(f"train gate rows {n_train} != frozen train basins {truth.axis_train.size}")
    block: dict[str, Any] = {}
    for index, process in enumerate(PROCESSES):
        weights = weights_train[:, index]
        t = truth.T[truth.axis_train, index]
        d = truth.d_train[truth.axis_train, index]
        p_thr = truth.p_matrix[truth.axis_train, index]
        primary = stratified_metrics(weights, t, d, p_thr)
        uniform = stratified_metrics(weights, t, d, p_thr, uniform_threshold=True)
        uniform_subset = {
            key: uniform[key]
            for key in (
                "n_high_d",
                "recall_high_d",
                "n_low_d",
                "shutdown_rate_low_d",
                "n_neg_d",
                "specificity",
                "ba_decidable",
                "full_truth_recovery_t1",
            )
        }
        primary.update(
            {
                "process": process,
                "process_label": PROCESS_LABELS[process],
                "p_threshold_used": float(p_thr[0]),
                "uniform_P_sensitivity": uniform_subset,
                "detection": detection_relationship(weights, t, d),
                "stratum": stratum,
                "n_train_basins": n_train,
            }
        )
        block[process] = primary
    return block


def h2_block(
    weights: np.ndarray,
    truth: ScenarioTruth,
    axis: CanonicalAxis,
    stratum: str,
) -> dict[str, Any]:
    """Per-process H2 metrics on the 135 holdout basins.

    Preregistered H2 primary metric (S2_PROTOCOL_v1.1_FROZEN.md 5.2): *unfiltered* balanced
    accuracy over all 135 holdout basins, i.e. recall over every T=1 holdout basin and
    specificity over every T=0 holdout basin, with NO D_J stratification ->
    ``ba_holdout_unfiltered``.

    The D_J^test-stratified decidable BA (High-D recall vs T=0 specificity) is kept as a
    secondary descriptive metric under ``ba_decidable_test_D_strata``; it must not be labelled
    "unfiltered" and is not the preregistered H2 quantity.
    """
    block: dict[str, Any] = {}
    for index, process in enumerate(PROCESSES):
        weights_hold = weights[axis.holdout_idx, index]
        t = truth.T[axis.holdout_idx, index]
        d_test = truth.d_test[axis.holdout_idx, index]
        p_thr = truth.p_matrix[axis.holdout_idx, index]
        primary = stratified_metrics(weights_hold, t, d_test, p_thr)
        uniform = stratified_metrics(weights_hold, t, d_test, p_thr, uniform_threshold=True)
        # unfiltered (preregistered) holdout BA: no D_J stratification
        recall_unfiltered = ratio(int((weights_hold[t == 1] > GATE_THRESHOLD).sum()), int((t == 1).sum()))
        specificity_unfiltered = primary["specificity"]
        if math.isnan(specificity_unfiltered):
            ba_holdout_unfiltered = recall_unfiltered
        elif math.isnan(recall_unfiltered):
            ba_holdout_unfiltered = specificity_unfiltered
        else:
            ba_holdout_unfiltered = 0.5 * (recall_unfiltered + specificity_unfiltered)
        primary.update(
            {
                "process": process,
                "process_label": PROCESS_LABELS[process],
                "n_pos": int((t == 1).sum()),
                "n_neg": int((t == 0).sum()),
                "recall_unfiltered_t1": recall_unfiltered,
                "specificity_unfiltered_t0": specificity_unfiltered,
                "ba_holdout_unfiltered": ba_holdout_unfiltered,
                "ba_holdout_unfiltered_definition": (
                    "0.5*(recall over all T=1 holdout basins + specificity over all T=0 holdout basins); "
                    "recall returned alone when no T=0 basin exists (Snow)"
                ),
                "ba_decidable_test_D_strata": primary["ba_decidable"],
                "ba_decidable_test_D_strata_definition": (
                    "secondary/descriptive only: 0.5*(High-D recall [T=1 & D_J^test > P] + specificity [T=0])"
                ),
                "p_threshold_used": float(p_thr[0]),
                "uniform_P_sensitivity": {
                    key: uniform[key]
                    for key in ("n_high_d", "recall_high_d", "n_low_d", "shutdown_rate_low_d", "n_neg_d")
                },
                "detection": detection_relationship(weights_hold, t, d_test),
                "stratum": stratum,
            }
        )
        block[process] = primary
    return block


def build_metrics_stage() -> dict[str, Any]:
    started = time.time()
    axis = load_canonical_axis()
    truths = {scenario: load_scenario_truth(scenario, axis) for scenario in SCENARIOS}
    process_order_evidence = verify_process_column_order(axis, truths)
    runs = discover_runs()

    inventory = []
    for run in runs:
        inventory.append(
            {
                "run_id": run.run_id,
                "experiment": run.experiment,
                "method": run.method,
                "scenario": run.scenario,
                "seed": run.seed,
                "batch": run.batch_name,
                "run_dir": str(run.run_dir),
                "has_best_checkpoint": run.best_checkpoint.is_file(),
                "has_all671_gates": has_all671_gates(run),
                "best_epoch": run.summary.get("best_epoch"),
                "final_epoch": run.summary.get("final_epoch"),
                "best_total_loss": run.summary.get("best_total_loss"),
                "stopped_by_early_stopping": run.summary.get("stopped_by_early_stopping"),
            }
        )

    h1_records: dict[str, Any] = {}
    h2_gate_arrays: dict[str, dict[str, np.ndarray]] = {}
    skipped: list[str] = []

    for run in runs:
        if not run.gates_train_best.is_file():
            skipped.append(f"{run.run_id}: missing p_struct_best_train.npy")
            continue
        truth = truths[run.scenario]
        gates_train = np.load(run.gates_train_best)
        if gates_train.shape != (axis.n_train, len(PROCESSES)):
            raise RuntimeError(f"{run.run_id}: unexpected p_struct_best_train shape {gates_train.shape}")
        record: dict[str, Any] = {
            "run_id": run.run_id,
            "method": run.method,
            "scenario": run.scenario,
            "seed": run.seed,
            "batch": run.batch_name,
            "best_epoch": run.summary.get("best_epoch"),
            "n_train_basins": axis.n_train,
        }
        record["H1_best"] = h1_block(gates_train, truth, stratum="train")
        if run.gates_train_final.is_file():
            gates_final = np.load(run.gates_train_final)
            record["H1_final"] = h1_block(gates_final, truth, stratum="train_final")
        h1_records[run.run_id] = record

        if not has_all671_gates(run):
            record["H2"] = None
            record["H2_note"] = (
                "p_struct_best_all671.npy absent and no stat/gates_all671_<RUN>.npy cache; "
                "run --stage nse to materialise holdout gates from best_checkpoint.pt"
            )
            continue
        gates_all, gate_source = load_all671_gates(run)
        if gates_all.shape != (axis.ids.size, len(PROCESSES)):
            raise RuntimeError(f"{run.run_id}: unexpected all-671 gate shape {gates_all.shape}")
        h2_gate_arrays[run.run_id] = {"holdout_pred": gates_all[axis.holdout_idx]}
        record["H2"] = h2_block(gates_all, truth, axis, stratum="holdout")
        record["H2_gate_source"] = gate_source

    # ------------------------------------------------------------------
    # B1: 2x2 factorial main effects with paired bootstrap
    # ------------------------------------------------------------------
    factorial = factorial_analysis(runs, truths, axis, h2_gate_arrays)

    # ------------------------------------------------------------------
    # B3: regionalization baselines
    # ------------------------------------------------------------------
    baselines = baseline_analysis(runs, truths, axis)

    # ------------------------------------------------------------------
    # B4/B5: ablation + random-phenology assembly
    # ------------------------------------------------------------------
    ablation = ablation_analysis(runs, h1_records, truths, axis)
    random_phenology = random_phenology_analysis(runs, h1_records)

    artifacts = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "metrics",
        "runtime_seconds": round(time.time() - started, 2),
        "canonical_axis": {
            "source": str(GAGE_ID_PATH),
            "split_source": str(SPLIT_PATH),
            "n_source": int(axis.ids.size),
            "n_train": axis.n_train,
            "n_holdout": axis.n_holdout,
            "train_ids_sha256": axis.split.get("train_ids_sha256"),
            "holdout_ids_sha256": axis.split.get("holdout_ids_sha256"),
        },
        "process_order_verification": process_order_evidence,
        "process_costs": PROCESS_COSTS,
        "run_inventory": inventory,
        "skipped_runs": skipped,
        "H1_best": h1_records,
        "factorial": factorial,
        "baselines": baselines,
        "method_ablation": ablation,
        "random_phenology": random_phenology,
    }
    write_metrics_outputs(artifacts)
    return artifacts


# --------------------------------------------------------------------------------------
# B1 factorial analysis
# --------------------------------------------------------------------------------------
def _scenario_pred_arrays(
    runs: Sequence[RunInfo],
    truths: Mapping[str, ScenarioTruth],
    axis: CanonicalAxis,
    gate_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """(experiment, seed) -> {scenario: holdout gate slice [135, 4]} for canonical BCE runs."""
    out: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for run in runs:
        if run.run_id not in gate_arrays:
            continue
        out.setdefault((run.experiment, run.seed), {})[run.scenario] = gate_arrays[run.run_id]["holdout_pred"]
    return out


def factorial_analysis(
    runs: Sequence[RunInfo],
    truths: Mapping[str, ScenarioTruth],
    axis: CanonicalAxis,
    gate_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = paired_index_matrix(axis.n_holdout, N_BOOT, rng)
    families = _scenario_pred_arrays(runs, truths, axis, gate_arrays)

    seed_results: dict[str, Any] = {}
    boot_effects: dict[str, dict[str, list[np.ndarray]]] = {}
    complete_seeds: list[int] = []

    for (experiment, seed), scenario_gates in sorted(families.items()):
        if set(scenario_gates) != set(SCENARIOS):
            continue
        if experiment != "E1":
            continue  # the 2x2 factorial is defined on the E1 family
        complete_seeds.append(seed)
        per_process: dict[str, Any] = {}
        seed_boot: dict[str, dict[str, np.ndarray]] = {}
        for index, process in enumerate(PROCESSES):
            ba_point: dict[str, float] = {}
            ba_boot: dict[str, np.ndarray] = {}
            for scenario in SCENARIOS:
                weights = scenario_gates[scenario][:, index]
                t = truths[scenario].T[axis.holdout_idx, index]
                t_bool = t == 1
                pred_bool = weights > GATE_THRESHOLD
                ba_point[scenario] = _ba_point(t_bool, pred_bool)
                ba_boot[scenario] = ba_bootstrap(t_bool, pred_bool, idx)
            effects_point = {
                name: float(value)
                for name, value in effect_of({s: np.array(ba_point[s]) for s in SCENARIOS}).items()
            }
            effects_boot = effect_of(ba_boot)
            seed_boot[process] = effects_boot
            blocks: dict[str, Any] = {}
            for name in ("me_attribute", "me_phenology", "interaction"):
                low, high = pct_ci(effects_boot[name])
                blocks[name] = {
                    "point": effects_point[name],
                    "ci_low": low,
                    "ci_high": high,
                    "n_holdout_basins": axis.n_holdout,
                    "n_bootstrap": N_BOOT,
                }
            per_process[process] = {
                "holdout_ba": ba_point,
                "n_pos": {scenario: int((truths[scenario].T[axis.holdout_idx, index] == 1).sum()) for scenario in SCENARIOS},
                "n_neg": {scenario: int((truths[scenario].T[axis.holdout_idx, index] == 0).sum()) for scenario in SCENARIOS},
                "effects": blocks,
            }
        per_process["seed"] = seed
        seed_results[str(seed)] = per_process
        boot_effects[str(seed)] = seed_boot

    pooled: dict[str, Any] = {}
    if complete_seeds:
        for index, process in enumerate(PROCESSES):
            pooled_process: dict[str, Any] = {
                "n_seeds": len(complete_seeds),
                "seeds": complete_seeds,
                "seed_status": mark_seed_pending(complete_seeds),
            }
            for name in ("me_attribute", "me_phenology", "interaction"):
                points = np.array([seed_results[str(s)][process]["effects"][name]["point"] for s in complete_seeds])
                pooled_samples = np.mean(
                    np.stack([boot_effects[str(s)][process][name] for s in complete_seeds], axis=0), axis=0
                )
                low, high = pct_ci(pooled_samples)
                pooled_process[name] = {
                    "point": float(points.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "seed_spread_sd": float(points.std(ddof=1)) if points.size > 1 else 0.0,
                    "seed_min": float(points.min()),
                    "seed_max": float(points.max()),
                    "n_seeds": len(complete_seeds),
                    "n_bootstrap": N_BOOT,
                }
            pooled[process] = pooled_process

    missing_scenarios = {
        str(seed): sorted(set(SCENARIOS) - set(scenario_gates))
        for (experiment, seed), scenario_gates in families.items()
        if experiment == "E1" and set(scenario_gates) != set(SCENARIOS)
    }
    return {
        "definition": {
            "me_attribute": "(BA_A + BA_C)/2 - (BA_B + BA_D)/2",
            "me_phenology": "(BA_B + BA_C)/2 - (BA_A + BA_D)/2",
            "interaction": "BA_C - BA_A - BA_B + BA_D",
            "bootstrap": f"paired basin bootstrap, {N_BOOT} resamples, one basin index set shared by all four scenarios",
            "basin_unit": f"{axis.n_holdout} holdout basins (never {axis.n_holdout}x4 basin-process pairs)",
        },
        "per_seed": seed_results,
        "pooled": pooled,
        "complete_seeds": complete_seeds,
        "partial_seeds": missing_scenarios,
        "audit_of_earlier_single_contrasts": audit_single_contrasts(seed_results),
    }


def _ba_point(t_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    n_pos = int(t_bool.sum())
    n_neg = int((~t_bool).sum())
    tp = int(np.logical_and(t_bool, pred_bool).sum())
    tn = int(np.logical_and(~t_bool, ~pred_bool).sum())
    recall = ratio(tp, n_pos)
    specificity = ratio(tn, n_neg)
    if math.isnan(specificity):
        return recall
    if math.isnan(recall):
        return specificity
    return 0.5 * (recall + specificity)


def audit_single_contrasts(seed_results: Mapping[str, Any]) -> dict[str, Any]:
    """Audit target: the earlier reports printed single pairwise contrasts as 'main effects'."""
    audit: dict[str, Any] = {}
    for seed, per_process in seed_results.items():
        audit[seed] = {}
        for process in PROCESSES:
            ba = per_process[process]["holdout_ba"]
            audit[seed][process] = {
                "single_contrast_A_minus_B": ba["A"] - ba["B"],
                "single_contrast_B_minus_A": ba["B"] - ba["A"],
                "preregistered_me_attribute": per_process[process]["effects"]["me_attribute"]["point"],
                "preregistered_me_phenology": per_process[process]["effects"]["me_phenology"]["point"],
            }
    return audit


# --------------------------------------------------------------------------------------
# B3 regionalization baselines
# --------------------------------------------------------------------------------------
def _load_x35() -> np.ndarray:
    import pickle

    with SOURCE_DATASET_PATH.open("rb") as handle:
        _, _, attributes = pickle.load(handle)
    attributes = np.asarray(attributes, dtype=np.float64)
    if attributes.shape[0] != 671 or attributes.shape[1] != 35:
        raise RuntimeError(f"unexpected X35 attribute shape {attributes.shape}")
    return attributes


def verify_attribute_axis(attributes: np.ndarray, axis: CanonicalAxis) -> dict[str, Any]:
    spec = json.loads(TRUTH_SPEC_PATH.read_text(encoding="utf-8"))["attributes"]
    train = axis.train_idx
    checks = {
        "slope_mean_p40_train": (float(np.percentile(attributes[train, spec["slope_mean"]["index"]], 40)), spec["slope_mean"]["threshold"]),
        "frac_forest_median_train": (float(np.median(attributes[train, spec["frac_forest"]["index"]])), spec["frac_forest"]["threshold"]),
        "gvf_diff_median_train": (float(np.median(attributes[train, spec["gvf_diff"]["index"]])), spec["gvf_diff"]["threshold"]),
        "aridity_median_train": (float(np.median(attributes[train, spec["aridity"]["index"]])), spec["aridity"]["median_train"]),
        "frac_snow_median_train": (float(np.median(attributes[train, spec["frac_snow"]["index"]])), spec["frac_snow"]["median_train"]),
    }
    out = {}
    for name, (observed, expected) in checks.items():
        out[name] = {"observed": observed, "truth_spec": expected, "match": bool(abs(observed - expected) < 1e-6)}
    return out


def baseline_analysis(
    runs: Sequence[RunInfo],
    truths: Mapping[str, ScenarioTruth],
    axis: CanonicalAxis,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = paired_index_matrix(axis.n_holdout, N_BOOT, rng)
    attributes = _load_x35()
    attribute_check = verify_attribute_axis(attributes, axis)
    x_train = attributes[axis.train_idx]
    x_hold = attributes[axis.holdout_idx]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    # 27 of 671 X35 rows carry NaN (root_depth_50: 24, geol_porosity: 3); they are filled with the
    # train mean after standardization, exactly as in evaluate_s2_batch1_retrain.py (np.nan_to_num).
    n_missing = int(np.isnan(attributes).any(axis=1).sum())
    x_train_norm = np.nan_to_num((x_train - mean) / std)
    x_hold_norm = np.nan_to_num((x_hold - mean) / std)

    run_by_scenario: dict[tuple[str, str, int], RunInfo] = {
        (run.experiment, run.scenario, run.seed): run for run in runs
    }

    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        truth = truths[scenario]
        for index, process in enumerate(PROCESSES):
            t_train = truth.T[axis.train_idx, index]
            t_hold = truth.T[axis.holdout_idx, index]
            t_train_bool = t_train == 1
            t_hold_bool = t_hold == 1
            n_pos = int(t_hold_bool.sum())
            n_neg = int((~t_hold_bool).sum())
            row_base = {
                "scenario": scenario,
                "process": process,
                "process_label": PROCESS_LABELS[process],
                "n_holdout": axis.n_holdout,
                "n_holdout_pos": n_pos,
                "n_holdout_neg": n_neg,
            }
            if len(np.unique(t_train)) < 2:
                records.append(
                    {
                        **row_base,
                        "approach": "all_approaches",
                        "ba": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "note": "single training class (Snow all ON); BA undefined on holdout, recall reported instead",
                        "recall_holdout": float(t_hold_bool.mean()),
                        "train_label_agreement_with_T": float("nan"),
                    }
                )
                continue

            # (a) 3-NN donor in X35 space using neighbour truth T
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(x_train_norm, t_train_bool.astype(int))
            pred_knn = knn.predict(x_hold_norm) == 1

            # (b) logistic regression supervised by the learner's own counterfactual Delta-J labels
            cf_run = run_by_scenario.get(("E1", scenario, 42))
            cf_labels = None
            cf_label_source = "unavailable"
            if cf_run is not None and (cf_run.run_dir / "q_best_train.npy").is_file():
                q = np.load(cf_run.run_dir / "q_best_train.npy")
                if q.shape == (axis.n_train, len(PROCESSES)):
                    cf_labels = q[:, index] > 0.5
                    cf_label_source = f"{cf_run.run_dir}/q_best_train.npy (q>0.5) [{cf_run.run_id}]"

            # (b') logistic regression supervised by teacher counterfactual Delta-J > P labels
            teacher_labels = (truth.d_train[axis.train_idx, index] > truth.p_matrix[axis.train_idx, index])

            # (c) truth-supervised oracle logistic regression
            variants = []
            if cf_labels is not None and len(np.unique(cf_labels)) > 1:
                variants.append(("LR_counterfactual_q_labels", cf_labels.astype(int), cf_label_source))
            if len(np.unique(teacher_labels)) > 1:
                variants.append(
                    (
                        "LR_teacher_DeltaJ_class_labels",
                        teacher_labels.astype(int),
                        f"{truth.path} (D_J_train > P)",
                    )
                )
            variants.append(("LR_truth_supervised_oracle", t_train_bool.astype(int), "evaluator_archive T (train)"))

            predictions: dict[str, np.ndarray] = {"3NN_donor_truth_T": pred_knn}
            for name, labels, _source in variants:
                model = LogisticRegression(max_iter=2000)
                model.fit(x_train_norm, labels)
                predictions[name] = model.predict(x_hold_norm) == 1
            label_agreements = {
                name: float((labels.astype(bool) == t_train_bool).mean()) for name, labels, _ in variants
            }
            label_sources = {name: source for name, _, source in variants}

            for name, pred in predictions.items():
                samples = ba_bootstrap(t_hold_bool, pred, idx)
                low, high = pct_ci(samples)
                row = {
                    **row_base,
                    "approach": name,
                    "ba": float(_ba_point(t_hold_bool, pred)),
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": N_BOOT,
                    "train_label_agreement_with_T": label_agreements.get(name, float("nan")),
                    "label_source": label_sources.get(name, "evaluator_archive T (train)"),
                }
                records.append(row)

    # Gap attribution for the ~78% subsurface holdout BA (scenario A)
    attribution = attribute_gap_attribution(runs, truths, axis, records, idx)

    return {
        "attribute_axis_verification": attribute_check,
        "x35_missing_rows_mean_filled": n_missing,
        "records": records,
        "gap_attribution": attribution,
    }


def attribute_gap_attribution(
    runs: Sequence[RunInfo],
    truths: Mapping[str, ScenarioTruth],
    axis: CanonicalAxis,
    records: Sequence[Mapping[str, Any]],
    idx: np.ndarray,
) -> dict[str, Any]:
    truth = truths["A"]
    index = PROCESSES.index("w_sub")
    t_hold_bool = truth.T[axis.holdout_idx, index] == 1
    by_approach: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if row["scenario"] == "A" and row["process"] == "w_sub":
            by_approach[row["approach"]] = row
    learned_run = next((run for run in runs if run.run_id == "E1_A_42"), None)
    learned_ba = None
    learned_source = None
    if learned_run is not None and has_all671_gates(learned_run):
        gates, source = load_all671_gates(learned_run)
        pred = gates[axis.holdout_idx, index] > GATE_THRESHOLD
        learned_ba = _ba_point(t_hold_bool, pred)
        learned_source = source
    out: dict[str, Any] = {
        "process": "w_sub",
        "scenario": "A",
        "n_holdout": axis.n_holdout,
        "learned_model_holdout_ba": learned_ba,
        "learned_model_source": learned_source,
    }
    for name in (
        "3NN_donor_truth_T",
        "LR_counterfactual_q_labels",
        "LR_truth_supervised_oracle",
        "LR_teacher_DeltaJ_class_labels",
    ):
        if name in by_approach:
            out[f"{name}_ba"] = by_approach[name]["ba"]
            out[f"{name}_train_label_agreement_with_T"] = by_approach[name].get("train_label_agreement_with_T")
    oracle = by_approach.get("LR_truth_supervised_oracle", {}).get("ba")
    learner_cf = by_approach.get("LR_counterfactual_q_labels", {}).get("ba")
    nn_truth = by_approach.get("3NN_donor_truth_T", {}).get("ba")
    teacher = by_approach.get("LR_teacher_DeltaJ_class_labels", {}).get("ba")
    if learned_ba is not None and oracle is not None and learner_cf is not None:
        out["gap_steps"] = {
            "total_oracle_minus_learned": oracle - learned_ba,
            "counterfactual_evidence_label_gap": oracle - learner_cf,
            "learner_gate_head_gap": learner_cf - learned_ba,
            "regionalization_mapping_gap_oracle_minus_3NN": (oracle - nn_truth) if nn_truth is not None else float("nan"),
            "teacher_DeltaJ_gt_P_label_gap_oracle_minus_teacherLR": (oracle - teacher) if teacher is not None else float("nan"),
        }
        out["gap_step_notes"] = {
            "total_oracle_minus_learned": "(truth-supervised LR oracle) - (learner E1_A_42 holdout BA)",
            "counterfactual_evidence_label_gap": (
                "(truth-supervised LR oracle) - (LR on the learner's own counterfactual targets q>0.5): "
                "price of counterfactual evidence not equalling the truth, holding the X35 mapping class fixed"
            ),
            "learner_gate_head_gap": (
                "(LR on learner q labels) - (learner's own holdout BA): residual attributable to the trained "
                "gate head (optimization/regularization/soft-gate thresholding) relative to a clean linear "
                "regionalization of the same supervision labels"
            ),
            "regionalization_mapping_gap_oracle_minus_3NN": (
                "(truth-supervised LR oracle) - (3-NN donor using neighbour truth T): if this is ~0 the X35 "
                "regionalization mapping is saturated and is NOT the limiting factor"
            ),
            "teacher_DeltaJ_gt_P_label_gap_oracle_minus_teacherLR": (
                "diagnostic only: 1[D_J_train > P] is not a valid truth surrogate for T=0 basins, because for "
                "T=0 the flip loss D_J is the loss of opening (Delta J_open = -D_J - P < 0), not an opening benefit"
            ),
        }
    out["counterfactual_label_stratification"] = counterfactual_label_stratification(truth, axis, learned_run)
    return out


def counterfactual_label_stratification(
    truth: ScenarioTruth,
    axis: CanonicalAxis,
    run: RunInfo | None,
) -> dict[str, Any]:
    """Where the learner's own counterfactual labels (q>0.5) deviate from the truth, by D stratum."""
    if run is None or not (run.run_dir / "q_best_train.npy").is_file():
        return {"available": False}
    q = np.load(run.run_dir / "q_best_train.npy")
    index = PROCESSES.index("w_sub")
    labels = q[:, index] > 0.5
    t = truth.T[axis.train_idx, index]
    d = truth.d_train[axis.train_idx, index]
    p_thr = truth.p_matrix[axis.train_idx, index]
    active = t == 1
    strata = {
        "T=1 High-D": active & (d > p_thr),
        "T=1 Low-D": active & (d >= 0) & (d <= p_thr),
        "T=1 Neg-D": active & (d < 0),
        "T=0": t == 0,
    }
    return {
        "available": True,
        "label_source": f"{run.run_dir}/q_best_train.npy (q>0.5, w_sub) [{run.run_id}]",
        "overall_agreement_with_T": float((labels == active).mean()),
        "by_stratum": {
            name: {
                "n": int(mask.sum()),
                "q_label_open_rate": ratio(int(labels[mask].sum()), int(mask.sum())),
                "agreement_with_T": ratio(
                    int((labels[mask] == active[mask]).sum()), int(mask.sum())
                ),
            }
            for name, mask in strata.items()
        },
    }


# --------------------------------------------------------------------------------------
# B4 / B5 assembly
# --------------------------------------------------------------------------------------
def ablation_analysis(
    runs: Sequence[RunInfo],
    h1_records: Mapping[str, Any],
    truths: Mapping[str, ScenarioTruth],
    axis: CanonicalAxis,
) -> dict[str, Any]:
    groups = {
        "Canonical_BCE": sorted(run.run_id for run in runs if run.experiment == "E1" and run.scenario == "A"),
        "Direct_Fit": sorted(run.run_id for run in runs if run.experiment == "E3" and run.scenario == "A"),
        "Plain_BCE": sorted(run.run_id for run in runs if run.experiment == "E4" and run.scenario == "A"),
    }
    rows: list[dict[str, Any]] = []
    stratified: list[dict[str, Any]] = []
    for method, run_ids in groups.items():
        seeds = []
        for run_id in run_ids:
            record = h1_records.get(run_id)
            if record is None:
                continue
            seeds.append(record["seed"])
            for index, process in enumerate(PROCESSES):
                metrics = record["H1_best"][process]
                rows.append(
                    {
                        "method": method,
                        "run_id": run_id,
                        "seed": record["seed"],
                        "process": process,
                        "process_label": PROCESS_LABELS[process],
                        "n_high_d": metrics["n_high_d"],
                        "recall_high_d": metrics["recall_high_d"],
                        "n_low_d": metrics["n_low_d"],
                        "shutdown_rate_low_d": metrics["shutdown_rate_low_d"],
                        "n_neg_d": metrics["n_neg_d"],
                        "n_t0": metrics["n_t0"],
                        "specificity": metrics["specificity"],
                        "ba_decidable": metrics["ba_decidable"],
                        "full_truth_recovery_t1": metrics["full_truth_recovery_t1"],
                        "auc": metrics["auc"],
                        "mean_weight": metrics["mean_weight"],
                        "std_weight": metrics["std_weight"],
                        "frac_weight_0.45_0.55": metrics["frac_weight_0.45_0.55"],
                        "w_q05": metrics["w_q05"],
                        "w_q25": metrics["w_q25"],
                        "w_q50": metrics["w_q50"],
                        "w_q75": metrics["w_q75"],
                        "w_q95": metrics["w_q95"],
                        "uniform_P_ba_decidable": metrics["uniform_P_sensitivity"]["ba_decidable"],
                        "uniform_P_shutdown_rate_low_d": metrics["uniform_P_sensitivity"]["shutdown_rate_low_d"],
                    }
                )
                if process in ("w_int", "w_snow"):
                    holdout = (record.get("H2") or {}).get(process)
                    stratified.append(
                        {
                            "method": method,
                            "run_id": run_id,
                            "process": process,
                            "region": "train_536",
                            "n_high_d": metrics["n_high_d"],
                            "recall_high_d": metrics["recall_high_d"],
                            "n_low_d": metrics["n_low_d"],
                            "shutdown_rate_low_d": metrics["shutdown_rate_low_d"],
                            "open_rate_low_d": metrics["open_rate_low_d"],
                            "mean_weight_low_d": metrics["mean_weight_low_d"],
                            "n_neg_d": metrics["n_neg_d"],
                            "open_rate_neg_d": metrics["open_rate_neg_d"],
                            "n_t0": metrics["n_t0"],
                            "specificity": metrics["specificity"],
                        }
                    )
                    if holdout:
                        stratified.append(
                            {
                                "method": method,
                                "run_id": run_id,
                                "process": process,
                                "region": "holdout_135_D_test_strata",
                                "n_high_d": holdout["n_high_d"],
                                "recall_high_d": holdout["recall_high_d"],
                                "n_low_d": holdout["n_low_d"],
                                "shutdown_rate_low_d": holdout["shutdown_rate_low_d"],
                                "open_rate_low_d": holdout["open_rate_low_d"],
                                "mean_weight_low_d": holdout["mean_weight_low_d"],
                                "n_neg_d": holdout["n_neg_d"],
                                "open_rate_neg_d": holdout["open_rate_neg_d"],
                                "n_t0": holdout["n_t0"],
                                "specificity": holdout["specificity"],
                            }
                        )
        groups[method] = {"run_ids": run_ids, "seeds": seeds, "seed_status": mark_seed_pending(seeds)}

    comparisons = [("E1_A_42", "E3_A_42"), ("E1_A_42", "E4_A_42"), ("E4_A_42", "E3_A_42")]
    paired = paired_method_differences(runs, truths, axis, comparisons)
    return {
        "groups": groups,
        "note": (
            "Raw training total loss is NOT comparable across methods: E3 (Direct_Fit) trains a "
            "different objective (MSE fit + 0.007*mean(p_struct)) so its loss_history scale is not "
            "on the same axis as the E1/E4 Fit-AIC + BCE total."
        ),
        "rows": rows,
        "stratified_w_int_w_snow": stratified,
        "paired_differences": paired,
        "cross_seed_stability": cross_seed_stability(h1_records, groups),
    }


def paired_method_differences(
    runs: Sequence[RunInfo],
    truths: Mapping[str, ScenarioTruth],
    axis: CanonicalAxis,
    comparisons: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Per-basin paired differences with 95% CIs (paired basin bootstrap, 10,000 resamples)."""
    rng = np.random.default_rng(BOOT_SEED)
    run_index = {run.run_id: run for run in runs}
    truth = truths["A"]
    rows: list[dict[str, Any]] = []
    for run_a, run_b in comparisons:
        a = run_index.get(run_a)
        b = run_index.get(run_b)
        if a is None or b is None:
            continue
        nse_a = load_nse_cache(a)
        nse_b = load_nse_cache(b)
        for index, process in enumerate(PROCESSES):
            t_train = truth.T[axis.train_idx, index]
            d_train = truth.d_train[axis.train_idx, index]
            p_train = truth.p_matrix[axis.train_idx, index]
            decidable = ((t_train == 1) & (d_train > p_train)) | (t_train == 0)
            if decidable.sum() == 0:
                continue
            weights_a = np.load(a.gates_train_best)[:, index]
            weights_b = np.load(b.gates_train_best)[:, index]
            correct_a = ((weights_a > GATE_THRESHOLD).astype(int) == t_train)[decidable].astype(float)
            correct_b = ((weights_b > GATE_THRESHOLD).astype(int) == t_train)[decidable].astype(float)
            idx = paired_index_matrix(int(decidable.sum()), N_BOOT, rng)
            samples = paired_mean_difference(correct_a, correct_b, idx)
            low, high = pct_ci(samples)
            rows.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "region": "train_536_decidable",
                    "metric": "per_basin_decision_correctness",
                    "process": process,
                    "n_paired": int(decidable.sum()),
                    "mean_a": float(correct_a.mean()),
                    "mean_b": float(correct_b.mean()),
                    "difference": float(correct_a.mean() - correct_b.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": N_BOOT,
                }
            )
        # holdout decision correctness + NSE/KGE use the full 135-basin holdout set
        gates_a = gates_all_for(a)
        gates_b = gates_all_for(b)
        idx_hold = paired_index_matrix(axis.n_holdout, N_BOOT, rng)
        for index, process in enumerate(PROCESSES):
            if gates_a is None or gates_b is None:
                continue
            t_hold = truth.T[axis.holdout_idx, index]
            correct_a = ((gates_a[axis.holdout_idx, index] > GATE_THRESHOLD).astype(int) == t_hold).astype(float)
            correct_b = ((gates_b[axis.holdout_idx, index] > GATE_THRESHOLD).astype(int) == t_hold).astype(float)
            samples = paired_mean_difference(correct_a, correct_b, idx_hold)
            low, high = pct_ci(samples)
            rows.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "region": "holdout_135",
                    "metric": "per_basin_decision_correctness",
                    "process": process,
                    "n_paired": axis.n_holdout,
                    "mean_a": float(correct_a.mean()),
                    "mean_b": float(correct_b.mean()),
                    "difference": float(correct_a.mean() - correct_b.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": N_BOOT,
                }
            )
            for metric_name, key in (("per_basin_NSE", "holdout_nse"), ("per_basin_KGE", "holdout_kge")):
                if nse_a is None or nse_b is None:
                    continue
                values_a = nse_a[key]
                values_b = nse_b[key]
                valid = np.isfinite(values_a) & np.isfinite(values_b)
                if int(valid.sum()) < MIN_VALID_DAYS:
                    continue
                idx_valid = np.where(valid)[0]
                pair_idx = paired_index_matrix(int(idx_valid.size), N_BOOT, rng)
                samples = (
                    values_a[idx_valid][pair_idx].mean(axis=1) - values_b[idx_valid][pair_idx].mean(axis=1)
                )
                low, high = pct_ci(samples)
                rows.append(
                    {
                        "run_a": run_a,
                        "run_b": run_b,
                        "region": "holdout_135",
                        "metric": metric_name,
                        "process": None,
                        "n_paired": int(valid.sum()),
                        "mean_a": float(np.nanmean(values_a[idx_valid])),
                        "mean_b": float(np.nanmean(values_b[idx_valid])),
                        "difference": float(np.nanmean(values_a[idx_valid]) - np.nanmean(values_b[idx_valid])),
                        "ci_low": low,
                        "ci_high": high,
                        "n_bootstrap": N_BOOT,
                    }
                )
        for metric_name, key in (("per_basin_NSE_train", "train_nse"), ("per_basin_KGE_train", "train_kge")):
            if nse_a is None or nse_b is None:
                continue
            values_a = nse_a[key]
            values_b = nse_b[key]
            valid = np.isfinite(values_a) & np.isfinite(values_b)
            if int(valid.sum()) < MIN_VALID_DAYS:
                continue
            idx_valid = np.where(valid)[0]
            pair_idx = paired_index_matrix(int(idx_valid.size), N_BOOT, rng)
            samples = values_a[idx_valid][pair_idx].mean(axis=1) - values_b[idx_valid][pair_idx].mean(axis=1)
            low, high = pct_ci(samples)
            rows.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "region": "train_536",
                    "metric": metric_name,
                    "process": None,
                    "n_paired": int(valid.sum()),
                    "mean_a": float(np.nanmean(values_a[idx_valid])),
                    "mean_b": float(np.nanmean(values_b[idx_valid])),
                    "difference": float(np.nanmean(values_a[idx_valid]) - np.nanmean(values_b[idx_valid])),
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": N_BOOT,
                }
            )
    return {
        "definition": "paired per-basin differences, identical basin index set inside each resample, "
                      f"{N_BOOT} percentile-bootstrap resamples",
        "rows": rows,
    }


def gates_all_for(run: RunInfo) -> np.ndarray | None:
    if has_all671_gates(run):
        return load_all671_gates(run)[0]
    return None


def load_nse_cache(run: RunInfo) -> dict[str, np.ndarray] | None:
    path = STAT_DIR / f"nse_kge_per_basin_{run.run_id}.npz"
    if not path.is_file():
        return None
    with np.load(path) as cached:
        return {key: cached[key] for key in ("train_nse", "train_kge", "holdout_nse", "holdout_kge")}


def cross_seed_stability(h1_records: Mapping[str, Any], groups: Mapping[str, Any]) -> dict[str, Any]:
    """Per-metric cross-seed mean/spread for the E1 scenario-A triple (the only replicated family)."""
    seeds = [seed for seed in (42, 43, 44) if f"E1_A_{seed}" in h1_records]
    out: dict[str, Any] = {"seeds": seeds, "seed_status": mark_seed_pending(seeds)}
    for process in PROCESSES:
        for field in ("ba_decidable", "recall_high_d", "specificity", "shutdown_rate_low_d"):
            values = []
            for seed in seeds:
                value = h1_records[f"E1_A_{seed}"]["H1_best"][process][field]
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    values.append(float(value))
            if values:
                out[f"{process}__{field}"] = {
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_seeds": len(values),
                }
    out["single_seed_families"] = {
        method: {"run_ids": block["run_ids"], "status": "SEED-PENDING (single seed 42)"}
        for method, block in groups.items()
        if len(block["seeds"]) < 2
    }
    return out


def random_phenology_analysis(
    runs: Sequence[RunInfo],
    h1_records: Mapping[str, Any],
) -> dict[str, Any]:
    scenario_a_train = []
    scenario_a_holdout = []
    for run in runs:
        record = h1_records.get(run.run_id)
        if record is None or run.scenario != "A" or run.experiment != "E1":
            continue
        h1 = record["H1_best"]["w_phen"]
        scenario_a_train.append(
            {
                "run_id": run.run_id,
                "seed": run.seed,
                "ba_decidable_train": h1["ba_decidable"],
                "recall_high_d_train": h1["recall_high_d"],
                "specificity_train": h1["specificity"],
                "n_high_d": h1["n_high_d"],
                "n_t0": h1["n_t0"],
            }
        )
        if record.get("H2"):
            h2 = record["H2"]["w_phen"]
            scenario_a_holdout.append(
                {
                    "run_id": run.run_id,
                    "seed": run.seed,
                    "ba_holdout": h2["ba_holdout_unfiltered"],
                    "ba_holdout_decidable_test_D_strata": h2["ba_decidable_test_D_strata"],
                    "recall_holdout": h2["recall_unfiltered_t1"],
                    "specificity_holdout": h2["specificity"],
                    "n_pos": h2["n_pos"],
                    "n_neg": h2["n_neg"],
                }
            )
    scenario_a_train.sort(key=lambda item: item["seed"])
    scenario_a_holdout.sort(key=lambda item: item["seed"])
    train_values = [item["ba_decidable_train"] for item in scenario_a_train]
    hold_values = [item["ba_holdout"] for item in scenario_a_holdout]
    return {
        "scenario_A_phenology_random": {
            "seeds": [item["seed"] for item in scenario_a_train],
            "seed_status": mark_seed_pending([item["seed"] for item in scenario_a_train]),
            "in_sample_train_decidable_ba_mean": float(np.mean(train_values)) if train_values else float("nan"),
            "in_sample_train_decidable_ba_sd": float(np.std(train_values, ddof=1)) if len(train_values) > 1 else 0.0,
            "out_of_sample_holdout_ba_mean": float(np.mean(hold_values)) if hold_values else float("nan"),
            "out_of_sample_holdout_ba_sd": float(np.std(hold_values, ddof=1)) if len(hold_values) > 1 else 0.0,
            "per_run_train": scenario_a_train,
            "per_run_holdout": scenario_a_holdout,
            "framing": (
                "in-sample structural recovery (536 training basins, D-stratified decidable BA) vs "
                "out-of-sample regionalizability (135 holdout basins, unfiltered BA)"
            ),
        },
        "scenario_D_random_processes": {
            "note": (
                "Scenario D has two random processes (w_sub and w_phen). Holdout BA below 50% is "
                "reported with a paired basin bootstrap CI; the CI is a one-sample CI of the BA "
                "functional on the 135 holdout basins."
            )
        },
    }


# --------------------------------------------------------------------------------------
# Output writers (metrics stage)
# --------------------------------------------------------------------------------------
def write_metrics_outputs(artifacts: Mapping[str, Any]) -> None:
    STAT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(STAT_DIR / "run_inventory.json", artifacts["run_inventory"])
    write_csv(
        STAT_DIR / "run_inventory.csv",
        artifacts["run_inventory"],
        ["run_id", "experiment", "method", "scenario", "seed", "batch", "best_epoch", "final_epoch", "best_total_loss",
         "stopped_by_early_stopping", "has_best_checkpoint", "has_all671_gates"],
    )
    write_json(STAT_DIR / "process_order_verification.json", artifacts["process_order_verification"])
    write_json(STAT_DIR / "h1_per_run_metrics.json", artifacts["H1_best"])

    h1_rows = []
    for run_id, record in artifacts["H1_best"].items():
        for phase in ("H1_best", "H1_final"):
            block = record.get(phase)
            if not block:
                continue
            for process in PROCESSES:
                metrics = block[process]
                h1_rows.append(
                    {
                        "run_id": run_id,
                        "method": record["method"],
                        "scenario": record["scenario"],
                        "seed": record["seed"],
                        "phase": phase,
                        "process": process,
                        "n_t1": metrics["n_t1"],
                        "n_t0": metrics["n_t0"],
                        "n_high_d": metrics["n_high_d"],
                        "recall_high_d": metrics["recall_high_d"],
                        "n_low_d": metrics["n_low_d"],
                        "shutdown_rate_low_d": metrics["shutdown_rate_low_d"],
                        "n_neg_d": metrics["n_neg_d"],
                        "specificity": metrics["specificity"],
                        "ba_decidable": metrics["ba_decidable"],
                        "full_truth_recovery_t1": metrics["full_truth_recovery_t1"],
                        "overall_accuracy": metrics["overall_accuracy"],
                        "auc": metrics["auc"],
                        "mean_weight": metrics["mean_weight"],
                        "p_threshold_used": metrics["p_threshold_used"],
                        "uniform_P_ba_decidable": metrics["uniform_P_sensitivity"]["ba_decidable"],
                        "uniform_P_n_high_d": metrics["uniform_P_sensitivity"]["n_high_d"],
                        "uniform_P_n_low_d": metrics["uniform_P_sensitivity"]["n_low_d"],
                    }
                )
    write_csv(
        STAT_DIR / "h1_per_run_metrics.csv",
        h1_rows,
        ["run_id", "method", "scenario", "seed", "phase", "process", "n_t1", "n_t0", "n_high_d", "recall_high_d",
         "n_low_d", "shutdown_rate_low_d", "n_neg_d", "specificity", "ba_decidable", "full_truth_recovery_t1",
         "overall_accuracy", "auc", "mean_weight", "p_threshold_used", "uniform_P_ba_decidable", "uniform_P_n_high_d",
         "uniform_P_n_low_d"],
    )

    h2_rows = []
    for run_id, record in artifacts["H1_best"].items():
        block = record.get("H2")
        if not block:
            continue
        for process in PROCESSES:
            metrics = block[process]
            h2_rows.append(
                {
                    "run_id": run_id,
                    "method": record["method"],
                    "scenario": record["scenario"],
                    "seed": record["seed"],
                    "process": process,
                    "n_pos": metrics["n_pos"],
                    "n_neg": metrics["n_neg"],
                    "recall": metrics["recall_unfiltered_t1"],
                    "specificity": metrics["specificity"],
                    "ba_holdout_unfiltered": metrics["ba_holdout_unfiltered"],
                    "ba_decidable_test_D_strata": metrics["ba_decidable_test_D_strata"],
                    "auc": metrics["auc"],
                    "n_high_d_test": metrics["n_high_d"],
                    "recall_high_d_test": metrics["recall_high_d"],
                    "n_low_d_test": metrics["n_low_d"],
                    "shutdown_rate_low_d_test": metrics["shutdown_rate_low_d"],
                    "n_neg_d_test": metrics["n_neg_d"],
                    "mean_weight": metrics["mean_weight"],
                }
            )
    write_json(STAT_DIR / "h2_per_run_metrics.json", {k: v.get("H2") for k, v in artifacts["H1_best"].items()})
    write_csv(
        STAT_DIR / "h2_per_run_metrics.csv",
        h2_rows,
        ["run_id", "method", "scenario", "seed", "process", "n_pos", "n_neg", "recall", "specificity",
         "ba_holdout_unfiltered", "ba_decidable_test_D_strata", "auc", "n_high_d_test", "recall_high_d_test",
         "n_low_d_test", "shutdown_rate_low_d_test", "n_neg_d_test", "mean_weight"],
    )

    write_json(STAT_DIR / "h2_factorial_bootstrap.json", artifacts["factorial"])
    factorial_rows = []
    for seed, per_process in artifacts["factorial"]["per_seed"].items():
        for process in PROCESSES:
            effects = per_process[process]["effects"]
            ba = per_process[process]["holdout_ba"]
            row = {
                "seed": seed,
                "process": process,
                "BA_A": ba["A"], "BA_B": ba["B"], "BA_C": ba["C"], "BA_D": ba["D"],
            }
            for name in ("me_attribute", "me_phenology", "interaction"):
                row[f"{name}_point"] = effects[name]["point"]
                row[f"{name}_ci_low"] = effects[name]["ci_low"]
                row[f"{name}_ci_high"] = effects[name]["ci_high"]
            factorial_rows.append(row)
    for process, pooled in artifacts["factorial"]["pooled"].items():
        row = {"seed": f"pooled(n={pooled['n_seeds']})", "process": process, "BA_A": "", "BA_B": "", "BA_C": "", "BA_D": ""}
        for name in ("me_attribute", "me_phenology", "interaction"):
            row[f"{name}_point"] = pooled[name]["point"]
            row[f"{name}_ci_low"] = pooled[name]["ci_low"]
            row[f"{name}_ci_high"] = pooled[name]["ci_high"]
        factorial_rows.append(row)
    write_csv(STAT_DIR / "h2_factorial_bootstrap.csv", factorial_rows)

    write_json(STAT_DIR / "baseline_regionalization.json", artifacts["baselines"])
    write_csv(STAT_DIR / "baseline_regionalization.csv", artifacts["baselines"]["records"])

    write_json(STAT_DIR / "method_ablation.json", artifacts["method_ablation"])
    write_csv(STAT_DIR / "method_ablation.csv", artifacts["method_ablation"]["rows"])
    write_csv(STAT_DIR / "method_ablation_stratified.csv", artifacts["method_ablation"]["stratified_w_int_w_snow"])
    write_csv(
        STAT_DIR / "method_ablation_paired_differences.csv",
        artifacts["method_ablation"]["paired_differences"]["rows"],
        ["run_a", "run_b", "region", "metric", "process", "n_paired", "mean_a", "mean_b", "difference", "ci_low",
         "ci_high", "n_bootstrap"],
    )

    write_json(STAT_DIR / "random_phenology.json", artifacts["random_phenology"])


# --------------------------------------------------------------------------------------
# Stage 2: forward-simulation NSE / KGE (reuses the formal evaluation path)
# --------------------------------------------------------------------------------------
def load_retrain_helpers() -> Any:
    """Import revision1/scripts/evaluate_s2_batch1_retrain.py to reuse its NSE/KGE helper."""
    path = ROOT / "evaluate/retrain_helpers.py"
    spec = importlib.util.spec_from_file_location("evaluate_s2_batch1_retrain", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_s2_batch1_retrain"] = module
    spec.loader.exec_module(module)
    return module


def build_strict_context():
    """Build the frozen scenario-A fold + model once (reuses strict.s2_loader / strict.trainer)."""
    import torch

    sys.path.insert(0, str(ROOT))
    from train.synthetic.strict.s2_loader import load_candidate_strict_fold
    from train.synthetic.strict.trainer import build_strict_model_and_loss, PROCESS_COSTS as COSTS, PROCESS_ORDER as ORDER

    if tuple(ORDER) != PROCESSES:
        raise RuntimeError(f"trainer PROCESS_ORDER {ORDER} != p_struct column order {PROCESSES}")
    if {k: float(v) for k, v in COSTS.items()} != PROCESS_COSTS:
        raise RuntimeError("trainer PROCESS_COSTS differ from the frozen cost map")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config.update({"mode": "train", "device": "cpu", "dtype": torch.float32})
    config["model"]["rho"] = 365
    config["model"]["warmup"] = 365
    config["model"]["nmul"] = 4
    config["delta_model"]["phy_model"]["nmul"] = 4
    config["confidence_weighted_cf_loss"] = True
    config["structure_optimizer"] = "none"
    config["train"]["batch_size"] = 100
    config["train"]["optimizer"] = "Adadelta"
    config["train"]["learning_rate"] = 1.0
    config["train"]["start_epoch"] = 0
    config["loss_function"]["aic_alpha"] = LAMBDA

    fold_data = load_candidate_strict_fold(
        config,
        manifest_path=SPLIT_PATH,
        cache_root=NSE_CACHE_ROOT,
        dataset_path=EVALUATOR_DIR / "scenario_A/dataset.pkl",
        gage_info_path=GAGE_ID_PATH,
        hydro_loader_source=HYDRO_LOADER_SOURCE,
    )
    model, _ = build_strict_model_and_loss(config, fold_data, device="cpu", model_dir=Path("/tmp/r1_019_model_build"))
    return config, fold_data, model


def build_test_window_sample(config: Mapping[str, Any], fold_data: Any, axis: CanonicalAxis) -> dict[str, Any]:
    """Build the holdout-basin x frozen test-window sample (1995-10-01..2010-09-30 + warmup).

    Validated against the loader-produced training sample: same attribute normalization
    statistics, same raw physical forcings, same day-of-year construction.
    """
    import pickle

    import torch

    dates = np.load(DATES_PATH).astype(str)
    with SOURCE_DATASET_PATH.open("rb") as handle:
        forcings, _, _ = pickle.load(handle)
    forcings = np.asarray(forcings, dtype=np.float32)

    stats = json.loads(Path(fold_data.stats_path).read_text(encoding="utf-8"))
    attributes = _load_x35()
    all_attributes = list(config["observations"]["all_attributes"])
    mean = np.array([stats[name][2] for name in all_attributes], dtype=np.float32)
    std = np.array([stats[name][3] for name in all_attributes], dtype=np.float32)
    std[std == 0] = 1.0
    # 27 X35 rows carry NaN (root_depth_50/geol_porosity); the frozen loader mean-fills them
    # (HydroLoader: c_nn_norm[c_nn_norm != c_nn_norm] = 0), so the same fill is applied here.
    attrs_norm = np.nan_to_num((attributes[axis.holdout_idx] - mean) / std)

    nn_forcings = list(config["model"]["nn"]["forcings"])
    all_forcings = list(config["observations"]["all_forcings"])
    forcing_idx = [all_forcings.index(name) for name in nn_forcings]
    # raw arrays are (basin, day, variable); model tensors are (day, basin, variable)
    x_nn_raw = np.transpose(
        forcings[axis.holdout_idx][:, TEST_WINDOW.start:TEST_WINDOW.stop][:, :, forcing_idx], (1, 0, 2)
    )
    x_nn_mean = np.array([stats[name][2] for name in nn_forcings], dtype=np.float32)
    x_nn_std = np.array([stats[name][3] for name in nn_forcings], dtype=np.float32)
    x_nn_std[x_nn_std == 0] = 1.0
    x_nn_norm = (x_nn_raw - x_nn_mean) / x_nn_std

    phy_forcings = list(config["model"]["phy"]["forcings"])
    phy_idx = [all_forcings.index(name) for name in phy_forcings]
    x_phy = np.transpose(
        forcings[axis.holdout_idx][:, TEST_WINDOW.start:TEST_WINDOW.stop][:, :, phy_idx], (1, 0, 2)
    )

    n_days = int(x_phy.shape[0])
    day_of_year = np.array(
        [datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates[TEST_WINDOW.start:TEST_WINDOW.stop]],
        dtype=np.float32,
    )
    doy = torch.tensor(day_of_year).view(n_days, 1, 1).repeat(1, len(axis.holdout_idx), 1)

    c_nn_norm = np.repeat(attrs_norm[None, :, :], n_days, axis=0)
    xc = np.concatenate([x_nn_norm, c_nn_norm], axis=2)
    return {
        "x_phy": torch.from_numpy(np.ascontiguousarray(x_phy.astype(np.float32))),
        "xc_nn_norm": torch.from_numpy(np.ascontiguousarray(xc.astype(np.float32))),
        "doy": doy,
    }


def validate_constructed_sample(config: Mapping[str, Any], fold_data: Any, axis: CanonicalAxis) -> dict[str, Any]:
    """Prove the hand-built sample equals the loader's frozen training sample (train window/basins)."""
    import pickle

    import torch

    stats = json.loads(Path(fold_data.stats_path).read_text(encoding="utf-8"))
    dates = np.load(DATES_PATH).astype(str)
    with SOURCE_DATASET_PATH.open("rb") as handle:
        forcings, _, attributes = pickle.load(handle)
    forcings = np.asarray(forcings, dtype=np.float32)
    attributes = np.asarray(attributes, dtype=np.float64)

    all_attributes = list(config["observations"]["all_attributes"])
    mean = np.array([stats[name][2] for name in all_attributes], dtype=np.float32)
    std = np.array([stats[name][3] for name in all_attributes], dtype=np.float32)
    std[std == 0] = 1.0

    train_ds = fold_data.train_dataset
    loader_xc = train_ds["xc_nn_norm"].numpy()
    constructed_attrs = np.nan_to_num((attributes[axis.train_idx] - mean) / std).astype(np.float32)
    loader_attrs = loader_xc[0, :, -35:]
    attrs_max_diff = float(np.max(np.abs(loader_attrs - constructed_attrs)))

    all_forcings = list(config["observations"]["all_forcings"])
    phy_idx = [all_forcings.index(name) for name in config["model"]["phy"]["forcings"]]
    constructed_x_phy = np.transpose(
        forcings[axis.train_idx][:, : loader_xc.shape[0]][:, :, phy_idx], (1, 0, 2)
    )
    loader_x_phy = train_ds["x_phy"].numpy()
    x_phy_max_diff = float(np.max(np.abs(loader_x_phy - constructed_x_phy)))

    day_of_year = np.array(
        [datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates[: loader_xc.shape[0]]], dtype=np.float32
    )
    constructed_doy = np.repeat(day_of_year[:, None, None], len(axis.train_idx), axis=1)
    doy_max_diff = float(np.max(np.abs(train_ds["doy"].numpy() - constructed_doy)))
    return {
        "attribute_normalization_max_abs_diff": attrs_max_diff,
        "x_phy_max_abs_diff": x_phy_max_diff,
        "doy_max_abs_diff": doy_max_diff,
        "date_axis": {"dates[0]": str(dates[0]), "dates[5478]": str(dates[5478]), "dates[10955]": str(dates[10955])},
    }


def nse_stage(runs: Sequence[RunInfo], axis: CanonicalAxis, force: bool) -> dict[str, Any]:
    import torch

    importlib = load_retrain_helpers()
    config, fold_data, model = build_strict_context()
    inner = next(iter(model.model_dict.values()))
    truths = {scenario: load_scenario_truth(scenario, axis) for scenario in SCENARIOS}
    validation = validate_constructed_sample(config, fold_data, axis)
    if validation["attribute_normalization_max_abs_diff"] > 1e-4 or validation["x_phy_max_abs_diff"] > 1e-4:
        raise RuntimeError(f"hand-built sample does not reproduce the loader sample: {validation}")

    train_sample = {key: value for key, value in fold_data.train_dataset.items()}
    test_sample = build_test_window_sample(config, fold_data, axis)

    truth_A = truths["A"]
    # archive arrays are (basin, day); model output is (day, basin)
    obs_train = truth_A.y_mm_d[axis.train_idx][:, TRAIN_SCORED].T
    obs_test = truth_A.y_mm_d[axis.holdout_idx][:, TEST_SCORED].T

    summary_rows = []
    cache_manifest = {"validation_of_constructed_sample": validation, "runs": {}}
    for run in runs:
        if not run.best_checkpoint.is_file():
            continue
        checkpoint_sha = sha256_file(run.best_checkpoint)
        cache_path = STAT_DIR / f"nse_kge_per_basin_{run.run_id}.npz"
        if cache_path.is_file() and not force:
            with np.load(cache_path) as cached:
                if str(cached["checkpoint_sha256"]) == checkpoint_sha:
                    cache_manifest["runs"][run.run_id] = {
                        "status": "cache-hit",
                        "cache": str(cache_path),
                        "checkpoint_sha256": checkpoint_sha,
                    }
                    summary_rows.append(_nse_row_from_cache(run, cache_path))
                    continue
        checkpoint = torch.load(run.best_checkpoint, map_location="cpu", weights_only=False)
        inner.load_state_dict(checkpoint["model_state_dict"])
        inner.eval()
        with torch.no_grad():
            sim_train = model(train_sample, eval=True)[next(iter(model.model_dict))]["streamflow"].numpy()[:, :, 0]
            sim_test = model(test_sample, eval=True)[next(iter(model.model_dict))]["streamflow"].numpy()[:, :, 0]

        gates_recomputed = _recompute_gates(inner, config, fold_data)
        gate_source = None
        gate_max_diff = float("nan")
        if run.gates_all671.is_file():
            gates_saved = np.load(run.gates_all671)
            gate_max_diff = float(np.max(np.abs(gates_saved - gates_recomputed)))
            gate_source = str(run.gates_all671)
        else:
            np.save(run.cached_gates_all671, gates_recomputed)
            gate_source = str(run.cached_gates_all671)

        train_nse, train_kge = _per_basin_nse_kge(sim_train, obs_train, importlib)
        test_nse, test_kge = _per_basin_nse_kge(sim_test, obs_test, importlib)
        np.savez(
            cache_path,
            train_basin_ids=axis.ids[axis.train_idx],
            holdout_basin_ids=axis.ids[axis.holdout_idx],
            train_nse=train_nse,
            train_kge=train_kge,
            holdout_nse=test_nse,
            holdout_kge=test_kge,
            train_n_calib_score_days=np.full(axis.n_train, sim_train.shape[0]),
            holdout_n_test_score_days=np.full(axis.n_holdout, sim_test.shape[0]),
            checkpoint_sha256=checkpoint_sha,
            gate_max_abs_diff_vs_saved=gate_max_diff,
            gates_used_source=gate_source,
        )
        cache_manifest["runs"][run.run_id] = {
            "status": "computed",
            "cache": str(cache_path),
            "checkpoint_sha256": checkpoint_sha,
            "gate_max_abs_diff_vs_saved": gate_max_diff,
            "gates_used_source": gate_source,
        }
        summary_rows.append(_nse_row_from_cache(run, cache_path))

    rows_by_run = {row["run_id"]: row for row in summary_rows}
    summary_rows = [rows_by_run[run_id] for run_id in sorted(rows_by_run)]
    write_csv(STAT_DIR / "nse_kge_summary.csv", summary_rows)
    write_json(STAT_DIR / "nse_kge_manifest.json", cache_manifest)
    return {"manifest": cache_manifest, "summary_rows": summary_rows}


def _recompute_gates(inner: Any, config: Mapping[str, Any], fold_data: Any) -> np.ndarray:
    import torch

    stats = json.loads(Path(fold_data.stats_path).read_text(encoding="utf-8"))
    attributes = _load_x35()
    all_attributes = list(config["observations"]["all_attributes"])
    mean = np.array([stats[name][2] for name in all_attributes], dtype=np.float32)
    std = np.array([stats[name][3] for name in all_attributes], dtype=np.float32)
    std[std == 0] = 1.0
    attrs_norm = torch.from_numpy(np.nan_to_num((attributes - mean) / std).astype(np.float32))
    with torch.no_grad():
        logits = inner.nn_model.get_structure_logits(attrs_norm).view(-1, 4, 2).clamp(-10.0, 10.0)
        return torch.sigmoid(logits[..., 1] - logits[..., 0]).numpy()


def _per_basin_nse_kge(sim: np.ndarray, obs: np.ndarray, retrain_module: Any) -> tuple[np.ndarray, np.ndarray]:
    n_basins = obs.shape[1]
    nse = np.full(n_basins, np.nan)
    kge = np.full(n_basins, np.nan)
    for index in range(n_basins):
        s = sim[:, index]
        o = obs[:, index]
        mask = np.isfinite(s) & np.isfinite(o)
        if int(mask.sum()) < MIN_VALID_DAYS:
            continue
        nse[index], kge[index] = retrain_module.compute_nse_kge(s[mask].astype(np.float64), o[mask].astype(np.float64))
    return nse, kge


def _nse_row_from_cache(run: RunInfo, cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path) as cached:
        train_nse = cached["train_nse"]
        train_kge = cached["train_kge"]
        hold_nse = cached["holdout_nse"]
        hold_kge = cached["holdout_kge"]
        gate_diff = float(cached["gate_max_abs_diff_vs_saved"])
    return {
        "run_id": run.run_id,
        "method": run.method,
        "scenario": run.scenario,
        "seed": run.seed,
        "train_region": "536 train basins x scored training window (1981-10-01..1995-09-30, 5113 days)",
        "holdout_region": "135 holdout basins x scored test window (1996-09-30..2010-09-30, 5113 days)",
        "n_train_basins": int(np.isfinite(train_nse).sum()),
        "n_holdout_basins": int(np.isfinite(hold_nse).sum()),
        "train_nse_mean": float(np.nanmean(train_nse)),
        "train_nse_median": float(np.nanmedian(train_nse)),
        "train_kge_mean": float(np.nanmean(train_kge)),
        "train_kge_median": float(np.nanmedian(train_kge)),
        "holdout_nse_mean": float(np.nanmean(hold_nse)),
        "holdout_nse_median": float(np.nanmedian(hold_nse)),
        "holdout_kge_mean": float(np.nanmean(hold_kge)),
        "holdout_kge_median": float(np.nanmedian(hold_kge)),
        "n_holdout_nse_gt_0.5": int(np.nansum(hold_nse > 0.5)),
        "gate_max_abs_diff_vs_saved_all671": gate_diff,
        "cache": str(cache_path),
    }


# --------------------------------------------------------------------------------------
# Stage 3: final report + summary
# --------------------------------------------------------------------------------------
def report_stage(metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if metrics is None:
        metrics = load_metrics_artifacts()
    nse_summary = load_nse_summary()
    report = render_report(metrics, nse_summary)
    REPORT_PATH.write_text(report, encoding="utf-8")
    summary = build_summary(metrics, nse_summary)
    write_json(SUMMARY_PATH, summary)
    return summary


def load_metrics_artifacts() -> dict[str, Any]:
    factorial = json.loads((STAT_DIR / "h2_factorial_bootstrap.json").read_text(encoding="utf-8"))
    h1 = json.loads((STAT_DIR / "h1_per_run_metrics.json").read_text(encoding="utf-8"))
    baselines = json.loads((STAT_DIR / "baseline_regionalization.json").read_text(encoding="utf-8"))
    ablation = json.loads((STAT_DIR / "method_ablation.json").read_text(encoding="utf-8"))
    random_phenology = json.loads((STAT_DIR / "random_phenology.json").read_text(encoding="utf-8"))
    inventory = json.loads((STAT_DIR / "run_inventory.json").read_text(encoding="utf-8"))
    process_order = json.loads((STAT_DIR / "process_order_verification.json").read_text(encoding="utf-8"))
    return {
        "factorial": factorial,
        "H1_best": h1,
        "baselines": baselines,
        "method_ablation": ablation,
        "random_phenology": random_phenology,
        "run_inventory": inventory,
        "process_order_verification": process_order,
    }


def load_nse_summary() -> list[dict[str, Any]]:
    path = STAT_DIR / "nse_kge_summary.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_report(metrics: Mapping[str, Any], nse_rows: Sequence[Mapping[str, Any]]) -> str:
    factorial = metrics["factorial"]
    h1 = metrics["H1_best"]
    baselines = metrics["baselines"]
    ablation = metrics["method_ablation"]
    random_phen = metrics["random_phenology"]
    inventory = metrics["run_inventory"]
    complete_seeds = factorial["complete_seeds"]

    lines: list[str] = []
    add = lines.append
    add("# R1-019 STAT 独立统计复核报告（S2 v1.1 受控合成真值实验，Batch-1）")
    add("")
    add(f"**生成时间：** {datetime.now().isoformat(timespec='seconds')}")
    add("**分析脚本：** `revision1/S2/stat/r1_019_stat_analysis.py`（可重跑、自动发现 run 目录）")
    add("**数据来源（全部原始 per-basin 数组，未复制历史报告文字）：**")
    add("")
    add("- `revision1/S2/batch1_results/<RUN_ID>/p_struct_best_train.npy`、`p_struct_final_train.npy`、"
        "`p_struct_best_all671.npy`、`summary.json`、`q_best_train.npy`、`best_checkpoint.pt`")
    add("- `revision1/S2/data/scenario_{A,B,C,D}/evaluator_archive.npz`（T, D_J_train, D_J_test, P_matrix, Y_mm_d, basin_ids）")
    add("- `data/gage_id.npy`（671 规范流域轴）、`data/camels_dates.npy`、`data/camels_dataset`（X35 属性）")
    add("- `revision1/analysis/R1_014_CANDIDATE_SPLIT.json`（536 train / 135 holdout，已冻结）")
    add("- `revision1/S2/truth_specification.json`（T/D 真值设计与计数）")
    add("")
    add("**复用的既有实现：** H1/H2 指标定义复用 `revision1/S2/evaluate_batch1_results.py`；"
        "NSE/KGE 计算复用 `revision1/scripts/evaluate_s2_batch1_retrain.py::compute_nse_kge`，"
        "checkpoint→门控预测路径与其 `predict_all_gates` 一致。")
    add("")
    add("## 0. 输入清单与决定性校验")
    add("")
    add(f"- 发现 run：{len(inventory)} 个（`{'`, `'.join(r['run_id'] for r in inventory)}`）")
    add(f"- 批次目录：{', '.join(sorted({r['batch'] for r in inventory}))}"
        + ("；`batch2_results/` 当前不存在（脚本已支持自动发现）" if not (S2_DIR / "batch2_results").exists() else ""))
    add("- 规范流域轴：`data/gage_id.npy` 671 个流域（升序），按 `R1_014_CANDIDATE_SPLIT.json` 切分为 "
        "536 train / 135 holdout；所有 per-basin 数组按该规范轴对齐（脚本启动即断言）。")
    add("- 过程列顺序校验：`p_struct` 四列顺序 = `('w_phen','w_int','w_snow','w_sub')`，"
        "与 `revision1/strict/trainer.py:PROCESS_ORDER` 及各情景 `truth_specification.json` 活跃计数逐列一致（见 "
        "`stat/process_order_verification.json`）。")
    add("- X35 属性轴校验：训练集 `slope_mean` p40、`frac_forest`/`gvf_diff`/`aridity`/`frac_snow` 中位数与 "
        "`truth_specification.json` 阈值完全一致（见 `stat/baseline_regionalization.json`）。")
    add("")
    add("### 0.1 检测阈值 P 的协议文本与实现差异（明确记录，不做静默选择）")
    add("")
    add("冻结实现（`revision1/strict/trainer.py` 第 34 行 `PROCESS_COSTS = {w_phen: 2.0, w_int: 2.0, w_snow: 2.0, "
        "w_sub: 1.0}`）与冻结生成器（`revision1/scripts/r1_017_generate_formal_s2.py`，`P = lambda*c_p*(n_bar/n_b)`）"
        "以及冻结 evaluator archive 的 `P_matrix` 一致使用**按过程**的 P：`w_sub = 0.007`，"
        "`w_phen = w_int = w_snow = 0.014`（四个情景完全相同）。"
        "唯一携带统一 0.007 的是 `S2_PROTOCOL_v1.1_FROZEN.md` 第 4.2 节对自己公式 `P = lambda*c_p*(n̄/n_b)` 的"
        "简化数值示例（`0.007 x 1.0 x 1.0 = 0.007`）。因此这是**协议文档的简化表述**，不是实现偏差，"
        "评估的权威取值是按过程 P；本报告以 archive `P_matrix` 为主定义，并同时给出统一 P=0.007 的敏感性结果。"
        "Batch-1 既有结果同样使用按过程 P（未使用错误阈值）。")
    add("")

    # ---------------- B2 (H1) ----------------
    add("## 1. B2 — H1 统一指标（536 训练流域，best checkpoint；分母/计数全部显式）")
    add("")
    add("定义（按过程 P）：`High-D = T=1 且 D_J^train > P_p`；`Low-D = T=1 且 0 <= D_J <= P_p`；"
        "`Neg-D = T=1 且 D_J < 0`；`Specificity = P(pred closed | T=0)`；"
        "`Decidable BA = 0.5*(Recall_High-D + Specificity)`；"
        "`Full truth recovery (all T=1) = P(pred open | T=1)`（覆盖 High-D/Low-D/Neg-D）。"
        "Snow 全为 T=1，无 specificity/BA/AUC，报 recall 及 High-D/Low-D 分层。")
    add("")
    add("### 1.1 Scenario A 三种子（E1_A_42/43/44）逐过程")
    add("")
    add("| 过程 | 种子 | High-D recall (N) | Low-D shutdown (N) | Neg-D open (N) | Specificity (N) | Decidable BA | Full T=1 recall | AUC |")
    add("|---|---|---|---|---|---|---|---|---|")
    for seed in sorted({record["seed"] for record in h1.values() if record.get("scenario") == "A" and record["method"] == "Canonical_BCE"}):
        run_id = f"E1_A_{seed}"
        record = h1.get(run_id)
        if record is None:
            continue
        for process in PROCESSES:
            m = record["H1_best"][process]
            add(
                f"| {PROCESS_LABELS[process]} | {seed} | "
                f"{fmt_pct(m['recall_high_d'])} ({m['tp_high_d']}/{m['n_high_d']}) | "
                f"{fmt_pct(m['shutdown_rate_low_d'])} ({m['n_low_d']}) | "
                f"{fmt_pct(m['open_rate_neg_d'])} ({m['n_neg_d']}) | "
                f"{fmt_pct(m['specificity'])} ({m['n_t0_correct']}/{m['n_t0']}) | "
                f"**{fmt_pct(m['ba_decidable'])}** | {fmt_pct(m['full_truth_recovery_t1'])} | {fmt(m['auc'])} |"
            )
    add("")
    seed_a = sorted(record["seed"] for record in h1.values() if record.get("scenario") == "A" and record["method"] == "Canonical_BCE")
    add("三种子汇总（均值 ± 样本标准差）：")
    add("")
    add("| 过程 | High-D recall | Specificity | Decidable BA | Low-D shutdown | Full T=1 recall |")
    add("|---|---|---|---|---|---|")
    for process in PROCESSES:
        def collect(field: str) -> list[float]:
            values = []
            for seed in seed_a:
                metrics = h1[f"E1_A_{seed}"]["H1_best"][process]
                value = metrics[field]
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    values.append(float(value))
            return values

        def ms(values: list[float]) -> str:
            if not values:
                return "n/a"
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            return f"{fmt_pct(mean)} ± {fmt_pct(sd, 2)}"

        add(
            f"| {PROCESS_LABELS[process]} | {ms(collect('recall_high_d'))} | {ms(collect('specificity'))} | "
            f"**{ms(collect('ba_decidable'))}** | {ms(collect('shutdown_rate_low_d'))} | {ms(collect('full_truth_recovery_t1'))} |"
        )
    add("")
    add("### 1.2 全部 run 的 H1 Decidable BA（含 final checkpoint）")
    add("")
    add("| run | method | 情景 | 种子 | phase | Phenology | Interception | Snow(recall) | Subsurface |")
    add("|---|---|---|---|---|---|---|---|---|")
    for run_id in sorted(h1):
        record = h1[run_id]
        for phase in ("H1_best", "H1_final"):
            block = record.get(phase)
            if not block:
                continue
            cells = []
            for process in PROCESSES:
                m = block[process]
                value = m["ba_decidable"]
                cells.append(fmt_pct(value) if not (isinstance(value, float) and math.isnan(value)) else "n/a")
            add(
                f"| {run_id} | {record['method']} | {record['scenario']} | {record['seed']} | "
                f"{'best' if phase == 'H1_best' else 'final'} | " + " | ".join(cells) + " |"
            )
    add("")
    add("### 1.3 统一 P=0.007 敏感性（同批流域，仅阈值不同）")
    add("")
    add("| run | 过程 | High-D N (per-process P → uniform P) | Low-D N (per-process P → uniform P) | Decidable BA (per-process P → uniform P) |")
    add("|---|---|---|---|---|")
    for run_id in sorted(h1):
        record = h1[run_id]
        for process in PROCESSES:
            m = record["H1_best"][process]
            u = m["uniform_P_sensitivity"]
            add(
                f"| {run_id} | {PROCESS_LABELS[process]} | {m['n_high_d']} → {u['n_high_d']} | "
                f"{m['n_low_d']} → {u['n_low_d']} | {fmt_pct(m['ba_decidable'])} → {fmt_pct(u['ba_decidable'])} |"
            )
    add("")
    add("### 1.4 连续 D_J → 预测权重关系（E1_A_42，536 训练流域）")
    add("")
    add("| 过程 | T=1: Pearson(D_J,w) | T=1: Spearman | T=1 均权 | T=0: Pearson(D_J,w) | T=0 均权 | 四分位均权 (Q1→Q4) |")
    add("|---|---|---|---|---|---|---|")
    for process in PROCESSES:
        det = h1["E1_A_42"]["H1_best"][process]["detection"]
        quant = det.get("mean_weight_by_d_quantile")
        add(
            f"| {PROCESS_LABELS[process]} | {fmt(det['pearson_d_w_t1'], 3)} | {fmt(det['spearman_d_w_t1'], 3)} | "
            f"{fmt(det['mean_weight_t1'], 3)} | {fmt(det.get('pearson_d_w_t0'), 3)} | {fmt(det.get('mean_weight_t0'), 3)} | "
            + (" → ".join(fmt(v, 3) for v in quant) if quant else "n/a") + " |"
        )
    add("")
    add("符号约定：`Delta J_open = (2T-1) D_J - P`。T=1 时 `D_J` 是开启收益；T=0 时 `D_J` 是“开启”的损失，"
        "不是开启收益（负号已由 `(2T-1)` 体现），因此 T=0 行的相关性与 T=1 行不可直接互相解释为同一方向的“证据”。")
    add("")

    # ---------------- B1 ----------------
    add("## 2. B1 — H2 2x2 因子主效应（135 留出流域，配对 bootstrap 10,000 次）")
    add("")
    add(f"主效应定义：`ME_sub = (BA_A+BA_C)/2 - (BA_B+BA_D)/2`；`ME_phen = (BA_B+BA_C)/2 - (BA_A+BA_D)/2`；"
        f"`I = BA_C - BA_A - BA_B + BA_D`。每个 resample 内**同一组留出流域索引**同时用于四个情景，"
        f"统计单位是 {metrics.get('canonical_axis', {}).get('n_holdout', 135)} 个流域（绝不把 135x4 当作独立样本）。")
    add("")
    add("### 2.1 留出 BA 矩阵（种子 42；每情景每过程的 N 与 BA）")
    add("")
    add("> **修正记录（本节起）：** 先前版本的 §2.1 把 `h2_block` 中名为 `ba_unfiltered` 的字段当作留出 BA 展示，"
        "但该字段实际是**测试期 D_J 分层后的 decidable BA**（`0.5*(High-D recall + T=0 specificity)`），"
        "既与冻结协议 §5.2 的预注册定义不符，也与 §2.2 主效应不一致。"
        "现已改为：H2 主指标 = **不过滤 D 的 unfiltered 留出 BA**（`ba_holdout_unfiltered`，即 §2.1 主表）；"
        "D 分层 decidable BA 保留为 `ba_decidable_test_D_strata` 副表（仅描述性）。主效应（§2.2）数值未变。")
    add("")
    add("**主表（预注册 H2 主指标）：不过滤 D 的 holdout balanced accuracy**，"
        "= 对全部 135 个留出流域中所有 T=1 流域的 recall 与所有 T=0 流域的 specificity 取 `0.5*(recall+specificity)`；"
        "Snow 无 T=0 流域，故只报 recall（等同于 BA）。")
    add("")
    add("| 情景 | 过程 | N (pos/neg) | Holdout BA (unfiltered) | 95% CI |")
    add("|---|---|---|---|---|")
    for scenario in SCENARIOS:
        record = h1.get(f"E1_{scenario}_42")
        if record is None or not record.get("H2"):
            continue
        for process in PROCESSES:
            h2m = record["H2"][process]
            tpos = h2m["n_pos"]
            tneg = h2m["n_neg"]
            ba = h2m["ba_holdout_unfiltered"]
            add(f"| {scenario} | {PROCESS_LABELS[process]} | {tpos}/{tneg} | {fmt_pct(ba)} | see §2.2 |")
    add("")
    add("**副表（仅描述性，非 H2 主指标）：测试期 D_J 分层 decidable BA** "
        "（`ba_decidable_test_D_strata` = `0.5*(High-D recall [T=1 且 D_J^test > P] + specificity [T=0])`；"
        "Snow 无 T=0，故此列为 High-D recall）。该值通过 D 过滤丢弃了 Low-D / Neg-D 流域，因此**高于**主表数值，"
        "不得与主表或 §2.2 主效应混用。")
    add("")
    add("| 情景 | 过程 | Unfiltered BA (主指标) | D-stratified decidable BA (描述性) | High-D N | Low-D N | Neg-D N |")
    add("|---|---|---|---|---|---|---|")
    for scenario in SCENARIOS:
        record = h1.get(f"E1_{scenario}_42")
        if record is None or not record.get("H2"):
            continue
        for process in PROCESSES:
            h2m = record["H2"][process]
            add(
                f"| {scenario} | {PROCESS_LABELS[process]} | {fmt_pct(h2m['ba_holdout_unfiltered'])} | "
                f"{fmt_pct(h2m['ba_decidable_test_D_strata'])} | {h2m['n_high_d']} | {h2m['n_low_d']} | {h2m['n_neg_d']} |"
            )
    add("")
    add("### 2.2 因子主效应与交互（逐种子 + 种子合并）")
    add("")
    add("| 过程 | 效应 | 种子 | 点估计 (95% CI) | N |")
    add("|---|---|---|---|---|")
    for process in PROCESSES:
        for seed, per_process in sorted(factorial["per_seed"].items(), key=lambda item: int(item[0])):
            effects = per_process[process]["effects"]
            for name, label in (("me_attribute", "ME_attribute"), ("me_phenology", "ME_phenology"), ("interaction", "I_interaction")):
                block = effects[name]
                add(
                    f"| {PROCESS_LABELS[process]} | {label} | {seed} | "
                    f"{fmt_signed_rate_ci(block['point'], block['ci_low'], block['ci_high'])} | "
                    f"{block['n_holdout_basins']} |"
                )
        pooled = factorial["pooled"].get(process)
        if pooled:
            for name, label in (("me_attribute", "ME_attribute"), ("me_phenology", "ME_phenology"), ("interaction", "I_interaction")):
                block = pooled[name]
                add(
                    f"| {PROCESS_LABELS[process]} | {label} | pooled (n_seeds={pooled['n_seeds']}, {pooled['seed_status']}) | "
                    f"{fmt_signed_rate_ci(block['point'], block['ci_low'], block['ci_high'])} | "
                    f"{block['n_seeds']} seeds x {block['n_seeds']} |"
                )
    add("")
    add("种子合并口径：合并点估计 = 各完整种子效应的算术平均；合并 CI = 在同一流域 resample 内先取各种子效应再对种子取平均后的百分位区间；"
        "种子级离散度（样本标准差/min/max）见 `stat/h2_factorial_bootstrap.json`。")
    add("")
    add("### 2.3 与既有报告的对比（审计目标，不作为本报告结论）")
    add("")
    add("| 过程 | 既有 `Main_Effect_A_minus_B`（单对对比） | 既有 `Main_Effect_B_minus_A`（单对对比） | 本报告 ME_sub 因子（(A+C)/2-(B+D)/2） | 本报告 ME_phen 因子（(B+C)/2-(A+D)/2） |")
    add("|---|---|---|---|---|")
    for process in PROCESSES:
        audit = factorial["audit_of_earlier_single_contrasts"].get("42", {}).get(process)
        if not audit:
            continue
        add(
            f"| {PROCESS_LABELS[process]} | {audit['single_contrast_A_minus_B'] * 100:+.2f} | "
            f"{audit['single_contrast_B_minus_A'] * 100:+.2f} | "
            f"{audit['preregistered_me_attribute'] * 100:+.2f} | {audit['preregistered_me_phenology'] * 100:+.2f} |"
        )
    add("")
    add("说明：`scientific_evaluation_full.json` 的 `Main_Effect_A_minus_B = +0.2781`、`Main_Effect_B_minus_A = +0.0836` "
        "是**单对对比**（A−B、B−A），不是预注册的 2x2 主效应；逐过程的 ME_sub 因子列即（w_sub 的）地下径流属性主效应，"
        "ME_phen 因子列即（w_phen 的）物候属性主效应。`w_int` 行给出同一情景对比作用在截留 BA 上的交叉诊断（22 个留出流域为正类，"
        "功效极低，仅作参考）。")
    add("")
    add("### 2.4 H2 留出流域的 D_J^test 分层（描述性副表；H2 主体 BA 不过滤 D，见 §2.1 主表）")
    add("")
    add("本表仅列 D 分层描述性指标（分层定义与 §2.1 副表一致）；**不能**将其与 §2.1 主表或 §2.2 主效应混用。")
    add("")
    add("| run | 过程 | High-D N (recall) | Low-D N (shutdown) | Neg-D N (open) |")
    add("|---|---|---|---|---|")
    for run_id in sorted(h1):
        record = h1[run_id]
        if not record.get("H2"):
            continue
        for process in ("w_int", "w_snow"):
            m = record["H2"][process]
            add(
                f"| {run_id} | {PROCESS_LABELS[process]} | {m['n_high_d']} ({fmt_pct(m['recall_high_d'])}) | "
                f"{m['n_low_d']} ({fmt_pct(m['shutdown_rate_low_d'])}) | {m['n_neg_d']} ({fmt_pct(m['open_rate_neg_d'])}) |"
            )
    add("")

    # ---------------- B3 ----------------
    add("## 3. B3 — 区域化基线（冻结 536/135 划分，仅在训练流域拟合）")
    add("")
    add("| 情景 | 过程 | 方法 | N (pos/neg) | Holdout BA (95% CI) | 训练标签与真值一致率 |")
    add("|---|---|---|---|---|---|")
    for row in baselines["records"]:
        if row["approach"] == "all_approaches":
            add(
                f"| {row['scenario']} | {PROCESS_LABELS[row['process']]} | 全部方法不可用（单类） | "
                f"{row['n_holdout_pos']}/{row['n_holdout_neg']} | recall {fmt_pct(row['recall_holdout'])} | n/a |"
            )
            continue
        agreement = row.get("train_label_agreement_with_T")
        add(
            f"| {row['scenario']} | {PROCESS_LABELS[row['process']]} | {row['approach']} | "
            f"{row['n_holdout_pos']}/{row['n_holdout_neg']} | "
            f"{fmt_rate_ci(float(row['ba']), float(row['ci_low']), float(row['ci_high']))} | "
            f"{fmt_pct(agreement) if agreement not in ('', None) else 'n/a'} |"
        )
    add("")
    attribution = baselines.get("gap_attribution", {})
    if attribution.get("gap_steps"):
        steps = attribution["gap_steps"]
        add("### 3.1 ~78% 地下径流留出 BA 与真值监督 oracle 的差距归因（情景 A，w_sub，135 留出流域）")
        add("")
        add("| 环节 | Holdout BA (unfiltered, 135 流域) | 训练标签与真值 T 一致率 | 含义 |")
        add("|---|---|---|---|")
        add(f"| 学习器自身（E1_A_42） | {fmt_pct(attribution.get('learned_model_holdout_ba'))} | n/a | 冻结模型 |")
        add(f"| LR(学习器反事实标签 q>0.5) | {fmt_pct(attribution.get('LR_counterfactual_q_labels_ba'))} | "
            f"{fmt_pct(attribution.get('LR_counterfactual_q_labels_train_label_agreement_with_T'))} | 学习器实际监督信号 + 干净 X35 映射 |")
        add(f"| LR(真值 T oracle) | {fmt_pct(attribution.get('LR_truth_supervised_oracle_ba'))} | 100.0% | X35 区域化上限 |")
        add(f"| 3-NN donor（邻居真值 T） | {fmt_pct(attribution.get('3NN_donor_truth_T_ba'))} | n/a | 非参数区域化参照 |")
        add(f"| LR(教师 1[D_J>P] 标签，仅诊断) | {fmt_pct(attribution.get('LR_teacher_DeltaJ_class_labels_ba'))} | "
            f"{fmt_pct(attribution.get('LR_teacher_DeltaJ_class_labels_train_label_agreement_with_T'))} | 该标签不是 T=0 的真值代理（见下） |")
        add("")
        add("| 差距项 | 数值 | 含义 |")
        add("|---|---|---|")
        add(f"| 总差距 = oracle − learner | {fmt_pct(steps['total_oracle_minus_learned'], 2)} | 需拆解的总量 |")
        add(f"| ↳ 反事实证据标签差距 = oracle − LR(q 标签) | {fmt_pct(steps['counterfactual_evidence_label_gap'], 2)} | 反事实证据不等于真值的代价（映射类别固定为 X35-LR） |")
        add(f"| ↳ 学习器门控头差距 = LR(q 标签) − learner | {fmt_pct(steps['learner_gate_head_gap'], 2)} | 训练后 gate head 相对同一监督标签的干净线性映射的残差 |")
        add(f"| ↳ 区域化映射差距 = oracle − 3NN(真值) | {fmt_pct(steps['regionalization_mapping_gap_oracle_minus_3NN'], 2)} | ≈0 则说明 X35 区域化映射已饱和，不是瓶颈 |")
        add(f"| ↳ 诊断：oracle − LR(1[D_J>P]) | {fmt_pct(steps['teacher_DeltaJ_gt_P_label_gap_oracle_minus_teacherLR'], 2)} | 反例：1[D_J>P] 对 T=0 无效 |")
        add("")
        add("归因结论：真值监督下用同一 X35 特征做区域化的上限（oracle LR = 3-NN = "
            f"{fmt_pct(attribution.get('LR_truth_supervised_oracle_ba'))}）远高于学习器的 "
            f"{fmt_pct(attribution.get('learned_model_holdout_ba'))}，而 oracle − 3NN(真值) = "
            f"{fmt_pct(steps['regionalization_mapping_gap_oracle_minus_3NN'], 2)}，说明**区域化映射本身几乎不损失精度**；"
            "差距主要落在**反事实证据标签质量**（唯一输入差异）与**门控头拟合残差**两项上，具体数值见上表。")
        add("")
        strat = attribution.get("counterfactual_label_stratification", {})
        if strat.get("available"):
            add(f"学习器反事实标签（{strat['label_source']}）在训练区与真值 T 的总体一致率："
                f"**{fmt_pct(strat['overall_agreement_with_T'])}**，按 D 分层：")
            add("")
            add("| 分层 | N | 标签判“开启”比例 | 与真值一致率 |")
            add("|---|---|---|---|")
            for name, block in strat["by_stratum"].items():
                add(f"| {name} | {block['n']} | {fmt_pct(block['q_label_open_rate'])} | {fmt_pct(block['agreement_with_T'])} |")
            add("")
            add("T=0 的符号约定在这里是决定性的：对 T=0 流域，`D_J` 是“开启”造成的拟合损失，"
                "净收益为 `Delta J_open = -D_J - P < 0`，因此 `1[D_J > P]` 并不是“应当开启”的标签；"
                "用它做监督会把 T=0 中 D_J 偏大的流域系统性错标（上表诊断行的训练一致率仅 "
                f"{fmt_pct(attribution.get('LR_teacher_DeltaJ_class_labels_train_label_agreement_with_T'))}）。")
            add("")

    # ---------------- B4 ----------------
    add("## 4. B4 — 方法消融（情景 A 同批流域：Canonical BCE vs Direct-fit vs Plain BCE）")
    add("")
    add(ablation["note"])
    add("")
    add("### 4.1 H1 指标（536 训练流域）")
    add("")
    add("| 方法 | run | 过程 | Decidable BA | High-D recall (N) | Specificity (N) | Low-D shutdown (N) | 均权 | 权重 sd | 45-55% 不确定占比 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in ablation["rows"]:
        add(
            f"| {row['method']} | {row['run_id']} | {PROCESS_LABELS[row['process']]} | {fmt_pct(row['ba_decidable'])} | "
            f"{fmt_pct(row['recall_high_d'])} ({row['n_high_d']}) | {fmt_pct(row['specificity'])} ({row['n_t0']}) | "
            f"{fmt_pct(row['shutdown_rate_low_d'])} ({row['n_low_d']}) | {fmt(row['mean_weight'], 3)} | "
            f"{fmt(row['std_weight'], 3)} | {fmt_pct(row['frac_weight_0.45_0.55'])} |"
        )
    add("")
    add("### 4.2 留出指标（同一 135 留出流域，不过滤 D 的 unfiltered BA）与 NSE/KGE（见 §6）")
    add("")
    add("> 修正说明：本表“留出 BA”列为不过滤 D 的 `ba_holdout_unfiltered`（预注册 H2 主指标）；"
        "右侧 High-D / Low-D 两列仍为测试期 D 分层描述性指标（`recall_high_d` / `shutdown_rate_low_d`）。")
    add("")
    add("| run | 过程 | 留出 BA (unfiltered) | 留出 High-D recall (N) | 留出 Low-D shutdown (N) |")
    add("|---|---|---|---|---|")
    for run_id in sorted(h1):
        record = h1[run_id]
        if record["method"] == "Old_Interception" or not record.get("H2"):
            continue
        for process in PROCESSES:
            m = record["H2"][process]
            add(
                f"| {run_id} | {PROCESS_LABELS[process]} | {fmt_pct(m['ba_holdout_unfiltered'])} | "
                f"{fmt_pct(m['recall_high_d'])} ({m['n_high_d']}) | {fmt_pct(m['shutdown_rate_low_d'])} ({m['n_low_d']}) |"
            )
    add("")
    add("### 4.3 Interception / Snow 的 High-D / Low-D 分层行为（按过程 P）")
    add("")
    add("| run | 过程 | 区域 | High-D N (recall) | Low-D N (shutdown / open) | Low-D 均权 | Neg-D N (open) | T=0 N (specificity) |")
    add("|---|---|---|---|---|---|---|---|")
    for row in ablation["stratified_w_int_w_snow"]:
        add(
            f"| {row['run_id']} | {PROCESS_LABELS[row['process']]} | {row['region']} | "
            f"{row['n_high_d']} ({fmt_pct(row['recall_high_d'])}) | "
            f"{row['n_low_d']} ({fmt_pct(row['shutdown_rate_low_d'])} / {fmt_pct(row['open_rate_low_d'])}) | "
            f"{fmt(row['mean_weight_low_d'], 3)} | {row['n_neg_d']} ({fmt_pct(row['open_rate_neg_d'])}) | "
            f"{row['n_t0']} ({fmt_pct(row['specificity'])}) |"
        )
    add("")
    add("### 4.4 逐流域配对差异（同一流域索引集内配对 bootstrap，10,000 次）")
    add("")
    add("| A vs B | 区域 | 指标 | 过程 | N 配对 | A 均值 | B 均值 | 差值 (95% CI) |")
    add("|---|---|---|---|---|---|---|---|")
    paired_rows = ablation["paired_differences"]["rows"]
    if not paired_rows:
        add("| n/a | n/a | n/a | n/a | 0 | n/a | n/a | 未验证（缺少 NSE 缓存或改至全 671 门控） |")
    for row in paired_rows:
        add(
            f"| {row['run_a']} vs {row['run_b']} | {row['region']} | {row['metric']} | "
            f"{PROCESS_LABELS[row['process']] if row['process'] else '-'} | {row['n_paired']} | "
            f"{fmt(row['mean_a'], 3)} | {fmt(row['mean_b'], 3)} | "
            f"{row['difference'] * 100:+.2f} pt [{row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f}] |"
        )
    add("")
    add("### 4.5 跨种子稳定性")
    add("")
    stab = ablation["cross_seed_stability"]
    add(f"E1 情景 A 多种子族：seeds {stab['seeds']}（{stab['seed_status']}）")
    add("")
    add("| 过程 | 指标 | 均值 ± 样本 sd | min–max | n_seeds |")
    add("|---|---|---|---|---|")
    for key, block in stab.items():
        if "__" not in key:
            continue
        process, field = key.split("__")
        add(
            f"| {PROCESS_LABELS[process]} | {field} | {fmt(block['mean'], 4)} ± {fmt(block['sd'], 4)} | "
            f"{fmt(block['min'], 4)}–{fmt(block['max'], 4)} | {block['n_seeds']} |"
        )
    add("")
    add("| 方法 | run 列表 | 种子 | 状态 |")
    add("|---|---|---|---|")
    for method, block in ablation["groups"].items():
        add(f"| {method} | {', '.join(block['run_ids']) or 'n/a'} | {', '.join(map(str, block['seeds'])) or 'n/a'} | {block['seed_status']} |")
    add("")

    # ---------------- B5 ----------------
    add("## 5. B5 — 随机物候与留出 BA < 50% 的随机过程")
    add("")
    add("> 修正说明：本节的“out-of-sample holdout BA”统一使用不过滤 D 的 `ba_holdout_unfiltered`"
        "（预注册 H2 主指标）；先前的 D 分层 decidable BA（如情景 A 物候 46.7%）不再用作“holdout BA”。")
    add("")
    phen = random_phen["scenario_A_phenology_random"]
    add(f"- 情景 A 的物候为**随机分层**真值。In-sample（536 训练流域，D 分层 decidable BA）："
        f"**{fmt_pct(phen['in_sample_train_decidable_ba_mean'])} ± {fmt_pct(phen['in_sample_train_decidable_ba_sd'], 2)}**"
        f"（{phen['seed_status']}）；Out-of-sample（135 留出流域，不过滤 D 的 unfiltered BA）："
        f"**{fmt_pct(phen['out_of_sample_holdout_ba_mean'])} ± {fmt_pct(phen['out_of_sample_holdout_ba_sd'], 2)}**。")
    add("- 解释严格限定为：in-sample 结构恢复（模型能拟合训练流域上的随机门控）vs out-of-sample 区域可推广性"
        "（随机规则不携带可迁移的属性信号，留出 BA 回落至随机水平）。")
    add("")
    add("| run | 种子 | in-sample decidable BA | out-of-sample holdout BA |")
    add("|---|---|---|---|")
    for item in phen["per_run_train"]:
        hold = next((h for h in phen["per_run_holdout"] if h["run_id"] == item["run_id"]), None)
        add(
            f"| {item['run_id']} | {item['seed']} | {fmt_pct(item['ba_decidable_train'])} | "
            f"{fmt_pct(hold['ba_holdout']) if hold else 'n/a'} |"
        )
    add("")
    add("### 5.1 情景 D 的两个随机过程（留出 BA < 50%）：配对 bootstrap CI")
    add("")
    add("| 过程 | Holdout BA | 95% CI | 与 50% 的差 (95% CI) | 种子 | 状态 |")
    add("|---|---|---|---|---|---|")
    for row in scenario_d_random_rows(metrics):
        add(
            f"| {PROCESS_LABELS[row['process']]} | {fmt_pct(row['ba'])} | "
            f"[{fmt_pct(row['ci_low'])}, {fmt_pct(row['ci_high'])}] | "
            f"{fmt_signed_rate_ci(row['delta_point'], row['delta_low'], row['delta_high'])} | "
            f"{row['seeds']} | {row['status']} |"
        )
    add("")

    # ---------------- NSE ----------------
    add("## 6. 训练区/留出区 NSE/KGE（前向模拟，复用正式评估路径）")
    add("")
    if not nse_rows:
        add("**未验证（UNVERIFIED）**：本次运行未生成 NSE/KGE 缓存（缺少 `best_checkpoint.pt` 或未执行 `--stage nse`）。"
            "对应指标在 `R1_019_STAT_SUMMARY.json` 中标记为 `unverified`，不做猜测。")
    else:
        add("口径：训练区 = 536 训练流域 × 计分训练期（1981-10-01..1995-09-30，5,113 天）；"
            "留出区 = 135 留出流域 × 计分测试期（1996-09-30..2010-09-30，5,113 天）；"
            "观测 = `evaluator_archive.npz::Y_mm_d`（mm/d，噪声实现与冻结协议一致），模拟由 `best_checkpoint.pt` 前向积分得到；"
            "NSE/KGE 逐流域计算后取均值/中位数（`compute_nse_kge` 复用自 `evaluate_s2_batch1_retrain.py`）。")
        add("")
        add("| run | 方法 | 训练区 NSE 均值/中位数 | 训练区 KGE 均值/中位数 | 留出区 NSE 均值/中位数 | 留出区 KGE 均值/中位数 | 留出 NSE>0.5 流域数 |")
        add("|---|---|---|---|---|---|---|")
        for row in nse_rows:
            add(
                f"| {row['run_id']} | {row['method']} | {fmt(float(row['train_nse_mean']), 3)} / {fmt(float(row['train_nse_median']), 3)} | "
                f"{fmt(float(row['train_kge_mean']), 3)} / {fmt(float(row['train_kge_median']), 3)} | "
                f"{fmt(float(row['holdout_nse_mean']), 3)} / {fmt(float(row['holdout_nse_median']), 3)} | "
                f"{fmt(float(row['holdout_kge_mean']), 3)} / {fmt(float(row['holdout_kge_median']), 3)} | "
                f"{row['n_holdout_nse_gt_0.5']}/{row['n_holdout_basins']} |"
            )
        add("")
        add("同时校验：由 checkpoint 重新推导的 671 流域门控与 run 目录内 `p_struct_best_all671.npy` 的最大绝对差记录在 "
            "`stat/nse_kge_summary.csv::gate_max_abs_diff_vs_saved_all671`（逐位一致性检查，见 §7）。")
    add("")

    # ---------------- provenance ----------------
    add("## 7. 复算来源与可追溯性")
    add("")
    add("| 指标 | 来源文件 |")
    add("|---|---|")
    add("| H1 (536 训练流域) 门控 | `batch1_results/<RUN>/p_struct_best_train.npy`（best）/`p_struct_final_train.npy`（final） |")
    add("| H2 (135 留出流域) 门控 | `batch1_results/<RUN>/p_struct_best_all671.npy`（本次用 checkpoint 重算逐值校验） |")
    add("| T / D_J^train / D_J^test / P_matrix | `S2/data/scenario_<S>/evaluator_archive.npz` |")
    add("| 观测（NSE/KGE） | `S2/data/scenario_A/evaluator_archive.npz::Y_mm_d` |")
    add("| 反事实 q 标签（B3b） | `batch1_results/<RUN>/q_best_train.npy` |")
    add("| X35 属性（B3） | `data/camels_dataset`（第二项 attributes），列序与 `truth_specification.json` 索引一致 |")
    add("| 536/135 划分 | `revision1/analysis/R1_014_CANDIDATE_SPLIT.json` |")
    add("| 前向模拟权重 | `batch1_results/<RUN>/best_checkpoint.pt` |")
    add("")
    add("所有数值均由本报告的脚本从上述原始数组重算；未使用 `batch1_results_VOID_GENERATOR_BUG/`。")
    add("")
    add("## 8. 未验证项与风险")
    add("")
    add("- 情景 B/C/D 目前只有 seed 42（Batch-2 未出现），因此 2x2 因子只对 seed 42 完整；"
        "种子合并估计当前等于单种子估计并标记 `SEED-PENDING`；批次 2 出现后重跑脚本即自动纳入。")
    add("- E3/E4（Direct-fit / Plain BCE）目前只有 seed 42，跨种子稳定性标记 `SEED-PENDING`。")
    add("- 本报告不使用任何“先前的报告文字数值”作为输入。")
    add("")
    return "\n".join(lines) + "\n"


def scenario_d_random_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Paired-bootstrap CI for scenario D's two random processes (recomputed here from arrays)."""
    axis = load_canonical_axis()
    truth = load_scenario_truth("D", axis)
    run_dir = S2_DIR / "batch1_results/E1_D_42"
    if not run_dir.is_dir():
        return []
    gates = np.load(run_dir / "p_struct_best_all671.npy")
    rng = np.random.default_rng(BOOT_SEED)
    idx = paired_index_matrix(axis.n_holdout, N_BOOT, rng)
    rows = []
    for process in ("w_sub", "w_phen"):
        index = PROCESSES.index(process)
        t_bool = truth.T[axis.holdout_idx, index] == 1
        pred = gates[axis.holdout_idx, index] > GATE_THRESHOLD
        samples = ba_bootstrap(t_bool, pred, idx)
        point = _ba_point(t_bool, pred)
        delta = samples - 0.5
        low, high = pct_ci(samples)
        dlow, dhigh = pct_ci(delta)
        rows.append(
            {
                "process": process,
                "ba": point,
                "ci_low": low,
                "ci_high": high,
                "delta_point": point - 0.5,
                "delta_low": dlow,
                "delta_high": dhigh,
                "seeds": "42",
                "status": "SEED-PENDING",
            }
        )
    return rows


def build_summary(metrics: Mapping[str, Any], nse_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factorial = metrics["factorial"]
    pooled = factorial["pooled"]
    ablation = metrics["method_ablation"]
    baselines = metrics["baselines"]
    h1 = metrics["H1_best"]

    h1_headline: dict[str, Any] = {}
    scenario_a = {seed: h1[f"E1_A_{seed}"] for seed in ("42", "43", "44") if f"E1_A_{seed}" in h1}
    for process in PROCESSES:
        values = []
        for record in scenario_a.values():
            value = record["H1_best"][process]["ba_decidable"]
            if not (isinstance(value, float) and math.isnan(value)):
                values.append(float(value))
        if values:
            h1_headline[process] = {
                "metric": "scenario_A_decidable_BA",
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_seeds": len(values),
                "counts_source": "batch1_results/E1_A_{42,43,44}/p_struct_best_train.npy + scenario_A/evaluator_archive.npz",
            }
    h2_main_effects = {}
    for process, block in pooled.items():
        h2_main_effects[process] = {
            "effect_definitions": {
                "me_attribute": "(BA_A + BA_C)/2 - (BA_B + BA_D)/2 on the unfiltered holdout BA",
                "me_phenology": "(BA_B + BA_C)/2 - (BA_A + BA_D)/2 on the unfiltered holdout BA",
                "interaction": "BA_C - BA_A - BA_B + BA_D on the unfiltered holdout BA",
                "statistic": "paired basin bootstrap over the 135 holdout basins, 10,000 resamples, "
                             "one basin index set shared by all four scenarios within a resample",
            },
            "me_attribute": {k: block["me_attribute"][k] for k in ("point", "ci_low", "ci_high", "n_seeds", "seed_spread_sd")},
            "me_phenology": {k: block["me_phenology"][k] for k in ("point", "ci_low", "ci_high", "n_seeds", "seed_spread_sd")},
            "interaction": {k: block["interaction"][k] for k in ("point", "ci_low", "ci_high", "n_seeds", "seed_spread_sd")},
            "seeds": block["seeds"],
            "seed_status": block["seed_status"],
        }
    nse_headline = None
    if nse_rows:
        nse_headline = {
            row["run_id"]: {
                "train_nse_mean": float(row["train_nse_mean"]),
                "holdout_nse_mean": float(row["holdout_nse_mean"]),
                "holdout_kge_mean": float(row["holdout_kge_mean"]),
            }
            for row in nse_rows
        }
    gaps = baselines.get("gap_attribution", {})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task": "R1-019-STAT (LANE B): independent statistical re-analysis of frozen S2 v1.1 batch-1",
        "input_provenance": {
            "learned_gates": "revision1/S2/batch1_results/<RUN>/p_struct_*_train.npy, p_struct_best_all671.npy",
            "truth": "revision1/S2/data/scenario_<S>/evaluator_archive.npz (T, D_J_train, D_J_test, P_matrix, Y_mm_d)",
            "split": "revision1/analysis/R1_014_CANDIDATE_SPLIT.json (536/135)",
            "attributes": "data/camels_dataset X35 (column order verified against truth_specification.json)",
        },
        "process_threshold_P": {
            "w_sub": 0.007,
            "w_phen": 0.014,
            "w_int": 0.014,
            "w_snow": 0.014,
            "statement": (
                "Per-process P from the frozen evaluator P_matrix (lambda*c_p, PROCESS_COSTS) is authoritative; "
                "the uniform P=0.007 carried only by S2_PROTOCOL_v1.1_FROZEN.md 4.2 prose is a simplified "
                "numeric example of the same formula and is reported as a sensitivity variant only."
            ),
        },
        "h2_main_effects": h2_main_effects,
        "h2_holdout_ba_unfiltered_scenario_matrix_seed42": {
            scenario: {process: h1[f"E1_{scenario}_42"]["H2"][process]["ba_holdout_unfiltered"] for process in PROCESSES}
            for scenario in SCENARIOS
            if f"E1_{scenario}_42" in h1 and h1[f"E1_{scenario}_42"].get("H2")
        },
        "h2_holdout_ba_decidable_test_D_strata_matrix_seed42": {
            scenario: {process: h1[f"E1_{scenario}_42"]["H2"][process]["ba_decidable_test_D_strata"] for process in PROCESSES}
            for scenario in SCENARIOS
            if f"E1_{scenario}_42" in h1 and h1[f"E1_{scenario}_42"].get("H2")
        },
        "post_review_fixes": [
            {
                "field": "H2 holdout BA",
                "was": (
                    "stat/h2_per_run_metrics.csv JSON key 'ba_unfiltered' held the D_J^test-stratified "
                    "decidable BA (0.5*(High-D recall + T=0 specificity)) instead of the preregistered "
                    "unfiltered holdout BA"
                ),
                "now": (
                    "'ba_holdout_unfiltered' = 0.5*(recall over all T=1 holdout basins + specificity over all "
                    "T=0 holdout basins), no D_J stratification (preregistered H2 metric); the D-stratified "
                    "value is kept separately as 'ba_decidable_test_D_strata' (descriptive only)"
                ),
                "h2_main_effects_unchanged": True,
            }
        ],
        "corrected_vs_reported_h2": {
            "reported_single_contrast_Subsurface_A_minus_B": 0.27808600509749937,
            "reported_single_contrast_Phenology_B_minus_A": 0.08362598770851626,
            "reported_source": "revision1/S2/batch1_results/scientific_evaluation_full.json (audit target only)",
            "preregistered_me_attribute_w_sub": pooled["w_sub"]["me_attribute"]["point"],
            "preregistered_me_phenology_w_phen": pooled["w_phen"]["me_phenology"]["point"],
        },
        "h1_headline_scenario_A": h1_headline,
        "baseline_gap_attribution_scenario_A_w_sub": gaps,
        "method_ablation": {method: block for method, block in ablation["groups"].items()},
        "nse_kge_headline": nse_headline,
        "unverified_or_pending": [
            "Scenario B/C/D have only seed 42 in batch1_results (batch2_results/ absent): factorial pooled estimate is SEED-PENDING.",
            "E3/E4 cross-seed stability is SEED-PENDING (single seed 42).",
            "NSE/KGE are marked unverified if stat/nse_kge_summary.csv is absent (no best_checkpoint.pt or --stage nse not run).",
        ],
        "artifacts": {
            "script": str(HERE),
            "h1_metrics": str(STAT_DIR / "h1_per_run_metrics.csv"),
            "h2_metrics": str(STAT_DIR / "h2_per_run_metrics.csv"),
            "h2_factorial": str(STAT_DIR / "h2_factorial_bootstrap.csv"),
            "baselines": str(STAT_DIR / "baseline_regionalization.csv"),
            "ablation": str(STAT_DIR / "method_ablation.csv"),
            "nse_kge": str(STAT_DIR / "nse_kge_summary.csv"),
            "report": str(REPORT_PATH),
        },
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="R1-019 STAT independent S2 v1.1 re-analysis")
    parser.add_argument("--stage", choices=("all", "metrics", "nse", "report"), default="all")
    parser.add_argument("--force-nse", action="store_true", help="recompute NSE even if a matching cache exists")
    args = parser.parse_args()

    STAT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = None
    if args.stage in ("all", "nse"):
        # NSE runs first so that the metrics stage can use the per-basin NSE/KGE caches for the
        # ablation paired differences and so that missing batch-2 all-671 gates get materialised.
        print("[r1_019_stat] stage=nse")
        axis = load_canonical_axis()
        runs = discover_runs()
        result = nse_stage(runs, axis, force=args.force_nse)
        print(f"[r1_019_stat] NSE/KGE cache updated for {len(result['summary_rows'])} runs")
    if args.stage in ("all", "metrics"):
        print("[r1_019_stat] stage=metrics")
        metrics = build_metrics_stage()
        print(f"[r1_019_stat] wrote metrics artifacts to {STAT_DIR}")
    if args.stage in ("all", "report"):
        print("[r1_019_stat] stage=report")
        summary = report_stage(metrics)
        print(f"[r1_019_stat] wrote {REPORT_PATH} and {SUMMARY_PATH}")
        print(json.dumps({k: summary[k] for k in ("h2_main_effects",)}, indent=2)[:2000])


if __name__ == "__main__":
    main()
