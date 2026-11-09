"""Candidate-only S2 truth allocation, target construction, and D diagnostics."""
from __future__ import annotations

from typing import Callable
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

from train.synthetic.noise import lognormal_ar1_multipliers

PROCESSES = ("w_phen", "w_int", "w_snow", "w_sub")
FEATURES = {"w_phen": "aridity", "w_int": "frac_forest", "w_snow": "frac_snow", "w_sub": "soil_porosity"}


def allocate_truth(basin_ids: np.ndarray, attributes: np.ndarray, attribute_names: list[str]) -> np.ndarray:
    """Allocate four time-invariant gates by ascending (feature, basin ID) ranks."""
    ids = np.asarray(basin_ids)
    attrs = np.asarray(attributes)
    if ids.ndim != 1 or attrs.ndim != 2 or len(ids) != len(attrs) or len(np.unique(ids)) != len(ids):
        raise ValueError("basin IDs and attributes must align and IDs must be unique")
    if len(set(attribute_names)) != len(attribute_names):
        raise ValueError("attribute_names contains duplicates")
    truth = np.zeros((len(ids), len(PROCESSES)), dtype=np.uint8)
    for col, process in enumerate(PROCESSES):
        feature = FEATURES[process]
        if feature not in attribute_names:
            raise ValueError(f"missing truth feature {feature}")
        values = attrs[:, attribute_names.index(feature)].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite truth feature {feature}")
        order = np.lexsort((ids, values))
        truth[order[len(ids) // 2:], col] = 1
    return truth


def candidate_spatial_split(basin_ids: np.ndarray, attributes: np.ndarray,
                            attribute_names: list[str], *, seed: int = 20261017) -> dict:
    """Create the contract's deterministic four-stratum 80/20 basin split."""
    from sklearn.model_selection import train_test_split
    ids = np.asarray(basin_ids)
    attrs = np.asarray(attributes)
    if ids.ndim != 1 or attrs.ndim != 2 or len(ids) != len(attrs) or len(np.unique(ids)) != len(ids):
        raise ValueError("basin IDs and attributes must align and IDs must be unique")
    bins = []
    for feature in ("aridity", "frac_snow"):
        if feature not in attribute_names:
            raise ValueError(f"missing split feature {feature}")
        values = attrs[:, attribute_names.index(feature)].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite split feature {feature}")
        order = np.lexsort((ids, values))
        ranked = np.zeros(len(ids), dtype=np.uint8)
        ranked[order[len(ids) // 2:]] = 1
        bins.append(ranked)
    strata = 2 * bins[0] + bins[1]
    train_ids, holdout_ids = train_test_split(ids, test_size=0.2, random_state=seed,
                                              stratify=strata)
    train_ids = np.sort(train_ids)
    holdout_ids = np.sort(holdout_ids)
    if len(train_ids) != 536 or len(holdout_ids) != 135:
        raise ValueError("candidate split did not yield the specified 536/135 partition")
    stratum_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(strata, return_counts=True))}
    id_text = lambda values: [int(x) for x in values]
    digest = lambda values: hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()
    source_ids = id_text(ids)
    train_ids_list = id_text(train_ids)
    holdout_ids_list = id_text(holdout_ids)
    return {"status": "CANDIDATE_DIAGNOSTIC_ONLY", "fold_id": "r1_014_candidate_s2", "seed": seed,
            "method": "sklearn.model_selection.train_test_split(test_size=0.2, random_state=20261017, stratify=stratum)",
            "stratum_formula": "2*aridity_median_bin + frac_snow_median_bin",
            "stratum_counts": stratum_counts, "train_count": len(train_ids),
            "holdout_count": len(holdout_ids),
            "source_ids": source_ids, "train_ids": train_ids_list, "holdout_ids": holdout_ids_list,
            "ordered_ids": source_ids, "ordered_train_ids": train_ids_list,
            "ordered_holdout_ids": holdout_ids_list,
            "source_ids_sha256": digest(ids), "train_ids_sha256": digest(train_ids),
            "holdout_ids_sha256": digest(holdout_ids),
            "train_ids_sha256_little_endian_int64": digest(train_ids),
            "holdout_ids_sha256_little_endian_int64": digest(holdout_ids),
            "numpy_version": np.__version__, "scikit_learn_version": __import__("sklearn").__version__}


def write_split_manifest(split: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")


def synthesize_observations(q_true: np.ndarray, basin_ids: np.ndarray, *, seed: int = 20261017) -> np.ndarray:
    """Multiply internal teacher streamflow (mm/day) by common noise; preserve units."""
    q = np.asarray(q_true, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != len(basin_ids):
        raise ValueError("q_true must have shape [basin, day]")
    multipliers = lognormal_ar1_multipliers(basin_ids, q.shape[1], seed=seed)
    return q * multipliers


def write_loader_dataset(forcings: np.ndarray, observations: np.ndarray, attributes: np.ndarray,
                         output_path: str | Path) -> None:
    """Write only HydroLoader's (forcings, target, attributes) schema."""
    forcings, observations, attributes = map(np.asarray, (forcings, observations, attributes))
    if forcings.ndim != 3 or observations.shape != (*forcings.shape[:2], 1):
        raise ValueError("expected forcings [basin, day, feature], target [basin, day, 1]")
    if attributes.ndim != 2 or len(attributes) != len(forcings):
        raise ValueError("attributes must align with basin axis")
    target = observations.astype(np.float32, copy=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("wb") as stream:
        pickle.dump((forcings, target, attributes), stream, protocol=pickle.HIGHEST_PROTOCOL)


def nse_contrast(q_true: np.ndarray, q_flip: np.ndarray, target: np.ndarray,
                 mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed NSE(q_true)-NSE(q_flip), validity, and support per basin."""
    q_true, q_flip, target = map(lambda x: np.asarray(x, dtype=np.float64), (q_true, q_flip, target))
    mask = np.asarray(mask, dtype=bool)
    if q_true.shape != q_flip.shape or q_true.shape != target.shape or q_true.shape != mask.shape or q_true.ndim != 2:
        raise ValueError("prediction, target and mask arrays must share [basin, day] shape")
    valid = mask & np.isfinite(q_true) & np.isfinite(q_flip) & np.isfinite(target)
    support = valid.sum(axis=1)
    out = np.full(q_true.shape[0], np.nan, dtype=np.float64)
    for b in range(q_true.shape[0]):
        keep = valid[b]
        if support[b] < 2:
            continue
        denom = np.sum((target[b, keep] - np.mean(target[b, keep])) ** 2)
        if not np.isfinite(denom) or denom <= 0:
            continue
        nse_true = 1.0 - np.sum((q_true[b, keep] - target[b, keep]) ** 2) / denom
        nse_flip = 1.0 - np.sum((q_flip[b, keep] - target[b, keep]) ** 2) / denom
        out[b] = nse_true - nse_flip
    return out, np.isfinite(out), support


def evaluate_truth_flips(simulate: Callable[[np.ndarray], np.ndarray], truth: np.ndarray,
                         observations: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Run truth plus four one-gate flips with common input/noise/mask.

    simulate accepts [basin, process] binary gates and returns [basin, day]
    mm/day predictions. Evaluator arrays are separate from the loader pickle.
    """
    truth = np.asarray(truth, dtype=np.uint8)
    if truth.ndim != 2 or truth.shape[1] != 4:
        raise ValueError("truth must have shape [basin, 4]")
    q0 = np.asarray(simulate(truth), dtype=np.float64)
    if q0.shape != observations.shape:
        raise ValueError("simulator output and observations do not align")
    d0, v0, s0 = [], [], []
    d, v, s = [], [], []
    for process in range(4):
        flipped = truth.copy()
        flipped[:, process] = 1 - flipped[:, process]
        qf = np.asarray(simulate(flipped), dtype=np.float64)
        # Both terms in each contrast use the identical case mask.
        noiseless_mask = np.asarray(mask, dtype=bool) & np.isfinite(q0) & np.isfinite(qf)
        noisy_mask = noiseless_mask & np.isfinite(observations)
        a, av, ass = nse_contrast(q0, qf, q0, noiseless_mask)
        b, bv, bs = nse_contrast(q0, qf, observations, noisy_mask)
        d0.append(a); v0.append(av); s0.append(ass)
        d.append(b); v.append(bv); s.append(bs)
    return {"q_true": q0, "D0": np.stack(d0, axis=1), "D": np.stack(d, axis=1),
            "D0_valid": np.stack(v0, axis=1), "D_valid": np.stack(v, axis=1),
            "D0_support": np.stack(s0, axis=1), "D_support": np.stack(s, axis=1)}
