"""Candidate-only S2 Strict loader adapter; never exposes evaluator truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fold import SpatialFold, load_strict_fold


def load_candidate_fold_manifest(path: Path) -> SpatialFold:
    """Load A's frozen candidate split; require explicit, ordered ID vectors."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = manifest.get("source_ids", manifest.get("ordered_ids"))
    train_ids = manifest.get("train_ids")
    holdout_ids = manifest.get("holdout_ids")
    if ids is None or train_ids is None or holdout_ids is None:
        raise ValueError("candidate split manifest must contain source_ids, train_ids, and holdout_ids")
    fold = SpatialFold.create(str(manifest.get("fold_id", "r1_014_candidate_s2")), ids, train_ids, holdout_ids)
    for name, observed in (("source_ids_sha256", fold.source_ids_sha256), ("train_ids_sha256", fold.train_ids_sha256), ("holdout_ids_sha256", fold.holdout_ids_sha256)):
        expected = manifest.get(name)
        if expected is not None and expected != observed:
            raise ValueError(f"candidate split {name} does not match ordered IDs")
    return fold


def load_candidate_strict_fold(
    config: Mapping[str, Any], *, manifest_path: Path, cache_root: Path,
    dataset_path: Path, gage_info_path: Path, hydro_loader_source: Path,
):
    """Route synthetic runoff through the existing train-fitted Strict loader."""
    fold = load_candidate_fold_manifest(manifest_path)
    with Path(dataset_path).open("rb") as handle:
        import pickle
        dataset = pickle.load(handle)
    if not isinstance(dataset, (tuple, list)) or len(dataset) != 3:
        raise ValueError("candidate dataset must use HydroLoader's (forcings, target, attributes) schema")
    forcing, target, attributes = (np.asarray(value) for value in dataset)
    source_ids = np.load(gage_info_path).astype(np.int64, copy=False)
    if tuple(map(int, source_ids)) != fold.source_ids:
        raise ValueError("candidate fold source IDs differ from canonical gage_info order")
    if forcing.shape[0] != len(source_ids) or target.shape[0] != len(source_ids) or attributes.shape[0] != len(source_ids):
        raise ValueError("candidate dataset basin axes do not match canonical IDs")
    if target.ndim not in (2, 3) or target.shape[1] != forcing.shape[1]:
        raise ValueError("candidate target time axis does not match forcings")
    return load_strict_fold(config, fold, cache_root=cache_root, data_path=Path(dataset_path),
                            gage_info_path=gage_info_path, hydro_loader_source=hydro_loader_source)
