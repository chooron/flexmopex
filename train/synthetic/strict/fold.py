"""Train-only spatial folds and train-fitted HydroLoader adapter.

Deliberately independent of ``run_model.py`` and ``CFTrainer`` so the synthetic
learner cannot construct a fold from holdout basins. HydroLoader, the model
implementation and the losses are the same project components used elsewhere.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from dmg.core.data.loaders.hydro_loader import HydroLoader


class StrictFoldError(ValueError):
    """Raised when a fold or its cached preprocessing identity is invalid."""



FLEXMOPEX_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = FLEXMOPEX_ROOT / "synthetic"
STRICT_RUN_ROOT = (REVISION_ROOT / "strict" / "runs").resolve()


def _validate_cache_root(cache_root: Path) -> Path:
    resolved = cache_root.resolve()
    if FLEXMOPEX_ROOT == resolved or FLEXMOPEX_ROOT in resolved.parents:
        if STRICT_RUN_ROOT not in resolved.parents:
            raise StrictFoldError(
                "in-package caches must be unique children of synthetic/strict/runs"
            )
    return resolved

def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_source_slice_sha256(data_path: Path, source_ids: np.ndarray, train_ids: np.ndarray) -> str:
    """Hash raw forcing/target/attribute rows for train IDs only, in canonical order."""
    with data_path.open("rb") as handle:
        forcings, target, attributes = pickle.load(handle)
    arrays = tuple(np.asarray(value) for value in (forcings, target, attributes))
    if any(array.ndim < 2 or int(array.shape[0]) != len(source_ids) for array in arrays):
        raise StrictFoldError("raw CAMELS arrays do not share the declared source basin axis")
    source_index = {int(basin_id): index for index, basin_id in enumerate(source_ids)}
    indices = [source_index[int(basin_id)] for basin_id in train_ids]
    digest = hashlib.sha256()
    metadata = {
        "train_ids_sha256": ordered_ids_sha256(train_ids),
        "array_shapes": [list(array.shape) for array in arrays],
        "array_dtypes": [array.dtype.str for array in arrays],
    }
    digest.update(_canonical_json(metadata))
    for array in arrays:
        for index in indices:
            row = np.ascontiguousarray(array[index])
            digest.update(memoryview(row).cast("B"))
    return digest.hexdigest()


def ordered_ids_sha256(ids: np.ndarray) -> str:
    values = np.asarray(ids, dtype="<i8")
    return _sha256_bytes(values.tobytes(order="C"))


def _as_ids(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise StrictFoldError(f"{label} must be a one-dimensional integer ID array")
    array = array.astype(np.int64, copy=False)
    if array.size == 0:
        raise StrictFoldError(f"{label} must not be empty")
    if len(np.unique(array)) != len(array):
        raise StrictFoldError(f"{label} contains duplicate IDs")
    return np.sort(array.copy())


@dataclass(frozen=True)
class SpatialFold:
    """A complete, immutable basin partition with ID-canonical local axes."""

    fold_id: str
    source_ids: tuple[int, ...]
    train_ids: tuple[int, ...]
    holdout_ids: tuple[int, ...]
    source_ids_sha256: str
    train_ids_sha256: str
    holdout_ids_sha256: str

    @classmethod
    def create(
        cls,
        fold_id: str,
        source_ids: Any,
        train_ids: Any,
        holdout_ids: Any,
    ) -> "SpatialFold":
        if not isinstance(fold_id, str) or not fold_id.strip():
            raise StrictFoldError("fold_id must be a non-empty string")
        source_raw = np.asarray(source_ids)
        if source_raw.ndim != 1 or source_raw.dtype.kind not in "iu":
            raise StrictFoldError("source_ids must be a one-dimensional integer ID array")
        source = source_raw.astype(np.int64, copy=False)
        if len(np.unique(source)) != len(source):
            raise StrictFoldError("source_ids contains duplicate IDs")
        train = _as_ids(train_ids, "train_ids")
        holdout = _as_ids(holdout_ids, "holdout_ids")
        source_set = set(map(int, source))
        train_set = set(map(int, train))
        holdout_set = set(map(int, holdout))
        if train_set & holdout_set:
            raise StrictFoldError("train_ids and holdout_ids overlap")
        if train_set | holdout_set != source_set:
            unknown = sorted((train_set | holdout_set) - source_set)
            missing = sorted(source_set - (train_set | holdout_set))
            raise StrictFoldError(f"fold does not partition source IDs; unknown={unknown[:5]}, missing={missing[:5]}")
        if not train_set or not holdout_set:
            raise StrictFoldError("train and holdout partitions must both be non-empty")
        return cls(
            fold_id=fold_id.strip(),
            source_ids=tuple(map(int, source)),
            train_ids=tuple(map(int, train)),
            holdout_ids=tuple(map(int, holdout)),
            source_ids_sha256=ordered_ids_sha256(source),
            train_ids_sha256=ordered_ids_sha256(train),
            holdout_ids_sha256=ordered_ids_sha256(holdout),
        )

    def mapping(self) -> dict[str, dict[int, int]]:
        """Return explicit source-axis and split-local ID-to-index maps."""
        return {
            "source": {basin_id: index for index, basin_id in enumerate(self.source_ids)},
            "train": {basin_id: index for index, basin_id in enumerate(self.train_ids)},
            "holdout": {basin_id: index for index, basin_id in enumerate(self.holdout_ids)},
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "source_ids": list(self.source_ids),
            "train_ids": list(self.train_ids),
            "holdout_ids": list(self.holdout_ids),
            "source_ids_sha256": self.source_ids_sha256,
            "train_ids_sha256": self.train_ids_sha256,
            "holdout_ids_sha256": self.holdout_ids_sha256,
            "axis_order": "ascending_numeric_basin_id",
        }


@dataclass
class FoldDatasets:
    fold: SpatialFold
    train_dataset: dict[str, Any]
    holdout_dataset: dict[str, Any]
    normalization_stats: dict[str, Any]
    cache_identity: dict[str, Any]
    cache_key: str
    stats_path: Path
    source_file_sha256: str
    source_train_slice_sha256: str


class _NoRefitHydroLoader(HydroLoader):
    """Holdout loader guard: reject any fallback to fitting statistics."""

    def _init_norm_stats(self, x_nn, c_nn, target):  # type: ignore[no-untyped-def]
        raise StrictFoldError(
            "holdout HydroLoader attempted to fit normalization statistics; train cache is incomplete or mismatched"
        )


def _config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    observations = config.get("observations", {})
    model = config.get("model", {})
    nn = model.get("nn", {})
    phy = model.get("phy", {})
    train = config.get("train", {})
    return {
        "observations": {
            "name": observations.get("name"),
            "data_source_role": "camels_671",
            "gage_info": str(observations.get("gage_info", "")),
            "target_unit": observations.get("target_unit"),
            "area_name": observations.get("area_name"),
            "all_forcings": observations.get("all_forcings"),
            "all_attributes": observations.get("all_attributes"),
        },
        "train_time": config.get("train_time"),
        "all_time": config.get("all_time"),
        "target": train.get("target"),
        "nn_forcings": nn.get("forcings"),
        "nn_attributes": nn.get("attributes"),
        "phy_forcings": phy.get("forcings"),
        "phy_attributes": phy.get("attributes"),
        "use_log_norm": model.get("use_log_norm"),
        "flow_regime": model.get("flow_regime"),
        "norm_target": model.get("norm_target"),
    }


def _attach_doy(dataset: dict[str, Any], start_date: str) -> None:
    start = datetime.strptime(start_date, "%Y/%m/%d")
    n_days = int(dataset["x_phy"].shape[0])
    days = [float((start + timedelta(days=i)).timetuple().tm_yday) for i in range(n_days)]
    dataset["doy"] = torch.tensor(days, dtype=torch.float32).view(n_days, 1, 1).repeat(
        1, int(dataset["x_phy"].shape[1]), 1
    )


def _subset_config(
    base_config: Mapping[str, Any],
    *,
    subset_path: Path,
    model_dir: Path,
    pretrained_model_dir: Path | None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["mode"] = "train"
    config["device"] = "cpu"
    config["dtype"] = torch.float32
    config["model_dir"] = str(model_dir.resolve())
    config["model_path"] = config["model_dir"]
    config["pretrained_model_dir"] = (
        str(pretrained_model_dir.resolve()) if pretrained_model_dir is not None else None
    )
    config["observations"] = copy.deepcopy(config["observations"])
    config["observations"]["subset_path"] = str(subset_path.resolve())
    return config


def _required_stats_keys(config: Mapping[str, Any]) -> set[str]:
    observations = config["observations"]
    model = config["model"]
    train = config["train"]
    return {
        *map(str, model["nn"].get("forcings", [])),
        *map(str, model["nn"].get("attributes", [])),
        *map(str, train.get("target", [])),
    }


def _assert_dataset_axis(dataset: Mapping[str, Any], expected_ids: np.ndarray, label: str) -> None:
    if "x_phy" not in dataset or "target" not in dataset or "xc_nn_norm" not in dataset:
        raise StrictFoldError(f"{label} dataset is missing required model tensors")
    for key in ("x_phy", "xc_nn_norm", "target"):
        value = dataset[key]
        if not torch.is_tensor(value) or value.ndim != 3 or int(value.shape[1]) != len(expected_ids):
            raise StrictFoldError(f"{label}.{key} basin axis does not match its ID vector")
    if "doy" not in dataset or int(dataset["doy"].shape[1]) != len(expected_ids):
        raise StrictFoldError(f"{label} day-of-year axis does not match its ID vector")


def verify_cache_identity(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Reject cache reuse unless all identity fields match exactly."""
    if _canonical_json(dict(stored)) != _canonical_json(dict(expected)):
        raise StrictFoldError("normalization cache identity mismatch; refusing reuse")


def load_strict_fold(
    base_config: Mapping[str, Any],
    fold: SpatialFold,
    *,
    cache_root: Path,
    data_path: Path,
    gage_info_path: Path,
    hydro_loader_source: Path,
) -> FoldDatasets:
    """Load physically separate train/holdout datasets with train-only stats.

    `cache_root` is an explicit temporary location for validation or a unique
    revision1/strict run cache for a future approved run. Existing cache entries
    are reused only after exact identity and statistics-hash verification.
    """
    cache_root = _validate_cache_root(cache_root)
    data_path = data_path.resolve()
    gage_info_path = gage_info_path.resolve()
    hydro_loader_source = hydro_loader_source.resolve()
    for path in (data_path, gage_info_path, hydro_loader_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_ids = np.load(gage_info_path).astype(np.int64, copy=False)
    if tuple(map(int, source_ids)) != fold.source_ids:
        raise StrictFoldError("fold source ID order differs from the gage_info file")
    train_ids = np.asarray(fold.train_ids, dtype=np.int64)
    holdout_ids = np.asarray(fold.holdout_ids, dtype=np.int64)

    config = copy.deepcopy(dict(base_config))
    config["observations"] = copy.deepcopy(config["observations"])
    config["observations"]["data_path"] = str(data_path)
    config["observations"]["gage_info"] = str(gage_info_path)
    projection = _config_projection(config)
    source_file_sha256 = sha256_file(data_path)
    source_train_slice_sha256 = _train_source_slice_sha256(data_path, source_ids, train_ids)
    identity = {
        "fold_id": fold.fold_id,
        "source_train_slice_sha256": source_train_slice_sha256,
        "source_gage_info_sha256": sha256_file(gage_info_path),
        "source_ids_sha256": fold.source_ids_sha256,
        "train_ids_sha256": fold.train_ids_sha256,
        "holdout_ids_sha256": fold.holdout_ids_sha256,
        "train_ids": list(fold.train_ids),
        "holdout_ids": list(fold.holdout_ids),
        "config_projection_sha256": _sha256_bytes(_canonical_json(projection)),
        "config_projection": projection,
        "hydro_loader_source_sha256": sha256_file(hydro_loader_source),
        "loader": f"{HydroLoader.__module__}.{HydroLoader.__name__}",
        "axis_order": "ascending_numeric_basin_id",
    }
    cache_key = _sha256_bytes(_canonical_json(identity))
    fold_root = cache_root / cache_key
    train_dir = fold_root / "train_norm"
    holdout_model_dir = fold_root / "holdout_loader"
    identity_path = fold_root / "normalization_identity.json"
    stats_path = train_dir / "normalization_statistics.json"
    train_ids_path = fold_root / "train_ids.json"
    holdout_ids_path = fold_root / "holdout_ids.json"

    fold_root.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    holdout_model_dir.mkdir(parents=True, exist_ok=True)
    train_ids_path.write_text(json.dumps(list(fold.train_ids)) + "\n", encoding="utf-8")
    holdout_ids_path.write_text(json.dumps(list(fold.holdout_ids)) + "\n", encoding="utf-8")

    reuse = identity_path.is_file()
    if reuse:
        stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        verify_cache_identity(stored_identity["identity"], identity)
        if not stats_path.is_file():
            raise StrictFoldError("cache identity exists but train normalization statistics are missing")
        if sha256_file(stats_path) != stored_identity.get("normalization_statistics_sha256"):
            raise StrictFoldError("cached normalization statistics hash mismatch")
    elif stats_path.exists():
        raise StrictFoldError("unbound normalization cache exists; refusing to use it")

    train_cfg = _subset_config(
        config,
        subset_path=train_ids_path,
        model_dir=train_dir,
        pretrained_model_dir=None,
    )
    train_loader = HydroLoader(train_cfg, test_split=False, overwrite=not reuse)
    train_dataset = train_loader.train_dataset
    if train_dataset is None:
        raise StrictFoldError("HydroLoader did not return a training dataset")
    train_axis = np.intersect1d(train_ids, source_ids)
    if not np.array_equal(train_axis, train_ids):
        raise StrictFoldError("train subset IDs do not map to the expected canonical loader axis")
    start_date = str(config["train"].get("start_time") or config["train_time"][0])
    _attach_doy(train_dataset, start_date)
    _assert_dataset_axis(train_dataset, train_ids, "train")
    if not stats_path.is_file():
        raise StrictFoldError("train HydroLoader did not create normalization_statistics.json")
    normalization_stats = json.loads(_canonical_json(train_loader.norm_stats))
    missing_keys = _required_stats_keys(config) - set(normalization_stats or {})
    if missing_keys:
        raise StrictFoldError(f"train normalization stats are incomplete: {sorted(missing_keys)}")
    stats_hash = sha256_file(stats_path)
    if not reuse:
        identity_path.write_text(
            json.dumps(
                {"identity": identity, "normalization_statistics_sha256": stats_hash},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if not reuse:
        # Apply the serialized stats on the returned train tensors too; the first
        # fit pass may retain higher-precision in-memory values than its JSON cache.
        del train_dataset, train_loader
        train_loader = HydroLoader(train_cfg, test_split=False, overwrite=False)
        train_dataset = train_loader.train_dataset
        if train_dataset is None:
            raise StrictFoldError("HydroLoader did not return the canonical-statistics training dataset")
        _attach_doy(train_dataset, start_date)
        _assert_dataset_axis(train_dataset, train_ids, "train")
        normalization_stats = json.loads(_canonical_json(train_loader.norm_stats))
        missing_keys = _required_stats_keys(config) - set(normalization_stats or {})
        if missing_keys:
            raise StrictFoldError(f"reloaded train normalization stats are incomplete: {sorted(missing_keys)}")
        if sha256_file(stats_path) != stats_hash:
            raise StrictFoldError("train normalization statistics changed while reloading canonical stats")

    holdout_cfg = _subset_config(
        config,
        subset_path=holdout_ids_path,
        model_dir=holdout_model_dir,
        pretrained_model_dir=train_dir,
    )
    # The guard ensures an incomplete cache cannot silently fit on holdout values.
    holdout_loader = _NoRefitHydroLoader(holdout_cfg, test_split=False, overwrite=False)
    holdout_dataset = holdout_loader.train_dataset
    if holdout_dataset is None:
        raise StrictFoldError("HydroLoader did not return the holdout application dataset")
    holdout_axis = np.intersect1d(holdout_ids, source_ids)
    if not np.array_equal(holdout_axis, holdout_ids):
        raise StrictFoldError("holdout subset IDs do not map to the expected canonical loader axis")
    _attach_doy(holdout_dataset, start_date)
    _assert_dataset_axis(holdout_dataset, holdout_ids, "holdout")
    if _canonical_json(holdout_loader.norm_stats) != _canonical_json(normalization_stats):
        raise StrictFoldError("holdout loader did not apply the exact train-fitted normalization stats")

    if set(fold.train_ids) & set(fold.holdout_ids):
        raise StrictFoldError("train and holdout axes overlap after loader construction")
    del train_loader, holdout_loader
    return FoldDatasets(
        fold=fold,
        train_dataset=train_dataset,
        holdout_dataset=holdout_dataset,
        normalization_stats=normalization_stats,
        cache_identity=identity,
        cache_key=cache_key,
        stats_path=stats_path,
        source_file_sha256=source_file_sha256,
        source_train_slice_sha256=source_train_slice_sha256,
    )
