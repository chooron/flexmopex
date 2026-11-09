from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from dmg.core.utils import initialize_config

log = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DOTENV_PATH = PROJECT_DIR / ".env"
load_dotenv(DOTENV_PATH, override=False)

__all__ = [
    'get_env_path',
    'load_config',
    'take_data_sample',
]


def get_env_path(name: str, *, default: str | Path | None = None) -> Path:
    """Return a path configured in flexmopex's .env file."""
    path_value = os.getenv(name)
    if path_value is None:
        if default is None:
            raise EnvironmentError(
                f"Please set '{name}' in {DOTENV_PATH} or the process environment."
            )
        path_value = str(default)
    return Path(path_value).expanduser()


def _normalize_none_like(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _preserve_trailing_separator(original: str, resolved: Path) -> str:
    resolved_str = str(resolved)
    if original.endswith(("/", "\\")):
        return resolved_str.rstrip("/\\") + "/"
    return resolved_str


def _resolve_input_path(path_str: str, *, base_dir: Path = REPO_ROOT) -> str:
    path_str = path_str.replace("\\", "/")
    path = Path(path_str).expanduser()
    if path.is_absolute() and path.exists():
        return _preserve_trailing_separator(path_str, path)

    data_dir = get_env_path("FLEXMOPEX_DATA_DIR", default=REPO_ROOT / "data")
    candidates = [
        Path.cwd() / path,
        base_dir / path,
        PROJECT_DIR / path,
    ]
    if "camels_data" in path_str or "camels_dataset" in path_str:
        candidates.extend(
            [
                data_dir / "camels_dataset",
                REPO_ROOT / "data/camels_dataset",
                Path("/workspace/my_deltamodel/data/camels_data"),
            ]
        )
    if path.name == "gage_id.npy":
        candidates.extend(
            [
                data_dir / "gage_id.npy",
                REPO_ROOT / "data/gage_id.npy",
                Path("/workspace/my_deltamodel/data/gage_id.npy"),
                Path("/workspace/my_deltamodel/data/camels_data/gage_id.npy"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return _preserve_trailing_separator(path_str, candidate.resolve())

    return _preserve_trailing_separator(path_str, (base_dir / path).resolve())


def _resolve_output_path(path_str: str, *, base_dir: Path = REPO_ROOT) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return _preserve_trailing_separator(path_str, path)
    if path.parts[:1] == ("train",):
        return _preserve_trailing_separator(path_str, (REPO_ROOT / path).resolve())
    return _preserve_trailing_separator(path_str, (base_dir / path).resolve())


def _normalize_runtime_paths(config: dict[str, Any]) -> None:
    observations_cfg = config.get("observations")
    if observations_cfg:
        for key in ("data_path", "gage_info", "subset_path", "train_path", "test_path"):
            if observations_cfg.get(key):
                observations_cfg[key] = _resolve_input_path(observations_cfg[key])


def _normalize_train_config(config: dict[str, Any]) -> None:
    train_cfg = config.setdefault("train", {})
    loss_cfg = deepcopy(config.get("loss_function") or train_cfg.get("loss_function") or {})
    if isinstance(loss_cfg, str):
        loss_cfg = {"name": loss_cfg, "model": loss_cfg}
    loss_name = loss_cfg.get("name") or loss_cfg.get("model")
    if loss_name:
        loss_cfg.setdefault("name", loss_name)
        loss_cfg.setdefault("model", loss_name)
        config["loss_function"] = loss_cfg
        train_cfg["loss_function"] = {"name": loss_name, "model": loss_name}

    lr_value = train_cfg.get("learning_rate", train_cfg.get("lr"))
    if lr_value is not None:
        train_cfg["learning_rate"] = lr_value
        train_cfg["lr"] = lr_value

    scheduler_name = _normalize_none_like(train_cfg.get("lr_scheduler"))
    train_cfg["lr_scheduler"] = scheduler_name
    train_cfg["lr_scheduler_params"] = deepcopy(train_cfg.get("lr_scheduler_params") or {})


def _normalize_model_config(config: dict[str, Any]) -> None:
    delta_model = config.get("delta_model")
    if delta_model is None and config.get("model") is None:
        raise KeyError("Configuration must define 'delta_model' or 'model'.")

    if config.get("model") is None:
        phy_cfg = deepcopy(delta_model.get("phy_model") or {})
        nn_cfg = deepcopy(delta_model.get("nn_model") or {})

        phy_names = deepcopy(phy_cfg.get("name") or phy_cfg.get("model") or [])
        phy_cfg["name"] = phy_names
        phy_cfg["model"] = deepcopy(phy_names)
        phy_cfg["directory"] = "train/models"

        nn_name = nn_cfg.get("name") or nn_cfg.get("model")
        if nn_name is not None:
            nn_cfg["name"] = nn_name
            nn_cfg["model"] = nn_name
        nn_cfg["directory"] = "train/models"
        if nn_cfg.get("dropout") is None and nn_cfg.get("dr") is not None:
            nn_cfg["dropout"] = nn_cfg["dr"]
        if nn_cfg.get("hidden_size") is None:
            for key in ("mlp_hidden_size", "lstm_hidden_size"):
                if nn_cfg.get(key) is not None:
                    nn_cfg["hidden_size"] = nn_cfg[key]
                    break

        config["model"] = {
            "rho": delta_model["rho"],
            "warm_up": phy_cfg.get("warm_up", delta_model["rho"]),
            "warmup": phy_cfg.get("warm_up", delta_model["rho"]),
            "use_log_norm": deepcopy(phy_cfg.get("use_log_norm", [])),
            "phy": phy_cfg,
            "nn": nn_cfg,
        }

    model_cfg = config["model"]
    model_cfg["warmup"] = model_cfg.get("warm_up", model_cfg.get("warmup", 0))
    model_cfg["warm_up"] = model_cfg["warmup"]

    phy_cfg = deepcopy(model_cfg.get("phy") or {})
    nn_cfg = deepcopy(model_cfg.get("nn") or {})
    phy_names = deepcopy(phy_cfg.get("name") or phy_cfg.get("model") or [])
    phy_cfg["name"] = phy_names
    phy_cfg["model"] = deepcopy(phy_names)

    nn_name = nn_cfg.get("name") or nn_cfg.get("model")
    if nn_name is not None:
        nn_cfg["name"] = nn_name
        nn_cfg["model"] = nn_name

    scheduler_name = _normalize_none_like(config["train"].get("lr_scheduler"))
    scheduler_params = deepcopy(config["train"].get("lr_scheduler_params") or {})
    model_cfg["phy"] = phy_cfg
    model_cfg["nn"] = nn_cfg
    config["delta_model"] = {
        "rho": model_cfg["rho"],
        "phy_model": deepcopy(phy_cfg),
        "nn_model": {
            **deepcopy(nn_cfg),
            "model": nn_cfg.get("name", nn_cfg.get("model")),
            "learning_rate": config["train"].get("learning_rate"),
            "lr_scheduler": scheduler_name,
            "lr_scheduler_params": scheduler_params,
        },
        "train": {
            "lr_scheduler": scheduler_name,
            "lr_scheduler_params": scheduler_params,
        },
    }


def _normalize_paths(config: dict[str, Any]) -> None:
    runtime_output = os.environ.get("FLEXMOPEX_OUTPUT_DIR")
    output_dir = config.get("output_dir") or runtime_output or config.get("save_path")
    trained_model = config.get("trained_model")
    if trained_model and not runtime_output:
        trained_model = _resolve_output_path(trained_model)
        config["trained_model"] = trained_model
    elif runtime_output:
        # A run always writes below the requested output root, never into the
        # checkpoint path named by the configuration.
        config.pop("trained_model", None)
        trained_model = None
    if not output_dir and trained_model:
        output_dir = os.path.dirname(os.path.normpath(trained_model))
    if not output_dir:
        output_dir = str(REPO_ROOT / "outputs")
    output_dir = _resolve_output_path(output_dir)

    model_dir = _resolve_output_path(config.get("model_dir") or trained_model or os.path.join(output_dir, "model"))
    plot_dir = _resolve_output_path(config.get("plot_dir") or os.path.join(output_dir, "plot"))
    sim_dir = _resolve_output_path(config.get("sim_dir") or config.get("out_path") or os.path.join(output_dir, "sim"))

    config["output_dir"] = output_dir
    config["model_dir"] = model_dir
    config["plot_dir"] = plot_dir
    config["sim_dir"] = sim_dir
    config["save_path"] = output_dir
    config["model_path"] = model_dir
    config["out_path"] = sim_dir


def _normalize_flexmopex_config(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    if config.get("mode") == "simulation":
        config["mode"] = "sim"
    if "sim" not in config and "simulation" in config:
        config["sim"] = deepcopy(config["simulation"])
    if "simulation" not in config and "sim" in config:
        config["simulation"] = deepcopy(config["sim"])

    seed = config.get("seed", config.get("random_seed", 0))
    config["seed"] = seed
    config["random_seed"] = seed
    config.setdefault("name", "flexmopex")
    config.setdefault("multimodel_type", None)
    config.setdefault("logging", None)
    config.setdefault("cache_states", False)
    config.setdefault("verbose", False)

    _normalize_runtime_paths(config)
    _normalize_train_config(config)
    _normalize_model_config(config)
    _normalize_paths(config)
    return config


def load_config(path: str) -> dict[str, Any]:
    """Parse and initialize configuration settings from yaml with Hydra.

    This loader is capable of handling config files in nonlinear directory
    structures.

    Parameters
    ----------
    config_path
        Path to the configuration file.

    Returns
    -------
    dict
        Formatted configuration settings.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        project_relative = PROJECT_DIR / path_obj
        path_obj = project_relative if project_relative.exists() else Path.cwd() / path_obj
    path_obj = path_obj.resolve()
    path_no_ext = path_obj.with_suffix("")
    parent_path = str(path_no_ext.parent)
    config_name = path_no_ext.name

    with hydra.initialize_config_dir(config_dir=parent_path, version_base='1.3'):
        config = hydra.compose(config_name=config_name)

    # Convert the OmegaConf object to a dict.
    config = OmegaConf.to_container(config, resolve=True)
    config = _normalize_flexmopex_config(config)

    # Convert date ranges / set device and dtype / create output dirs.
    config = initialize_config(config, make_dirs=False)
    _normalize_paths(config)
    config["random_seed"] = config["seed"]
    config["simulation"] = deepcopy(config["sim"])

    return config


def take_data_sample(
        config: dict,
        dataset_dict: dict[str, torch.Tensor],
        days: int = 730,
        basins: int = 100,
) -> dict[str, torch.Tensor]:
    """Take sample of data.

    Parameters
    ----------
    config
        Configuration settings.
    dataset_dict
        Dictionary containing dataset tensors.
    days
        Number of days to sample.
    basins
        Number of basins to sample.

    Returns
    -------
    dict
        Dictionary containing sampled dataset tensors.
    """
    dataset_sample = {}

    for key, value in dataset_dict.items():
        if value.ndim == 3:
            # Determine warm-up period based on the key
            if key in ['x_phy', 'xc_nn_norm']:
                warm_up = 0
            else:
                warm_up = config['delta_model']['phy_model']['warm_up']

            # Clone and detach the tensor to avoid the warning
            dataset_sample[key] = value[warm_up:days, :basins, :].clone().detach().to(
                dtype=torch.float32, device=config['device'])

        elif value.ndim == 2:
            # Clone and detach the tensor to avoid the warning
            dataset_sample[key] = value[:basins, :].clone().detach().to(
                dtype=torch.float32, device=config['device'])

        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    # Adjust the 'target' tensor based on the configuration
    if ('HBV1_1p' in config['delta_model']['phy_model']['model'] and
            config['delta_model']['phy_model']['use_warmup_mode'] and
            config['multimodel_type'] == 'none'):
        pass  # Keep 'warmup' days for dHBV1.1p
    else:
        warm_up = config['delta_model']['phy_model']['warm_up']
        dataset_sample['target'] = dataset_sample['target'][warm_up:days, :basins]

    return dataset_sample
