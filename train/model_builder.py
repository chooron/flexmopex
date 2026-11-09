from __future__ import annotations

import copy
from importlib import import_module
from typing import Any

import torch


PHY_MODEL_MODULES: dict[str, str] = {
    "FixedWeightMopex": "train.models.fixed_weight_mopex",
    "Mopex": "train.models.mopex",
}

NN_MODEL_MODULES: dict[str, str] = {
    "ParamRoutingNet": "train.models.parameter_nets",
    "LearnedStructureNetPureAttrEncoder": "train.models.structure_net",
}


def _load_module(module_path: str):
    return import_module(module_path)


def get_phy_model_names(config: dict[str, Any]) -> list[str]:
    phy_names = config["model"].get("phy", {}).get("name", [])
    if not phy_names:
        raise ValueError("FlexMopexModelHandler requires config['model']['phy']['name'].")
    return phy_names if isinstance(phy_names, list) else [phy_names]


def _resolve_model_class(
    component_name: str,
    *,
    model_map: dict[str, str],
    kind: str,
) -> type[torch.nn.Module]:
    module_path = model_map.get(component_name)
    if module_path is None:
        available = ", ".join(sorted(model_map))
        raise ImportError(f"Unknown flexmopex {kind} '{component_name}'. Available: {available}")

    module = _load_module(module_path)
    component_class = getattr(module, component_name, None)
    if component_class is None:
        raise ImportError(f"Module '{module_path}' does not define '{component_name}'.")
    return component_class


def build_phy_config(config: dict[str, Any]) -> dict[str, Any]:
    phy_config = copy.deepcopy(config["model"]["phy"])
    phy_config.setdefault("variables", phy_config.get("forcings", []))
    return phy_config


def build_nn_config(
    config: dict[str, Any],
    phy_model: torch.nn.Module,
) -> dict[str, Any]:
    nn_config = copy.deepcopy(config["model"]["nn"])
    forcings = nn_config.get("forcings", [])
    attributes = nn_config.get("attributes", [])
    sequence_input_size = len(forcings) + len(attributes)
    nn_config.setdefault("nx", sequence_input_size)
    nn_config.setdefault("nx1", sequence_input_size)
    nn_config.setdefault("nx2", len(attributes))
    nn_config.setdefault("ny", getattr(phy_model, "learnable_param_count", 0))
    nn_config.setdefault("nmul", config["model"]["phy"].get("nmul", 1))
    nn_config.setdefault("dr", nn_config.get("dropout", 0.0))
    nn_config.setdefault("hidden_size", nn_config.get("mlp_hidden_size", 128))
    return nn_config


def build_phy_model(
    config: dict[str, Any],
    phy_model_name: str,
    *,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    phy_class = _resolve_model_class(
        phy_model_name,
        model_map=PHY_MODEL_MODULES,
        kind="physics model",
    )
    return phy_class(build_phy_config(config), device=device)


def build_nn_model(
    config: dict[str, Any],
    phy_model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    nn_name = config["model"]["nn"]["name"]
    nn_class = _resolve_model_class(
        nn_name,
        model_map=NN_MODEL_MODULES,
        kind="neural model",
    )
    nn_config = build_nn_config(config, phy_model)
    if hasattr(nn_class, "build_by_config"):
        return nn_class.build_by_config(nn_config, device)
    return nn_class(
        input_dim=nn_config["nx2"],
        hidden_dim=nn_config["hidden_size"],
        dropout=nn_config.get("dr", 0.0),
        nmul=nn_config["nmul"],
        device=str(device or config.get("device", "cpu")),
    )
