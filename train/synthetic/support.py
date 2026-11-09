"""Component loading and teacher simulation helpers for dataset generation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from train.local_model_handler import FlexMopexModelHandler

ROOT = Path(__file__).resolve().parents[2]


def load_teacher(teacher_dir: Path, device: str):
    """Strictly load the frozen teacher snapshot and its best checkpoint."""
    from train.models.mopex import mopex_step

    teacher_dir = Path(teacher_dir)
    config = json.loads((teacher_dir / "config_snapshot.json").read_text())
    if config["model"]["phy"]["nmul"] != 4 or config["model"]["nn"]["nmul"] != 4:
        raise RuntimeError("materialized teacher snapshot is not the verified nmul=4 model")
    config["device"] = device
    config["mode"] = "train_test"
    config["model_dir"] = str(teacher_dir / "model")
    config["model"]["phy"]["warm_up"] = 365
    config["model"]["phy"]["disable_compile"] = True
    if str(config["model"]["phy"].get("interception_semantics", "")).upper() == "S0":
        # Candidate E-S0 teacher checkpoint (used for the E2 cross-semantic dataset).
        config["model"]["phy"]["teacher_mode"] = True
        config["model"]["phy"]["teacher_step"] = "es0"
        config["delta_model"]["phy_model"]["teacher_mode"] = True
        config["delta_model"]["phy_model"]["teacher_step"] = "es0"
    handler = FlexMopexModelHandler(config, device=device, verbose=False)
    inner = next(iter(handler.model_dict.values()))
    state = torch.load(teacher_dir / "model/best_checkpoint.pt", map_location=device, weights_only=True)
    inner.load_state_dict(state, strict=True)
    handler.eval()
    inner.eval()
    if getattr(inner.phy_model, "step_fn", None) is None:
        inner.phy_model.step_fn = mopex_step
    return config, handler, inner


def prepare_dataset(path: Path, config: dict, gage_info: Path, scratch: Path | None = None):
    from dmg.core.data.loaders.hydro_loader import HydroLoader
    import tempfile

    cfg = copy.deepcopy(config)
    root = Path(scratch) if scratch is not None else Path(tempfile.mkdtemp(prefix="synthetic-loader-"))
    for key, suffix in (("output_dir", ""), ("save_path", ""), ("model_dir", "model"),
                        ("sim_dir", "sim"), ("out_path", "sim"), ("plot_dir", "plot")):
        cfg[key] = str(root / suffix)
        (root / suffix).mkdir(parents=True, exist_ok=True)
    cfg["mode"] = "train_test"
    cfg["device"] = "cpu"
    cfg["dtype"] = torch.float32
    cfg["observations"]["data_path"] = str(path)
    cfg["observations"]["gage_info"] = str(gage_info)
    cfg["observations"]["subset_path"] = None
    cfg["pretrained_model_dir"] = str(config.get("model_dir", ""))
    cfg["model"]["output_unit"] = "mm/d"
    loader = HydroLoader(cfg, test_split=False, overwrite=False)
    dataset = loader.dataset
    if dataset is None:
        raise RuntimeError("HydroLoader returned no all-period dataset")
    return loader, dataset


def simulate_gate_batch(inner, batch, weights, parameters=None):
    """Run the teacher network and physics loop under prescribed structural gates."""
    phy = inner.phy_model
    if "c_nn_norm" not in batch:
        batch["c_nn_norm"] = batch["xc_nn_norm"][0, :, -35:]
    if parameters is None:
        parameters = inner.nn_model(batch)
    mopex_params = phy._descale_mopex_params(parameters["params"])
    routing_params = phy._descale_routing_params(parameters["gamma_uh"])
    P, T, PET, doy, n_steps, n_grid = phy._prepare_forcings(batch)
    q = phy._run_weighted_loop(P, T, PET, doy, mopex_params, weights, n_steps, n_grid)
    return phy._apply_routing(q.mean(-1), routing_params).squeeze(-1)
