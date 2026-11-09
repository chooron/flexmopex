"""Train-isolated counterfactual trainer for the synthetic experiments.

The target and loss equations follow ``train/models/cf_trainer.py`` and
``train/models/nse_dyn_aic_batch_loss.py``; the trainer never sees holdout basins.
"""
from __future__ import annotations

import copy
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from dmg.core.data.data import random_index
from dmg.core.data.samplers.hydro_sampler import HydroSampler

from train.local_model_handler import FlexMopexModelHandler
from train.models.nse_dyn_aic_batch_loss import NseDynAicBatchLoss
from train.models.pub_trainer import PubTrainer

from .fold import FoldDatasets, SpatialFold, StrictFoldError

PROCESS_ORDER = ("w_phen", "w_int", "w_snow", "w_sub")
PROCESS_COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
EPS = 1e-12
REVISION_ROOT = Path(__file__).resolve().parents[1]
STRICT_RUN_ROOT = (REVISION_ROOT / "strict" / "runs").resolve()


@dataclass
class StrictTargets:
    basin_ids: tuple[int, ...]
    n_valid_by_basin: torch.Tensor
    n_valid_total: float
    delta_j: torch.Tensor
    temperature: dict[str, float]
    q: torch.Tensor
    confidence: torch.Tensor

    def validate_axis(self, fold: SpatialFold) -> None:
        if self.basin_ids != fold.train_ids:
            raise StrictFoldError("counterfactual target basin axis differs from fold train IDs")
        expected = (len(fold.train_ids), len(PROCESS_ORDER))
        if tuple(self.q.shape) != expected or tuple(self.confidence.shape) != expected:
            raise StrictFoldError(f"target arrays must have shape {expected}")


class StrictCounterfactualTargetGenerator:
    """Train-only R15 delta-J/q adapter; it cannot accept a full-pool axis."""

    def __init__(self, config: Mapping[str, Any], device: str | torch.device) -> None:
        self.config = dict(config)
        self.device = torch.device(device)
        loss_cfg = self.config.get("loss_function") or self.config.get("train", {}).get("loss_function") or {}
        self.aic_alpha = float(loss_cfg.get("aic_alpha", 0.01)) if isinstance(loss_cfg, dict) else 0.01

    @torch.no_grad()
    def generate(self, model: FlexMopexModelHandler, data: FoldDatasets) -> StrictTargets:
        fold = data.fold
        td = data.train_dataset
        basin_count = len(fold.train_ids)
        if int(td["x_phy"].shape[1]) != basin_count or int(td["target"].shape[1]) != basin_count:
            raise StrictFoldError("target generation input is not physically train-only")
        if "doy" not in td:
            raise StrictFoldError("train dataset is missing day-of-year inputs")
        if len(PROCESS_ORDER) != 4 or set(PROCESS_COSTS) != set(PROCESS_ORDER):
            raise RuntimeError("counterfactual process order/cost configuration is invalid")

        dpl_model = next(iter(model.model_dict.values()))
        phy = dpl_model.phy_model
        nn_model = dpl_model.nn_model
        modes = (phy.training, nn_model.training)
        phy.eval()
        nn_model.eval()
        try:
            n_forcings = len(self.config["model"]["nn"].get("forcings", []))
            if n_forcings != 3:
                raise StrictFoldError(f"R15 target adapter expects the source formula's three NN forcings, got {n_forcings}")
            n_attr = int(td["xc_nn_norm"].shape[-1]) - n_forcings
            if n_attr <= 0:
                raise StrictFoldError("normalized input does not contain static attributes")
            attrs = td["xc_nn_norm"][0, :, -n_attr:].to(self.device)
            target = td["target"][:, :, 0].to(self.device)
            y_obs = target.detach().cpu().numpy()
            std_train = (np.nanstd(y_obs, axis=0) + 0.1).astype(np.float32)
            std_t = torch.from_numpy(std_train).to(self.device)
            n_valid = (~torch.isnan(target)).sum(dim=0).float()
            n_total = float(n_valid.sum().item())
            if n_total <= 0 or not bool(torch.isfinite(n_valid).all().item()):
                raise StrictFoldError("train-only valid-observation counts are invalid")

            p_raw = nn_model({"c_nn_norm": attrs})
            learned_weights = F.softmax(p_raw["weights"].view(basin_count, 4, 2).clamp(-10.0, 10.0), dim=-1)[..., 1]
            mopex_params = phy._descale_mopex_params(p_raw["params"])
            routing = phy._descale_routing_params(p_raw["gamma_uh"])
            if bool(getattr(phy, "is_dflex", False)):
                base_weights = (learned_weights > 0.5).to(learned_weights.dtype)
            else:
                base_weights = learned_weights.detach().clone()
            removed = set(getattr(phy, "removed_processes", set()))

            repeats = 2
            params_rep = {key: value.repeat(repeats, 1) for key, value in mopex_params.items()}
            routing_rep = {key: value.repeat(repeats) for key, value in routing.items()}
            sample_rep = {
                "x_phy": td["x_phy"].repeat(1, repeats, 1).to(self.device),
                "doy": td["doy"].repeat(1, repeats, 1).to(self.device),
                "c_nn_norm": attrs.repeat(repeats, 1),
            }
            delta_j = torch.zeros((basin_count, 4), dtype=target.dtype, device=self.device)
            q = torch.zeros_like(delta_j)
            temperature: dict[str, float] = {}
            std_dev = std_t.view(1, 1, basin_count)

            for process_index, process in enumerate(PROCESS_ORDER):
                if process in removed:
                    temperature[process] = 0.0
                    continue
                weights_cf = base_weights.repeat(repeats, 1)
                weights_cf[:basin_count, process_index] = 0.0
                weights_cf[basin_count:, process_index] = 1.0
                P, t_forcing, pet, doy, n_steps, _ = phy._prepare_forcings(sample_rep)
                Q = phy._run_weighted_loop(P, t_forcing, pet, doy, params_rep, weights_cf, n_steps, basin_count * repeats)
                Qr = phy._apply_routing(Q.mean(-1), routing_rep)
                n_out = int(Qr.shape[0])
                predictions = Qr[:, : basin_count * repeats, 0].view(n_out, 2, basin_count)
                observed = target[phy.warm_up : phy.warm_up + n_out].view(n_out, 1, basin_count)
                squared = (predictions - observed) ** 2 / (std_dev**2)
                mask = ~torch.isnan(observed)
                valid_repeated = mask.sum(dim=0).clamp(min=1)
                squared = torch.where(mask, squared, torch.zeros_like(squared))
                fit = squared.sum(dim=0) / valid_repeated
                fit_off, fit_on = fit[0], fit[1]
                aic_penalty = (
                    self.aic_alpha
                    * PROCESS_COSTS[process]
                    * (n_total / (float(basin_count) * torch.clamp(n_valid, min=1.0)))
                )
                delta = fit_off - fit_on - aic_penalty
                delta_j[:, process_index] = delta
                finite_nonzero = delta[torch.isfinite(delta) & (delta != 0)]
                if finite_nonzero.numel():
                    scale = max(float(torch.median(torch.abs(finite_nonzero)).item()), 1e-4)
                else:
                    scale = 0.02
                temperature[process] = scale
                q[:, process_index] = torch.sigmoid(delta / scale)

            confidence = 2.0 * torch.abs(q - 0.5)
            result = StrictTargets(
                basin_ids=fold.train_ids,
                n_valid_by_basin=n_valid.detach(),
                n_valid_total=n_total,
                delta_j=delta_j.detach(),
                temperature=temperature,
                q=q.detach(),
                confidence=confidence.detach(),
            )
            result.validate_axis(fold)
            return result
        finally:
            phy.train(modes[0])
            nn_model.train(modes[1])


class StrictTrainOnlySampler:
    """HydroSampler-compatible random sampling over a train-only basin axis."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.base = HydroSampler(self.config)
        self.rho = int(self.config["model"]["rho"])
        self.warmup = int(self.config["model"].get("warmup", 0))
        self.device = torch.device(self.config["device"])

    def get_training_sample(
        self,
        dataset: Mapping[str, torch.Tensor],
        ngrid_train: int | None,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        n_basins = int(dataset["x_phy"].shape[1])
        if ngrid_train not in (None, n_basins):
            raise StrictFoldError("sampler basin count must equal the physically train-only dataset axis")
        batch_size = int(self.config["train"]["batch_size"])
        indices, starts = random_index(
            n_basins,
            int(nt),
            (batch_size, self.rho),
            warmup=self.warmup,
        )
        return self.sample_for_indices(dataset, indices, starts)

    def sample_for_indices(
        self,
        dataset: Mapping[str, torch.Tensor],
        indices: Sequence[int] | np.ndarray,
        starts: Sequence[int] | np.ndarray,
    ) -> dict[str, torch.Tensor]:
        basin_indices = np.asarray(indices, dtype=np.int64)
        time_starts = np.asarray(starts, dtype=np.int64)
        n_basins = int(dataset["x_phy"].shape[1])
        if basin_indices.ndim != 1 or time_starts.ndim != 1 or len(basin_indices) != len(time_starts):
            raise StrictFoldError("sampler indices/starts must be equal-length vectors")
        if not len(basin_indices) or basin_indices.min() < 0 or basin_indices.max() >= n_basins:
            raise StrictFoldError("sampler emitted an index outside the train-local basin axis")
        if np.any(time_starts < self.warmup) or np.any(time_starts + self.rho > int(dataset["x_phy"].shape[0])):
            raise StrictFoldError("sampler emitted a time window outside the train dataset")
        sample: dict[str, torch.Tensor] = {
            "x_phy": self.base.select_subset(dataset["x_phy"], basin_indices, time_starts),
            "c_phy": dataset["c_phy"][basin_indices].to(self.device),
            "c_nn": dataset["c_nn"][basin_indices].to(self.device),
            "xc_nn_norm": self.base.select_subset(
                dataset["xc_nn_norm"], basin_indices, time_starts, has_grad=False
            ),
            "target": self.base.select_subset(dataset["target"], basin_indices, time_starts),
            "batch_sample": torch.as_tensor(basin_indices, dtype=torch.long, device=self.device),
        }
        if "doy" in dataset:
            sample["doy"] = self.base.select_subset(dataset["doy"], basin_indices, time_starts)
        return sample


def _structure_bce(
    model: FlexMopexModelHandler,
    sample: Mapping[str, torch.Tensor],
    targets: StrictTargets,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    dpl_model = next(iter(model.model_dict.values()))
    nn_model = dpl_model.nn_model
    batch_sample = sample["batch_sample"].long().to(targets.q.device)
    if batch_sample.numel() == 0 or int(batch_sample.min()) < 0 or int(batch_sample.max()) >= len(targets.basin_ids):
        raise StrictFoldError("batch_sample is not in the local train/q axis")
    q_batch = targets.q[batch_sample]
    attr_count = int(nn_model.backbone[0].in_features)
    attrs = sample["xc_nn_norm"][0, :, -attr_count:].to(targets.q.device)
    if hasattr(nn_model, "get_structure_logits"):
        raw_logits = nn_model.get_structure_logits(attrs)
    elif hasattr(nn_model, "structure_encoder"):
        raw_logits = nn_model.structure_encoder(attrs)
    else:
        raise StrictFoldError(f"{type(nn_model).__name__} has no source-supported structure-logit path")
    logits = torch.clamp(raw_logits, min=-10.0, max=10.0).view(-1, 4, 2)
    probability = torch.sigmoid(logits[..., 1] - logits[..., 0])
    if bool(config.get("confidence_weighted_cf_loss", False)):
        confidence = (2.0 * torch.abs(q_batch - 0.5)).detach()
        elementwise = F.binary_cross_entropy(probability, q_batch, reduction="none")
        loss = torch.mean(torch.sum(confidence * elementwise, dim=0) / (confidence.sum(dim=0) + EPS))
    else:
        loss = F.binary_cross_entropy(probability, q_batch)
    return loss, probability


@dataclass
class StrictTrainer(PubTrainer):
    """Actual future training entry; updates require an explicit runtime opt-in."""

    def __init__(
        self,
        config: dict[str, Any],
        model: FlexMopexModelHandler,
        data: FoldDatasets,
        loss_func: NseDynAicBatchLoss,
        output_dir: Path,
    ) -> None:
        self.fold_data = data
        self.fold = data.fold
        self.output_dir = Path(output_dir).resolve()
        self._validate_output_dir(self.output_dir)
        if int(data.train_dataset["x_phy"].shape[1]) != len(self.fold.train_ids):
            raise StrictFoldError("StrictTrainer train dataset is not aligned with its fold IDs")
        run_config = copy.deepcopy(config)
        run_config["mode"] = "train"
        run_config["output_dir"] = str(self.output_dir)
        run_config["save_path"] = str(self.output_dir)
        run_config["model_dir"] = str(self.output_dir / "model")
        run_config["model_path"] = run_config["model_dir"]
        run_config["pretrained_model_dir"] = str(data.stats_path.parent)
        run_config["train"] = copy.deepcopy(config["train"])
        run_config["train"]["start_epoch"] = 0
        if str(config.get("structure_optimizer", "none")).lower() != "none":
            raise StrictFoldError("Strict adapter currently supports the source config's unified optimizer only")
        self.strict_config = run_config
        self.cf_generator = StrictCounterfactualTargetGenerator(run_config, run_config["device"])
        self.cached_targets: StrictTargets | None = None
        super().__init__(
            config=run_config,
            model=model,
            train_dataset=data.train_dataset,
            eval_dataset=None,
            loss_func=loss_func,
            optimizer=None,
            scheduler=None,
            verbose=False,
        )
        self.sampler = StrictTrainOnlySampler(run_config)
        if self.start_epoch != 0:
            raise StrictFoldError("Strict training output path is not fresh; implicit resume is prohibited")

    @staticmethod
    def _validate_output_dir(output_dir: Path) -> None:
        if output_dir == STRICT_RUN_ROOT or STRICT_RUN_ROOT not in output_dir.parents:
            raise StrictFoldError(f"Strict output must be a unique child of {STRICT_RUN_ROOT}")
        if not output_dir.name or any(token in output_dir.parts for token in ("FLEXMOPEX_PAPER_DATA", "formal_671", "loro")):
            raise StrictFoldError("Strict output path overlaps a frozen/formal/historical run namespace")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise StrictFoldError("Strict output directory is not empty; implicit reuse is prohibited")

    def train(self, *, authorize_updates: bool = False) -> None:  # type: ignore[override]
        if not authorize_updates:
            raise PermissionError("training is held; pass authorize_updates=True only after separate explicit authorization")
        self._validate_output_dir(self.output_dir)
        super().train()


    def evaluate(self) -> None:
        raise PermissionError("holdout evaluation is outside the authorized Strict training path")

    @staticmethod
    def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
        terms = [p.grad.detach().float().norm().square() for p in parameters if p.grad is not None]
        return float(torch.sqrt(torch.stack(terms).sum()).item()) if terms else 0.0

    def _training_minibatch(self, sample: Mapping[str, torch.Tensor], *, diagnose: bool = False) -> dict[str, Any]:
        """Run the shared source batch equation; optional diagnostics do not change its update."""
        if self.cached_targets is None:
            raise StrictFoldError("counterfactual targets must be generated before a minibatch")
        self.optimizer.zero_grad(set_to_none=True)
        _ = self.model(dict(sample))
        fit_aic = self.model.calc_loss(dict(sample), loss_func=self.loss_func)
        cf_bce, _ = _structure_bce(self.model, sample, self.cached_targets, self.strict_config)
        total = fit_aic + float(self.strict_config.get("cf_loss_weight", 1.0)) * cf_bce
        result: dict[str, Any] = {"fit_aic": float(fit_aic.detach().item()), "cf_bce": float(cf_bce.detach().item()),
                                  "total": float(total.detach().item()),
                                  "finite_counts": {"fit_aic": int(torch.isfinite(fit_aic).sum().item()),
                                                    "cf_bce": int(torch.isfinite(cf_bce).sum().item()),
                                                    "total": int(torch.isfinite(total).sum().item())}}
        if not bool(torch.isfinite(total).item()):
            self.optimizer.zero_grad(set_to_none=True)
            result["updated"] = False
            result["reason"] = "nonfinite_loss"
            return result
        inner = next(iter(self.model.model_dict.values()))
        named = [(name, p) for name, p in inner.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        branch_grads: dict[str, dict[str, float]] = {}
        if diagnose:
            gradient_finite_counts: dict[str, int] = {}
            for label, component in (("fit_aic", fit_aic), ("cf_bce", cf_bce)):
                grads = torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)
                branch_grads[label] = {
                    name: float(grad.detach().float().norm().item())
                    for (name, _), grad in zip(named, grads) if grad is not None
                }
                gradient_finite_counts[label] = sum(int(torch.isfinite(grad).sum().item()) for grad in grads if grad is not None)
            result["gradient_finite_counts"] = gradient_finite_counts
        total.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in parameters):
            self.optimizer.zero_grad(set_to_none=True)
            result["updated"] = False
            result["reason"] = "nonfinite_gradient"
            return result
        pre_clip_norm = self._gradient_norm(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        post_clip_norm = self._gradient_norm(parameters)
        before = {name: p.detach().clone() for name, p in named} if diagnose else {}
        self.optimizer.step()
        if diagnose:
            update_norms = {name: float((p.detach() - before[name]).float().norm().item()) for name, p in named}
            result.update({"branch_grad_norms": branch_grads, "gradient_norm_before_clip": pre_clip_norm,
                           "gradient_norm_after_clip": post_clip_norm, "parameter_update_norms": update_norms})
        self.optimizer.zero_grad(set_to_none=True)
        result["updated"] = True
        return result

    def smoke_steps(self, *, authorize_smoke_updates: bool = False, max_steps: int = 0) -> list[dict[str, Any]]:
        """Run an explicitly opted-in, fixed-init, update-only smoke (no epoch/checkpoint path)."""
        if not authorize_smoke_updates:
            raise PermissionError("smoke updates require explicit authorize_smoke_updates=True")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 3:
            raise ValueError("smoke max_steps must be an integer in 1..3")
        if self.optimizer is None:
            raise RuntimeError("source optimizer was not initialized")
        from dmg.core.data.data import create_training_grid
        _, _, n_timesteps = create_training_grid(self.train_dataset["xc_nn_norm"], self.strict_config)
        records: list[dict[str, Any]] = []
        for step in range(1, int(max_steps) + 1):
            sample = self.sampler.get_training_sample(self.train_dataset, len(self.fold.train_ids), n_timesteps)
            local = sample["batch_sample"].detach().cpu().tolist()
            records.append({"step": step, "batch_ids": [int(self.fold.train_ids[i]) for i in local],
                            "unique_count": len(set(local)), "window_shape": list(sample["x_phy"].shape),
                            **self._training_minibatch(sample, diagnose=True)})
            if not records[-1]["updated"]:
                raise StrictFoldError(f"smoke step {step} did not update: {records[-1].get('reason')}")
        return records

    def save_checkpoint(self, path: Path | str, step: int = 0) -> Path:
        """Save full model, optimizer, sampler, RNG, and target state for bitwise resume."""
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        inner = next(iter(self.model.model_dict.values()))
        checkpoint = {
            "step": int(step),
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()},
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            },
            "cached_targets": copy.deepcopy(self.cached_targets),
        }
        torch.save(checkpoint, target_path)
        return target_path

    def load_checkpoint(self, path: Path | str) -> int:
        """Load full state and restore RNG, sampler, model, optimizer, and targets."""
        target_path = Path(path).resolve()
        device = self.strict_config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(target_path, map_location=device, weights_only=False)
        inner = next(iter(self.model.model_dict.values()))
        inner.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        rng = checkpoint.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch_cpu" in rng:
            cpu_state = rng["torch_cpu"]
            if not isinstance(cpu_state, torch.Tensor):
                cpu_state = torch.as_tensor(cpu_state, dtype=torch.uint8)
            torch.set_rng_state(cpu_state.cpu().to(torch.uint8))
        if "torch_cuda" in rng and torch.cuda.is_available() and rng["torch_cuda"]:
            cuda_states = []
            for s in rng["torch_cuda"]:
                if not isinstance(s, torch.Tensor):
                    s = torch.as_tensor(s, dtype=torch.uint8)
                cuda_states.append(s.cpu().to(torch.uint8))
            torch.cuda.set_rng_state_all(cuda_states)
        if "cached_targets" in checkpoint:
            self.cached_targets = checkpoint["cached_targets"]
        return int(checkpoint.get("step", 0))

    def _train_one_epoch(self, epoch: int, n_minibatch: int, n_timesteps: int) -> None:
        self.cached_targets = self.cf_generator.generate(self.model, self.fold_data)
        self.cached_targets.validate_axis(self.fold)
        self.total_loss = 0.0
        fit_total = 0.0
        bce_total = 0.0
        finite_batches = 0
        started = time.perf_counter()
        for _batch in range(1, n_minibatch + 1):
            sample = self.sampler.get_training_sample(
                self.train_dataset,
                ngrid_train=len(self.fold.train_ids),
                nt=n_timesteps,
            )
            result = self._training_minibatch(sample)
            if not result["updated"]:
                continue
            self.total_loss += float(result["total"])
            fit_total += float(result["fit_aic"])
            bce_total += float(result["cf_bce"])
            finite_batches += 1
        if self.use_scheduler:
            self.scheduler.step()
        divisor = max(1, finite_batches)
        self._final_loss = self.total_loss / divisor
        self.fit_aic_loss = fit_total / divisor
        self.cf_bce_loss = bce_total / divisor
        self.epoch_seconds = time.perf_counter() - started


def build_strict_model_and_loss(
    config: Mapping[str, Any],
    data: FoldDatasets,
    *,
    device: str | torch.device,
    model_dir: Path,
    checkpoint_path: Path | None = None,
) -> tuple[FlexMopexModelHandler, NseDynAicBatchLoss]:
    """Build the original model/loss with train-only loss statistics.

    A supplied checkpoint is loaded read-only into the in-memory model. It is
    never written or used as an output directory.
    """
    cfg = copy.deepcopy(dict(config))
    cfg["device"] = str(device)
    cfg["mode"] = "train"
    cfg["model_dir"] = str(model_dir.resolve())
    cfg["model_path"] = cfg["model_dir"]
    cfg["pretrained_model_dir"] = str(data.stats_path.parent)
    cfg["train"] = copy.deepcopy(config["train"])
    cfg["train"]["start_epoch"] = 0
    cfg["dtype"] = torch.float32
    model = FlexMopexModelHandler(cfg, device=torch.device(device), verbose=False)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if len(model.model_dict) != 1:
            raise StrictFoldError("fixed-checkpoint validation expects exactly one Flex-MOPEX model")
        next(iter(model.model_dict.values())).load_state_dict(state, strict=True)
    model.eval()
    loss_cfg = cfg.get("loss_function") or cfg.get("train", {}).get("loss_function") or {}
    aic_alpha = float(loss_cfg.get("aic_alpha", 0.01)) if isinstance(loss_cfg, dict) else 0.01
    loss = NseDynAicBatchLoss(
        cfg,
        device=device,
        y_obs=data.train_dataset["target"],
        aic_alpha=aic_alpha,
    )
    return model, loss
