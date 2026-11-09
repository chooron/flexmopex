"""Self-contained PubTrainer for LORO experiments.

This version avoids importing from dmg (which requires cartopy) by using
dmg utilities only where they are safe (no cartopy dep), and reimplementing
the minimal training loop needed for FlexMOPEX LORO runs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import tqdm

from train.models.early_stopping import EarlyStoppingController
log = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert non-finite numeric leaves to JSON null without changing finite values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value

class PubTrainer:
    """Trainer adapted for PUB (Predictions in Ungauged Basins) / LORO experiments.

    Designed to work with FlexMopexModelHandler and FlexMopexPubSampler.
    Does NOT inherit from BaseTrainer to avoid dmg cartopy dependency.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.is_in_train = False
        self.sampler = None  # Set externally after construction
        self.start_epoch = 0
        self.predictions = None
        self.early_stopping = EarlyStoppingController.from_config(config)
        self.training_history: list[dict[str, Any]] = []
        self.stop_epoch = 0
        self.best_epoch = None
        self.nonfinite_batch_skips = 0
        self.nonfinite_batch_skips_by_epoch: dict[int, int] = {}
        self.total_batches_seen = 0
        self.finite_batches_seen = 0
        if "train" in config.get("mode", "train"):
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            self.epochs = self.config["train"]["epochs"]

            # Loss function (caller should pass it directly)
            if self.loss_func is not None:
                self.model.loss_func = self.loss_func

            # Optimizer
            self.optimizer = optimizer or self._init_optimizer()

            # LR scheduler
            lr_scheduler_name = config.get("delta_model", {}).get("nn_model", {}).get("lr_scheduler")
            if lr_scheduler_name:
                self.use_scheduler = True
                self.scheduler = scheduler or self._init_scheduler()
            else:
                self.use_scheduler = False

            self._load_states()

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    def _init_optimizer(self) -> torch.optim.Optimizer:
        name = self.config["train"]["optimizer"]
        lr = self.config["delta_model"]["nn_model"]["learning_rate"]
        optimizer_dict = {
            "Adadelta": torch.optim.Adadelta,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        cls = optimizer_dict.get(name)
        if cls is None:
            raise ValueError(f"Optimizer '{name}' not recognized.")
        return cls(self.model.get_parameters(), lr=lr)

    def _init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        name = self.config["delta_model"]["nn_model"]["lr_scheduler"]
        params = self.config["delta_model"]["nn_model"].get("lr_scheduler_params", {})
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        }
        cls = scheduler_dict.get(name)
        if cls is None:
            raise ValueError(f"Scheduler '{name}' not recognized.")
        return cls(self.optimizer, **params)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def _load_states(self) -> None:
        path = self.config.get("model_path", "")
        if not path or not os.path.isdir(path):
            self.start_epoch = 0
            return
        state_files = []
        for file in os.listdir(path):
            match = re.match(r"train(?:er)?_state_ep(\d+)\.pt$", file)
            if match:
                state_files.append((int(match.group(1)), file))
        if state_files:
            epoch, file = max(state_files)
            checkpoint = torch.load(os.path.join(path, file), weights_only=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.load_model(epoch=epoch)
            self.start_epoch = epoch + 1
            print(f"Loaded checkpoint from epoch {epoch}")
            if self.scheduler and checkpoint.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "random_state" in checkpoint:
                torch.set_rng_state(checkpoint["random_state"])
            cuda_state = checkpoint.get("cuda_random_state", checkpoint.get("cuda_state"))
            if torch.cuda.is_available() and cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)
            return
        self.start_epoch = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Train the model for all epochs."""
        from dmg.core.data import create_training_grid  # safe: no cartopy dep

        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset["xc_nn_norm"], self.config
        )
        log.info(f"LORO training: epochs {self.start_epoch}–{self.epochs}")
        self.stop_epoch = max(self.start_epoch - 1, 0)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_one_epoch(epoch, n_minibatch, n_timesteps)
            self.stop_epoch = epoch
            self._write_structural_diagnostics(epoch)
            if self._record_training_epoch(epoch):
                break
        self._finalize_training_metadata(self.stop_epoch)
    def _model_checkpoint_path(self, epoch: int) -> str:
        model_name = next(iter(self.model.model_dict)).lower()
        return os.path.join(self.config["model_path"], f"{model_name}_ep{int(epoch)}.pt")

    def _write_structural_diagnostics(self, epoch: int) -> None:
        """Persist the per-epoch soft structure gates for the LORO/PUB path.

        Mirrors :meth:`CFTrainer._write_structural_diagnostics`: the diagnostic array is
        ``sigmoid(z1 - z0)``, which equals the effective ``softmax(logits)[..., 1]`` gate
        used by the physics loop in eval mode.  Fixed-weight endpoints (LORO base/full)
        route through ``ParamRoutingNet`` and expose no structure head, so for those runs
        the method is a no-op rather than a fabricated constant array.
        """
        try:
            if not getattr(self, "model", None) or not getattr(self.model, "model_dict", None):
                return
            dpl_model = next(iter(self.model.model_dict.values()))
            nn = dpl_model.nn_model
            if not hasattr(nn, "get_structure_logits"):
                return
            n_attr = int(nn.backbone[0].in_features)
            attrs = self.train_dataset["xc_nn_norm"][0, :, -n_attr:]
            device = getattr(dpl_model, "device", "cpu")
            attrs = attrs.to(device)
            was_training = nn.training
            nn.eval()
            try:
                with torch.no_grad():
                    raw_weights = nn.get_structure_logits(attrs)
                    logits = raw_weights.view(raw_weights.shape[0], 4, 2).clamp(-10.0, 10.0)
                    p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])
            finally:
                if was_training:
                    nn.train()
        except Exception as exc:  # diagnostics must never abort training
            log.warning("structural diagnostics skipped at epoch %s: %s", epoch, exc)
            return
        diag_dir = Path(self.config["model_path"]).parent / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        np.save(diag_dir / f"p_struct_epoch_{int(epoch):03d}.npy", p_struct.detach().cpu().numpy())

    def _checkpoint_alias(self, epoch: int, alias: str) -> None:
        source = self._model_checkpoint_path(epoch)
        if not os.path.exists(source):
            self.model.save_model(epoch)
        shutil.copy2(source, os.path.join(self.config["model_path"], f"{alias}_checkpoint.pt"))

    def _record_training_epoch(self, epoch: int) -> bool:
        value = float(self.total_loss / max(1, self.config["train"].get("n_minibatch", 1)))
        # _train_one_epoch stores the true average in _final_loss when available.
        value = float(getattr(self, "_final_loss", value))
        should_stop = self.early_stopping.update(epoch, value)
        row = self.early_stopping.history[-1]
        self.training_history.append(row)
        self._checkpoint_alias(epoch, "last")
        if row["improved"]:
            self.best_epoch = int(epoch)
            self._checkpoint_alias(epoch, "best")
            self._checkpoint_alias(epoch, "final")
        if should_stop:
            print(
                f"[Early stopping] epoch={epoch} best_epoch={self.best_epoch} "
                f"monitor={value:.6f} patience={self.early_stopping.patience}",
                flush=True,
            )
        return should_stop

    def _finalize_training_metadata(self, stop_epoch: int) -> None:
        self.early_stopping.finalize(stop_epoch, int(self.epochs))
        self.stop_epoch = int(self.early_stopping.stop_epoch or stop_epoch)
        self.best_epoch = self.early_stopping.best_epoch
        if self.early_stopping.enabled and self.best_epoch is None:
            raise RuntimeError("No eligible LORO best epoch was recorded.")
        selected_epoch = self.best_epoch if self.early_stopping.enabled else self.stop_epoch
        if selected_epoch and self.early_stopping.enabled:
            self.model.load_model(epoch=int(selected_epoch))
            self._checkpoint_alias(int(selected_epoch), "final")
        if selected_epoch:
            self.config.setdefault("test", {})["test_epoch"] = int(selected_epoch)
        payload = {
            "protocol": "training_side_early_stopping",
            "selected_checkpoint": "best_checkpoint.pt" if self.early_stopping.enabled else "last_checkpoint.pt",
            "best_epoch": self.best_epoch,
            "stop_epoch": self.stop_epoch,
            "early_stop_reason": self.early_stopping.reason,
            "monitor": self.early_stopping.monitor,
            "monitor_values": self.training_history,
            "nonfinite_batch_skips": self.nonfinite_batch_skips,
            "nonfinite_batch_skips_by_epoch": self.nonfinite_batch_skips_by_epoch,
            "total_batches_seen": self.total_batches_seen,
            "finite_batches_seen": self.finite_batches_seen,
            "early_stopping": self.early_stopping.as_dict(),
        }
        output_dir = Path(self.config["model_path"]).parent
        safe_payload = _json_safe(payload)
        (output_dir / "early_stopping.json").write_text(
            json.dumps(safe_payload, indent=2, allow_nan=False) + "\n"
        )
        (output_dir / "train_loss_history.json").write_text(
            json.dumps(_json_safe(self.training_history), indent=2, allow_nan=False) + "\n"
        )

    def _train_one_epoch(self, epoch: int, n_minibatch: int, n_timesteps: int) -> None:
        start_time = time.perf_counter()
        self.total_loss = 0.0
        finite_batches = 0
        skipped_batches = 0

        for mb in tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        ):
            self.total_batches_seen += 1
            dataset_sample = self.sampler.get_training_sample(
                dataset=self.train_dataset, ngrid_train=None, nt=n_timesteps
            )
            _ = self.model(dataset_sample)
            loss_dict_before = dict(getattr(self.model, "loss_dict", {}))
            loss = self.model.calc_loss(dataset_sample, loss_func=self.loss_func)
            loss_is_finite = bool(torch.isfinite(loss).all().item())
            if not loss_is_finite:
                for key, value in loss_dict_before.items():
                    self.model.loss_dict[key] = value
                self.optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                self.nonfinite_batch_skips += 1
                log.warning(
                    "Skipping non-finite batch: epoch=%s batch=%s loss=%s",
                    epoch,
                    mb,
                    loss.detach().item(),
                )
                continue

            loss.backward()
            parameters = [p for p in self.model.get_parameters() if p.requires_grad]
            gradients_finite = all(
                p.grad is None or bool(torch.isfinite(p.grad).all().item())
                for p in parameters
            )
            if not gradients_finite:
                self.optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                self.nonfinite_batch_skips += 1
                log.warning(
                    "Skipping non-finite gradient batch: epoch=%s batch=%s loss=%s",
                    epoch,
                    mb,
                    loss.detach().item(),
                )
                continue
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            finite_batches += 1
            self.finite_batches_seen += 1
            self.total_loss += loss.item()

            if self.verbose and mb % 10 == 0:
                tqdm.tqdm.write(f"Epoch {epoch}, batch {mb} | loss: {loss.item():.6f}")

        self.nonfinite_batch_skips_by_epoch[int(epoch)] = skipped_batches
        if self.use_scheduler:
            self.scheduler.step()

        self._final_loss = (
            self.total_loss / finite_batches if finite_batches else float("nan")
        )
        self._log_epoch_stats(epoch, getattr(self.model, "loss_dict", {}), max(finite_batches, 1), start_time)

        save_epoch = self.config["train"].get("save_epoch", 10)
        if epoch % save_epoch == 0:
            self.model.save_model(epoch)
            self._save_train_state(epoch)

    def _save_train_state(self, epoch: int) -> None:
        from dmg.core.utils.utils import save_train_state  # safe: no cartopy dep
        save_train_state(
            self.config["model_path"],
            epoch=epoch,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            clear_prior=False,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> None:
        """Evaluate on all holdout (val) basins basin-by-basin."""
        from dmg.core.calc.metrics import Metrics  # safe: no cartopy dep
        from dmg.core.utils.utils import save_outputs  # safe: no cartopy dep

        self.is_in_train = False
        val_indices = self.sampler.val_indices
        log.info(f"Evaluating on {len(val_indices)} holdout basins.")

        # Use the actual key from model_dict rather than config, since
        # apply_runtime_overrides may write to a different config path than
        # what model_builder.get_phy_model_names() reads.
        model_name = next(iter(self.model.model_dict))
        all_preds: list[dict] = []

        for basin_idx in tqdm.tqdm(
            val_indices, desc="Evaluating holdout basins", leave=False, dynamic_ncols=True
        ):
            sample = self._get_val_sample(basin_idx)
            pred = self.model(sample, eval=True)
            pred_basin = {
                k: v.cpu().detach() for k, v in pred[model_name].items()
            }
            all_preds.append(pred_basin)

        self.predictions = self._concat_preds(all_preds)

        observations = self.eval_dataset["target"][:, val_indices, :]
        save_outputs(self.config, [self.predictions], observations)
        self._calc_metrics(self.predictions, val_indices)

    def _get_val_sample(self, basin_idx: int) -> dict:
        """Get full time-series sample for one validation basin."""
        dataset = self.eval_dataset
        device = self.config["device"]
        sample = {}
        for key, value in dataset.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    sample[key] = value[:, [basin_idx], :].to(device)
                elif value.ndim == 2:
                    sample[key] = value[[basin_idx], :].to(device)
                else:
                    sample[key] = value.to(device)
            else:
                sample[key] = value
        return sample

    def _concat_preds(self, preds: list[dict]) -> dict:
        """Concatenate per-basin predictions along basin dimension (dim=1)."""
        result = {}
        for key in preds[0]:
            tensors = [p[key] for p in preds]
            dim = 1 if tensors[0].ndim == 3 else 0
            result[key] = torch.cat(tensors, dim=dim)
        return result

    def _calc_metrics(self, predictions: dict, val_indices) -> None:
        from dmg.core.calc.metrics import Metrics  # safe: no cartopy dep

        target_name = self.config["train"]["target"][0]
        preds = predictions[target_name]  # (T, N, 1) or (T, N)
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()

        warmup = self.config["delta_model"]["phy_model"]["warm_up"]
        obs = self.eval_dataset["target"][:, val_indices, :].cpu().numpy()
        obs = obs[warmup:, :, :]
        if obs.ndim == 3:
            obs_2d = obs[:, :, 0]
        else:
            obs_2d = obs
        if preds.ndim == 3:
            preds_2d = preds[:, :, 0]
        else:
            preds_2d = preds

        # Align lengths
        n = min(obs_2d.shape[0], preds_2d.shape[0])
        metrics = Metrics(
            np.swapaxes(preds_2d[:n], 1, 0),
            np.swapaxes(obs_2d[:n], 1, 0),
        )
        metrics.dump_metrics(self.config["out_path"])

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_epoch_stats(
        self, epoch: int, loss_dict: dict, n_minibatch: int, start_time: float
    ) -> None:
        avg = {k: v / n_minibatch for k, v in loss_dict.items()} if loss_dict else {}
        loss_str = ", ".join(f"{k}: {v:.6f}" for k, v in avg.items()) or f"total: {self.total_loss / n_minibatch:.6f}"
        elapsed = time.perf_counter() - start_time
        mem_mb = 0
        try:
            device = self.config.get("device", "cpu")
            if str(device).startswith("cuda"):
                mem_mb = int(torch.cuda.memory_reserved(device=device) / 1e6)
        except Exception:
            pass
        log.info(f"Epoch {epoch}: {loss_str} | {elapsed:.1f}s | {mem_mb}MB GPU")
        print(f"[Epoch {epoch}/{self.epochs}] {loss_str} | {elapsed:.1f}s")
