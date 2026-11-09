"""Counterfactual Structural Supervision (R15).

Provides:
  - CounterfactualTargetGenerator: Computes detached basin-specific structural evidence DeltaJ
    and soft targets q = sigmoid(DeltaJ / T) at the start of each epoch.
  - CFTrainer: Trainer that orchestrates epoch-start target refresh, detached BCE loss L_CF
    for weights_head, and gradient routing.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from train.my_trainer import MyTrainer

log = logging.getLogger(__name__)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
EPS = 1e-12


def _extract_aic_alpha(config: dict[str, Any]) -> float:
    loss_cfg = config.get("loss_function") or config.get("train", {}).get("loss_function") or {}
    if isinstance(loss_cfg, dict):
        return float(loss_cfg.get("aic_alpha", 0.01))
    return 0.01


class CounterfactualTargetGenerator:
    """Computes detached counterfactual structural targets q for all training basins."""

    def __init__(
        self,
        config: dict[str, Any],
        device: str | torch.device = "cuda:0",
        aic_alpha: float | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if aic_alpha is not None:
            self.aic_alpha = float(aic_alpha)
        else:
            self.aic_alpha = _extract_aic_alpha(config)
        self.costs = COSTS
    @torch.no_grad()
    def generate_targets(
        self,
        model: Any,
        train_dataset: dict[str, torch.Tensor],
        eval_dataset: dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes DeltaJ, T, and q for all B training basins.

        Returns:
            q_tensor: [B, 4] tensor of soft targets in [0, 1] on self.device
            diagnostics: dictionary with T scales, mean q, entropy, etc.
        """
        # Determine physics and NN model
        dpl_model = next(iter(model.model_dict.values()))
        phy = dpl_model.phy_model
        nn = dpl_model.nn_model

        phy.eval()
        nn.eval()

        td = train_dataset
        B = td["x_phy"].shape[1]
        n_attr = td["xc_nn_norm"].shape[-1] - 3
        attrs = td["xc_nn_norm"][0, :, -n_attr:].to(self.device)
        n_timesteps = td["x_phy"].shape[0]

        # Use training dataset targets & valid days
        y_obs = td["target"][:, :, 0].cpu().numpy()
        std_train = (np.nanstd(y_obs, axis=0) + 0.1).astype(np.float32)
        std_t = torch.from_numpy(std_train).to(self.device)

        y_t_dev = td["target"][:, :, 0].to(self.device)
        n_valid_b = (~torch.isnan(y_t_dev)).sum(dim=0).float()  # [B]
        N = float(n_valid_b.sum().item())
        n_valid_b_cpu = n_valid_b.cpu().numpy()

        # Baseline parameters from current NN model
        p_raw = nn({"c_nn_norm": attrs})
        w_learn = F.softmax(p_raw["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
        mopex_params = phy._descale_mopex_params(p_raw["params"])
        routing = phy._descale_routing_params(p_raw["gamma_uh"])
        # DFlex counterfactuals must use the same hard-gated simulator as its
        # training forward; CFlex retains the continuous baseline weights.
        if bool(getattr(phy, "is_dflex", False)):
            base_w = (w_learn > 0.5).to(w_learn.dtype)
        else:
            base_w = w_learn.detach().clone()
        removed_processes = set(getattr(phy, "removed_processes", set()))

        # Process-wise counterfactual evaluation (S=2 per process for low VRAM footprint)
        S = 2
        params_rep = {k: v.repeat(S, 1) for k, v in mopex_params.items()}
        routing_rep = {k: v.repeat(S) for k, v in routing.items()}
        sample_rep = {
            "x_phy": td["x_phy"].repeat(1, S, 1).to(self.device),
            "doy": td["doy"].repeat(1, S, 1).to(self.device),
            "c_nn_norm": attrs.repeat(S, 1).to(self.device),
        }

        L_fit_off_list = []
        L_fit_on_list = []
        std_dev = std_t.view(1, 1, B)
        q_list = [None] * len(PROCESSES)
        diag = {}

        for p_idx in range(4):
            proc = PROCESSES[p_idx]
            if proc in removed_processes:
                q_p = torch.zeros(B, device=self.device)
                zero_fit = torch.zeros(B, device=self.device)
                L_fit_off_list.append(zero_fit.unsqueeze(0))
                L_fit_on_list.append(zero_fit.unsqueeze(0))
                q_list[p_idx] = q_p.unsqueeze(-1)
                diag[proc] = {
                    "removed": True,
                    "T_scale": 0.0,
                    "q_mean": 0.0,
                    "q_median": 0.0,
                    "q_std": 0.0,
                    "c_mean": 1.0,
                    "effective_n_samples": float(B),
                }
                continue
            w_cf = base_w.repeat(S, 1)
            w_cf[0:B, p_idx] = 0.0      # OFF
            w_cf[B:2*B, p_idx] = 1.0    # ON

            P, T_forcing, PET, doy, n_steps, _ = phy._prepare_forcings(sample_rep)
            Q = phy._run_weighted_loop(P, T_forcing, PET, doy, params_rep, w_cf, n_steps, B * S)
            Qr = phy._apply_routing(Q.mean(-1), routing_rep)  # [n_out, B * S, 1]
            n_out = Qr.shape[0]

            Qr_arr = Qr[:, :B * S, 0].view(n_out, 2, B)  # [n_out, 2 (off/on), B]
            obs_t = y_t_dev[phy.warm_up:phy.warm_up + n_out].view(n_out, 1, B)

            sq_res = (Qr_arr - obs_t) ** 2 / (std_dev ** 2)
            mask = ~torch.isnan(obs_t)
            n_valid_rep = mask.sum(dim=0).clamp(min=1)
            sq_res = torch.where(mask, sq_res, torch.zeros_like(sq_res))
            fit_arr = sq_res.sum(dim=0) / n_valid_rep  # [2, B]

            L_fit_off_list.append(fit_arr[0].unsqueeze(0))
            L_fit_on_list.append(fit_arr[1].unsqueeze(0))

        L_fit_off = torch.cat(L_fit_off_list, dim=0)  # [4, B]
        L_fit_on = torch.cat(L_fit_on_list, dim=0)    # [4, B]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Compute DeltaJ per process
        # aic_penalty = AIC_ALPHA * cost * (N / (B * n_valid_b))
        for p_idx, proc in enumerate(PROCESSES):
            if proc in removed_processes:
                continue
            cost = self.costs[proc]
            aic_pen = self.aic_alpha * cost * (N / (B * torch.clamp(n_valid_b, min=1.0)))  # [B]
            fit_diff = L_fit_off[p_idx] - L_fit_on[p_idx]  # [B]
            delta_J = fit_diff - aic_pen  # [B]

            dJ_np = delta_J.cpu().numpy()
            finite_mask = np.isfinite(dJ_np) & (dJ_np != 0)
            if finite_mask.sum() > 0:
                T_p = float(np.median(np.abs(dJ_np[finite_mask])))
                T_p = max(T_p, 1e-4)
            else:
                T_p = 0.02

            # Soft target q = sigmoid(DeltaJ / T_p)
            q_p = torch.sigmoid(delta_J / T_p)
            q_list[p_idx] = q_p.unsqueeze(-1)

            q_p_np = q_p.cpu().numpy()
            entropy = float(-np.nanmean(q_p_np * np.log(np.maximum(q_p_np, 1e-12)) + (1 - q_p_np) * np.log(np.maximum(1 - q_p_np, 1e-12))))

            # Confidence c = 2 * |q - 0.5|
            c_p_np = 2.0 * np.abs(q_p_np - 0.5)
            sum_c = float(np.sum(c_p_np))
            sum_c_sq = float(np.sum(c_p_np ** 2))
            n_eff = float((sum_c ** 2) / (sum_c_sq + 1e-12))

            # Concentration fractions
            sorted_c = np.sort(c_p_np)[::-1]
            k5 = max(1, int(0.05 * B))
            k10 = max(1, int(0.10 * B))
            k20 = max(1, int(0.20 * B))
            top5_frac = float(np.sum(sorted_c[:k5]) / (sum_c + 1e-12))
            top10_frac = float(np.sum(sorted_c[:k10]) / (sum_c + 1e-12))
            top20_frac = float(np.sum(sorted_c[:k20]) / (sum_c + 1e-12))

            diag[proc] = {
                "T_scale": T_p,
                "delta_J_median": float(np.nanmedian(dJ_np)),
                "delta_J_mean": float(np.nanmean(dJ_np)),
                "q_mean": float(np.nanmean(q_p_np)),
                "q_median": float(np.nanmedian(q_p_np)),
                "q_std": float(np.nanstd(q_p_np)),
                "c_mean": float(np.nanmean(c_p_np)),
                "c_median": float(np.nanmedian(c_p_np)),
                "c_std": float(np.nanstd(c_p_np)),
                "effective_n_samples": n_eff,
                "top5_loss_weight_fraction": top5_frac,
                "top10_loss_weight_fraction": top10_frac,
                "top20_loss_weight_fraction": top20_frac,
                "frac_q_gt05": float(np.nanmean(q_p_np > 0.5)),
                "delta_positive_fraction": float(np.nanmean(dJ_np > 0.0)),
                "frac_q_gt08": float(np.nanmean(q_p_np > 0.8)),
                "entropy": entropy,
            }
        q_tensor = torch.cat(q_list, dim=-1)  # [B, 4]
        return q_tensor.detach(), diag


class CFTrainer(MyTrainer):
    """Trainer with Counterfactual Structural Supervision (R15).

    Features:
      - Epoch-start deterministic evaluation of DeltaJ and soft targets q
      - Detached BCE loss L_CF on deterministic gate logit contrast
      - Pure fit/routing gradients for backbone and hydrologic parameter heads
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cf_generator = CounterfactualTargetGenerator(
            self.config,
            device=self.config.get("device", "cuda:0"),
        )
        print(f"[CFTrainer] Structural complexity lambda (aic_alpha) = {self.cf_generator.aic_alpha}")
        self.cached_q: Optional[torch.Tensor] = None
        self.epoch_cf_diag: Dict[str, Any] = {}
        self.cf_loss_weight: float = float(self.config.get("cf_loss_weight", 1.0))
        self.confidence_weighted_cf_loss: bool = bool(self.config.get("confidence_weighted_cf_loss", False))
        self._last_epoch_refreshed: int = -1

        # Dual-optimizer support (R17):
        # By default, weights_head uses dedicated Adam optimizer if configured
        self.structure_opt_name = str(self.config.get("structure_optimizer", "Adam"))
        self.structure_lr = float(self.config.get("structure_lr", 0.01))
        self.structure_weight_decay = float(self.config.get("structure_weight_decay", 1e-4))

        # Find structure parameters across model_dict
        self.weights_head_params = []
        for model in getattr(self.model, "model_dict", {}).values():
            nn_m = getattr(model, "nn_model", None)
            if nn_m is not None:
                if hasattr(nn_m, "structure_parameters"):
                    self.weights_head_params.extend(list(nn_m.structure_parameters()))
                elif hasattr(nn_m, "structure_encoder"):
                    self.weights_head_params.extend(list(nn_m.structure_encoder.parameters()))
                elif hasattr(nn_m, "heads") and "weights" in nn_m.heads:
                    self.weights_head_params.extend(list(nn_m.heads["weights"].parameters()))
        self.weights_head_param_ids = {id(p) for p in self.weights_head_params}
        self.primary_params = [p for p in self.model.get_parameters() if id(p) not in self.weights_head_param_ids]

        if self.structure_opt_name.lower() == "adam" and len(self.weights_head_params) > 0:
            # Re-initialize primary optimizer to exclude weights_head parameters
            opt_cls = getattr(torch.optim, self.config["train"]["optimizer"])
            self.optimizer = opt_cls(
                self.primary_params,
                lr=self.config["train"]["learning_rate"],
                weight_decay=self.config["train"].get("weight_decay", 0.0),
            )
            self.structure_optimizer = torch.optim.Adam(
                self.weights_head_params,
                lr=self.structure_lr,
                weight_decay=self.structure_weight_decay,
            )
            print(f"[CFTrainer Dual Optimizer] Primary: {self.config['train']['optimizer']} (lr={self.config['train']['learning_rate']}, {len(self.primary_params)} params) | "
                  f"Structure: Adam (lr={self.structure_lr}, {len(self.weights_head_params)} params)")
        else:
            self.structure_optimizer = None
            total_p = len(list(self.model.get_parameters()))
            print(f"[CFTrainer Unified Optimizer] Single {self.config['train']['optimizer']} optimizer (lr={self.config['train']['learning_rate']}, total {total_p} params including structure encoder)")

    def refresh_targets_if_needed(self, epoch: int) -> None:
        if self._last_epoch_refreshed == epoch and self.cached_q is not None:
            return

        t0 = time.perf_counter()
        q_tensor, diag = self.cf_generator.generate_targets(
            self.model,
            self.train_dataset,
        )
        self.cached_q = q_tensor.to(self.config.get("device", "cuda:0"))
        self.epoch_cf_diag = diag
        self._last_epoch_refreshed = epoch
        t_refresh = time.perf_counter() - t0

        # Emit progress
        int_diag = diag.get("w_int", {})
        phen_diag = diag.get("w_phen", {})
        snow_diag = diag.get("w_snow", {})
        sub_diag = diag.get("w_sub", {})
        print(f"[R15 Target Refresh Ep {epoch}] t={t_refresh:.2f}s | "
              f"w_int: T={int_diag.get('T_scale', 0):.4f}, frac_ON={int_diag.get('frac_q_gt05', 0)*100:.1f}%, mean_q={int_diag.get('q_mean', 0):.3f} | "
              f"w_phen: frac_ON={phen_diag.get('frac_q_gt05', 0)*100:.1f}% | "
              f"w_snow: frac_ON={snow_diag.get('frac_q_gt05', 0)*100:.1f}% | "
              f"w_sub: frac_ON={sub_diag.get('frac_q_gt05', 0)*100:.1f}%", flush=True)

    @torch.no_grad()
    def _write_structural_diagnostics(
        self,
        epoch: int,
        *,
        total_loss: float,
        fit_loss: float,
        cf_loss: float,
    ) -> None:
        """Persist per-epoch structural scores and hard gates for CF/DFlex audits."""
        if self.cached_q is None:
            return

        dpl_model = next(iter(self.model.model_dict.values()))
        nn = dpl_model.nn_model
        n_attr = int(nn.backbone[0].in_features)
        attrs = self.train_dataset["xc_nn_norm"][0, :, -n_attr:].to(self.cached_q.device)
        was_training = nn.training
        nn.eval()
        try:
            raw_weights = nn.get_structure_logits(attrs)
            logits = raw_weights.view(raw_weights.shape[0], 4, 2).clamp(-10.0, 10.0)
            p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])
        finally:
            if was_training:
                nn.train()

        q = self.cached_q.detach()
        hard = (p_struct > 0.5).to(p_struct.dtype)
        c = 2.0 * torch.abs(q - 0.5)
        bce_elem = F.binary_cross_entropy(p_struct, q, reduction="none")
        rows: dict[str, Any] = {}
        for index, proc in enumerate(PROCESSES):
            p = p_struct[:, index]
            target = q[:, index] > 0.5  # equivalent to DeltaJ > 0
            pred = hard[:, index] > 0.5
            tp = float((pred & target).sum().item())
            fp = float((pred & ~target).sum().item())
            fn = float((~pred & target).sum().item())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            weight_sum = float(c[:, index].sum().item())
            rows[proc] = {
                "p_struct_mean": float(p.mean().item()),
                "p_struct_median": float(p.median().item()),
                "p_struct_std": float(p.std(unbiased=False).item()),
                "p_struct_min": float(p.min().item()),
                "p_struct_max": float(p.max().item()),
                "hard_on_fraction": float(hard[:, index].mean().item()),
                "delta_positive_fraction": float(target.float().mean().item()),
                "bce": float((c[:, index] * bce_elem[:, index]).sum().item() / (weight_sum + EPS)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        output_dir = Path(self.config.get("output_dir", self.config.get("save_path", ".")))
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        np.save(diag_dir / f"p_struct_epoch_{epoch:03d}.npy", p_struct.cpu().numpy())
        np.save(diag_dir / f"hard_gates_epoch_{epoch:03d}.npy", hard.cpu().numpy())
        np.save(diag_dir / f"q_epoch_{epoch:03d}.npy", q.cpu().numpy())
        record = {
            "epoch": int(epoch),
            "variant": "DFlex-CF/BCE" if bool(getattr(dpl_model.phy_model, "is_dflex", False)) else "CFlex-CF/BCE",
            "total_loss": float(total_loss),
            "fit_loss": float(fit_loss),
            "bce_loss": float(cf_loss),
            "processes": rows,
        }
        with (output_dir / "structural_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            "[CF structural Ep %d] " % epoch
            + " | ".join(
                f"{proc}: p={rows[proc]['p_struct_mean']:.3f}, hard={rows[proc]['hard_on_fraction']:.3f}, "
                f"delta+={rows[proc]['delta_positive_fraction']:.3f}, F1={rows[proc]['f1']:.3f}"
                for proc in PROCESSES
            ),
            flush=True,
        )

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        self.refresh_targets_if_needed(epoch)

        start_time = time.perf_counter()
        self.current_epoch = epoch
        self.total_loss = 0.0
        total_fit_aic_loss = 0.0
        total_cf_loss = 0.0
        nonfinite_batches = 0

        dpl_model = next(iter(self.model.model_dict.values()))
        nn = dpl_model.nn_model
        phy = dpl_model.phy_model

        for mb in range(1, n_minibatch + 1):
            self.current_batch = mb

            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            # Forward pass through model (eval=False for training)
            out_dict = self.model(dataset_sample)
            loss_fit_aic = self.model.calc_loss(dataset_sample)

            # Compute L_CF on current minibatch
            # Sample indices in [0, B-1]
            batch_sample = dataset_sample["batch_sample"]
            if isinstance(batch_sample, np.ndarray):
                batch_sample = torch.from_numpy(batch_sample).long().to(self.cached_q.device)
            elif isinstance(batch_sample, torch.Tensor):
                batch_sample = batch_sample.long().to(self.cached_q.device)

            q_batch = self.cached_q[batch_sample]  # [B_batch, 4]

            # Deterministic two-logit contrast with stopgrad to backbone
            # In LearnedStructureNetCF, nn.heads["weights"] already receives shared.detach()
            # If standard LearnedStructureNet is used, we ensure stopgrad explicitly:
            attrs_b = dataset_sample["xc_nn_norm"][0, :, -nn.backbone[0].in_features:].to(self.cached_q.device) if "xc_nn_norm" in dataset_sample else dataset_sample["c_nn_norm"].to(self.cached_q.device)
            # Model owns structure-logit extraction from static attributes
            if hasattr(nn, "get_structure_logits"):
                raw_weights = nn.get_structure_logits(attrs_b)
            elif hasattr(nn, "structure_encoder"):
                raw_weights = nn.structure_encoder(attrs_b)
            elif hasattr(nn, "heads") and "weights" in nn.heads:
                with torch.no_grad():
                    shared_detached = nn.backbone(attrs_b)
                raw_weights = nn.heads["weights"](shared_detached)
            else:
                raise ValueError(f"NN model {type(nn)} does not support structure logit extraction.")
            raw_weights = torch.clamp(raw_weights, min=-10.0, max=10.0)
            logits = raw_weights.view(raw_weights.shape[0], 4, 2)
            # Contrast z_on - z_off
            z_contrast = logits[..., 1] - logits[..., 0]  # [B_batch, 4]
            p_struct = torch.sigmoid(z_contrast)          # [B_batch, 4]
            if self.confidence_weighted_cf_loss:
                # Bounded confidence: c = 2 * |q - 0.5|
                c_batch = (2.0 * torch.abs(q_batch - 0.5)).detach()  # [B_batch, 4]
                bce_elem = F.binary_cross_entropy(p_struct, q_batch, reduction="none")  # [B_batch, 4]
                sum_c = torch.sum(c_batch, dim=0)  # [4]
                weighted_bce = torch.sum(c_batch * bce_elem, dim=0) / (sum_c + 1e-12)  # [4]
                loss_cf = torch.mean(weighted_bce)  # scalar across 4 processes
            else:
                loss_cf = F.binary_cross_entropy(p_struct, q_batch)
            loss = loss_fit_aic + self.cf_loss_weight * loss_cf

            if not torch.isfinite(loss).item():
                nonfinite_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in self.model.get_parameters()
            )
            if not gradients_finite:
                nonfinite_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.structure_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.primary_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.weights_head_params, max_norm=1.0)
                self.optimizer.step()
                self.structure_optimizer.step()
                self.optimizer.zero_grad()
                self.structure_optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.get_parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.total_loss += loss.item()
            total_fit_aic_loss += loss_fit_aic.item()
            total_cf_loss += loss_cf.item()

        if nonfinite_batches:
            log.warning(
                "Skipped %d non-finite minibatch update(s) in epoch %d",
                nonfinite_batches,
                epoch,
            )

        if self.use_scheduler:
            self.scheduler.step()

        self._final_loss = self.total_loss / max(n_minibatch, 1)
        mean_fit_aic = total_fit_aic_loss / max(n_minibatch, 1)
        mean_cf = total_cf_loss / max(n_minibatch, 1)
        self._write_structural_diagnostics(
            epoch,
            total_loss=self._final_loss,
            fit_loss=mean_fit_aic,
            cf_loss=mean_cf,
        )

        # Log epoch stats
        t_epoch = time.perf_counter() - start_time
        print(f"[R15 Epoch {epoch:2d}/{self.epochs}] t={t_epoch:.1f}s | "
              f"Loss_total={self._final_loss:.4f} | Loss_fit_aic={mean_fit_aic:.4f} | Loss_CF={mean_cf:.4f}", flush=True)

        # Save model and trainer states
        if epoch % self.config['train']['save_epoch'] == 0:
            self.model.save_model(epoch)
            from dmg.core.utils.utils import save_train_state
            save_train_state(
                self._model_dir(),
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=True,
            )
