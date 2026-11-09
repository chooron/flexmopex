#!/usr/bin/env python3
"""Run a single S2 v1.1 formal experiment with full logging, early stopping, and checkpointing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.synthetic.strict.s2_loader import load_candidate_strict_fold
from train.synthetic.strict import trainer as strict_trainer
from train.synthetic.strict.trainer import (
    PROCESS_ORDER,
    StrictTrainer,
    build_strict_model_and_loss,
)
from dmg.core.data.data import create_training_grid

FLEX = ROOT / "train"
S2_DIR = ROOT / "inputs" / "synthetic"
CONFIG = Path(os.environ.get(
    "FLEXMOPEX_LEARNER_CONFIG",
    str(ROOT / "inputs" / "trained" / "normal" / "flex" / "ref" / "seed_42" / "config_snapshot.json")))
SPLIT = FLEX / "synthetic" / "split.json"
_DATA = Path(os.environ.get("FLEXMOPEX_DATA_DIR", str(ROOT / "data")))
IDS = Path(os.environ.get("FLEXMOPEX_GAGE_ID", str(_DATA / "gage_id.npy")))
import dmg.core.data.loaders.hydro_loader as _hl_mod
HYDRO_SOURCE = Path(_hl_mod.__file__).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single S2 formal experiment")
    parser.add_argument("--run_id", type=str, required=True, help="e.g. E1_A_42")
    # Batch-1/2 use the four frozen scenarios. Batch-3 (E2 / E5) needs additional
    # scenario names plus an explicit dataset path; both extensions are additive and
    # leave the frozen A/B/C/D behaviour byte-for-byte unchanged.
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--method", type=str, required=True, choices=["Canonical_BCE", "Plain_BCE", "Direct_Fit", "Matched_Direct_Fit", "Strict_Direct_Fit"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_epochs", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--preflight_only", action="store_true", help="Run one minibatch update without saving a checkpoint")
    parser.add_argument("--preflight_batch_size", type=int, default=2)
    args = parser.parse_args()

    t_start = time.perf_counter()
    out_dir = Path(args.output_dir).resolve()
    protected = [(ROOT / p).resolve() for p in ("data", "inputs", "inputs/evidence", "inputs/trained", "inputs/synthetic")]
    if any(out_dir == base or base in out_dir.parents for base in protected):
        raise ValueError(f"training output overlaps input data: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_trainer.STRICT_RUN_ROOT = out_dir
    log_file = out_dir / "train.log"

    def log(msg: str) -> None:
        t_stamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{t_stamp} {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log(f"=== Starting Experiment {args.run_id} ===")
    log(f"Scenario: {args.scenario} | Method: {args.method} | Seed: {args.seed} | Device: {args.device}")
    log(f"Output Dir: {out_dir}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    # Load configuration
    cfg = json.loads(CONFIG.read_text())
    # Snapshot identities refer to the archived class name; use the local model.
    cfg["model"]["phy"]["name"] = ["Mopex"]
    cfg["model"]["phy"]["model"] = ["Mopex"]
    cfg["delta_model"]["phy_model"]["model"] = ["Mopex"]
    cfg.update({
        "mode": "train",
        "device": args.device,
        "dtype": torch.float32,
    })
    cfg["model"]["rho"] = 365
    cfg["model"]["warmup"] = 365
    cfg["model"]["nmul"] = 4
    cfg["delta_model"]["phy_model"]["nmul"] = 4
    cfg["confidence_weighted_cf_loss"] = (args.method == "Canonical_BCE")
    if args.method == "Strict_Direct_Fit":
        # Gumbel-matched Direct-fit (R1-036, P0-Final). Keep the unified
        # forward pass -- hence the identical training-mode Gumbel-softmax gate
        # sampling and the identical physics/parameter path -- but disable
        # counterfactual supervision so that the fit loss and the k_p-weighted
        # complexity charge both reach the structure encoder. No BCE term is
        # computed or applied in this method.
        cfg["model"]["phy"]["counterfactual_supervision"] = False
        cfg["delta_model"]["phy_model"]["counterfactual_supervision"] = False
        cfg["counterfactual_supervision"] = False
    cfg["structure_optimizer"] = "none"
    cfg["train"]["batch_size"] = 100
    if args.preflight_only:
        if args.preflight_batch_size < 1:
            parser.error("--preflight_batch_size must be positive")
        cfg["train"]["batch_size"] = args.preflight_batch_size
        cfg["model"]["phy"]["disable_compile"] = True
    cfg["train"]["optimizer"] = "Adadelta"
    cfg["train"]["learning_rate"] = 1.0
    cfg["train"]["start_epoch"] = 0
    cfg["train"]["epochs"] = args.max_epochs
    cfg["loss_function"]["aic_alpha"] = 0.007
    cfg["cf_loss_weight"] = 1.0

    dataset_path = (
        Path(args.dataset_path).resolve()
        if args.dataset_path
        else S2_DIR / f"scenario_{args.scenario}/dataset.pkl"
    )
    log(f"Dataset: {dataset_path} (SHA-256: {sha256_file(dataset_path)})")

    # Load dataset & split
    strict_root = out_dir / "work"
    strict_root.mkdir(parents=True, exist_ok=True)
    cache_dir = strict_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = load_candidate_strict_fold(
        cfg,
        manifest_path=SPLIT,
        cache_root=cache_dir,
        dataset_path=dataset_path,
        gage_info_path=IDS,
        hydro_loader_source=HYDRO_SOURCE,
    )
    train_ids = list(data.fold.train_ids)
    holdout_ids = list(data.fold.holdout_ids)
    log(f"Dataset split loaded: {len(train_ids)} train basins, {len(holdout_ids)} holdout basins")

    model, loss_func = build_strict_model_and_loss(cfg, data, device=args.device, model_dir=strict_root / "model_build")
    trainer = StrictTrainer(cfg, model, data, loss_func, strict_root / "trainer_output")

    n_samples, n_minibatch, n_timesteps = create_training_grid(data.train_dataset["xc_nn_norm"], trainer.strict_config)
    log(f"Grid: n_samples={n_samples}, n_minibatch={n_minibatch}, n_timesteps={n_timesteps}")

    loss_history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    # Custom minibatch logic based on method
    if args.method == "Plain_BCE":
        def custom_minibatch(sample: Mapping[str, torch.Tensor], diagnose: bool = False) -> dict[str, Any]:
            trainer.optimizer.zero_grad(set_to_none=True)
            _ = trainer.model(dict(sample))
            fit_aic = trainer.model.calc_loss(dict(sample), loss_func=trainer.loss_func)
            
            # Plain unweighted BCE
            dpl = next(iter(trainer.model.model_dict.values()))
            attrs_b = sample["xc_nn_norm"][0, :, -35:].to(args.device)
            raw_w = dpl.nn_model.get_structure_logits(attrs_b).view(-1, 4, 2).clamp(-10.0, 10.0)
            p_struct = torch.sigmoid(raw_w[..., 1] - raw_w[..., 0])
            batch_sample = sample["batch_sample"].long().to(args.device)
            q_batch = trainer.cached_targets.q[batch_sample]
            plain_bce = F.binary_cross_entropy(p_struct, q_batch, reduction="mean")
            
            total = fit_aic + float(trainer.strict_config.get("cf_loss_weight", 1.0)) * plain_bce
            
            # Gradient isolation
            inner_m = next(iter(trainer.model.model_dict.values()))
            params = [p for _, p in inner_m.named_parameters() if p.requires_grad]
            
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            return {
                "updated": True,
                "total": float(total.detach().item()),
                "fit_aic": float(fit_aic.detach().item()),
                "cf_bce": float(plain_bce.detach().item()),
            }
        trainer._training_minibatch = custom_minibatch

    elif args.method == "Direct_Fit":
        def custom_minibatch(sample: Mapping[str, torch.Tensor], diagnose: bool = False) -> dict[str, Any]:
            trainer.optimizer.zero_grad(set_to_none=True)
            dpl = next(iter(trainer.model.model_dict.values()))
            attrs_b = sample["xc_nn_norm"][0, :, -35:].to(args.device)
            raw_w = dpl.nn_model.get_structure_logits(attrs_b).view(-1, 4, 2).clamp(-10.0, 10.0)
            p_struct = torch.sigmoid(raw_w[..., 1] - raw_w[..., 0])  # differentiable soft gates [B, 4]
            
            # Forward physics loop with soft gates
            P, T_f, PET, doy, n_steps, _ = dpl.phy_model._prepare_forcings(dict(sample))
            p_raw = dpl.nn_model({"c_nn_norm": attrs_b})
            mopex_params = dpl.phy_model._descale_mopex_params(p_raw["params"])
            routing = dpl.phy_model._descale_routing_params(p_raw["gamma_uh"])
            
            B_b = sample["x_phy"].shape[1]
            Q = dpl.phy_model._run_weighted_loop(P, T_f, PET, doy, mopex_params, p_struct, n_steps, B_b)
            Qr = dpl.phy_model._apply_routing(Q.mean(-1), routing)
            out_streamflow = Qr[:, :B_b, 0]
            
            # Fit loss against synthetic target flow
            target_flow = sample["target"][:, :, 0].to(args.device)
            tar_trimmed = target_flow[dpl.phy_model.warm_up : dpl.phy_model.warm_up + out_streamflow.shape[0]]
            mask = ~torch.isnan(tar_trimmed)
            fit_loss = F.mse_loss(out_streamflow[mask], tar_trimmed[mask])
            
            complexity_penalty = 0.007 * p_struct.mean()
            total = fit_loss + complexity_penalty
            
            inner_m = next(iter(trainer.model.model_dict.values()))
            params = [p for _, p in inner_m.named_parameters() if p.requires_grad]
            
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            return {
                "updated": True,
                "total": float(total.detach().item()),
                "fit_aic": float(fit_loss.detach().item()),
                "cf_bce": float(complexity_penalty.detach().item()),
            }
        trainer._training_minibatch = custom_minibatch

    elif args.method == "Matched_Direct_Fit":
        def custom_minibatch(sample: Mapping[str, torch.Tensor], diagnose: bool = False) -> dict[str, Any]:
            trainer.optimizer.zero_grad(set_to_none=True)
            dpl = next(iter(trainer.model.model_dict.values()))
            attrs_b = sample["xc_nn_norm"][0, :, -35:].to(args.device)
            raw_w = dpl.nn_model.get_structure_logits(attrs_b).view(-1, 4, 2).clamp(-10.0, 10.0)
            p_struct = torch.sigmoid(raw_w[..., 1] - raw_w[..., 0])  # differentiable soft gates [B, 4]

            # Forward physics loop with soft gates
            P, T_f, PET, doy, n_steps, _ = dpl.phy_model._prepare_forcings(dict(sample))
            p_raw = dpl.nn_model({"c_nn_norm": attrs_b})
            mopex_params = dpl.phy_model._descale_mopex_params(p_raw["params"])
            routing = dpl.phy_model._descale_routing_params(p_raw["gamma_uh"])

            B_b = sample["x_phy"].shape[1]
            Q = dpl.phy_model._run_weighted_loop(P, T_f, PET, doy, mopex_params, p_struct, n_steps, B_b)
            Qr = dpl.phy_model._apply_routing(Q.mean(-1), routing)
            out_streamflow = Qr[:, :B_b, 0]

            # Standardized fit loss (Methods Eq. 7) + process-weighted complexity penalty (Methods Eq. 8)
            tar_trimmed = sample["target"][dpl.phy_model.warm_up : dpl.phy_model.warm_up + out_streamflow.shape[0]]
            weights_dict = {
                "w_phen": p_struct[:, 0:1],
                "w_int": p_struct[:, 1:2],
                "w_snow": p_struct[:, 2:3],
                "w_sub": p_struct[:, 3:4],
            }
            total = trainer.loss_func(
                out_streamflow,
                tar_trimmed,
                sample_ids=sample["batch_sample"],
                weights=weights_dict,
            )

            # Component breakdown for logging (identical definition to NseDynAicBatchLoss)
            pred_f, tar_f = trainer.loss_func._format(out_streamflow, tar_trimmed)
            sample_ids_np = sample["batch_sample"].cpu().numpy().astype(int)
            std_batch = torch.tensor(
                np.tile(trainer.loss_func.std[sample_ids_np].T, (tar_f.shape[0], 1)),
                dtype=torch.float32,
                device=args.device,
            )
            mask = ~torch.isnan(tar_f)
            sq_res = (pred_f[mask] - tar_f[mask]) ** 2
            fit_loss = torch.mean(sq_res / (std_batch[mask] + trainer.loss_func.eps) ** 2)
            complexity_penalty = total - fit_loss

            inner_m = next(iter(trainer.model.model_dict.values()))
            params = [p for _, p in inner_m.named_parameters() if p.requires_grad]

            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            return {
                "updated": True,
                "total": float(total.detach().item()),
                "fit_aic": float(fit_loss.detach().item()),
                "cf_bce": float(complexity_penalty.detach().item()),
            }
        trainer._training_minibatch = custom_minibatch

    elif args.method == "Strict_Direct_Fit":
        def custom_minibatch(sample: Mapping[str, torch.Tensor], diagnose: bool = False) -> dict[str, Any]:
            """Gumbel-matched Direct-fit: unified forward, no BCE, gates trainable.

            The forward pass is the model's own (``counterfactual_supervision``
            disabled), so the structure gates used in the physics loop are drawn
            from the same relaxed Gumbel-softmax sampler Canonical uses, and the
            AIC/complexity outputs are not detached. The only loss is
            ``NseDynAicBatchLoss`` = Eq. (7) fit term + lambda * sum_p k_p w_p.
            """
            trainer.optimizer.zero_grad(set_to_none=True)
            _ = trainer.model(dict(sample))
            total = trainer.model.calc_loss(dict(sample), loss_func=trainer.loss_func)
            if not bool(torch.isfinite(total).item()):
                trainer.optimizer.zero_grad(set_to_none=True)
                return {"updated": False, "reason": "nonfinite_loss",
                        "total": float("nan"), "fit_aic": float("nan"), "cf_bce": 0.0}
            # Component split for logging only (no effect on the update).
            dpl = next(iter(trainer.model.model_dict.values()))
            name = next(iter(trainer.model.model_dict))
            with torch.no_grad():
                out = trainer.model.output_dict[name]
                pred = out[trainer.model.target_names[0]]
                pred, target = trainer.model._trim(pred, sample["target"])
                w = {k: v for k, v in out.items() if k.startswith("w_")}
                fit_only = trainer.loss_func(pred, target, sample_ids=sample["batch_sample"], weights=None)
            inner = next(iter(trainer.model.model_dict.values()))
            params = [p for _, p in inner.named_parameters() if p.requires_grad]
            total.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in params):
                trainer.optimizer.zero_grad(set_to_none=True)
                return {"updated": False, "reason": "nonfinite_gradient",
                        "total": float(total.detach().item()), "fit_aic": float(fit_only.item()),
                        "cf_bce": 0.0}
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            return {
                "updated": True,
                "total": float(total.detach().item()),
                "fit_aic": float(fit_only.item()),
                "cf_bce": float(total.detach().item()) - float(fit_only.item()),
            }
        trainer._training_minibatch = custom_minibatch

    if args.preflight_only:
        log("=== Running Preflight Minibatch Check ===")
        if args.method in ("Canonical_BCE", "Plain_BCE"):
            log("Generating counterfactual targets for BCE preflight...")
            trainer.cached_targets = trainer.cf_generator.generate(trainer.model, trainer.fold_data)
        sample = trainer.sampler.get_training_sample(
            trainer.train_dataset,
            ngrid_train=len(trainer.fold.train_ids),
            nt=n_timesteps,
        )
        res = trainer._training_minibatch(sample)
        log(f"Preflight check completed: updated={res.get('updated')}, total_loss={res.get('total'):.6f}, fit={res.get('fit_aic'):.6f}, complexity={res.get('cf_bce'):.6f}")
        log("=== Preflight Finished Successfully (one minibatch update; no checkpoint saved) ===")
        return
    # Training loop
    stopped_by_early_stopping = False

    for epoch in range(1, args.max_epochs + 1):
        ep_t0 = time.perf_counter()

        # Target refresh for BCE methods
        if args.method in ("Canonical_BCE", "Plain_BCE"):
            ref_t0 = time.perf_counter()
            trainer.cached_targets = trainer.cf_generator.generate(trainer.model, trainer.fold_data)
            ref_sec = time.perf_counter() - ref_t0
        else:
            ref_sec = 0.0

        # Minibatch loop
        trainer.total_loss = 0.0
        fit_total = 0.0
        bce_total = 0.0
        finite_batches = 0

        for mb in range(1, n_minibatch + 1):
            sample = trainer.sampler.get_training_sample(
                trainer.train_dataset,
                ngrid_train=len(trainer.fold.train_ids),
                nt=n_timesteps,
            )
            res = trainer._training_minibatch(sample)
            if res["updated"]:
                trainer.total_loss += float(res["total"])
                fit_total += float(res["fit_aic"])
                bce_total += float(res["cf_bce"])
                finite_batches += 1

        divisor = max(1, finite_batches)
        ep_total = trainer.total_loss / divisor
        ep_fit = fit_total / divisor
        ep_bce = bce_total / divisor
        ep_sec = time.perf_counter() - ep_t0

        # Evaluate gate predictions on all 671 basins
        inner = next(iter(trainer.model.model_dict.values()))
        with torch.no_grad():
            inner.eval()
            # Extract static attributes for train set
            train_attrs = data.train_dataset["xc_nn_norm"][0, :, -35:].to(args.device)
            p_train = torch.sigmoid(inner.nn_model.get_structure_logits(train_attrs).view(-1, 4, 2).clamp(-10.0, 10.0)[..., 1] - inner.nn_model.get_structure_logits(train_attrs).view(-1, 4, 2).clamp(-10.0, 10.0)[..., 0]).mean(dim=0).cpu().tolist()
            inner.train()

        ep_rec = {
            "epoch": epoch,
            "seconds": ep_sec,
            "target_refresh_seconds": ref_sec,
            "loss_total": ep_total,
            "loss_fit_aic": ep_fit,
            "loss_cf_bce": ep_bce,
            "p_struct_train_mean": {proc: p_train[i] for i, proc in enumerate(PROCESS_ORDER)},
        }
        loss_history.append(ep_rec)

        log(f"Epoch {epoch:03d}/{args.max_epochs:03d} ({ep_sec:.1f}s): Total={ep_total:.4f} (Fit={ep_fit:.4f}, BCE={ep_bce:.4f}) | p_struct_tr={list(np.round(p_train, 3))}")

        # Best model checkpoint check
        if ep_total < best_loss:
            best_loss = ep_total
            best_epoch = epoch
            patience_counter = 0
            best_ckpt_file = out_dir / "best_checkpoint.pt"
            trainer.save_checkpoint(best_ckpt_file, step=epoch)
            
            # Save predictions at best epoch
            with torch.no_grad():
                inner.eval()
                all_raw_attrs = data.train_dataset["xc_nn_norm"][0, :, -35:].to(args.device)
                logits_all = inner.nn_model.get_structure_logits(all_raw_attrs).view(-1, 4, 2).clamp(-10.0, 10.0)
                p_best_train = torch.sigmoid(logits_all[..., 1] - logits_all[..., 0]).cpu().numpy()
                np.save(out_dir / "p_struct_best_train.npy", p_best_train)
                if trainer.cached_targets is not None:
                    np.save(out_dir / "q_best_train.npy", trainer.cached_targets.q.cpu().numpy())
                inner.train()
        else:
            if epoch >= args.min_epochs:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log(f"Early stopping triggered at Epoch {epoch}! Best Epoch was {best_epoch} with Total Loss {best_loss:.4f}.")
                    stopped_by_early_stopping = True
                    break

    # Save final checkpoint
    final_epoch = loss_history[-1]["epoch"]
    final_ckpt_file = out_dir / "final_checkpoint.pt"
    trainer.save_checkpoint(final_ckpt_file, step=final_epoch)
    
    with torch.no_grad():
        inner.eval()
        train_attrs = data.train_dataset["xc_nn_norm"][0, :, -35:].to(args.device)
        logits_final = inner.nn_model.get_structure_logits(train_attrs).view(-1, 4, 2).clamp(-10.0, 10.0)
        p_final_train = torch.sigmoid(logits_final[..., 1] - logits_final[..., 0]).cpu().numpy()
        np.save(out_dir / "p_struct_final_train.npy", p_final_train)
        if trainer.cached_targets is not None:
            np.save(out_dir / "q_final_train.npy", trainer.cached_targets.q.cpu().numpy())
        inner.train()

    total_time = time.perf_counter() - t_start
    log(f"=== Experiment {args.run_id} Completed in {total_time:.1f}s ({total_time/3600.0:.2f}h) ===")
    log(f"Final Epoch: {final_epoch} | Best Epoch: {best_epoch} (Best Loss: {best_loss:.4f}) | Early Stopped: {stopped_by_early_stopping}")

    # Write loss_history.json
    with open(out_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)

    # Save summary.json
    summary = {
        "run_id": args.run_id,
        "scenario": args.scenario,
        "method": args.method,
        "seed": args.seed,
        "device": args.device,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "total_runtime_seconds": total_time,
        "final_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_total_loss": float(best_loss),
        "stopped_by_early_stopping": stopped_by_early_stopping,
        "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "checkpoints": {
            "best": {
                "path": str(best_ckpt_file.name),
                "bytes": best_ckpt_file.stat().st_size,
                "sha256": sha256_file(best_ckpt_file),
                "epoch": best_epoch,
            },
            "final": {
                "path": str(final_ckpt_file.name),
                "bytes": final_ckpt_file.stat().st_size,
                "sha256": sha256_file(final_ckpt_file),
                "epoch": final_epoch,
            }
        }
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
