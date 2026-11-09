import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer
from train.models.early_stopping import EarlyStoppingController

log = logging.getLogger(__name__)


class MyTrainer(BaseTrainer):
    """Generic, unified trainer for neural networks and differentiable models.

    Inspired by the Hugging Face Trainer class.
    
    Retrieves and formats data, initializes optimizers/schedulers/loss functions,
    and runs training and testing/inference loops.
    
    Parameters
    ----------
    config
        Configuration settings for the model and experiment.
    model
        Learnable model object. If not provided, a new model is initialized.
    train_dataset
        Training dataset dictionary.
    eval_dataset
        Testing/inference dataset dictionary.
    dataset
        Inference dataset dictionary.
    loss_func
        Loss function object. If not provided, a new loss function is initialized.
    optimizer
        Optimizer object for learning model states. If not provided, a new
        optimizer is initialized.
    scheduler
        Learning rate scheduler. If not provided, a new scheduler is initialized.
    verbose
        Whether to print verbose output.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
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
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = import_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False
        self.early_stopping = EarlyStoppingController.from_config(config)
        self.training_history: list[dict[str, Any]] = []
        self.stop_epoch = 0
        self.best_epoch = None
        self.best_checkpoint_path: str | None = None
        self.last_checkpoint_path: str | None = None
        if 'train' in config['mode']:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            log.info("Initializing experiment")
            self.epochs = self.config['train']['epochs']

            # Loss function
            self.loss_func = loss_func or load_criterion(
                self.train_dataset['target'],
                config['loss_function'],
                device=config['device'],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()
            if config['delta_model']['nn_model']['lr_scheduler']:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            # Resume model training by loading prior states.
            # self.start_epoch = self.config['train']['start_epoch'] + 1
            # if self.start_epoch > 1:
            self.load_states()
        elif 'test' in config['mode']:
            self.load_test_states()

    def _model_dir(self) -> str:
        return self.config.get('model_dir', self.config['model_path'])

    def _primary_model_name(self) -> str:
        model_name = self.config.get('model', {}).get('phy', {}).get('name')
        if model_name is None:
            model_name = self.config['delta_model']['phy_model']['model']
        if isinstance(model_name, list):
            model_name = model_name[0]
        return str(model_name).lower()

    def _find_latest_model_epoch(self, directory: str) -> tuple[int, str | None]:
        if not os.path.isdir(directory):
            return 0, None

        pattern = re.compile(rf"{re.escape(self._primary_model_name())}_ep(\d+)\.pt$")
        latest_epoch = 0
        latest_path = None

        for file_name in os.listdir(directory):
            match = pattern.fullmatch(file_name)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = os.path.join(directory, file_name)

        return latest_epoch, latest_path

    def _find_latest_train_state(self, directory: str) -> tuple[int, str | None]:
        if not os.path.isdir(directory):
            return 0, None

        latest_epoch = 0
        latest_path = None

        for file_name in os.listdir(directory):
            if 'train_state' not in file_name:
                continue

            file_path = os.path.join(directory, file_name)
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
            except Exception as exc:
                log.warning("Skipping unreadable train state %s: %s", file_path, exc)
                continue

            epoch = int(checkpoint.get('epoch', 0))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = file_path

        return latest_epoch, latest_path

    def _model_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self._model_dir(), f"{self._primary_model_name()}_ep{int(epoch)}.pt")

    def _sync_output_dir(self, path: Path | str) -> tuple[str, str]:
        path_str = str(path)
        original = (self.config['out_path'], self.config['sim_dir'])
        self.config['out_path'] = path_str
        self.config['sim_dir'] = path_str
        return original

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer.
        
        Adding additional optimizers is possible by extending the optimizer_dict.

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer object.
        """
        name = self.config['train']['optimizer']
        optimizer_dict = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'Adadelta': torch.optim.Adadelta,
            'RMSprop': torch.optim.RMSprop,
        }

        # Fetch optimizer class
        cls = optimizer_dict[name]
        if cls is None:
            raise ValueError(f"Optimizer '{name}' not recognized. "
                             f"Available options are: {list(optimizer_dict.keys())}")

        # Initialize
        try:
            self.optimizer = cls(
                self.model.get_parameters(),
                lr=self.config['train']['learning_rate'],
                weight_decay=self.config['train'].get('weight_decay', 0.0)
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing optimizer: {e}") from e
        return self.optimizer

    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Initialize a learning rate scheduler for the optimizer.
        
        torch.optim.lr_scheduler.LRScheduler
            Initialized learning rate scheduler object.
        """
        name = self.config['delta_model']['train']['lr_scheduler']
        scheduler_dict = {
            'StepLR': torch.optim.lr_scheduler.StepLR,
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        }

        # Fetch scheduler class
        cls = scheduler_dict[name]
        if cls is None:
            raise ValueError(f"Scheduler '{name}' not recognized. "
                             f"Available options are: {list(scheduler_dict.keys())}")

        # Initialize
        try:
            self.scheduler = cls(
                self.optimizer,
                **self.config['delta_model']['train']['lr_scheduler_params'],
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_states(self) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint to resume
        training if a checkpoint file exists.
        """
        path = self._model_dir()
        os.makedirs(path, exist_ok=True)
        self.start_epoch = 1

        latest_model_epoch, latest_model_path = self._find_latest_model_epoch(path)
        if latest_model_path is None:
            log.info("No checkpoint found in %s; training from scratch.", path)
            return

        self.model.load_model(epoch=latest_model_epoch)
        self.start_epoch = latest_model_epoch + 1
        self.config['train']['start_epoch'] = latest_model_epoch

        latest_state_epoch, latest_state_path = self._find_latest_train_state(path)
        if latest_state_path is not None and latest_state_epoch == latest_model_epoch:
            checkpoint = torch.load(latest_state_path, map_location='cpu')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            scheduler_state = checkpoint.get('scheduler_state_dict')
            if self.scheduler and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)

            random_state = checkpoint.get('random_state')
            if random_state is not None:
                torch.set_rng_state(random_state)

            if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
        elif latest_state_path is not None:
            log.warning(
                "Latest trainer state is at epoch %d but latest model checkpoint is at epoch %d; "
                "resuming from model weights only.",
                latest_state_epoch,
                latest_model_epoch,
            )
        else:
            log.warning(
                "Found model checkpoint at epoch %d but no trainer state; resuming from weights only.",
                latest_model_epoch,
            )

        if latest_model_epoch >= int(self.epochs):
            log.info(
                "Found completed checkpoint at epoch %d/%d; training will be skipped.",
                latest_model_epoch,
                self.epochs,
            )
        else:
            log.info(
                "Resuming training from epoch %d/%d using checkpoint epoch %d.",
                self.start_epoch,
                self.epochs,
                latest_model_epoch,
            )
        print(f"Loaded checkpoint from epoch {latest_model_epoch}")

    def _save_final_checkpoint_if_needed(self) -> None:
        """Preserve legacy final-epoch behavior when early stopping is disabled."""
        if self.early_stopping.enabled:
            return
        final_epoch = int(self.epochs)
        model_path = self._model_checkpoint_path(final_epoch)
        latest_state_epoch, _ = self._find_latest_train_state(self._model_dir())

        if os.path.exists(model_path) and latest_state_epoch == final_epoch:
            return

        if not os.path.exists(model_path):
            self.model.save_model(final_epoch)
        if latest_state_epoch != final_epoch:
            save_train_state(
                self._model_dir(),
                epoch=final_epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=False,
            )

    def _checkpoint_alias(self, epoch: int, alias: str) -> str:
        """Copy an epoch checkpoint to a stable best/last/final alias."""
        source = self._model_checkpoint_path(epoch)
        if not os.path.exists(source):
            self.model.save_model(epoch)
        destination = os.path.join(self._model_dir(), f"{alias}_checkpoint.pt")
        shutil.copy2(source, destination)
        return destination

    def _record_training_epoch(self, epoch: int) -> bool:
        """Record train loss, create aliases, and return the stop decision."""
        value = float(getattr(self, "_final_loss", float("nan")))
        should_stop = self.early_stopping.update(epoch, value)
        row = self.early_stopping.history[-1]
        self.training_history.append(row)
        self.last_checkpoint_path = self._checkpoint_alias(epoch, "last")
        if row["improved"]:
            self.best_epoch = int(epoch)
            self.best_checkpoint_path = self._checkpoint_alias(epoch, "best")
            self._checkpoint_alias(epoch, "final")
        if should_stop:
            self.stop_epoch = int(epoch)
            print(
                f"[Early stopping] epoch={epoch} best_epoch={self.best_epoch} "
                f"monitor={value:.6f} patience={self.early_stopping.patience}",
                flush=True,
            )
        return should_stop

    def _finalize_training_metadata(self, stop_epoch: int) -> None:
        """Restore the selected checkpoint and persist machine-readable metadata."""
        self.early_stopping.finalize(stop_epoch, int(self.epochs))
        self.stop_epoch = int(self.early_stopping.stop_epoch or stop_epoch)
        self.best_epoch = self.early_stopping.best_epoch
        if self.early_stopping.enabled and self.best_epoch is None:
            raise RuntimeError(
                "Early stopping was enabled but no eligible best epoch was recorded; "
                "check min_epochs/max_epochs."
            )
        selected_epoch = self.best_epoch if self.early_stopping.enabled else self.stop_epoch
        if selected_epoch is not None and selected_epoch > 0:
            if self.early_stopping.enabled:
                self.model.load_model(epoch=int(selected_epoch))
                self._checkpoint_alias(int(selected_epoch), "final")
            self.config.setdefault("test", {})["test_epoch"] = int(selected_epoch)

        payload = {
            "protocol": "training_side_early_stopping",
            "selected_checkpoint": "best_checkpoint.pt" if self.early_stopping.enabled else "last_checkpoint.pt",
            "best_epoch": self.best_epoch,
            "stop_epoch": self.stop_epoch,
            "early_stop_reason": self.early_stopping.reason,
            "monitor": self.early_stopping.monitor,
            "monitor_values": self.training_history,
            "early_stopping": self.early_stopping.as_dict(),
        }
        output_path = Path(self._model_dir()).parent / "early_stopping.json"
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        history_path = Path(self._model_dir()).parent / "train_loss_history.json"
        history_path.write_text(json.dumps(self.training_history, indent=2, allow_nan=False) + "\n")

    def load_test_states(self) -> None:
        """Load model states for testing using the configured test epoch."""
        path = self._model_dir()
        test_epoch = self.config['test'].get('test_epoch', None)

        if test_epoch is None:
            raise ValueError("'test_epoch' must be set in config['test'].")

        model_name = self.config['delta_model']['phy_model']['model']
        if isinstance(model_name, list):
            model_name = model_name[0]
        checkpoint_file = f"{str(model_name).lower()}_ep{int(test_epoch)}.pt"
        checkpoint_path = os.path.join(path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{checkpoint_path} not found.")

        self.model.load_model(epoch=int(test_epoch))
        print(f"Loaded test checkpoint: {checkpoint_path}")

    def _emit_progress(self, message: str) -> None:
        """Emit progress even when the entry script does not configure logging."""
        if log.hasHandlers() and log.isEnabledFor(logging.INFO):
            log.info(message)
        else:
            print(message, flush=True)

    def train(self) -> None:
        """Train the model."""
        self.is_in_train = True

        if self.start_epoch > self.epochs:
            log.info(
                "Skipping training because checkpoint already reached epoch %d.",
                self.start_epoch - 1,
            )
            return

        # Setup a training grid (number of samples, minibatches, and timesteps)
        # 根据 data_sampler 类型选择合适的训练网格计算函数

        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'],
            self.config,
        )

        n_basins = self.train_dataset['xc_nn_norm'].shape[1]
        optimizer_name = self.config['train']['optimizer']
        lr = self.config['train']['learning_rate']
        scheduler_name = self.config['delta_model']['nn_model'].get('lr_scheduler', 'None')
        self._emit_progress(
            f"[Train Start] epochs={self.epochs} | optimizer={optimizer_name} | "
            f"lr={lr} | scheduler={scheduler_name} | n_basins={n_basins}"
        )
        self._emit_progress(
            f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs"
        )
        sys.stdout.flush()
        sys.stderr.flush()

        self._train_start_time = time.perf_counter()
        self._final_loss = 0.0

        # Training loop. The monitor is always the actual optimization loss;
        # validation/test data is never touched by the stopping decision.
        self.stop_epoch = max(self.start_epoch - 1, 0)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(
                epoch,
                n_samples,
                n_minibatch,
                n_timesteps,
            )
            self.stop_epoch = epoch
            if self._record_training_epoch(epoch):
                break

        self._finalize_training_metadata(self.stop_epoch)
        self._save_final_checkpoint_if_needed()
        total_time = time.perf_counter() - self._train_start_time
        self._emit_progress(
            f"[Train End] total_time={total_time:.1f}s | final_loss={self._final_loss:.4f}"
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        """Train model for one epoch.
        
        Parameters
        ----------
        epoch
            Current epoch number.
        n_samples
            Number of samples in the training dataset.
        n_minibatch
            Number of minibatches in the training dataset.
        n_timesteps
            Number of timesteps in the training dataset.
        """
        start_time = time.perf_counter()

        self.current_epoch = epoch
        self.total_loss = 0.0
        nonfinite_batches = 0

        # Iterate through epoch in minibatches.
        for mb in range(1, n_minibatch + 1):
            self.current_batch = mb

            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            # Forward pass through model.
            _ = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample)

            # A rare basin/seed combination can produce a non-finite first
            # forward or gradient.  Do not poison the optimizer state with a
            # NaN update; skip that minibatch and continue the epoch.
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
            
            # phy_model = self.model.model_dict['HbvTriton'].phy_model
            # if len(phy_model.grad_error_list) > 0:
            #     print(f"检测到梯度爆炸的参数: {phy_model.grad_error_list}")
            #     raise RuntimeError("Gradient NaN/Inf detected!")
            # if self.config['train'].get('clip_grad_norm', False):
            #     # Add gradient clipping here
            torch.nn.utils.clip_grad_norm_(self.model.get_parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.total_loss += loss.item()

        if nonfinite_batches:
            log.warning(
                "Skipped %d non-finite minibatch update(s) in epoch %d",
                nonfinite_batches,
                epoch,
            )

        if self.use_scheduler:
            self.scheduler.step()

        self._final_loss = self.total_loss / max(n_minibatch, 1)
        self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

        # Save model and trainer states.
        if epoch % self.config['train']['save_epoch'] == 0:
            self.model.save_model(epoch)
            save_train_state(
                self._model_dir(),
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=True,
            )


    def _get_num_start(self) -> int:
        """Return the multi-start count for evaluation output alignment."""
        model_name = self.config["delta_model"]["phy_model"]["model"]
        _model_name = model_name[0] if isinstance(model_name, list) else model_name
        _nn = getattr(self.model.model_dict.get(_model_name, None), "nn_model", None)
        return getattr(_nn, "num_start", 1)

    def _evaluate_dataset(
        self,
        dataset: dict[str, torch.Tensor],
        out_path: Path,
    ) -> None:
        """Evaluate one dataset and save predictions/metrics under out_path."""
        observations = dataset['target']
        if self._get_num_start() > 1:
            observations = observations.repeat_interleave(self._get_num_start(), dim=1)

        n_samples = dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Validating Model: Forwarding {len(batch_start)} batches -> {out_path}")
        batch_predictions = self._forward_loop(dataset, batch_start, batch_end)

        original_out_path, original_sim_dir = self._sync_output_dir(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            log.info("Saving model outputs + Calculating metrics")
            save_outputs(self.config, batch_predictions, observations)
            self.predictions = self._batch_data(batch_predictions)
            self.calc_metrics(batch_predictions, observations)
        finally:
            self.config['out_path'] = original_out_path
            self.config['sim_dir'] = original_sim_dir

    def evaluate(self) -> None:
        """Run model evaluation on both train and eval datasets when available."""
        self.is_in_train = False

        base_outpath = Path(self.config['out_path']).parents[0]
        test_epoch = self.config['test'].get('test_epoch', '')
        datasets_to_eval = []

        if self.train_dataset is not None:
            train_start = self.config['train'].get('start_time', '1980/01/01')
            train_end = self.config['train'].get('end_time', '1995/12/31')
            folder = base_outpath / (
                f"train{train_start.split('/')[0]}-{train_end.split('/')[0]}_Ep{test_epoch}"
            )
            datasets_to_eval.append((self.train_dataset, folder))

        if self.eval_dataset is not None:
            eval_start = self.config['test'].get('start_time', '1995/01/01')
            eval_end = self.config['test'].get('end_time', '2010/12/31')
            folder = base_outpath / (
                f"test{eval_start.split('/')[0]}-{eval_end.split('/')[0]}_Ep{test_epoch}"
            )
            datasets_to_eval.append((self.eval_dataset, folder))

        for dataset, out_path in datasets_to_eval:
            self._evaluate_dataset(dataset, out_path)
            print(f"Metrics and predictions saved to {out_path}")

    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch
        n_samples = self.dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['sim']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        # Save predictions
        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions

    def _batch_data(
            self,
            batch_list: list[dict[str, torch.Tensor]],
            target_key: str = None,
    ) -> None:
        """Merge batch data into a single dictionary.
        
        Parameters
        ----------
        batch_list
            List of dictionaries from each forward batch containing inputs and
            model predictions.
        target_key
            Key to extract from each batch dictionary.
        """
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()

            for key in batch_list[0].keys():
                if len(batch_list[0][key].shape) == 3:
                    pass
                else:
                    pass
                data[key] = torch.cat([d[key] for d in batch_list], dim=1).cpu().numpy()
            return data

        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def _forward_loop(
            self,
            data: dict[str, torch.Tensor],
            batch_start: NDArray,
            batch_end: NDArray,
    ):
        """Forward loop used in model evaluation and inference.

        Parameters
        ----------
        data
            Dictionary containing model input data.
        batch_start
            Start indices for each batch.
        batch_end
            End indices for each batch.
        """
        # Track predictions accross batches
        batch_predictions = []
        # Save the batch predictions
        model_name = self.config['delta_model']['phy_model']['model'][0]
        for i in tqdm.tqdm(range(len(batch_start)), desc='Forwarding', leave=False, dynamic_ncols=True):
            self.current_batch = i

            # Select a batch of data
            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[i],
                batch_end[i],
            )
            if self.config['test']['split_dataset']:
                total_time_steps = dataset_sample["x_phy"].shape[0]
                # split to 730
                prediction_time_chunks = []
                prediction_length = self.config['delta_model']['rho']
                warmup_length = self.config['delta_model']['phy_model']['warm_up']
                # subtime_length = prediction_length + warmup_length
                time_starts = range(0, total_time_steps - prediction_length - warmup_length + 1, prediction_length)
                for t_start in time_starts:
                    t_end = t_start + prediction_length + warmup_length
                    time_window_input = {
                        key: tensor[t_start:t_end, ...] if len(tensor.shape) > 2 else tensor
                        for key, tensor in dataset_sample.items()
                    }
                    prediction_window = self.model(time_window_input, eval=True)
                    prediction_valid_part = {
                        key: tensor[warmup_length:, ...].cpu().detach()
                        if tensor.shape[0] > warmup_length else tensor.cpu().detach()
                        for key, tensor in prediction_window[model_name].items()
                    }
                    prediction_time_chunks.append(prediction_valid_part)
                collated_chunks = {key: [] for key in prediction_time_chunks[0]}
                for chunk in prediction_time_chunks:
                    for key, ten in chunk.items():
                        collated_chunks[key].append(ten)
                prediction = {
                    key: torch.cat(tensors, dim=0) for key, tensors in collated_chunks.items()
                }
                batch_predictions.append(prediction)
            else:
                prediction = self.model(dataset_sample, eval=True)
                prediction = {
                    key: tensor.cpu().detach() for key, tensor in prediction[model_name].items()
                }
                batch_predictions.append(prediction)
        return batch_predictions

    def calc_metrics(
            self,
            batch_predictions: list[dict[str, torch.Tensor]],
            observations: torch.Tensor,
    ) -> None:
        """Calculate and save model performance metrics.

        Parameters
        ----------
        batch_predictions
            List of dictionaries containing model predictions.
        observations
            Target variable observation data.
        """
        target_name = self.config['train']['target'][0]
        predictions = self._batch_data(batch_predictions, target_name)
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)

        # Remove warm-up data
        target = target[self.config['delta_model']['phy_model']['warm_up']:, :]
        target = target[:len(predictions),:]

        # Compute metrics
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )

        # Save all metrics and aggregated statistics.
        metrics.dump_metrics(self.config['out_path'])

    def _log_epoch_stats(
            self,
            epoch: int,
            loss_dict: dict[str, float],
            n_minibatch: int,
            start_time: float,
    ) -> None:
        """Log statistics after each epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        loss_dict
            Dictionary containing loss values.
        n_minibatch
            Number of minibatches.
        start_time
            Start time of the epoch.
        """
        log_interval = self.config['train'].get('log_interval', 1)
        if epoch % log_interval != 0:
            return

        avg_loss = getattr(self, '_final_loss', self.total_loss / max(n_minibatch, 1))
        elapsed = time.perf_counter() - start_time
        if torch.cuda.is_available() and str(self.config['device']).startswith('cuda'):
            mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
        else:
            mem_aloc = 0
        lr = self.optimizer.param_groups[0]['lr']

        self._emit_progress(
            f"[Epoch {epoch:>4}/{self.epochs}] loss={avg_loss:.4f} | "
            f"lr={lr:.2e} | time={elapsed:.1f}s | mem={mem_aloc}MB"
        )
        sys.stdout.flush()
        sys.stderr.flush()
