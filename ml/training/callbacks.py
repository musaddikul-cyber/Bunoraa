"""
Training Callbacks

Callbacks for training pipeline.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger("bunoraa.ml.training")


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: Any) -> None:
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        pass
    
    def on_batch_begin(self, trainer: Any, batch: int) -> None:
        pass
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float]) -> None:
        pass
    
    def on_validation_begin(self, trainer: Any) -> None:
        pass
    
    def on_validation_end(self, trainer: Any, logs: Dict[str, float]) -> None:
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping based on validation metric."""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0001,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.stopped = False
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current = logs.get(self.monitor)
        
        if current is None:
            logger.warning(f"Early stopping: metric {self.monitor} not found in logs")
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
            
            if self.restore_best_weights and hasattr(trainer, "model"):
                import copy
                self.best_weights = copy.deepcopy(trainer.model.state_dict())
            
            logger.info(f"Improvement: {self.monitor} = {current:.6f}")
        else:
            self.counter += 1
            logger.info(
                f"No improvement for {self.counter} epochs "
                f"(best: {self.best_value:.6f} at epoch {self.best_epoch})"
            )
            
            if self.counter >= self.patience:
                self.stopped = True
                trainer.stop_training = True
                logger.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_train_end(self, trainer: Any) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")


class CheckpointCallback(Callback):
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_frequency: int = 1,
        max_to_keep: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.max_to_keep = max_to_keep
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints: List[Path] = []
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value
        else:
            return current > self.best_value
    
    def _save_checkpoint(
        self,
        trainer: Any,
        epoch: int,
        logs: Dict[str, float],
        is_best: bool = False
    ):
        """Save a checkpoint."""
        import torch
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filename = f"best_model_epoch_{epoch}_{timestamp}.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict() if hasattr(trainer, "optimizer") else None,
            "logs": logs,
            "config": trainer.config if hasattr(trainer, "config") else {},
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        self.saved_checkpoints.append(checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints."""
        while len(self.saved_checkpoints) > self.max_to_keep:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        current = logs.get(self.monitor)
        
        if current is None:
            if not self.save_best_only and epoch % self.save_frequency == 0:
                self._save_checkpoint(trainer, epoch, logs, is_best=False)
            return
        
        is_best = self._is_improvement(current)
        
        if is_best:
            self.best_value = current
            self._save_checkpoint(trainer, epoch, logs, is_best=True)
        elif not self.save_best_only and epoch % self.save_frequency == 0:
            self._save_checkpoint(trainer, epoch, logs, is_best=False)


class MetricsCallback(Callback):
    """Track and log training metrics."""
    
    def __init__(
        self,
        log_frequency: int = 10,
        log_to_file: Optional[str] = None,
        metrics_to_track: Optional[List[str]] = None
    ):
        self.log_frequency = log_frequency
        self.log_to_file = Path(log_to_file) if log_to_file else None
        self.metrics_to_track = metrics_to_track
        
        self.history: Dict[str, List[float]] = {}
        self.batch_logs: List[Dict[str, Any]] = []
        self.epoch_logs: List[Dict[str, Any]] = []
    
    def on_train_begin(self, trainer: Any) -> None:
        self.history = {}
        self.batch_logs = []
        self.epoch_logs = []
        
        logger.info("Training started")
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float]) -> None:
        self.batch_logs.append({"batch": batch, **logs})
        
        if batch % self.log_frequency == 0:
            log_msg = f"Batch {batch}"
            for key, value in logs.items():
                if self.metrics_to_track is None or key in self.metrics_to_track:
                    log_msg += f" - {key}: {value:.6f}"
            logger.debug(log_msg)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        # Add to history
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        self.epoch_logs.append({"epoch": epoch, **logs})
        
        # Log
        log_msg = f"Epoch {epoch}"
        for key, value in logs.items():
            if self.metrics_to_track is None or key in self.metrics_to_track:
                log_msg += f" - {key}: {value:.6f}"
        logger.info(log_msg)
    
    def on_train_end(self, trainer: Any) -> None:
        logger.info("Training completed")
        
        # Save to file if requested
        if self.log_to_file:
            self.log_to_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.log_to_file, "w") as f:
                json.dump({
                    "history": self.history,
                    "epoch_logs": self.epoch_logs,
                    "batch_logs": self.batch_logs,
                }, f, indent=2)
            
            logger.info(f"Saved metrics to {self.log_to_file}")
    
    def get_history(self) -> Dict[str, List[float]]:
        return self.history
    
    def get_best_epoch(self, metric: str, mode: str = "min") -> int:
        if metric not in self.history:
            return 0
        
        values = self.history[metric]
        if mode == "min":
            return int(values.index(min(values)))
        else:
            return int(values.index(max(values)))


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduling during training."""
    
    def __init__(
        self,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 3,
        min_lr: float = 1e-6,
        max_lr: Optional[float] = None
    ):
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.scheduler = None
    
    def on_train_begin(self, trainer: Any) -> None:
        import torch.optim as optim
        
        optimizer = trainer.optimizer
        total_epochs = trainer.config.get("epochs", 100)
        
        if self.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - self.warmup_epochs,
                eta_min=self.min_lr
            )
        elif self.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
        elif self.scheduler_type == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in trainer.optimizer.param_groups:
                if self.max_lr:
                    param_group["lr"] = self.min_lr + warmup_factor * (self.max_lr - self.min_lr)
        elif self.scheduler:
            self.scheduler.step()
        
        # Log current LR
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        logs["learning_rate"] = current_lr
        logger.debug(f"Learning rate: {current_lr:.6f}")


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, trainer: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            logger.info(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        if self.writer:
            for key, value in logs.items():
                self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, trainer: Any) -> None:
        if self.writer:
            self.writer.close()


class GradientClippingCallback(Callback):
    """Clip gradients during training."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float]) -> None:
        import torch.nn.utils as utils
        
        total_norm = utils.clip_grad_norm_(
            trainer.model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        
        logs["grad_norm"] = total_norm.item() if hasattr(total_norm, "item") else total_norm


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)
    
    def on_batch_begin(self, trainer: Any, batch: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch)
    
    def on_batch_end(self, trainer: Any, batch: int, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch, logs)
    
    def on_validation_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_validation_begin(trainer)
    
    def on_validation_end(self, trainer: Any, logs: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(trainer, logs)
