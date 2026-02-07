"""
Model Trainer

Training loop and utilities for model training.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .callbacks import CallbackList, Callback, EarlyStoppingCallback, MetricsCallback

logger = logging.getLogger("bunoraa.ml.training")


@dataclass
class TrainerConfig:
    """Configuration for model training."""
    # Basic training
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Optimizer
    optimizer: str = "adamw"  # adam, adamw, sgd, rmsprop
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # cosine, step, exponential, none
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 5
    
    # Evaluation
    eval_frequency: int = 1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0001
    
    # Device
    device: str = "auto"  # auto, cuda, cpu, mps


class ModelTrainer:
    """
    Universal trainer for PyTorch models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        loss_fn: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics = metrics or {}
        self.callbacks = CallbackList(callbacks)
        
        # Set device
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.stop_training = False
        self.best_metric = None
        
        # Add default callbacks
        self._add_default_callbacks()
    
    def _get_device(self) -> torch.device:
        """Get appropriate device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        params = self.model.parameters()
        
        if self.config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "rmsprop":
            return optim.RMSprop(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        warmup_steps = self.config.warmup_epochs * (num_training_steps // self.config.epochs)
        
        if self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_training_steps - warmup_steps,
                eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif self.config.lr_scheduler == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
    
    def _add_default_callbacks(self):
        """Add default callbacks."""
        # Metrics logging
        self.callbacks.append(MetricsCallback())
        
        # Early stopping
        if self.config.early_stopping:
            self.callbacks.append(EarlyStoppingCallback(
                monitor="val_loss",
                patience=self.config.patience,
                min_delta=self.config.min_delta
            ))
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (overrides config)
        
        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs
        num_training_steps = len(train_loader) * epochs
        
        # Create scheduler
        self._create_scheduler(num_training_steps)
        
        # Callbacks
        self.callbacks.on_train_begin(self)
        
        history = {"train_loss": [], "val_loss": []}
        
        try:
            for epoch in range(epochs):
                if self.stop_training:
                    break
                
                self.current_epoch = epoch
                self.callbacks.on_epoch_begin(self, epoch)
                
                # Training
                train_loss = self._train_epoch(train_loader)
                history["train_loss"].append(train_loss)
                
                logs = {"train_loss": train_loss}
                
                # Validation
                if val_loader and (epoch + 1) % self.config.eval_frequency == 0:
                    val_loss, val_metrics = self._validate(val_loader)
                    history["val_loss"].append(val_loss)
                    logs["val_loss"] = val_loss
                    logs.update(val_metrics)
                
                self.callbacks.on_epoch_end(self, epoch, logs)
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
        
        finally:
            self.callbacks.on_train_end(self)
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            self.callbacks.on_batch_begin(self, batch_idx)
            
            # Move to device
            batch = self._to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                    loss = self._compute_loss(outputs, batch)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                loss = self._compute_loss(outputs, batch)
                
                loss.backward()
                
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            batch_logs = {"loss": loss.item()}
            self.callbacks.on_batch_end(self, batch_idx, batch_logs)
        
        return total_loss / num_batches
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        self.callbacks.on_validation_begin(self)
        
        total_loss = 0.0
        num_batches = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)
                
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect for metrics
                if self.metrics:
                    all_outputs.append(outputs)
                    all_targets.append(batch.get("target") or batch.get("label"))
        
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        metrics = {}
        if self.metrics and all_outputs:
            outputs = torch.cat(all_outputs, dim=0) if isinstance(all_outputs[0], torch.Tensor) else all_outputs
            targets = torch.cat(all_targets, dim=0) if all_targets and isinstance(all_targets[0], torch.Tensor) else all_targets
            
            for name, metric_fn in self.metrics.items():
                metrics[f"val_{name}"] = metric_fn(outputs, targets)
        
        logs = {"val_loss": avg_loss, **metrics}
        self.callbacks.on_validation_end(self, logs)
        
        return avg_loss, metrics
    
    def _compute_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss from outputs and batch."""
        if isinstance(outputs, dict):
            # Model returns dict with predictions
            return self.loss_fn(outputs, batch)
        else:
            # Model returns tensor
            target_key = "target" if "target" in batch else "label"
            if target_key in batch:
                return self.loss_fn(outputs, batch[target_key])
            return outputs  # Assume model returns loss directly
    
    def _to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            return [
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in batch
            ]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch
    
    def save_checkpoint(self, path: str, include_optimizer: bool = True):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        
        if include_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            if self.scaler:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if load_optimizer and self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if load_optimizer and self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


class TrainingPipeline:
    """
    High-level training pipeline for end-to-end model training.
    """
    
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_config: Dict[str, Any],
        trainer_config: TrainerConfig,
        output_dir: str = "training_output"
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.model = None
        self.trainer = None
        self.history = None
    
    def run(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Returns:
            Training results
        """
        start_time = time.time()
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = self.model_class(**self.model_config)
        
        # Create trainer
        self.trainer = ModelTrainer(
            model=self.model,
            config=self.trainer_config
        )
        
        # Train
        logger.info("Starting training...")
        self.history = self.trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        test_results = {}
        if test_loader:
            logger.info("Evaluating on test set...")
            test_loss, test_metrics = self.trainer._validate(test_loader)
            test_results = {"test_loss": test_loss, **test_metrics}
        
        # Save final model
        model_path = self.output_dir / "final_model.pt"
        self.trainer.save_checkpoint(str(model_path))
        
        training_time = time.time() - start_time
        
        results = {
            "training_time": training_time,
            "epochs_trained": self.trainer.current_epoch + 1,
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_loss": self.history["val_loss"][-1] if self.history.get("val_loss") else None,
            "test_results": test_results,
            "model_path": str(model_path),
        }
        
        logger.info(f"Training completed in {training_time:.2f}s")
        
        return results
    
    def load_best_model(self) -> nn.Module:
        """Load the best model from training."""
        checkpoint_dir = self.output_dir / "checkpoints"
        
        # Find best checkpoint
        best_checkpoint = None
        for f in checkpoint_dir.glob("best_*.pt"):
            best_checkpoint = f
            break
        
        if best_checkpoint:
            self.trainer.load_checkpoint(str(best_checkpoint))
        
        return self.model
