"""
Base Classes for ML Models and Neural Networks

Provides abstract base classes that define the interface for all ML models,
ensuring consistency, proper versioning, and standardized training/inference.
"""

import os
import json
import logging
import pickle
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

import torch

logger = logging.getLogger("bunoraa.ml")


@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    model_name: str
    model_version: str
    model_type: str
    framework: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Training info
    training_data_hash: str = ""
    num_training_samples: int = 0
    num_validation_samples: int = 0
    training_time_seconds: float = 0.0
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Config
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    
    # Deployment
    is_production: bool = False
    deployment_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "framework": self.framework,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "training_data_hash": self.training_data_hash,
            "num_training_samples": self.num_training_samples,
            "num_validation_samples": self.num_validation_samples,
            "training_time_seconds": self.training_time_seconds,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "is_production": self.is_production,
            "deployment_timestamp": self.deployment_timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(**data)


class BaseMLModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    Provides standardized interface for:
    - Training and validation
    - Prediction and inference
    - Model serialization and loading
    - Metrics tracking
    - Feature importance
    """
    
    MODEL_TYPE = "base"
    FRAMEWORK = "sklearn"
    
    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.version = version
        self.config = config or {}
        
        self.model = None
        self.is_fitted = False
        self._feature_names: List[str] = []
        self._target_names: List[str] = []
        
        self.metadata = ModelMetadata(
            model_name=model_name,
            model_version=version,
            model_type=self.MODEL_TYPE,
            framework=self.FRAMEWORK,
            hyperparameters=self.config
        )
        
        self._setup()
    
    def _setup(self):
        """Initialize model-specific components. Override in subclasses."""
        pass
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> "BaseMLModel":
        """
        Train the model on the given data.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation set
            **kwargs: Additional training arguments
        
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            **kwargs: Additional prediction arguments
        
        Returns:
            Predictions array
        """
        pass
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Input features
        
        Returns:
            Probability predictions
        """
        raise NotImplementedError("predict_proba not implemented for this model")
    
    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
        
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return {}
    
    def save(self, path: Union[str, Path]) -> Path:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model
        
        Returns:
            Path to saved model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Update and save metadata
        self.metadata.updated_at = datetime.utcnow().isoformat()
        self.metadata.feature_names = self._feature_names
        self.metadata.target_names = self._target_names
        
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseMLModel":
        """
        Load model from disk.
        
        Args:
            path: Directory containing saved model
        
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path) as f:
            metadata_dict = json.load(f)
        
        # Load config
        config_path = path / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            model_name=metadata_dict["model_name"],
            version=metadata_dict["model_version"],
            config=config
        )
        
        # Load model
        model_path = path / "model.pkl"
        with open(model_path, 'rb') as f:
            instance.model = pickle.load(f)
        
        instance.metadata = ModelMetadata.from_dict(metadata_dict)
        instance._feature_names = metadata_dict.get("feature_names", [])
        instance._target_names = metadata_dict.get("target_names", [])
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def get_model_hash(self) -> str:
        """Generate hash for model versioning."""
        data = f"{self.model_name}_{self.version}_{json.dumps(self.config, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()[:12]


class BaseNeuralNetwork(BaseMLModel):
    """
    Base class for PyTorch neural network models.
    
    Provides:
    - Standard training loop with validation
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Checkpointing
    """
    
    FRAMEWORK = "pytorch"
    
    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name, version, config)
        
        # Training state
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history: List[Dict[str, float]] = []
    
    def _setup(self):
        """Setup PyTorch-specific components."""
        try:
            import torch
            
            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            
            logger.info(f"Using device: {self.device}")
        except ImportError:
            logger.warning("PyTorch not available. Install with: pip install torch")
            self.device = None
    
    @abstractmethod
    def build_model(self) -> "torch.nn.Module":
        """
        Build and return the neural network architecture.
        
        Returns:
            PyTorch nn.Module
        """
        pass
    
    @abstractmethod
    def get_loss_function(self) -> "torch.nn.Module":
        """
        Get the loss function for training.
        
        Returns:
            PyTorch loss function
        """
        pass
    
    def get_optimizer(self, model: "torch.nn.Module") -> "torch.optim.Optimizer":
        """Get optimizer for training."""
        import torch.optim as optim
        
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 0.0001)
        optimizer_name = self.config.get("optimizer", "adamw")
        
        if optimizer_name.lower() == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer: "torch.optim.Optimizer", num_training_steps: int):
        """Get learning rate scheduler."""
        import torch.optim.lr_scheduler as lr_scheduler
        
        scheduler_name = self.config.get("scheduler", "cosine")
        warmup_steps = self.config.get("warmup_steps", 1000)
        
        if scheduler_name == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - warmup_steps
            )
        elif scheduler_name == "linear":
            return lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=num_training_steps
            )
        elif scheduler_name == "step":
            return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        return None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> "BaseNeuralNetwork":
        """
        Train the neural network.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional (X_val, y_val) tuple
            **kwargs: Additional arguments (epochs, batch_size, etc.)
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        self.model = self.model.to(self.device)
        
        # Get training config
        epochs = kwargs.get("epochs", self.config.get("epochs", 100))
        batch_size = kwargs.get("batch_size", self.config.get("batch_size", 32))
        early_stopping_patience = self.config.get("early_stopping_patience", 10)
        gradient_clip_norm = self.config.get("gradient_clip_norm", 1.0)
        use_mixed_precision = self.config.get("mixed_precision", True) and self.device.type == "cuda"
        
        # Create data loaders
        train_tensor_x = torch.FloatTensor(X)
        train_tensor_y = torch.FloatTensor(y) if y.dtype == np.float64 else torch.LongTensor(y)
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_tensor_x = torch.FloatTensor(X_val)
            val_tensor_y = torch.FloatTensor(y_val) if y_val.dtype == np.float64 else torch.LongTensor(y_val)
            val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training components
        loss_fn = self.get_loss_function()
        self.optimizer = self.get_optimizer(self.model)
        
        num_training_steps = epochs * len(train_loader)
        self.scheduler = self.get_scheduler(self.optimizer, num_training_steps)
        
        # Mixed precision scaler
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        start_time = datetime.now()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = loss_fn(outputs, batch_y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = loss_fn(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Early stopping check
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.epochs_without_improvement = 0
                    # Save best model state
                    self._best_model_state = self.model.state_dict().copy()
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        # Restore best model
                        self.model.load_state_dict(self._best_model_state)
                        break
            else:
                avg_val_loss = None
            
            # Record history
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            }
            self.training_history.append(epoch_metrics)
            
            if epoch % 10 == 0:
                val_str = f", val_loss: {avg_val_loss:.4f}" if avg_val_loss else ""
                logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {avg_train_loss:.4f}{val_str}")
        
        # Update metadata
        training_time = (datetime.now() - start_time).total_seconds()
        self.metadata.training_time_seconds = training_time
        self.metadata.num_training_samples = len(X)
        if validation_data:
            self.metadata.num_validation_samples = len(validation_data[0])
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions."""
        import torch
        
        self.model.eval()
        batch_size = kwargs.get("batch_size", self.config.get("batch_size", 64))
        
        predictions = []
        tensor_x = torch.FloatTensor(X)
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = tensor_x[i:i + batch_size].to(self.device)
                outputs = self.model(batch)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Evaluate model on test data."""
        predictions = self.predict(X)
        
        # Basic metrics
        metrics = {}
        
        # For regression
        if predictions.shape[-1] == 1 or len(predictions.shape) == 1:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            predictions = predictions.flatten()
            metrics["mse"] = float(mean_squared_error(y, predictions))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(y, predictions))
            metrics["r2"] = float(r2_score(y, predictions))
        else:
            # For classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            pred_classes = np.argmax(predictions, axis=1)
            metrics["accuracy"] = float(accuracy_score(y, pred_classes))
            try:
                metrics["precision"] = float(precision_score(y, pred_classes, average="weighted"))
                metrics["recall"] = float(recall_score(y, pred_classes, average="weighted"))
                metrics["f1"] = float(f1_score(y, pred_classes, average="weighted"))
            except Exception:
                pass
        
        self.metadata.metrics = metrics
        return metrics
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save PyTorch model."""
        import torch
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = path / "model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "training_history": self.training_history,
            "best_val_loss": self.best_val_loss,
        }, model_path)
        
        # Save metadata
        self.metadata.updated_at = datetime.utcnow().isoformat()
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Neural network saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseNeuralNetwork":
        """Load PyTorch model."""
        import torch
        
        path = Path(path)
        
        # Load config and metadata
        with open(path / "config.json") as f:
            config = json.load(f)
        
        with open(path / "metadata.json") as f:
            metadata_dict = json.load(f)
        
        # Create instance
        instance = cls(
            model_name=metadata_dict["model_name"],
            version=metadata_dict["model_version"],
            config=config
        )
        
        # Build and load model
        instance.model = instance.build_model()
        
        checkpoint = torch.load(path / "model.pt", map_location=instance.device)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.model = instance.model.to(instance.device)
        
        instance.training_history = checkpoint.get("training_history", [])
        instance.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        instance.metadata = ModelMetadata.from_dict(metadata_dict)
        instance.is_fitted = True
        
        logger.info(f"Neural network loaded from {path}")
        return instance
