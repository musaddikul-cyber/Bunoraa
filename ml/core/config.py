"""
ML Configuration and Hyperparameters Management
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    RECOMMENDATION = "recommendation"
    SEGMENTATION = "segmentation"
    FRAUD_DETECTION = "fraud_detection"
    DEMAND_FORECAST = "demand_forecast"
    PRICE_OPTIMIZATION = "price_optimization"
    CHURN_PREDICTION = "churn_prediction"
    SEARCH_RANKING = "search_ranking"
    IMAGE_CLASSIFICATION = "image_classification"
    NLP = "nlp"


class Framework(Enum):
    """ML Framework options."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    TRANSFORMERS = "transformers"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    optimizer: str = "adamw"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelArchitectureConfig:
    """Neural network architecture configuration."""
    # Embedding dimensions
    user_embedding_dim: int = 128
    product_embedding_dim: int = 128
    category_embedding_dim: int = 64
    
    # Transformer settings
    num_attention_heads: int = 8
    num_transformer_layers: int = 6
    feedforward_dim: int = 512
    dropout: float = 0.1
    
    # MLP settings
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "gelu"
    use_batch_norm: bool = True
    
    # Sequence models
    max_sequence_length: int = 100
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    bidirectional: bool = True


@dataclass
class InferenceConfig:
    """Inference settings."""
    batch_size: int = 64
    use_gpu: bool = True
    quantize: bool = False
    cache_predictions: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_concurrent_requests: int = 100
    timeout_seconds: int = 30


@dataclass
class MLConfig:
    """Main ML configuration."""
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "artifacts")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Model versioning
    model_version: str = "v2.0.0"
    experiment_name: str = "bunoraa_ml"
    
    # Configs
    training: TrainingConfig = field(default_factory=TrainingConfig)
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Feature store
    use_feature_store: bool = True
    feature_store_backend: str = "redis"
    
    # MLflow tracking
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    enable_mlflow: bool = True
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    num_gpus: int = 1
    
    def __post_init__(self):
        """Create necessary directories."""
        for path_attr in ['models_dir', 'data_dir', 'logs_dir']:
            path = getattr(self, path_attr)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        return convert(asdict(self))
    
    def save(self, path: Optional[Path] = None):
        """Save config to JSON."""
        path = path or self.base_dir / "config.json"
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "MLConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def get_device(self) -> str:
        """Get the best available device."""
        if self.device != "auto":
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        
        return "cpu"


# Singleton config instance
_config: Optional[MLConfig] = None


def get_config() -> MLConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = MLConfig()
    return _config


def set_config(config: MLConfig):
    """Set the global config instance."""
    global _config
    _config = config
