"""
Bunoraa Machine Learning & Neural Network Models Package

This package provides comprehensive ML/NN solutions for:
- Product recommendations (hybrid, collaborative filtering, content-based)
- Customer segmentation and clustering
- Demand forecasting and price optimization
- Fraud detection and anomaly detection
- Customer churn prediction and lifetime value
- Search relevance and NLP
- Image recognition for products
- Automatic data collection and training

Architecture:
- Base classes for consistent model interface
- Model registry for version control
- Automated training pipelines
- Real-time inference engines
- A/B testing framework
- Data collection and tracking middleware
- Auto-training system

Author: Bunoraa AI Team
Version: 2.0.0
"""

from .core.registry import ModelRegistry
from .core.base import BaseMLModel, BaseNeuralNetwork
from .core.inference import InferenceEngine
from .core.config import MLConfig
from .core.metrics import MetricsTracker
from .core.feature_store import FeatureStore
from .core.settings import get_default_settings, get_celery_beat_schedule

__version__ = "2.0.0"
__all__ = [
    "ModelRegistry",
    "BaseMLModel",
    "BaseNeuralNetwork",
    "InferenceEngine",
    "MLConfig",
    "MetricsTracker",
    "FeatureStore",
    "get_default_settings",
    "get_celery_beat_schedule",
]
