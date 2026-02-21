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

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

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

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "ModelRegistry": ("ml.core.registry", "ModelRegistry"),
    "BaseMLModel": ("ml.core.base", "BaseMLModel"),
    "BaseNeuralNetwork": ("ml.core.base", "BaseNeuralNetwork"),
    "InferenceEngine": ("ml.core.inference", "InferenceEngine"),
    "MLConfig": ("ml.core.config", "MLConfig"),
    "MetricsTracker": ("ml.core.metrics", "MetricsTracker"),
    "FeatureStore": ("ml.core.feature_store", "FeatureStore"),
    "get_default_settings": ("ml.core.settings", "get_default_settings"),
    "get_celery_beat_schedule": ("ml.core.settings", "get_celery_beat_schedule"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'ml' has no attribute '{name}'")
    module_path, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
