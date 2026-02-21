"""
Neural Network Models Package

Comprehensive deep learning models for e-commerce:
- Product embeddings
- User embeddings
- Recommendation systems
- Demand forecasting
- Price optimization
- Fraud detection
- Customer churn prediction
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    # Embeddings
    "ProductEmbeddingModel",
    "UserEmbeddingModel",
    
    # Recommenders
    "NeuralCollaborativeFiltering",
    "DeepFM",
    "TwoTowerRecommender",
    "SequenceRecommender",
    
    # Forecasting
    "DemandForecaster",
    "PriceOptimizer",
    
    # Fraud
    "FraudDetector",
    
    # Customer
    "ChurnPredictor",
    "CustomerLifetimeValue",
    
    # Search
    "SemanticSearchModel",
    "QueryEncoder",
    
    # Vision
    "ProductImageClassifier",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "ProductEmbeddingModel": ("ml.models.embeddings", "ProductEmbeddingModel"),
    "UserEmbeddingModel": ("ml.models.embeddings", "UserEmbeddingModel"),
    "NeuralCollaborativeFiltering": ("ml.models.recommender", "NeuralCollaborativeFiltering"),
    "DeepFM": ("ml.models.recommender", "DeepFM"),
    "TwoTowerRecommender": ("ml.models.recommender", "TwoTowerRecommender"),
    "SequenceRecommender": ("ml.models.recommender", "SequenceRecommender"),
    "DemandForecaster": ("ml.models.forecasting", "DemandForecaster"),
    "PriceOptimizer": ("ml.models.forecasting", "PriceOptimizer"),
    "FraudDetector": ("ml.models.fraud", "FraudDetector"),
    "ChurnPredictor": ("ml.models.churn", "ChurnPredictor"),
    "CustomerLifetimeValue": ("ml.models.churn", "CustomerLifetimeValue"),
    "SemanticSearchModel": ("ml.models.search", "SemanticSearchModel"),
    "QueryEncoder": ("ml.models.search", "QueryEncoder"),
    "ProductImageClassifier": ("ml.models.vision", "ProductImageClassifier"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'ml.models' has no attribute '{name}'")
    module_path, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
