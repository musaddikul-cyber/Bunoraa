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

from .embeddings import ProductEmbeddingModel, UserEmbeddingModel
from .recommender import (
    NeuralCollaborativeFiltering,
    DeepFM,
    TwoTowerRecommender,
    SequenceRecommender
)
from .forecasting import DemandForecaster, PriceOptimizer
from .fraud import FraudDetector
from .churn import ChurnPredictor, CustomerLifetimeValue
from .search import SemanticSearchModel, QueryEncoder
from .vision import ProductImageClassifier

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
