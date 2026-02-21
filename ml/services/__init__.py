"""
Django Integration Services

Services for integrating ML models with Django applications.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "RecommendationService",
    "SearchService",
    "AnalyticsService",
    "FraudService",
    "PersonalizationService",
    "ChatModelService",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "RecommendationService": ("ml.services.recommendation_service", "RecommendationService"),
    "SearchService": ("ml.services.search_service", "SearchService"),
    "AnalyticsService": ("ml.services.analytics_service", "AnalyticsService"),
    "FraudService": ("ml.services.fraud_service", "FraudService"),
    "PersonalizationService": ("ml.services.personalization_service", "PersonalizationService"),
    "ChatModelService": ("ml.services.chat_model_service", "ChatModelService"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'ml.services' has no attribute '{name}'")
    module_path, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
