"""
ML Data Collection Package

Comprehensive silent data collection for ML training.
Collects user behavior, product interactions, and conversion data.
"""

from .collector import DataCollector
from .events import EventTracker
from .user_profile import UserProfileCollector
from .product_analytics import ProductAnalyticsCollector

__all__ = [
    "DataCollector",
    "EventTracker",
    "UserProfileCollector",
    "ProductAnalyticsCollector",
]
