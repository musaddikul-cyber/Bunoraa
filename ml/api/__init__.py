"""
ML API Package

REST API endpoints for ML services.
"""

from .views import (
    PersonalizedRecommendationsView,
    SimilarProductsView,
    FrequentlyBoughtTogetherView,
    CartRecommendationsView,
    PopularProductsView,
    SemanticSearchView,
    AutocompleteView,
    VisualSearchView,
    PersonalizedHomepageView,
    UserProfileView,
    NextBestActionView,
    DemandForecastView,
    PriceRecommendationsView,
    CustomerSegmentsView,
    ProductInsightsView,
    AnalyticsDashboardView,
    OrderRiskAssessmentView,
    UserRiskAssessmentView,
    FraudDashboardView,
    ChurnPredictionView,
    TriggerTrainingView,
    ModelHealthView,
)

__all__ = [
    # Recommendations
    "PersonalizedRecommendationsView",
    "SimilarProductsView",
    "FrequentlyBoughtTogetherView",
    "CartRecommendationsView",
    "PopularProductsView",
    # Search
    "SemanticSearchView",
    "AutocompleteView",
    "VisualSearchView",
    # Personalization
    "PersonalizedHomepageView",
    "UserProfileView",
    "NextBestActionView",
    # Analytics
    "DemandForecastView",
    "PriceRecommendationsView",
    "CustomerSegmentsView",
    "ProductInsightsView",
    "AnalyticsDashboardView",
    # Fraud
    "OrderRiskAssessmentView",
    "UserRiskAssessmentView",
    "FraudDashboardView",
    # Churn
    "ChurnPredictionView",
    # Training
    "TriggerTrainingView",
    "ModelHealthView",
]
