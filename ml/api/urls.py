"""
ML API URL Configuration

URL patterns for ML API endpoints including tracking, recommendations, search, and predictions.
"""

from django.urls import path

from .views import (
    # Tracking (Django Views)
    MLTrackingAPIView,
    MLPredictionsAPIView,
    # Recommendations
    PersonalizedRecommendationsView,
    SimilarProductsView,
    VisuallySimilarProductsView,
    FrequentlyBoughtTogetherView,
    CartRecommendationsView,
    PopularProductsView,
    # Search
    SemanticSearchView,
    AutocompleteView,
    VisualSearchView,
    # Personalization
    PersonalizedHomepageView,
    UserProfileView,
    NextBestActionView,
    # Analytics
    DemandForecastView,
    PriceRecommendationsView,
    CustomerSegmentsView,
    ProductInsightsView,
    AnalyticsDashboardView,
    # Fraud
    OrderRiskAssessmentView,
    UserRiskAssessmentView,
    FraudDashboardView,
    # Churn
    ChurnPredictionView,
    # Training
    TriggerTrainingView,
    ModelHealthView,
)

app_name = "ml_api"

urlpatterns = [
    # ==========================================================================
    # TRACKING ENDPOINTS (Frontend JS library)
    # ==========================================================================
    
    # POST /ml/track/ - Receive tracking events from frontend
    path(
        "track/",
        MLTrackingAPIView.as_view(),
        name="tracking"
    ),
    
    # POST /ml/predict/ - ML predictions (demand, fraud, churn)
    path(
        "predict/",
        MLPredictionsAPIView.as_view(),
        name="predictions"
    ),
    
    # ==========================================================================
    # RECOMMENDATION ENDPOINTS
    # ==========================================================================
    
    # GET /ml/recommendations/ - Get personalized recommendations
    path(
        "recommendations/",
        PersonalizedRecommendationsView.as_view(),
        name="personalized-recommendations"
    ),
    
    # GET /ml/recommendations/popular/ - Get popular products
    path(
        "recommendations/popular/",
        PopularProductsView.as_view(),
        name="popular-products"
    ),
    
    # GET /ml/recommendations/similar/<product_id>/ - Get similar products
    path(
        "recommendations/similar/<int:product_id>/",
        SimilarProductsView.as_view(),
        name="similar-products"
    ),

    # GET /ml/recommendations/visual-similar/<int:product_id>/ - Get visually similar products
    path(
        "recommendations/visual-similar/<int:product_id>/",
        VisuallySimilarProductsView.as_view(),
        name="visual-similar-products"
    ),
    
    # GET /ml/recommendations/fbt/<product_id>/ - Frequently bought together
    path(
        "recommendations/fbt/<int:product_id>/",
        FrequentlyBoughtTogetherView.as_view(),
        name="frequently-bought-together"
    ),
    
    # POST /ml/recommendations/cart/ - Cart-based recommendations
    path(
        "recommendations/cart/",
        CartRecommendationsView.as_view(),
        name="cart-recommendations"
    ),
    
    # ==========================================================================
    # SEARCH ENDPOINTS
    # ==========================================================================
    
    # GET /ml/search/ - Semantic product search
    path(
        "search/",
        SemanticSearchView.as_view(),
        name="semantic-search"
    ),
    
    # GET /ml/search/autocomplete/ - Search autocomplete
    path(
        "search/autocomplete/",
        AutocompleteView.as_view(),
        name="autocomplete"
    ),
    
    # POST /ml/search/visual/ - Visual (image) search
    path(
        "search/visual/",
        VisualSearchView.as_view(),
        name="visual-search"
    ),
    
    # ==========================================================================
    # PERSONALIZATION ENDPOINTS
    # ==========================================================================
    
    # GET /ml/personalization/homepage/ - Personalized homepage
    path(
        "personalization/homepage/",
        PersonalizedHomepageView.as_view(),
        name="personalized-homepage"
    ),
    
    # GET /ml/personalization/profile/ - User profile
    path(
        "personalization/profile/",
        UserProfileView.as_view(),
        name="user-profile"
    ),
    
    # GET /ml/personalization/next-action/ - Next best action
    path(
        "personalization/next-action/",
        NextBestActionView.as_view(),
        name="next-best-action"
    ),
    
    # ==========================================================================
    # ANALYTICS ENDPOINTS
    # ==========================================================================
    
    # GET /ml/analytics/forecast/ - Demand forecast
    path(
        "analytics/forecast/",
        DemandForecastView.as_view(),
        name="demand-forecast"
    ),
    
    # GET /ml/analytics/pricing/ - Price recommendations
    path(
        "analytics/pricing/",
        PriceRecommendationsView.as_view(),
        name="price-recommendations"
    ),
    
    # GET /ml/analytics/segments/ - Customer segments
    path(
        "analytics/segments/",
        CustomerSegmentsView.as_view(),
        name="customer-segments"
    ),
    
    # GET /ml/analytics/products/<product_id>/ - Product insights
    path(
        "analytics/products/<int:product_id>/",
        ProductInsightsView.as_view(),
        name="product-insights"
    ),
    
    # GET /ml/analytics/dashboard/ - Analytics dashboard
    path(
        "analytics/dashboard/",
        AnalyticsDashboardView.as_view(),
        name="analytics-dashboard"
    ),
    
    # ==========================================================================
    # FRAUD DETECTION ENDPOINTS
    # ==========================================================================
    
    # POST /ml/fraud/assess-order/ - Assess order risk
    path(
        "fraud/assess-order/",
        OrderRiskAssessmentView.as_view(),
        name="order-risk-assessment"
    ),
    
    # GET /ml/fraud/assess-user/<user_id>/ - Assess user risk
    path(
        "fraud/assess-user/<int:user_id>/",
        UserRiskAssessmentView.as_view(),
        name="user-risk-assessment"
    ),
    
    # GET /ml/fraud/dashboard/ - Fraud dashboard
    path(
        "fraud/dashboard/",
        FraudDashboardView.as_view(),
        name="fraud-dashboard"
    ),
    
    # ==========================================================================
    # CHURN ENDPOINTS
    # ==========================================================================
    
    # GET /ml/churn/<user_id>/ - Churn prediction
    path(
        "churn/<int:user_id>/",
        ChurnPredictionView.as_view(),
        name="churn-prediction"
    ),
    
    # ==========================================================================
    # ADMIN/TRAINING ENDPOINTS
    # ==========================================================================
    
    # POST /ml/admin/train/ - Trigger training
    path(
        "admin/train/",
        TriggerTrainingView.as_view(),
        name="trigger-training"
    ),
    
    # GET /ml/admin/health/ - Model health
    path(
        "admin/health/",
        ModelHealthView.as_view(),
        name="model-health"
    ),
]
