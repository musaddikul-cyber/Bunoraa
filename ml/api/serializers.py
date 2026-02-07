"""
ML API Serializers

DRF serializers for ML API endpoints.
"""

try:
    from rest_framework import serializers
    DRF_AVAILABLE = True
except ImportError:
    DRF_AVAILABLE = False
    # Mock serializer
    class serializers:
        class Serializer:
            pass
        class CharField:
            def __init__(self, *args, **kwargs): pass
        class IntegerField:
            def __init__(self, *args, **kwargs): pass
        class FloatField:
            def __init__(self, *args, **kwargs): pass
        class BooleanField:
            def __init__(self, *args, **kwargs): pass
        class ListField:
            def __init__(self, *args, **kwargs): pass
        class DictField:
            def __init__(self, *args, **kwargs): pass


# =============================================================================
# RECOMMENDATION SERIALIZERS
# =============================================================================

class RecommendationItemSerializer(serializers.Serializer):
    """Serializer for a recommendation item."""
    
    product_id = serializers.IntegerField()
    score = serializers.FloatField()
    source = serializers.CharField(required=False)
    reason = serializers.CharField(required=False)


class PersonalizedRecommendationRequestSerializer(serializers.Serializer):
    """Request serializer for personalized recommendations."""
    
    limit = serializers.IntegerField(default=20, min_value=1, max_value=100)
    category_id = serializers.IntegerField(required=False, allow_null=True)
    platform = serializers.CharField(default="web")


class SimilarProductsRequestSerializer(serializers.Serializer):
    """Request serializer for similar products."""
    
    limit = serializers.IntegerField(default=10, min_value=1, max_value=50)
    type = serializers.ChoiceField(
        choices=["content", "collaborative", "hybrid"],
        default="hybrid"
    )


class CartRecommendationRequestSerializer(serializers.Serializer):
    """Request serializer for cart recommendations."""
    
    product_ids = serializers.ListField(
        child=serializers.IntegerField(),
        min_length=1
    )
    limit = serializers.IntegerField(default=5, min_value=1, max_value=20)


# =============================================================================
# SEARCH SERIALIZERS
# =============================================================================

class SearchRequestSerializer(serializers.Serializer):
    """Request serializer for product search."""
    
    q = serializers.CharField(max_length=500)
    page = serializers.IntegerField(default=1, min_value=1)
    page_size = serializers.IntegerField(default=20, min_value=1, max_value=100)
    sort = serializers.ChoiceField(
        choices=["relevance", "price_asc", "price_desc", "newest", "rating"],
        default="relevance"
    )
    category = serializers.IntegerField(required=False, allow_null=True)
    min_price = serializers.FloatField(required=False, allow_null=True)
    max_price = serializers.FloatField(required=False, allow_null=True)
    in_stock = serializers.BooleanField(default=False)


class SearchResultSerializer(serializers.Serializer):
    """Serializer for search results."""
    
    product_id = serializers.IntegerField()
    name = serializers.CharField()
    price = serializers.FloatField()
    score = serializers.FloatField()
    highlighted_name = serializers.CharField(required=False)


class AutocompleteResultSerializer(serializers.Serializer):
    """Serializer for autocomplete results."""
    
    text = serializers.CharField()
    type = serializers.CharField()  # product, category, search
    id = serializers.IntegerField(required=False)
    score = serializers.FloatField()


# =============================================================================
# PERSONALIZATION SERIALIZERS
# =============================================================================

class UserProfileSerializer(serializers.Serializer):
    """Serializer for user profile."""
    
    segment = serializers.CharField()
    preferences = serializers.DictField()
    favorite_categories = serializers.ListField(child=serializers.IntegerField())
    price_sensitivity = serializers.FloatField()
    average_order_value = serializers.FloatField()
    lifetime_value = serializers.FloatField()
    churn_probability = serializers.FloatField()


class PersonalizedHomepageSerializer(serializers.Serializer):
    """Serializer for personalized homepage."""
    
    hero_products = serializers.ListField()
    featured_categories = serializers.ListField()
    recommended_for_you = serializers.ListField()
    based_on_history = serializers.ListField()
    trending_in_category = serializers.ListField()
    deals_for_you = serializers.ListField()


class NextBestActionSerializer(serializers.Serializer):
    """Serializer for next best action."""
    
    action_type = serializers.CharField()
    priority = serializers.FloatField()
    context = serializers.DictField()
    message = serializers.CharField()
    cta_text = serializers.CharField()
    cta_url = serializers.CharField()


# =============================================================================
# ANALYTICS SERIALIZERS
# =============================================================================

class DemandForecastSerializer(serializers.Serializer):
    """Serializer for demand forecast."""
    
    product_id = serializers.IntegerField()
    forecasts = serializers.ListField()  # List of daily forecasts
    total_predicted = serializers.FloatField()
    confidence_lower = serializers.FloatField()
    confidence_upper = serializers.FloatField()


class PriceRecommendationSerializer(serializers.Serializer):
    """Serializer for price recommendation."""
    
    product_id = serializers.IntegerField()
    current_price = serializers.FloatField()
    recommended_price = serializers.FloatField()
    expected_uplift = serializers.FloatField()
    price_elasticity = serializers.FloatField()
    confidence = serializers.FloatField()


class CustomerSegmentSerializer(serializers.Serializer):
    """Serializer for customer segment."""
    
    segment_id = serializers.IntegerField()
    name = serializers.CharField()
    size = serializers.IntegerField()
    characteristics = serializers.DictField()
    average_ltv = serializers.FloatField()
    churn_rate = serializers.FloatField()


class ProductInsightSerializer(serializers.Serializer):
    """Serializer for product insights."""
    
    product_id = serializers.IntegerField()
    performance_score = serializers.FloatField()
    demand_trend = serializers.CharField()  # growing, stable, declining
    price_optimization = serializers.DictField()
    customer_segments = serializers.ListField()
    related_products = serializers.ListField()
    seasonality = serializers.DictField()


# =============================================================================
# FRAUD SERIALIZERS
# =============================================================================

class FraudOrderDataSerializer(serializers.Serializer):
    """Serializer for order data in fraud assessment."""
    
    order_id = serializers.IntegerField(required=False)
    total_amount = serializers.FloatField()
    item_count = serializers.IntegerField()
    shipping_address = serializers.DictField()
    billing_address = serializers.DictField()
    payment_method = serializers.CharField()
    device_fingerprint = serializers.CharField(required=False, allow_null=True)


class FraudUserDataSerializer(serializers.Serializer):
    """Serializer for user data in fraud assessment."""
    
    user_id = serializers.IntegerField()
    account_age_days = serializers.IntegerField()
    order_count = serializers.IntegerField()
    email_verified = serializers.BooleanField()
    phone_verified = serializers.BooleanField()


class RiskAssessmentRequestSerializer(serializers.Serializer):
    """Request serializer for risk assessment."""
    
    order = FraudOrderDataSerializer()
    user = FraudUserDataSerializer()


class RiskAssessmentResponseSerializer(serializers.Serializer):
    """Response serializer for risk assessment."""
    
    risk_score = serializers.FloatField()
    risk_level = serializers.ChoiceField(choices=["low", "medium", "high", "critical"])
    factors = serializers.ListField(child=serializers.DictField())
    recommendation = serializers.CharField()
    is_blocked = serializers.BooleanField()
    needs_review = serializers.BooleanField()


# =============================================================================
# CHURN SERIALIZERS
# =============================================================================

class ChurnPredictionSerializer(serializers.Serializer):
    """Serializer for churn prediction."""
    
    user_id = serializers.IntegerField()
    churn_probability = serializers.FloatField()
    risk_level = serializers.CharField()
    days_to_churn = serializers.IntegerField(required=False, allow_null=True)
    reasons = serializers.ListField(child=serializers.CharField())
    retention_actions = serializers.ListField(child=serializers.DictField())


# =============================================================================
# TRAINING SERIALIZERS
# =============================================================================

class TrainingRequestSerializer(serializers.Serializer):
    """Request serializer for training trigger."""
    
    model_type = serializers.ChoiceField(
        choices=[
            "all", "recommendation", "embeddings", "forecasting",
            "fraud", "churn", "search"
        ],
        default="all"
    )


class TrainingStatusSerializer(serializers.Serializer):
    """Serializer for training status."""
    
    model_type = serializers.CharField()
    status = serializers.ChoiceField(
        choices=["queued", "running", "completed", "failed"]
    )
    started_at = serializers.DateTimeField(required=False, allow_null=True)
    completed_at = serializers.DateTimeField(required=False, allow_null=True)
    metrics = serializers.DictField(required=False)
    error = serializers.CharField(required=False, allow_null=True)


class ModelHealthSerializer(serializers.Serializer):
    """Serializer for model health status."""
    
    model_name = serializers.CharField()
    version = serializers.CharField()
    is_healthy = serializers.BooleanField()
    last_training = serializers.DateTimeField(required=False, allow_null=True)
    performance_metrics = serializers.DictField()
    drift_detected = serializers.BooleanField()
    warnings = serializers.ListField(child=serializers.CharField())
