"""
Analytics API serializers
"""
from rest_framework import serializers

from ..models import DailyStat, ProductStat


class DailyStatSerializer(serializers.ModelSerializer):
    """Serializer for daily statistics."""
    
    class Meta:
        model = DailyStat
        fields = [
            'date', 'page_views', 'unique_visitors', 'new_visitors',
            'returning_visitors', 'product_views', 'products_added_to_cart',
            'orders_count', 'orders_revenue', 'average_order_value',
            'checkout_starts', 'checkout_completions', 'conversion_rate',
            'cart_abandonment_rate', 'new_registrations'
        ]


class OverviewStatSerializer(serializers.Serializer):
    """Serializer for overview statistics."""
    total_revenue = serializers.DecimalField(max_digits=12, decimal_places=2)
    total_orders = serializers.IntegerField()
    total_visitors = serializers.IntegerField()
    avg_order_value = serializers.DecimalField(max_digits=10, decimal_places=2)
    avg_conversion_rate = serializers.DecimalField(max_digits=5, decimal_places=2)
    period_days = serializers.IntegerField()


class RevenueChartSerializer(serializers.Serializer):
    """Serializer for revenue chart data."""
    date = serializers.DateField()
    revenue = serializers.DecimalField(max_digits=12, decimal_places=2)
    orders = serializers.IntegerField()


class TopProductSerializer(serializers.Serializer):
    """Serializer for top products."""
    product__id = serializers.UUIDField()
    product__name = serializers.CharField()
    product__slug = serializers.CharField()
    total_revenue = serializers.DecimalField(max_digits=12, decimal_places=2)
    total_orders = serializers.IntegerField()
    total_views = serializers.IntegerField()


class PopularSearchSerializer(serializers.Serializer):
    """Serializer for popular searches."""
    query = serializers.CharField()
    count = serializers.IntegerField()
    avg_results = serializers.FloatField()


class DeviceBreakdownSerializer(serializers.Serializer):
    """Serializer for device breakdown."""
    device_type = serializers.CharField()
    count = serializers.IntegerField()


class CartAnalyticsSerializer(serializers.Serializer):
    """Serializer for cart analytics."""
    add_to_cart = serializers.IntegerField()
    checkout_started = serializers.IntegerField()
    checkout_completed = serializers.IntegerField()
    abandonment_rate = serializers.FloatField()


class TrackEventSerializer(serializers.Serializer):
    """Serializer for tracking events."""
    event_type = serializers.ChoiceField(choices=[
        'page_view', 'product_view', 'search', 'cart_add', 
        'cart_remove', 'checkout_start'
    ])
    product_id = serializers.UUIDField(required=False)
    query = serializers.CharField(required=False)
    source = serializers.CharField(required=False)
    metadata = serializers.DictField(required=False)
