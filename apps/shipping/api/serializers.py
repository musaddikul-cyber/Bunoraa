"""
Shipping API Serializers
"""
from rest_framework import serializers
from ..models import (
    ShippingZone, ShippingCarrier, ShippingMethod, ShippingRate,
    Shipment, ShipmentEvent, ShippingSettings
)
from apps.i18n.models import Currency


class ShippingCarrierSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingCarrier
        fields = ['id', 'name', 'code', 'logo', 'website', 'supports_tracking']


class ShippingMethodSerializer(serializers.ModelSerializer):
    carrier = ShippingCarrierSerializer(read_only=True)
    delivery_estimate = serializers.ReadOnlyField()
    
    class Meta:
        model = ShippingMethod
        fields = [
            'id', 'name', 'code', 'description', 'carrier',
            'min_delivery_days', 'max_delivery_days', 'delivery_estimate',
            'is_express', 'requires_signature'
        ]


class ShippingZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingZone
        fields = ['id', 'name', 'countries', 'states']


class CurrencySimpleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Currency
        fields = ['code', 'symbol', 'decimal_places', 'symbol_position']


class ShippingRateSerializer(serializers.ModelSerializer):
    zone = ShippingZoneSerializer(read_only=True)
    method = ShippingMethodSerializer(read_only=True)
    currency = CurrencySimpleSerializer(read_only=True)
    
    class Meta:
        model = ShippingRate
        fields = [
            'id', 'zone', 'method', 'rate_type', 'base_rate',
            'free_shipping_threshold', 'currency'
        ]


class ShippingRateCalculationSerializer(serializers.Serializer):
    """Serializer for rate calculation request."""
    country = serializers.CharField(max_length=100)
    state = serializers.CharField(max_length=100, required=False, allow_blank=True)
    postal_code = serializers.CharField(max_length=20, required=False, allow_blank=True)
    subtotal = serializers.DecimalField(max_digits=10, decimal_places=2, default=0)
    weight = serializers.DecimalField(max_digits=10, decimal_places=2, default=0)
    item_count = serializers.IntegerField(default=1, min_value=1)
    product_ids = serializers.ListField(
        child=serializers.UUIDField(),
        required=False,
        default=list
    )


class AvailableShippingMethodSerializer(serializers.Serializer):
    """Serializer for available shipping method response."""
    id = serializers.UUIDField()
    code = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField()
    carrier = serializers.DictField(allow_null=True)
    rate = serializers.DecimalField(max_digits=10, decimal_places=2)
    rate_display = serializers.CharField()
    is_free = serializers.BooleanField()
    delivery_estimate = serializers.CharField()
    min_days = serializers.IntegerField()
    max_days = serializers.IntegerField()
    is_express = serializers.BooleanField()
    requires_signature = serializers.BooleanField()


class ShipmentEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShipmentEvent
        fields = ['id', 'status', 'description', 'location', 'occurred_at']


class ShipmentSerializer(serializers.ModelSerializer):
    carrier = ShippingCarrierSerializer(read_only=True)
    method = ShippingMethodSerializer(read_only=True)
    events = ShipmentEventSerializer(many=True, read_only=True)
    
    class Meta:
        model = Shipment
        fields = [
            'id', 'carrier', 'method', 'tracking_number', 'tracking_url',
            'status', 'weight', 'dimensions', 'shipping_cost',
            'shipped_at', 'estimated_delivery', 'delivered_at',
            'signature_required', 'signed_by', 'events', 'created_at'
        ]


class ShipmentCreateSerializer(serializers.Serializer):
    """Serializer for creating a shipment."""
    order_id = serializers.UUIDField()
    carrier_id = serializers.UUIDField(required=False, allow_null=True)
    method_id = serializers.UUIDField(required=False, allow_null=True)
    tracking_number = serializers.CharField(max_length=100, required=False, allow_blank=True)
    weight = serializers.DecimalField(max_digits=10, decimal_places=2, required=False, allow_null=True)
    dimensions = serializers.DictField(required=False, default=dict)


class ShipmentUpdateSerializer(serializers.Serializer):
    """Serializer for updating a shipment."""
    tracking_number = serializers.CharField(max_length=100, required=False)
    carrier_id = serializers.UUIDField(required=False)
    status = serializers.ChoiceField(choices=Shipment.STATUS_CHOICES, required=False)


class TrackingUpdateSerializer(serializers.Serializer):
    """Serializer for adding tracking event."""
    status = serializers.CharField(max_length=50)
    description = serializers.CharField()
    location = serializers.CharField(max_length=255, required=False, allow_blank=True)
    occurred_at = serializers.DateTimeField(required=False)


class ShippingSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShippingSettings
        fields = [
            'origin_city', 'origin_state', 'origin_country',
            'default_weight_unit', 'default_dimension_unit',
            'show_delivery_estimates', 'show_carrier_logos',
            'enable_free_shipping', 'free_shipping_threshold',
            'handling_days'
        ]
