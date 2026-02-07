"""
Orders API serializers
"""
from rest_framework import serializers
from ..models import Order, OrderItem, OrderStatusHistory


class OrderItemSerializer(serializers.ModelSerializer):
    """Serializer for order item."""
    line_total = serializers.ReadOnlyField()
    
    class Meta:
        model = OrderItem
        fields = [
            'id', 'product_name', 'product_sku', 'variant_name',
            'product_image', 'unit_price', 'quantity', 'line_total'
        ]


class OrderStatusHistorySerializer(serializers.ModelSerializer):
    """Serializer for order status history."""
    changed_by_email = serializers.SerializerMethodField()
    
    class Meta:
        model = OrderStatusHistory
        fields = ['id', 'old_status', 'new_status', 'changed_by_email', 'notes', 'created_at']
    
    def get_changed_by_email(self, obj):
        return obj.changed_by.email if obj.changed_by else None


class OrderSerializer(serializers.ModelSerializer):
    """Serializer for order."""
    items = OrderItemSerializer(many=True, read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    shipping_method_display = serializers.CharField(source='get_shipping_method_display', read_only=True)
    payment_method_display = serializers.CharField(source='get_payment_method_display', read_only=True)
    item_count = serializers.ReadOnlyField()
    is_paid = serializers.ReadOnlyField()
    can_cancel = serializers.ReadOnlyField()
    shipping_address = serializers.SerializerMethodField()
    billing_address = serializers.SerializerMethodField()
    pickup_location = serializers.SerializerMethodField()
    
    class Meta:
        model = Order
        fields = [
            'id', 'order_number', 'email', 'phone',
            'status', 'status_display',
            'shipping_address', 'billing_address',
            'shipping_method', 'shipping_method_display', 'shipping_cost',
            'pickup_location',
            'tracking_number', 'tracking_url',
            'payment_method', 'payment_method_display', 'payment_status',
            'subtotal', 'discount', 'tax', 'total',
            'currency', 'exchange_rate',
            'payment_fee_amount', 'payment_fee_label',
            'gift_wrap', 'gift_wrap_cost',
            'coupon_code', 'customer_notes',
            'items', 'item_count', 'is_paid', 'can_cancel',
            'created_at', 'confirmed_at', 'shipped_at', 'delivered_at'
        ]
    
    def get_shipping_address(self, obj):
        return {
            'first_name': obj.shipping_first_name,
            'last_name': obj.shipping_last_name,
            'address_line_1': obj.shipping_address_line_1,
            'address_line_2': obj.shipping_address_line_2,
            'city': obj.shipping_city,
            'state': obj.shipping_state,
            'postal_code': obj.shipping_postal_code,
            'country': obj.shipping_country,
        }
    
    def get_billing_address(self, obj):
        return {
            'first_name': obj.billing_first_name,
            'last_name': obj.billing_last_name,
            'address_line_1': obj.billing_address_line_1,
            'address_line_2': obj.billing_address_line_2,
            'city': obj.billing_city,
            'state': obj.billing_state,
            'postal_code': obj.billing_postal_code,
            'country': obj.billing_country,
        }

    def get_pickup_location(self, obj):
        location = getattr(obj, 'pickup_location', None)
        if not location:
            return None
        return {
            'id': str(location.id),
            'name': location.name,
            'full_address': location.full_address,
            'phone': location.phone,
            'email': location.email,
        }


class OrderDetailSerializer(OrderSerializer):
    """Detailed order serializer with history."""
    status_history = OrderStatusHistorySerializer(many=True, read_only=True)
    
    class Meta(OrderSerializer.Meta):
        fields = OrderSerializer.Meta.fields + ['status_history', 'admin_notes']


class OrderListSerializer(serializers.ModelSerializer):
    """Simplified serializer for order list."""
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    item_count = serializers.ReadOnlyField()
    
    class Meta:
        model = Order
        fields = [
            'id', 'order_number', 'status', 'status_display',
            'total', 'item_count', 'created_at'
        ]


class CancelOrderSerializer(serializers.Serializer):
    """Serializer for cancelling order."""
    reason = serializers.CharField(required=False, allow_blank=True, max_length=500)


class UpdateOrderStatusSerializer(serializers.Serializer):
    """Serializer for updating order status (admin)."""
    status = serializers.ChoiceField(choices=Order.STATUS_CHOICES)
    notes = serializers.CharField(required=False, allow_blank=True, max_length=500)


class AddTrackingSerializer(serializers.Serializer):
    """Serializer for adding tracking information."""
    tracking_number = serializers.CharField(max_length=100)
    tracking_url = serializers.URLField(required=False, allow_blank=True)
