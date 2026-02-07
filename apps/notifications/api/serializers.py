"""
Notifications API serializers
"""
from rest_framework import serializers

from ..models import Notification, NotificationPreference, BackInStockNotification


class NotificationSerializer(serializers.ModelSerializer):
    """Serializer for notifications."""
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    
    class Meta:
        model = Notification
        fields = [
            'id', 'type', 'type_display', 'title', 'message', 'url',
            'reference_type', 'reference_id', 'is_read', 'read_at',
            'created_at'
        ]
        read_only_fields = ['id', 'type', 'title', 'message', 'url', 
                           'reference_type', 'reference_id', 'created_at']


class NotificationPreferenceSerializer(serializers.ModelSerializer):
    """Serializer for notification preferences."""
    
    class Meta:
        model = NotificationPreference
        fields = [
            'email_order_updates', 'email_shipping_updates', 'email_promotions',
            'email_newsletter', 'email_reviews', 'email_price_drops', 'email_back_in_stock',
            'sms_enabled', 'sms_order_updates', 'sms_shipping_updates', 'sms_promotions',
            'push_enabled', 'push_order_updates', 'push_promotions'
        ]


class MarkReadSerializer(serializers.Serializer):
    """Serializer for marking notifications as read."""
    notification_ids = serializers.ListField(
        child=serializers.UUIDField(),
        required=False,
        help_text='List of notification IDs to mark as read. If empty, marks all as read.'
    )


class PushTokenSerializer(serializers.Serializer):
    """Serializer for push token registration."""
    token = serializers.CharField(max_length=500)
    device_type = serializers.ChoiceField(choices=['ios', 'android', 'web'])
    device_name = serializers.CharField(max_length=100, required=False, allow_blank=True)


class BackInStockNotificationSerializer(serializers.ModelSerializer):
    """Serializer for back-in-stock notification requests."""
    
    class Meta:
        model = BackInStockNotification
        fields = ['product', 'variant', 'email']
