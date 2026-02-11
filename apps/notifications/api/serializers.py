"""
Notifications API serializers
"""
from rest_framework import serializers

from ..models import (
    Notification,
    NotificationPreference,
    NotificationDelivery,
    NotificationTemplate,
    NotificationCategory,
    NotificationPriority,
    NotificationChannel,
    NotificationType,
    BackInStockNotification,
)


class NotificationSerializer(serializers.ModelSerializer):
    """Serializer for notifications."""
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    category_display = serializers.CharField(source='get_category_display', read_only=True)
    priority_display = serializers.CharField(source='get_priority_display', read_only=True)
    delivery_status = serializers.CharField(source='status', read_only=True)
    
    class Meta:
        model = Notification
        fields = [
            'id', 'type', 'type_display', 'category', 'category_display',
            'priority', 'priority_display', 'status', 'delivery_status', 'title', 'message', 'url',
            'reference_type', 'reference_id', 'is_read', 'read_at',
            'channels_requested', 'channels_sent', 'metadata', 'created_at'
        ]
        read_only_fields = ['id', 'type', 'title', 'message', 'url', 
                           'reference_type', 'reference_id', 'created_at']


class NotificationPreferenceSerializer(serializers.ModelSerializer):
    """Serializer for notification preferences."""
    
    class Meta:
        model = NotificationPreference
        fields = [
            'email_enabled', 'email_order_updates', 'email_shipping_updates', 'email_promotions',
            'email_newsletter', 'email_reviews', 'email_price_drops', 'email_back_in_stock',
            'sms_enabled', 'sms_order_updates', 'sms_shipping_updates', 'sms_promotions',
            'push_enabled', 'push_order_updates', 'push_promotions',
            'digest_frequency', 'quiet_hours_start', 'quiet_hours_end', 'timezone',
            'marketing_opt_in', 'transactional_opt_in', 'per_type_overrides'
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
    device_type = serializers.ChoiceField(choices=['ios', 'android', 'web', 'fcm_web'])
    device_name = serializers.CharField(max_length=100, required=False, allow_blank=True)
    platform = serializers.CharField(max_length=50, required=False, allow_blank=True)
    app_version = serializers.CharField(max_length=50, required=False, allow_blank=True)
    locale = serializers.CharField(max_length=10, required=False, allow_blank=True)
    timezone = serializers.CharField(max_length=50, required=False, allow_blank=True)
    browser = serializers.CharField(max_length=100, required=False, allow_blank=True)
    user_agent = serializers.CharField(max_length=500, required=False, allow_blank=True)


class BackInStockNotificationSerializer(serializers.ModelSerializer):
    """Serializer for back-in-stock notification requests."""
    
    class Meta:
        model = BackInStockNotification
        fields = ['product', 'variant', 'email']


class NotificationDeliverySerializer(serializers.ModelSerializer):
    class Meta:
        model = NotificationDelivery
        fields = [
            'id', 'notification', 'channel', 'provider', 'external_id',
            'status', 'attempts', 'error', 'scheduled_for', 'sent_at',
            'delivered_at', 'opened_at', 'clicked_at', 'created_at'
        ]
        read_only_fields = fields


class NotificationTemplateSerializer(serializers.ModelSerializer):
    class Meta:
        model = NotificationTemplate
        fields = [
            'id', 'name', 'notification_type', 'channel', 'language',
            'subject', 'body', 'html_template', 'text_template', 'is_active',
            'created_at', 'updated_at'
        ]


class NotificationBroadcastSerializer(serializers.Serializer):
    """Serializer for admin broadcast notifications."""
    notification_type = serializers.ChoiceField(
        choices=NotificationType.choices,
        default=NotificationType.GENERAL,
    )
    title = serializers.CharField(max_length=200)
    message = serializers.CharField()
    url = serializers.URLField(required=False, allow_blank=True)
    category = serializers.ChoiceField(
        choices=NotificationCategory.choices,
        required=False,
        allow_null=True,
    )
    priority = serializers.ChoiceField(
        choices=NotificationPriority.choices,
        required=False,
        allow_null=True,
    )
    user_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        help_text='Optional list of user IDs to broadcast to. If omitted, sends to all active users.'
    )
    channels = serializers.ListField(
        child=serializers.ChoiceField(choices=NotificationChannel.choices),
        required=False,
        help_text='Optional list of channels to send (email, sms, push, in_app).'
    )
    metadata = serializers.DictField(required=False)
    dedupe_key = serializers.CharField(max_length=100, required=False, allow_blank=True)


class NotificationUnsubscribeSerializer(serializers.Serializer):
    """Serializer for unsubscribe requests."""
    token = serializers.CharField()
