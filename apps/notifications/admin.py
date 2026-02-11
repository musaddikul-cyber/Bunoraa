"""
Notifications admin configuration
"""
from django.contrib import admin
from core.admin_mixins import EnhancedModelAdmin
from .models import (
    Notification,
    NotificationPreference,
    NotificationTemplate,
    NotificationDelivery,
    EmailTemplate,
    EmailLog,
    PushToken,
    BackInStockNotification,
)


@admin.register(Notification)
class NotificationAdmin(EnhancedModelAdmin):
    list_display = [
        'user', 'type', 'category', 'priority', 'status', 'title', 'is_read', 'created_at'
    ]
    list_filter = ['type', 'category', 'priority', 'status', 'is_read', 'created_at']
    search_fields = ['user__email', 'title', 'message']
    readonly_fields = ['id', 'created_at', 'read_at', 'channels_sent']
    
    fieldsets = (
        (None, {
            'fields': ('id', 'user', 'type', 'category', 'priority', 'status', 'title', 'message', 'url')
        }),
        ('Reference', {
            'fields': ('reference_type', 'reference_id', 'metadata', 'dedupe_key', 'expires_at')
        }),
        ('Status', {
            'fields': ('is_read', 'read_at', 'channels_requested', 'channels_sent')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    actions = ['mark_as_read', 'mark_as_unread']
    
    def mark_as_read(self, request, queryset):
        from django.utils import timezone
        queryset.update(is_read=True, read_at=timezone.now())
        self.message_user(request, f'{queryset.count()} notifications marked as read.')
    mark_as_read.short_description = 'Mark selected notifications as read'
    
    def mark_as_unread(self, request, queryset):
        queryset.update(is_read=False, read_at=None)
        self.message_user(request, f'{queryset.count()} notifications marked as unread.')
    mark_as_unread.short_description = 'Mark selected notifications as unread'


@admin.register(NotificationPreference)
class NotificationPreferenceAdmin(EnhancedModelAdmin):
    list_display = [
        'user', 'email_enabled', 'email_promotions', 'sms_enabled', 'push_enabled', 'digest_frequency'
    ]
    list_filter = ['email_enabled', 'email_promotions', 'sms_enabled', 'push_enabled', 'digest_frequency']
    search_fields = ['user__email']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('id', 'user')
        }),
        ('Email Preferences', {
            'fields': (
                'email_enabled', 'email_order_updates', 'email_shipping_updates', 'email_promotions',
                'email_newsletter', 'email_reviews', 'email_price_drops', 'email_back_in_stock'
            )
        }),
        ('SMS Preferences', {
            'fields': (
                'sms_enabled', 'sms_order_updates', 'sms_shipping_updates', 'sms_promotions'
            )
        }),
        ('Push Preferences', {
            'fields': ('push_enabled', 'push_order_updates', 'push_promotions')
        }),
        ('Delivery Rules', {
            'fields': (
                'digest_frequency', 'quiet_hours_start', 'quiet_hours_end', 'timezone',
                'marketing_opt_in', 'transactional_opt_in', 'per_type_overrides'
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(NotificationTemplate)
class NotificationTemplateAdmin(EnhancedModelAdmin):
    list_display = ['name', 'notification_type', 'channel', 'language', 'is_active', 'updated_at']
    list_filter = ['channel', 'language', 'is_active']
    search_fields = ['name', 'subject']
    readonly_fields = ['id', 'created_at', 'updated_at']

    fieldsets = (
        (None, {
            'fields': ('id', 'name', 'notification_type', 'channel', 'language', 'is_active')
        }),
        ('Content', {
            'fields': ('subject', 'body', 'html_template', 'text_template')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(NotificationDelivery)
class NotificationDeliveryAdmin(EnhancedModelAdmin):
    list_display = [
        'notification', 'channel', 'status', 'attempts', 'provider', 'external_id', 'sent_at', 'created_at'
    ]
    list_filter = ['channel', 'status', 'provider']
    search_fields = ['notification__user__email', 'external_id']
    readonly_fields = [
        'id', 'notification', 'channel', 'provider', 'external_id',
        'status', 'attempts', 'error', 'scheduled_for', 'sent_at',
        'delivered_at', 'opened_at', 'clicked_at', 'created_at', 'updated_at'
    ]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(EmailTemplate)
class EmailTemplateAdmin(EnhancedModelAdmin):
    list_display = ['name', 'notification_type', 'subject', 'is_active', 'updated_at']
    list_filter = ['is_active', 'notification_type']
    search_fields = ['name', 'subject']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('id', 'name', 'notification_type', 'is_active')
        }),
        ('Content', {
            'fields': ('subject', 'html_template', 'text_template')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(EmailLog)
class EmailLogAdmin(EnhancedModelAdmin):
    list_display = [
        'recipient_email', 'notification_type', 'subject', 'status', 'sent_at', 'created_at'
    ]
    list_filter = ['status', 'notification_type', 'created_at']
    search_fields = ['recipient_email', 'subject']
    readonly_fields = [
        'id', 'recipient_email', 'recipient_user', 'notification_type',
        'subject', 'status', 'error_message', 'sent_at', 'opened_at',
        'clicked_at', 'reference_type', 'reference_id', 'created_at'
    ]
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(PushToken)
class PushTokenAdmin(EnhancedModelAdmin):
    list_display = ['user', 'device_type', 'device_name', 'platform', 'browser', 'is_active', 'last_used_at']
    list_filter = ['device_type', 'platform', 'is_active']
    search_fields = ['user__email', 'device_name', 'token']
    readonly_fields = ['id', 'created_at', 'last_used_at']


@admin.register(BackInStockNotification)
class BackInStockNotificationAdmin(EnhancedModelAdmin):
    list_display = ('product', 'variant_display', 'user_display', 'email', 'is_notified', 'created_at')
    list_filter = ('is_notified', 'created_at')
    search_fields = ('product__name', 'user__email', 'email')
    readonly_fields = ('product', 'variant', 'user', 'email', 'created_at', 'notified_at')

    actions = ['mark_as_notified', 'mark_as_unnotified']

    def variant_display(self, obj):
        return obj.variant if obj.variant else "N/A"
    variant_display.short_description = "Variant"

    def user_display(self, obj):
        return obj.user.email if obj.user else "N/A"
    user_display.short_description = "User"

    def mark_as_notified(self, request, queryset):
        from django.utils import timezone
        updated = queryset.update(is_notified=True, notified_at=timezone.now())
        self.message_user(request, f'{updated} notifications marked as notified.')
    mark_as_notified.short_description = 'Mark selected as notified'

    def mark_as_unnotified(self, request, queryset):
        updated = queryset.update(is_notified=False, notified_at=None)
        self.message_user(request, f'{updated} notifications marked as unnotified.')
    mark_as_unnotified.short_description = 'Mark selected as unnotified'
