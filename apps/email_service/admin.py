"""
Email Service Admin Configuration
==================================

Django admin interface for managing the email service.
"""

from django.contrib import admin
from django.utils.html import format_html

from .models import (
    APIKey, SenderDomain, SenderIdentity, EmailTemplate, TemplateVersion,
    EmailMessage, EmailAttachment, EmailEvent, Suppression,
    UnsubscribeGroup, UnsubscribePreference, Webhook, WebhookLog, DailyStats
)


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    """Admin for API Keys."""
    
    list_display = [
        'name', 'user', 'key_prefix', 'permission',
        'is_active', 'rate_limit_per_minute', 'created_at'
    ]
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'user__email', 'user__username']
    readonly_fields = ['key_prefix', 'key_hash', 'created_at', 'updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('user', 'name', 'key_prefix', 'key_hash')
        }),
        ('Permissions', {
            'fields': ('permission', 'allowed_ips')
        }),
        ('Limits', {
            'fields': ('rate_limit_per_minute', 'rate_limit_per_hour', 'rate_limit_per_day', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    def key_preview(self, obj):
        """Show first 8 characters of key prefix."""
        return f"{obj.key_prefix}..."
    key_preview.short_description = 'Key'


@admin.register(SenderDomain)
class SenderDomainAdmin(admin.ModelAdmin):
    """Admin for Sender Domains."""
    
    list_display = [
        'domain', 'user', 'verification_status', 'spf_verified',
        'dkim_verified', 'dmarc_verified', 'created_at'
    ]
    list_filter = ['verification_status', 'spf_verified', 'dkim_verified', 'dmarc_verified']
    search_fields = ['domain', 'user__email', 'user__username']
    readonly_fields = [
        'dkim_selector', 'dkim_public_key', 'dkim_private_key',
        'verification_token', 'created_at', 'updated_at'
    ]
    
    fieldsets = (
        (None, {
            'fields': ('user', 'domain')
        }),
        ('Verification Status', {
            'fields': ('verification_status', 'verification_token')
        }),
        ('SPF', {
            'fields': ('spf_verified', 'dns_records')
        }),
        ('DKIM', {
            'fields': ('dkim_verified', 'dkim_selector', 'dkim_public_key', 'dkim_private_key')
        }),
        ('DMARC', {
            'fields': ('dmarc_verified',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(SenderIdentity)
class SenderIdentityAdmin(admin.ModelAdmin):
    """Admin for Sender Identities."""
    
    list_display = [
        'email', 'from_name', 'domain', 'verification_status', 'created_at'
    ]
    list_filter = ['verification_status', 'domain']
    search_fields = ['email', 'from_name', 'reply_to']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(EmailTemplate)
class EmailTemplateAdmin(admin.ModelAdmin):
    """Admin for Email Templates."""
    
    list_display = [
        'name', 'user', 'is_active',
        'version_count', 'created_at', 'updated_at'
    ]
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'description', 'user__email']
    readonly_fields = ['created_at', 'updated_at']
    
    def version_count(self, obj):
        return obj.versions.count()
    version_count.short_description = 'Versions'


@admin.register(TemplateVersion)
class TemplateVersionAdmin(admin.ModelAdmin):
    """Admin for Template Versions."""
    
    list_display = ['template', 'version', 'subject', 'created_at']
    list_filter = ['created_at']
    search_fields = ['template__name', 'subject']
    readonly_fields = ['created_at']


class EmailAttachmentInline(admin.TabularInline):
    """Inline for Email Attachments."""
    
    model = EmailAttachment
    extra = 0
    readonly_fields = ['filename', 'content_type', 'size']


class EmailEventInline(admin.TabularInline):
    """Inline for Email Events."""
    
    model = EmailEvent
    extra = 0
    readonly_fields = ['event_type', 'ip_address', 'user_agent', 'url', 'timestamp']
    ordering = ['-timestamp']


@admin.register(EmailMessage)
class EmailMessageAdmin(admin.ModelAdmin):
    """Admin for Email Messages."""
    
    list_display = [
        'message_id_short', 'subject_short', 'from_email', 'to_emails_short',
        'status_badge', 'sent_at'
    ]
    list_filter = ['status', 'sent_at']
    search_fields = ['message_id', 'subject', 'from_email', 'to_emails']
    readonly_fields = [
        'message_id', 'created_at', 'updated_at', 'sent_at', 'delivered_at'
    ]
    date_hierarchy = 'sent_at'
    inlines = [EmailAttachmentInline, EmailEventInline]
    
    fieldsets = (
        ('Message Info', {
            'fields': ('message_id', 'user', 'from_email', 'from_name')
        }),
        ('Email Content', {
            'fields': ('to_emails', 'cc_emails', 
                      'bcc_emails', 'reply_to', 'subject')
        }),
        ('Body', {
            'fields': ('html_body', 'text_body'),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('status', 'priority', 'scheduled_at')
        }),
        ('Tracking', {
            'fields': ('track_opens', 'track_clicks')
        }),
        ('Delivery', {
            'fields': ('sent_at', 'delivered_at', 'smtp_response', 'error_message', 'retry_count')
        }),
        ('Metadata', {
            'fields': ('categories', 'custom_args', 'headers'),
            'classes': ('collapse',)
        }),
    )
    
    def message_id_short(self, obj):
        return obj.message_id[:12] + '...'
    message_id_short.short_description = 'Message ID'
    
    def subject_short(self, obj):
        return obj.subject[:50] + '...' if len(obj.subject) > 50 else obj.subject
    subject_short.short_description = 'Subject'
    
    def to_emails_short(self, obj):
        # Display recipient email
        return obj.to_email if obj.to_email else '-'
    to_emails_short.short_description = 'To'
    
    def status_badge(self, obj):
        colors = {
            'pending': '#6c757d',
            'queued': '#17a2b8',
            'sending': '#ffc107',
            'sent': '#28a745',
            'delivered': '#20c997',
            'opened': '#007bff',
            'clicked': '#6f42c1',
            'bounced': '#dc3545',
            'dropped': '#fd7e14',
            'deferred': '#ffc107',
            'spam': '#dc3545',
            'unsubscribed': '#6c757d',
        }
        color = colors.get(obj.status, '#6c757d')
        return format_html(
            '<span style="padding: 3px 8px; border-radius: 3px; '
            'background-color: {}; color: white; font-size: 11px;">{}</span>',
            color, obj.status.upper()
        )
    status_badge.short_description = 'Status'


@admin.register(EmailEvent)
class EmailEventAdmin(admin.ModelAdmin):
    """Admin for Email Events."""
    
    list_display = [
        'message_short', 'event_type_badge', 'ip_address',
        'user_agent_short', 'timestamp'
    ]
    list_filter = ['event_type', 'timestamp']
    search_fields = ['message__message_id', 'ip_address']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'
    
    def message_short(self, obj):
        return obj.message.message_id[:12] + '...'
    message_short.short_description = 'Message'
    
    def event_type_badge(self, obj):
        colors = {
            'delivered': '#28a745',
            'opened': '#007bff',
            'clicked': '#6f42c1',
            'bounced': '#dc3545',
            'dropped': '#fd7e14',
            'deferred': '#ffc107',
            'spam': '#dc3545',
            'unsubscribed': '#6c757d',
        }
        color = colors.get(obj.event_type, '#6c757d')
        return format_html(
            '<span style="padding: 3px 8px; border-radius: 3px; '
            'background-color: {}; color: white; font-size: 11px;">{}</span>',
            color, obj.event_type.upper()
        )
    event_type_badge.short_description = 'Event Type'
    
    def user_agent_short(self, obj):
        if obj.user_agent:
            return obj.user_agent[:50] + '...' if len(obj.user_agent) > 50 else obj.user_agent
        return '-'
    user_agent_short.short_description = 'User Agent'


@admin.register(Suppression)
class SuppressionAdmin(admin.ModelAdmin):
    """Admin for Suppressions."""
    
    list_display = ['email', 'suppression_type', 'reason_short', 'created_at']
    list_filter = ['suppression_type', 'created_at']
    search_fields = ['email', 'reason']
    readonly_fields = ['created_at']
    
    def reason_short(self, obj):
        return obj.reason[:50] + '...' if len(obj.reason) > 50 else obj.reason
    reason_short.short_description = 'Reason'


@admin.register(UnsubscribeGroup)
class UnsubscribeGroupAdmin(admin.ModelAdmin):
    """Admin for Unsubscribe Groups."""
    
    list_display = ['name', 'user', 'is_default', 'preference_count', 'created_at']
    list_filter = ['is_default', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    
    def preference_count(self, obj):
        return obj.preferences.count()
    preference_count.short_description = 'Subscribers'


@admin.register(UnsubscribePreference)
class UnsubscribePreferenceAdmin(admin.ModelAdmin):
    """Admin for Unsubscribe Preferences."""
    
    list_display = ['email', 'group', 'is_unsubscribed', 'created_at']
    list_filter = ['is_unsubscribed', 'group', 'created_at']
    search_fields = ['email']
    readonly_fields = ['created_at', 'updated_at']


class WebhookLogInline(admin.TabularInline):
    """Inline for Webhook Logs."""
    
    model = WebhookLog
    extra = 0
    readonly_fields = ['event_type', 'response_status', 'attempt', 'created_at']
    ordering = ['-created_at']


@admin.register(Webhook)
class WebhookAdmin(admin.ModelAdmin):
    """Admin for Webhooks."""
    
    list_display = [
        'url_short', 'user', 'event_types_display', 'is_active', 'created_at'
    ]
    list_filter = ['is_active', 'created_at']
    search_fields = ['url', 'user__email']
    readonly_fields = ['secret', 'created_at', 'updated_at']
    inlines = [WebhookLogInline]
    
    def url_short(self, obj):
        return obj.url[:50] + '...' if len(obj.url) > 50 else obj.url
    url_short.short_description = 'URL'
    
    def event_types_display(self, obj):
        events = obj.events if isinstance(obj.events, list) else []
        return ', '.join(events[:3]) + ('...' if len(events) > 3 else '')
    event_types_display.short_description = 'Events'


@admin.register(WebhookLog)
class WebhookLogAdmin(admin.ModelAdmin):
    """Admin for Webhook Logs."""
    
    list_display = [
        'webhook', 'event_type', 'status_badge', 'attempt', 'created_at'
    ]
    list_filter = ['created_at']
    search_fields = ['webhook__url']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    
    def status_badge(self, obj):
        if obj.response_status and 200 <= obj.response_status < 300:
            color = '#28a745'
        elif obj.response_status:
            color = '#dc3545'
        else:
            color = '#6c757d'
        return format_html(
            '<span style="padding: 3px 8px; border-radius: 3px; '
            'background-color: {}; color: white; font-size: 11px;">{}</span>',
            color, obj.response_status or 'N/A'
        )
    status_badge.short_description = 'Status'


@admin.register(DailyStats)
class DailyStatsAdmin(admin.ModelAdmin):
    """Admin for Daily Statistics."""
    
    list_display = [
        'date', 'user', 'sent',
        'delivered', 'opened', 'clicked', 'bounced', 'delivery_rate'
    ]
    list_filter = ['date', 'user']
    search_fields = ['user__email']
    readonly_fields = ['date']
    date_hierarchy = 'date'
    ordering = ['-date']
    
    def delivery_rate(self, obj):
        if obj.sent > 0:
            rate = (obj.delivered / obj.sent) * 100
            color = '#28a745' if rate >= 95 else '#ffc107' if rate >= 85 else '#dc3545'
            return format_html(
                '<span style="color: {}; font-weight: bold;">{:.1f}%</span>',
                color, rate
            )
        return '-'
    delivery_rate.short_description = 'Delivery Rate'


# Admin Site Customization
admin.site.site_header = "Email Service Administration"
admin.site.site_title = "Email Service Admin"
admin.site.index_title = "Email Service Dashboard"
