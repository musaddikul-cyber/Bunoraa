"""
Email Service API Serializers
==============================

REST API serializers for the email service.
"""

from rest_framework import serializers
from django.utils.translation import gettext_lazy as _
from .models import (
    APIKey, SenderDomain, SenderIdentity, EmailTemplate,
    EmailMessage, EmailEvent, Suppression, UnsubscribeGroup,
    Webhook, DailyStats
)


# =============================================================================
# API KEYS
# =============================================================================

class APIKeySerializer(serializers.ModelSerializer):
    """Serializer for API keys (without exposing the actual key)."""
    
    class Meta:
        model = APIKey
        fields = [
            'id', 'name', 'key_prefix', 'permission', 'allowed_ips',
            'rate_limit_per_minute', 'rate_limit_per_hour', 'rate_limit_per_day',
            'is_active', 'last_used_at', 'expires_at', 'created_at'
        ]
        read_only_fields = ['id', 'key_prefix', 'last_used_at', 'created_at']


class APIKeyCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating API keys."""
    
    class Meta:
        model = APIKey
        fields = ['name', 'permission', 'allowed_ips', 'expires_at']
    
    def create(self, validated_data):
        user = self.context['request'].user
        api_key, full_key = APIKey.create_key(user=user, **validated_data)
        # Store full key for response (shown only once)
        api_key._full_key = full_key
        return api_key
    
    def to_representation(self, instance):
        data = APIKeySerializer(instance).data
        # Include full key only on creation
        if hasattr(instance, '_full_key'):
            data['api_key'] = instance._full_key
        return data


# =============================================================================
# SENDER DOMAINS
# =============================================================================

class SenderDomainSerializer(serializers.ModelSerializer):
    """Serializer for sender domains."""
    
    is_fully_verified = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = SenderDomain
        fields = [
            'id', 'domain', 'verification_status', 'dns_records',
            'dkim_verified', 'spf_verified', 'dmarc_verified',
            'is_fully_verified', 'is_default', 'is_active',
            'verified_at', 'last_checked_at', 'created_at'
        ]
        read_only_fields = [
            'id', 'verification_status', 'dns_records', 'dkim_verified',
            'spf_verified', 'dmarc_verified', 'verified_at', 'last_checked_at', 'created_at'
        ]


class SenderDomainCreateSerializer(serializers.Serializer):
    """Serializer for adding a sender domain."""
    
    domain = serializers.CharField(max_length=255)
    
    def validate_domain(self, value):
        # Normalize domain
        value = value.lower().strip()
        
        # Basic validation
        if not value or '.' not in value:
            raise serializers.ValidationError(_("Invalid domain name"))
        
        # Check for existing domain
        if SenderDomain.objects.filter(domain=value).exists():
            raise serializers.ValidationError(_("Domain already registered"))
        
        return value


# =============================================================================
# SENDER IDENTITIES
# =============================================================================

class SenderIdentitySerializer(serializers.ModelSerializer):
    """Serializer for sender identities."""
    
    class Meta:
        model = SenderIdentity
        fields = [
            'id', 'email', 'from_name', 'reply_to', 'domain',
            'verification_status', 'is_default', 'is_active',
            'verified_at', 'created_at'
        ]
        read_only_fields = ['id', 'verification_status', 'verified_at', 'created_at']


# =============================================================================
# EMAIL TEMPLATES
# =============================================================================

class EmailTemplateSerializer(serializers.ModelSerializer):
    """Serializer for email templates."""
    
    class Meta:
        model = EmailTemplate
        fields = [
            'id', 'name', 'template_id', 'description', 'subject',
            'html_content', 'text_content', 'content_type', 'variables',
            'is_active', 'version', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'version', 'created_at', 'updated_at']


class EmailTemplateListSerializer(serializers.ModelSerializer):
    """Lighter serializer for template lists."""
    
    class Meta:
        model = EmailTemplate
        fields = [
            'id', 'name', 'template_id', 'description', 'subject',
            'content_type', 'is_active', 'version', 'updated_at'
        ]


# =============================================================================
# EMAIL MESSAGES
# =============================================================================

class EmailRecipientSerializer(serializers.Serializer):
    """Serializer for email recipient."""
    
    email = serializers.EmailField()
    name = serializers.CharField(max_length=100, required=False, allow_blank=True)


class EmailAttachmentSerializer(serializers.Serializer):
    """Serializer for email attachment."""
    
    filename = serializers.CharField(max_length=255)
    content = serializers.CharField(help_text=_("Base64 encoded content"))
    content_type = serializers.CharField(max_length=100, default='application/octet-stream')
    content_id = serializers.CharField(max_length=255, required=False, allow_blank=True)


class SendEmailSerializer(serializers.Serializer):
    """
    Serializer for sending emails via API.
    Similar to SendGrid's Mail Send API.
    """
    
    # Recipients
    to = serializers.ListField(
        child=EmailRecipientSerializer(),
        min_length=1,
        max_length=1000
    )
    cc = serializers.ListField(
        child=EmailRecipientSerializer(),
        required=False,
        max_length=100
    )
    bcc = serializers.ListField(
        child=EmailRecipientSerializer(),
        required=False,
        max_length=100
    )
    
    # Sender
    from_email = serializers.EmailField(required=False)
    from_name = serializers.CharField(max_length=100, required=False, allow_blank=True)
    reply_to = serializers.EmailField(required=False, allow_blank=True)
    
    # Content
    subject = serializers.CharField(max_length=255)
    html_body = serializers.CharField(required=False, allow_blank=True)
    text_body = serializers.CharField(required=False, allow_blank=True)
    
    # Template
    template_id = serializers.SlugField(required=False)
    template_data = serializers.DictField(required=False, default=dict)
    
    # Attachments
    attachments = serializers.ListField(
        child=EmailAttachmentSerializer(),
        required=False,
        max_length=10
    )
    
    # Headers
    headers = serializers.DictField(
        child=serializers.CharField(),
        required=False,
        default=dict
    )
    
    # Categorization
    categories = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False,
        max_length=10
    )
    tags = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False,
        max_length=10
    )
    
    # Metadata
    metadata = serializers.DictField(required=False, default=dict)
    
    # Scheduling
    send_at = serializers.DateTimeField(required=False)
    
    # Tracking
    track_opens = serializers.BooleanField(default=True)
    track_clicks = serializers.BooleanField(default=True)
    
    def validate(self, data):
        # Ensure content or template is provided
        has_content = data.get('html_body') or data.get('text_body')
        has_template = data.get('template_id')
        
        if not has_content and not has_template:
            raise serializers.ValidationError(
                _("Either content (html_body/text_body) or template_id is required")
            )
        
        # Validate template exists
        if has_template:
            try:
                user = self.context['request'].user
                EmailTemplate.objects.get(
                    template_id=data['template_id'],
                    user=user,
                    is_active=True
                )
            except EmailTemplate.DoesNotExist:
                raise serializers.ValidationError(
                    {'template_id': _("Template not found")}
                )
        
        return data


class EmailMessageSerializer(serializers.ModelSerializer):
    """Serializer for email message details."""
    
    events = serializers.SerializerMethodField()
    
    class Meta:
        model = EmailMessage
        fields = [
            'id', 'message_id', 'from_email', 'from_name', 'to_email', 'to_name',
            'cc', 'bcc', 'reply_to', 'subject', 'status', 'bounce_type',
            'bounce_reason', 'categories', 'tags', 'metadata',
            'scheduled_at', 'sent_at', 'delivered_at', 'opened_at', 'clicked_at',
            'attempt_count', 'error_message', 'created_at', 'events'
        ]
        read_only_fields = fields
    
    def get_events(self, obj):
        events = obj.events.all()[:20]
        return EmailEventSerializer(events, many=True).data


class EmailMessageListSerializer(serializers.ModelSerializer):
    """Lighter serializer for message lists."""
    
    class Meta:
        model = EmailMessage
        fields = [
            'id', 'message_id', 'to_email', 'subject', 'status',
            'sent_at', 'opened_at', 'created_at'
        ]


# =============================================================================
# EMAIL EVENTS
# =============================================================================

class EmailEventSerializer(serializers.ModelSerializer):
    """Serializer for email events."""
    
    class Meta:
        model = EmailEvent
        fields = [
            'id', 'event_type', 'timestamp', 'url', 'ip_address',
            'country', 'city', 'device_type', 'browser', 'os', 'data'
        ]


# =============================================================================
# SUPPRESSIONS
# =============================================================================

class SuppressionSerializer(serializers.ModelSerializer):
    """Serializer for suppressions."""
    
    class Meta:
        model = Suppression
        fields = [
            'id', 'email', 'suppression_type', 'reason',
            'is_active', 'created_at', 'removed_at'
        ]
        read_only_fields = ['id', 'created_at', 'removed_at']


class SuppressionCreateSerializer(serializers.Serializer):
    """Serializer for adding suppressions."""
    
    emails = serializers.ListField(
        child=serializers.EmailField(),
        min_length=1,
        max_length=1000
    )
    suppression_type = serializers.ChoiceField(
        choices=Suppression.SuppressionType.choices,
        default=Suppression.SuppressionType.MANUAL
    )
    reason = serializers.CharField(required=False, allow_blank=True)


# =============================================================================
# UNSUBSCRIBE GROUPS
# =============================================================================

class UnsubscribeGroupSerializer(serializers.ModelSerializer):
    """Serializer for unsubscribe groups."""
    
    class Meta:
        model = UnsubscribeGroup
        fields = [
            'id', 'name', 'description', 'is_default', 'is_active',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


# =============================================================================
# WEBHOOKS
# =============================================================================

class WebhookSerializer(serializers.ModelSerializer):
    """Serializer for webhooks."""
    
    class Meta:
        model = Webhook
        fields = [
            'id', 'name', 'url', 'events', 'is_active',
            'total_sent', 'total_failed', 'last_sent_at', 'last_error',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'total_sent', 'total_failed', 'last_sent_at',
            'last_error', 'created_at', 'updated_at'
        ]


class WebhookCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating webhooks."""
    
    class Meta:
        model = Webhook
        fields = ['name', 'url', 'events']
    
    def validate_events(self, value):
        valid_events = [e[0] for e in Webhook.EventType.choices]
        for event in value:
            if event not in valid_events:
                raise serializers.ValidationError(
                    _(f"Invalid event type: {event}")
                )
        return value


# =============================================================================
# STATISTICS
# =============================================================================

class DailyStatsSerializer(serializers.ModelSerializer):
    """Serializer for daily statistics."""
    
    open_rate = serializers.FloatField(read_only=True)
    click_rate = serializers.FloatField(read_only=True)
    bounce_rate = serializers.FloatField(read_only=True)
    
    class Meta:
        model = DailyStats
        fields = [
            'date', 'sent', 'delivered', 'opened', 'clicked',
            'bounced', 'dropped', 'spam_reports', 'unsubscribes',
            'unique_opens', 'unique_clicks',
            'open_rate', 'click_rate', 'bounce_rate'
        ]


class StatsOverviewSerializer(serializers.Serializer):
    """Serializer for statistics overview."""
    
    period = serializers.CharField()
    total_sent = serializers.IntegerField()
    total_delivered = serializers.IntegerField()
    total_opened = serializers.IntegerField()
    total_clicked = serializers.IntegerField()
    total_bounced = serializers.IntegerField()
    total_spam_reports = serializers.IntegerField()
    total_unsubscribes = serializers.IntegerField()
    
    delivery_rate = serializers.FloatField()
    open_rate = serializers.FloatField()
    click_rate = serializers.FloatField()
    bounce_rate = serializers.FloatField()
    
    daily_stats = DailyStatsSerializer(many=True)
