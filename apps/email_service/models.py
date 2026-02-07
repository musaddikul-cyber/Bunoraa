"""
Email Service Models
====================

Database models for the email service provider.
"""

import hashlib
import hmac
import secrets
import uuid
from datetime import timedelta
from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.core.validators import MinValueValidator, MaxValueValidator


# =============================================================================
# API KEYS & AUTHENTICATION
# =============================================================================

class APIKey(models.Model):
    """
    API keys for authenticating email service requests.
    Similar to SendGrid API keys.
    """
    
    class Permission(models.TextChoices):
        FULL_ACCESS = 'full', _('Full Access')
        MAIL_SEND = 'mail_send', _('Mail Send Only')
        READ_ONLY = 'read', _('Read Only')
        TEMPLATES = 'templates', _('Templates Only')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(_('Key Name'), max_length=100)
    key_prefix = models.CharField(_('Key Prefix'), max_length=8, db_index=True)
    key_hash = models.CharField(_('Key Hash'), max_length=64)
    
    # Ownership
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='email_api_keys'
    )
    
    # Permissions
    permission = models.CharField(
        _('Permission'),
        max_length=20,
        choices=Permission.choices,
        default=Permission.MAIL_SEND
    )
    allowed_ips = models.JSONField(
        _('Allowed IPs'),
        default=list,
        blank=True,
        help_text=_('List of IP addresses allowed to use this key')
    )
    
    # Rate limiting
    rate_limit_per_minute = models.PositiveIntegerField(default=100)
    rate_limit_per_hour = models.PositiveIntegerField(default=1000)
    rate_limit_per_day = models.PositiveIntegerField(default=10000)
    
    # Status
    is_active = models.BooleanField(_('Active'), default=True)
    last_used_at = models.DateTimeField(_('Last Used'), null=True, blank=True)
    expires_at = models.DateTimeField(_('Expires At'), null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('API Key')
        verbose_name_plural = _('API Keys')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.key_prefix}...)"
    
    @classmethod
    def generate_key(cls):
        """Generate a new API key. Returns (key_prefix, full_key, key_hash)."""
        # Generate a secure random key (similar to SendGrid format)
        key = f"BN.{secrets.token_urlsafe(32)}"
        prefix = key[:8]
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return prefix, key, key_hash
    
    @classmethod
    def create_key(cls, user, name, permission=Permission.MAIL_SEND, **kwargs):
        """Create a new API key and return the full key (shown only once)."""
        prefix, full_key, key_hash = cls.generate_key()
        api_key = cls.objects.create(
            user=user,
            name=name,
            key_prefix=prefix,
            key_hash=key_hash,
            permission=permission,
            **kwargs
        )
        return api_key, full_key
    
    @classmethod
    def verify_key(cls, key):
        """Verify an API key and return the APIKey object if valid."""
        if not key or len(key) < 8:
            return None
        
        prefix = key[:8]
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        try:
            api_key = cls.objects.get(
                key_prefix=prefix,
                key_hash=key_hash,
                is_active=True
            )
            
            # Check expiration
            if api_key.expires_at and api_key.expires_at < timezone.now():
                return None
            
            # Update last used
            api_key.last_used_at = timezone.now()
            api_key.save(update_fields=['last_used_at'])
            
            return api_key
        except cls.DoesNotExist:
            return None
    
    def has_permission(self, required_permission):
        """Check if this key has the required permission."""
        if self.permission == self.Permission.FULL_ACCESS:
            return True
        return self.permission == required_permission


# =============================================================================
# SENDER DOMAINS & VERIFICATION
# =============================================================================

class SenderDomain(models.Model):
    """
    Verified sender domains with DNS records for SPF, DKIM, and DMARC.
    """
    
    class VerificationStatus(models.TextChoices):
        PENDING = 'pending', _('Pending')
        VERIFIED = 'verified', _('Verified')
        FAILED = 'failed', _('Failed')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='sender_domains'
    )
    
    # Domain info
    domain = models.CharField(_('Domain'), max_length=255, unique=True)
    
    # Verification
    verification_status = models.CharField(
        _('Status'),
        max_length=20,
        choices=VerificationStatus.choices,
        default=VerificationStatus.PENDING
    )
    verification_token = models.CharField(_('Verification Token'), max_length=64)
    verified_at = models.DateTimeField(_('Verified At'), null=True, blank=True)
    
    # DNS Records (for display to user)
    dns_records = models.JSONField(
        _('DNS Records'),
        default=dict,
        help_text=_('Required DNS records for verification')
    )
    
    # DKIM
    dkim_selector = models.CharField(_('DKIM Selector'), max_length=50, default='bunoraa')
    dkim_public_key = models.TextField(_('DKIM Public Key'), blank=True)
    dkim_private_key = models.TextField(_('DKIM Private Key'), blank=True)
    dkim_verified = models.BooleanField(_('DKIM Verified'), default=False)
    
    # SPF
    spf_verified = models.BooleanField(_('SPF Verified'), default=False)
    
    # DMARC
    dmarc_verified = models.BooleanField(_('DMARC Verified'), default=False)
    
    # Settings
    is_default = models.BooleanField(_('Default Domain'), default=False)
    is_active = models.BooleanField(_('Active'), default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_checked_at = models.DateTimeField(_('Last Checked'), null=True, blank=True)
    
    class Meta:
        verbose_name = _('Sender Domain')
        verbose_name_plural = _('Sender Domains')
        ordering = ['-is_default', 'domain']
    
    def __str__(self):
        return self.domain
    
    def save(self, *args, **kwargs):
        if not self.verification_token:
            self.verification_token = secrets.token_hex(32)
        if not self.dns_records:
            self.generate_dns_records()
        super().save(*args, **kwargs)
    
    def generate_dns_records(self):
        """Generate required DNS records for verification."""
        self.dns_records = {
            'verification': {
                'type': 'TXT',
                'host': f'_bunoraa.{self.domain}',
                'value': f'bunoraa-verify={self.verification_token}'
            },
            'spf': {
                'type': 'TXT',
                'host': self.domain,
                'value': 'v=spf1 include:_spf.bunoraa.com ~all'
            },
            'dkim': {
                'type': 'TXT',
                'host': f'{self.dkim_selector}._domainkey.{self.domain}',
                'value': f'v=DKIM1; k=rsa; p={self.dkim_public_key[:100]}...' if self.dkim_public_key else 'Generate DKIM keys first'
            },
            'dmarc': {
                'type': 'TXT',
                'host': f'_dmarc.{self.domain}',
                'value': 'v=DMARC1; p=quarantine; rua=mailto:dmarc@bunoraa.com'
            }
        }
    
    @property
    def is_fully_verified(self):
        """Check if domain is fully verified with all DNS records."""
        return (
            self.verification_status == self.VerificationStatus.VERIFIED and
            self.dkim_verified and
            self.spf_verified
        )


class SenderIdentity(models.Model):
    """
    Verified sender email addresses or identities.
    """
    
    class VerificationStatus(models.TextChoices):
        PENDING = 'pending', _('Pending')
        VERIFIED = 'verified', _('Verified')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='sender_identities'
    )
    domain = models.ForeignKey(
        SenderDomain,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='identities'
    )
    
    # Identity info
    email = models.EmailField(_('Email Address'), unique=True)
    from_name = models.CharField(_('From Name'), max_length=100)
    reply_to = models.EmailField(_('Reply To'), blank=True)
    
    # Verification
    verification_status = models.CharField(
        _('Status'),
        max_length=20,
        choices=VerificationStatus.choices,
        default=VerificationStatus.PENDING
    )
    verification_token = models.CharField(_('Verification Token'), max_length=64)
    verified_at = models.DateTimeField(_('Verified At'), null=True, blank=True)
    
    # Settings
    is_default = models.BooleanField(_('Default Identity'), default=False)
    is_active = models.BooleanField(_('Active'), default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('Sender Identity')
        verbose_name_plural = _('Sender Identities')
        ordering = ['-is_default', 'email']
    
    def __str__(self):
        return f"{self.from_name} <{self.email}>"
    
    def save(self, *args, **kwargs):
        if not self.verification_token:
            self.verification_token = secrets.token_hex(32)
        super().save(*args, **kwargs)


# =============================================================================
# EMAIL TEMPLATES
# =============================================================================

class EmailTemplate(models.Model):
    """
    Reusable email templates with versioning.
    """
    
    class ContentType(models.TextChoices):
        HTML = 'html', _('HTML')
        TEXT = 'text', _('Plain Text')
        BOTH = 'both', _('HTML & Text')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='email_templates'
    )
    
    # Template info
    name = models.CharField(_('Template Name'), max_length=100)
    template_id = models.SlugField(_('Template ID'), max_length=100, unique=True)
    description = models.TextField(_('Description'), blank=True)
    
    # Content
    subject = models.CharField(_('Subject Line'), max_length=255)
    html_content = models.TextField(_('HTML Content'), blank=True)
    text_content = models.TextField(_('Text Content'), blank=True)
    content_type = models.CharField(
        _('Content Type'),
        max_length=10,
        choices=ContentType.choices,
        default=ContentType.HTML
    )
    
    # Variables (for dynamic content)
    variables = models.JSONField(
        _('Template Variables'),
        default=list,
        help_text=_('List of variable names used in the template')
    )
    
    # Settings
    is_active = models.BooleanField(_('Active'), default=True)
    is_system = models.BooleanField(_('System Template'), default=False)
    
    # Versioning
    version = models.PositiveIntegerField(_('Version'), default=1)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('Email Template')
        verbose_name_plural = _('Email Templates')
        ordering = ['name']
        unique_together = ['user', 'template_id']
    
    def __str__(self):
        return f"{self.name} (v{self.version})"
    
    def render(self, context=None):
        """Render template with given context."""
        from django.template import Template, Context
        
        context = context or {}
        
        html_body = ''
        text_body = ''
        
        if self.html_content:
            template = Template(self.html_content)
            html_body = template.render(Context(context))
        
        if self.text_content:
            template = Template(self.text_content)
            text_body = template.render(Context(context))
        
        subject = self.subject
        for key, value in context.items():
            subject = subject.replace(f'{{{{{key}}}}}', str(value))
        
        return {
            'subject': subject,
            'html_body': html_body,
            'text_body': text_body
        }


class TemplateVersion(models.Model):
    """
    Historical versions of email templates.
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    template = models.ForeignKey(
        EmailTemplate,
        on_delete=models.CASCADE,
        related_name='versions'
    )
    
    version = models.PositiveIntegerField(_('Version'))
    subject = models.CharField(_('Subject Line'), max_length=255)
    html_content = models.TextField(_('HTML Content'), blank=True)
    text_content = models.TextField(_('Text Content'), blank=True)
    
    # Metadata
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    change_notes = models.TextField(_('Change Notes'), blank=True)
    
    class Meta:
        verbose_name = _('Template Version')
        verbose_name_plural = _('Template Versions')
        ordering = ['-version']
        unique_together = ['template', 'version']


# =============================================================================
# EMAIL MESSAGES & DELIVERY
# =============================================================================

class EmailMessage(models.Model):
    """
    Individual email messages sent through the service.
    """
    
    class Status(models.TextChoices):
        QUEUED = 'queued', _('Queued')
        SENDING = 'sending', _('Sending')
        SENT = 'sent', _('Sent')
        DELIVERED = 'delivered', _('Delivered')
        OPENED = 'opened', _('Opened')
        CLICKED = 'clicked', _('Clicked')
        BOUNCED = 'bounced', _('Bounced')
        FAILED = 'failed', _('Failed')
        DROPPED = 'dropped', _('Dropped')
        SPAM = 'spam', _('Marked as Spam')
        UNSUBSCRIBED = 'unsubscribed', _('Unsubscribed')
    
    class BounceType(models.TextChoices):
        HARD = 'hard', _('Hard Bounce')
        SOFT = 'soft', _('Soft Bounce')
        BLOCK = 'block', _('Blocked')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    api_key = models.ForeignKey(
        APIKey,
        on_delete=models.SET_NULL,
        null=True,
        related_name='messages'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='email_messages'
    )
    
    # Message ID (for tracking)
    message_id = models.CharField(_('Message ID'), max_length=255, unique=True, db_index=True)
    
    # Sender
    from_email = models.EmailField(_('From Email'))
    from_name = models.CharField(_('From Name'), max_length=100, blank=True)
    reply_to = models.EmailField(_('Reply To'), blank=True)
    
    # Recipients
    to_email = models.EmailField(_('To Email'), db_index=True)
    to_name = models.CharField(_('To Name'), max_length=100, blank=True)
    cc = models.JSONField(_('CC'), default=list, blank=True)
    bcc = models.JSONField(_('BCC'), default=list, blank=True)
    
    # Content
    subject = models.CharField(_('Subject'), max_length=255)
    html_body = models.TextField(_('HTML Body'), blank=True)
    text_body = models.TextField(_('Text Body'), blank=True)
    
    # Template
    template = models.ForeignKey(
        EmailTemplate,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='messages'
    )
    template_data = models.JSONField(_('Template Data'), default=dict, blank=True)
    
    # Attachments
    has_attachments = models.BooleanField(_('Has Attachments'), default=False)
    
    # Headers
    headers = models.JSONField(_('Custom Headers'), default=dict, blank=True)
    
    # Categorization
    categories = models.JSONField(_('Categories'), default=list, blank=True)
    tags = models.JSONField(_('Tags'), default=list, blank=True)
    
    # Metadata
    metadata = models.JSONField(_('Metadata'), default=dict, blank=True)
    ip_address = models.GenericIPAddressField(_('Sender IP'), null=True, blank=True)
    user_agent = models.CharField(_('User Agent'), max_length=500, blank=True)
    
    # Status
    status = models.CharField(
        _('Status'),
        max_length=20,
        choices=Status.choices,
        default=Status.QUEUED,
        db_index=True
    )
    
    # Bounce info
    bounce_type = models.CharField(
        _('Bounce Type'),
        max_length=10,
        choices=BounceType.choices,
        blank=True
    )
    bounce_reason = models.TextField(_('Bounce Reason'), blank=True)
    
    # Timing
    scheduled_at = models.DateTimeField(_('Scheduled At'), null=True, blank=True)
    sent_at = models.DateTimeField(_('Sent At'), null=True, blank=True)
    delivered_at = models.DateTimeField(_('Delivered At'), null=True, blank=True)
    opened_at = models.DateTimeField(_('First Opened'), null=True, blank=True)
    clicked_at = models.DateTimeField(_('First Clicked'), null=True, blank=True)
    
    # Delivery attempts
    attempt_count = models.PositiveSmallIntegerField(_('Attempt Count'), default=0)
    last_attempt_at = models.DateTimeField(_('Last Attempt'), null=True, blank=True)
    next_retry_at = models.DateTimeField(_('Next Retry'), null=True, blank=True)
    error_message = models.TextField(_('Error Message'), blank=True)
    
    # SMTP response
    smtp_response = models.TextField(_('SMTP Response'), blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('Email Message')
        verbose_name_plural = _('Email Messages')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status', 'created_at']),
            models.Index(fields=['to_email', 'created_at']),
            models.Index(fields=['status', 'next_retry_at']),
        ]
    
    def __str__(self):
        return f"{self.subject} → {self.to_email}"
    
    def save(self, *args, **kwargs):
        if not self.message_id:
            self.message_id = f"{uuid.uuid4().hex}@bunoraa.com"
        super().save(*args, **kwargs)
    
    def get_tracking_pixel_url(self):
        """Get URL for open tracking pixel."""
        from django.urls import reverse
        return reverse('email_service:track_open', kwargs={'message_id': self.message_id})
    
    def get_click_tracking_url(self, original_url):
        """Get click tracking URL."""
        import base64
        from django.urls import reverse
        encoded_url = base64.urlsafe_b64encode(original_url.encode()).decode()
        return reverse('email_service:track_click', kwargs={
            'message_id': self.message_id,
            'url': encoded_url
        })


class EmailAttachment(models.Model):
    """
    Email attachments.
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(
        EmailMessage,
        on_delete=models.CASCADE,
        related_name='attachments'
    )
    
    filename = models.CharField(_('Filename'), max_length=255)
    content_type = models.CharField(_('Content Type'), max_length=100)
    size = models.PositiveIntegerField(_('Size (bytes)'))
    
    # Storage
    file = models.FileField(_('File'), upload_to='email_attachments/%Y/%m/')
    
    # Inline attachment
    content_id = models.CharField(_('Content ID'), max_length=255, blank=True)
    is_inline = models.BooleanField(_('Inline Attachment'), default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('Email Attachment')
        verbose_name_plural = _('Email Attachments')


# =============================================================================
# TRACKING & EVENTS
# =============================================================================

class EmailEvent(models.Model):
    """
    Email delivery and engagement events.
    """
    
    class EventType(models.TextChoices):
        QUEUED = 'queued', _('Queued')
        SENT = 'sent', _('Sent')
        DELIVERED = 'delivered', _('Delivered')
        DEFERRED = 'deferred', _('Deferred')
        BOUNCED = 'bounced', _('Bounced')
        DROPPED = 'dropped', _('Dropped')
        OPENED = 'opened', _('Opened')
        CLICKED = 'clicked', _('Clicked')
        SPAM_REPORT = 'spam', _('Spam Report')
        UNSUBSCRIBE = 'unsubscribe', _('Unsubscribe')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(
        EmailMessage,
        on_delete=models.CASCADE,
        related_name='events'
    )
    
    event_type = models.CharField(
        _('Event Type'),
        max_length=20,
        choices=EventType.choices,
        db_index=True
    )
    
    # Event data
    timestamp = models.DateTimeField(_('Timestamp'), default=timezone.now, db_index=True)
    
    # Click data
    url = models.URLField(_('Clicked URL'), max_length=2048, blank=True)
    
    # Client info
    ip_address = models.GenericIPAddressField(_('IP Address'), null=True, blank=True)
    user_agent = models.CharField(_('User Agent'), max_length=500, blank=True)
    
    # Geo data
    country = models.CharField(_('Country'), max_length=2, blank=True)
    region = models.CharField(_('Region'), max_length=100, blank=True)
    city = models.CharField(_('City'), max_length=100, blank=True)
    
    # Device info
    device_type = models.CharField(_('Device Type'), max_length=50, blank=True)
    browser = models.CharField(_('Browser'), max_length=50, blank=True)
    os = models.CharField(_('Operating System'), max_length=50, blank=True)
    
    # Additional data
    data = models.JSONField(_('Event Data'), default=dict, blank=True)
    
    class Meta:
        verbose_name = _('Email Event')
        verbose_name_plural = _('Email Events')
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['message', 'event_type', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.event_type} - {self.message.to_email}"


# =============================================================================
# SUPPRESSION LISTS
# =============================================================================

class Suppression(models.Model):
    """
    Email suppression list (unsubscribes, bounces, spam reports).
    """
    
    class SuppressionType(models.TextChoices):
        BOUNCE_HARD = 'bounce_hard', _('Hard Bounce')
        BOUNCE_SOFT = 'bounce_soft', _('Soft Bounce')
        SPAM_REPORT = 'spam', _('Spam Report')
        UNSUBSCRIBE = 'unsubscribe', _('Unsubscribe')
        MANUAL = 'manual', _('Manually Added')
        INVALID = 'invalid', _('Invalid Email')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='email_suppressions'
    )
    
    email = models.EmailField(_('Email Address'), db_index=True)
    suppression_type = models.CharField(
        _('Type'),
        max_length=20,
        choices=SuppressionType.choices
    )
    
    # Source
    source_message = models.ForeignKey(
        EmailMessage,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    reason = models.TextField(_('Reason'), blank=True)
    
    # Status
    is_active = models.BooleanField(_('Active'), default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    removed_at = models.DateTimeField(_('Removed At'), null=True, blank=True)
    removed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='removed_suppressions'
    )
    
    class Meta:
        verbose_name = _('Suppression')
        verbose_name_plural = _('Suppressions')
        ordering = ['-created_at']
        unique_together = ['user', 'email', 'suppression_type']
    
    def __str__(self):
        return f"{self.email} ({self.suppression_type})"


class UnsubscribeGroup(models.Model):
    """
    Unsubscribe groups for granular email preferences.
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='unsubscribe_groups'
    )
    
    name = models.CharField(_('Group Name'), max_length=100)
    description = models.TextField(_('Description'), blank=True)
    
    is_default = models.BooleanField(_('Default Group'), default=False)
    is_active = models.BooleanField(_('Active'), default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('Unsubscribe Group')
        verbose_name_plural = _('Unsubscribe Groups')
        ordering = ['name']
    
    def __str__(self):
        return self.name


class UnsubscribePreference(models.Model):
    """
    User unsubscribe preferences per group.
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(_('Email Address'), db_index=True)
    group = models.ForeignKey(
        UnsubscribeGroup,
        on_delete=models.CASCADE,
        related_name='preferences'
    )
    
    is_unsubscribed = models.BooleanField(_('Unsubscribed'), default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('Unsubscribe Preference')
        verbose_name_plural = _('Unsubscribe Preferences')
        unique_together = ['email', 'group']


# =============================================================================
# WEBHOOKS
# =============================================================================

class Webhook(models.Model):
    """
    Webhook configurations for delivery notifications.
    """
    
    class EventType(models.TextChoices):
        ALL = 'all', _('All Events')
        DELIVERED = 'delivered', _('Delivered')
        BOUNCED = 'bounced', _('Bounced')
        DROPPED = 'dropped', _('Dropped')
        OPENED = 'opened', _('Opened')
        CLICKED = 'clicked', _('Clicked')
        SPAM = 'spam', _('Spam Report')
        UNSUBSCRIBE = 'unsubscribe', _('Unsubscribe')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='email_webhooks'
    )
    
    name = models.CharField(_('Webhook Name'), max_length=100)
    url = models.URLField(_('Webhook URL'), max_length=500)
    
    # Events to trigger
    events = models.JSONField(
        _('Events'),
        default=list,
        help_text=_('List of event types to trigger this webhook')
    )
    
    # Security
    secret = models.CharField(_('Signing Secret'), max_length=64)
    
    # Settings
    is_active = models.BooleanField(_('Active'), default=True)
    
    # Stats
    total_sent = models.PositiveIntegerField(_('Total Sent'), default=0)
    total_failed = models.PositiveIntegerField(_('Total Failed'), default=0)
    last_sent_at = models.DateTimeField(_('Last Sent'), null=True, blank=True)
    last_error = models.TextField(_('Last Error'), blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = _('Webhook')
        verbose_name_plural = _('Webhooks')
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} → {self.url}"
    
    def save(self, *args, **kwargs):
        if not self.secret:
            self.secret = secrets.token_hex(32)
        super().save(*args, **kwargs)
    
    def sign_payload(self, payload):
        """Create HMAC signature for webhook payload."""
        import json
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        signature = hmac.new(
            self.secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"


class WebhookLog(models.Model):
    """
    Webhook delivery logs.
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    webhook = models.ForeignKey(
        Webhook,
        on_delete=models.CASCADE,
        related_name='logs'
    )
    
    # Request
    event_type = models.CharField(_('Event Type'), max_length=50)
    payload = models.JSONField(_('Payload'))
    
    # Response
    response_status = models.PositiveSmallIntegerField(_('Response Status'), null=True)
    response_body = models.TextField(_('Response Body'), blank=True)
    response_time_ms = models.PositiveIntegerField(_('Response Time (ms)'), null=True)
    
    # Status
    success = models.BooleanField(_('Success'), default=False)
    error_message = models.TextField(_('Error'), blank=True)
    
    # Retry
    attempt = models.PositiveSmallIntegerField(_('Attempt'), default=1)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('Webhook Log')
        verbose_name_plural = _('Webhook Logs')
        ordering = ['-created_at']


# =============================================================================
# ANALYTICS & STATISTICS
# =============================================================================

class DailyStats(models.Model):
    """
    Daily email statistics per user.
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='email_daily_stats'
    )
    
    date = models.DateField(_('Date'), db_index=True)
    
    # Counts
    sent = models.PositiveIntegerField(_('Sent'), default=0)
    delivered = models.PositiveIntegerField(_('Delivered'), default=0)
    opened = models.PositiveIntegerField(_('Opened'), default=0)
    clicked = models.PositiveIntegerField(_('Clicked'), default=0)
    bounced = models.PositiveIntegerField(_('Bounced'), default=0)
    dropped = models.PositiveIntegerField(_('Dropped'), default=0)
    spam_reports = models.PositiveIntegerField(_('Spam Reports'), default=0)
    unsubscribes = models.PositiveIntegerField(_('Unsubscribes'), default=0)
    
    # Unique counts
    unique_opens = models.PositiveIntegerField(_('Unique Opens'), default=0)
    unique_clicks = models.PositiveIntegerField(_('Unique Clicks'), default=0)
    
    class Meta:
        verbose_name = _('Daily Statistics')
        verbose_name_plural = _('Daily Statistics')
        ordering = ['-date']
        unique_together = ['user', 'date']
    
    def __str__(self):
        return f"{self.user} - {self.date}"
    
    @property
    def open_rate(self):
        """Calculate open rate percentage."""
        if self.delivered == 0:
            return 0
        return round((self.unique_opens / self.delivered) * 100, 2)
    
    @property
    def click_rate(self):
        """Calculate click rate percentage."""
        if self.delivered == 0:
            return 0
        return round((self.unique_clicks / self.delivered) * 100, 2)
    
    @property
    def bounce_rate(self):
        """Calculate bounce rate percentage."""
        if self.sent == 0:
            return 0
        return round((self.bounced / self.sent) * 100, 2)
