"""
Notifications models
"""
import uuid
from django.db import models
from django.conf import settings
from django.db.models import Q
from django.core.exceptions import ValidationError
from apps.catalog.models import Product, ProductVariant


class NotificationType(models.TextChoices):
    """Notification types."""
    ORDER_PLACED = 'order_placed', 'Order Placed'
    ORDER_CONFIRMED = 'order_confirmed', 'Order Confirmed'
    ORDER_SHIPPED = 'order_shipped', 'Order Shipped'
    ORDER_DELIVERED = 'order_delivered', 'Order Delivered'
    ORDER_CANCELLED = 'order_cancelled', 'Order Cancelled'
    ORDER_REFUNDED = 'order_refunded', 'Order Refunded'
    PAYMENT_RECEIVED = 'payment_received', 'Payment Received'
    PAYMENT_FAILED = 'payment_failed', 'Payment Failed'
    REVIEW_APPROVED = 'review_approved', 'Review Approved'
    REVIEW_REJECTED = 'review_rejected', 'Review Rejected'
    PRICE_DROP = 'price_drop', 'Price Drop'
    BACK_IN_STOCK = 'back_in_stock', 'Back In Stock'
    WISHLIST_SALE = 'wishlist_sale', 'Wishlist Item On Sale'
    ACCOUNT_CREATED = 'account_created', 'Account Created'
    PASSWORD_RESET = 'password_reset', 'Password Reset'
    PROMO_CODE = 'promo_code', 'Promo Code'
    GENERAL = 'general', 'General'


class NotificationChannel(models.TextChoices):
    """Notification delivery channels."""
    EMAIL = 'email', 'Email'
    SMS = 'sms', 'SMS'
    PUSH = 'push', 'Push Notification'
    IN_APP = 'in_app', 'In-App Notification'


class NotificationCategory(models.TextChoices):
    """High-level notification categories."""
    MARKETING = 'marketing', 'Marketing'
    TRANSACTIONAL = 'transactional', 'Transactional'
    SYSTEM = 'system', 'System'


class NotificationPriority(models.TextChoices):
    """Notification priority levels."""
    LOW = 'low', 'Low'
    NORMAL = 'normal', 'Normal'
    HIGH = 'high', 'High'
    URGENT = 'urgent', 'Urgent'


class NotificationStatus(models.TextChoices):
    """Overall notification delivery status."""
    PENDING = 'pending', 'Pending'
    PROCESSING = 'processing', 'Processing'
    SENT = 'sent', 'Sent'
    PARTIAL = 'partial', 'Partial'
    FAILED = 'failed', 'Failed'
    SKIPPED = 'skipped', 'Skipped'


class DeliveryStatus(models.TextChoices):
    """Per-channel delivery status."""
    PENDING = 'pending', 'Pending'
    QUEUED = 'queued', 'Queued'
    BATCHED = 'batched', 'Batched'
    SENT = 'sent', 'Sent'
    FAILED = 'failed', 'Failed'
    SKIPPED = 'skipped', 'Skipped'


class Notification(models.Model):
    """User notification model."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='notifications'
    )
    type = models.CharField(
        max_length=50,
        choices=NotificationType.choices,
        default=NotificationType.GENERAL
    )
    title = models.CharField(max_length=200)
    message = models.TextField()
    url = models.URLField(blank=True, null=True, help_text='Link to relevant page')
    
    # Reference to related object
    reference_type = models.CharField(max_length=50, blank=True, null=True)
    reference_id = models.CharField(max_length=100, blank=True, null=True)
    
    # Status
    is_read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    
    # Delivery tracking
    channels_requested = models.JSONField(default=list, help_text='Channels requested for delivery')
    channels_sent = models.JSONField(default=list, help_text='Channels notification was sent to')
    
    # Classification
    category = models.CharField(
        max_length=20,
        choices=NotificationCategory.choices,
        default=NotificationCategory.TRANSACTIONAL
    )
    priority = models.CharField(
        max_length=10,
        choices=NotificationPriority.choices,
        default=NotificationPriority.NORMAL
    )
    status = models.CharField(
        max_length=20,
        choices=NotificationStatus.choices,
        default=NotificationStatus.PENDING
    )
    dedupe_key = models.CharField(max_length=100, blank=True, null=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    metadata = models.JSONField(default=dict, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['user', 'is_read']),
            models.Index(fields=['type']),
            models.Index(fields=['dedupe_key']),
            models.Index(fields=['category']),
        ]
    
    def __str__(self):
        return f"{self.user.email} - {self.title}"
    
    def mark_as_read(self):
        """Mark notification as read."""
        from django.utils import timezone
        if not self.is_read:
            self.is_read = True
            self.read_at = timezone.now()
            self.save(update_fields=['is_read', 'read_at'])

    def update_status_from_deliveries(self):
        """Update overall status based on delivery records."""
        try:
            deliveries = self.deliveries.all()
        except Exception:
            return

        if not deliveries.exists():
            return

        statuses = set(deliveries.values_list('status', flat=True))
        if statuses.issubset({DeliveryStatus.SENT}):
            self.status = NotificationStatus.SENT
        elif DeliveryStatus.FAILED in statuses and DeliveryStatus.SENT in statuses:
            self.status = NotificationStatus.PARTIAL
        elif statuses.issubset({DeliveryStatus.FAILED, DeliveryStatus.SKIPPED}):
            self.status = NotificationStatus.FAILED
        elif DeliveryStatus.PENDING in statuses or DeliveryStatus.QUEUED in statuses or DeliveryStatus.BATCHED in statuses:
            self.status = NotificationStatus.PROCESSING
        else:
            self.status = NotificationStatus.PENDING

        self.save(update_fields=['status', 'updated_at'])


class NotificationPreference(models.Model):
    """User notification preferences."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='notification_preferences'
    )
    
    # Email preferences
    email_enabled = models.BooleanField(default=True)
    email_order_updates = models.BooleanField(default=True)
    email_shipping_updates = models.BooleanField(default=True)
    email_promotions = models.BooleanField(default=True)
    email_newsletter = models.BooleanField(default=True)
    email_reviews = models.BooleanField(default=True)
    email_price_drops = models.BooleanField(default=True)
    email_back_in_stock = models.BooleanField(default=True)
    
    # SMS preferences
    sms_enabled = models.BooleanField(default=False)
    sms_order_updates = models.BooleanField(default=True)
    sms_shipping_updates = models.BooleanField(default=True)
    sms_promotions = models.BooleanField(default=False)
    
    # Push preferences
    push_enabled = models.BooleanField(default=True)
    push_order_updates = models.BooleanField(default=True)
    push_promotions = models.BooleanField(default=True)
    
    # Digest & quiet hours
    DIGEST_IMMEDIATE = 'immediate'
    DIGEST_HOURLY = 'hourly'
    DIGEST_DAILY = 'daily'
    DIGEST_WEEKLY = 'weekly'
    DIGEST_NEVER = 'never'
    DIGEST_CHOICES = [
        (DIGEST_IMMEDIATE, 'Immediate'),
        (DIGEST_HOURLY, 'Hourly'),
        (DIGEST_DAILY, 'Daily'),
        (DIGEST_WEEKLY, 'Weekly'),
        (DIGEST_NEVER, 'Never'),
    ]
    digest_frequency = models.CharField(
        max_length=20,
        choices=DIGEST_CHOICES,
        default=DIGEST_IMMEDIATE
    )
    quiet_hours_start = models.TimeField(null=True, blank=True)
    quiet_hours_end = models.TimeField(null=True, blank=True)
    timezone = models.CharField(max_length=50, default='Asia/Dhaka')
    marketing_opt_in = models.BooleanField(default=True)
    transactional_opt_in = models.BooleanField(default=True)
    per_type_overrides = models.JSONField(default=dict, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = 'Notification Preferences'
    
    def __str__(self):
        return f"{self.user.email} - Notification Preferences"


class NotificationTemplate(models.Model):
    """Unified templates for email, SMS, push, and in-app notifications."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    notification_type = models.CharField(max_length=50, choices=NotificationType.choices)
    channel = models.CharField(max_length=20, choices=NotificationChannel.choices)
    language = models.CharField(max_length=10, default='en')
    
    subject = models.CharField(max_length=200, blank=True)
    body = models.TextField(blank=True, help_text='Body for SMS/push/in-app')
    html_template = models.TextField(blank=True, help_text='HTML template for email')
    text_template = models.TextField(blank=True, help_text='Plain text template for email')
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['notification_type', 'channel', 'language']),
        ]
        unique_together = ('notification_type', 'channel', 'language')
    
    def __str__(self):
        return f"{self.notification_type} ({self.channel}) [{self.language}]"


class NotificationDelivery(models.Model):
    """Delivery status per channel."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    notification = models.ForeignKey(
        Notification,
        on_delete=models.CASCADE,
        related_name='deliveries'
    )
    channel = models.CharField(max_length=20, choices=NotificationChannel.choices)
    provider = models.CharField(max_length=50, blank=True, null=True)
    external_id = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=20, choices=DeliveryStatus.choices, default=DeliveryStatus.PENDING)
    attempts = models.PositiveSmallIntegerField(default=0)
    error = models.TextField(blank=True, null=True)
    scheduled_for = models.DateTimeField(null=True, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    clicked_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['channel', 'status']),
            models.Index(fields=['notification', 'channel']),
        ]
    
    def __str__(self):
        return f"{self.notification_id} - {self.channel} ({self.status})"


class EmailTemplate(models.Model):
    """Email template model."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    notification_type = models.CharField(
        max_length=50,
        choices=NotificationType.choices,
        unique=True
    )
    subject = models.CharField(max_length=200)
    html_template = models.TextField(help_text='HTML template with placeholders')
    text_template = models.TextField(help_text='Plain text template with placeholders')
    
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name


class EmailLog(models.Model):
    """Email delivery log."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recipient_email = models.EmailField()
    recipient_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='email_logs'
    )
    notification_type = models.CharField(
        max_length=50,
        choices=NotificationType.choices
    )
    subject = models.CharField(max_length=200)
    notification = models.ForeignKey(
        Notification,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='email_logs'
    )
    delivery = models.ForeignKey(
        'NotificationDelivery',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='email_logs'
    )
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_SENT = 'sent'
    STATUS_FAILED = 'failed'
    STATUS_BOUNCED = 'bounced'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_SENT, 'Sent'),
        (STATUS_FAILED, 'Failed'),
        (STATUS_BOUNCED, 'Bounced'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    error_message = models.TextField(blank=True, null=True)
    
    # Tracking
    sent_at = models.DateTimeField(null=True, blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    clicked_at = models.DateTimeField(null=True, blank=True)
    
    # Reference
    reference_type = models.CharField(max_length=50, blank=True, null=True)
    reference_id = models.CharField(max_length=100, blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Email Logs'
        indexes = [
            models.Index(fields=['recipient_email', '-created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.recipient_email} - {self.subject}"


class BackInStockNotification(models.Model):
    """
    Records a customer's request to be notified when a product or variant is back in stock.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='back_in_stock_requests')
    variant = models.ForeignKey(
        ProductVariant,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='back_in_stock_requests',
        help_text='Specific variant of the product requested (if applicable).'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='back_in_stock_subscriptions',
        help_text='Authenticated user who requested the notification.'
    )
    email = models.EmailField(
        blank=True,
        help_text='Email address to notify if user is not authenticated.'
    )
    
    is_notified = models.BooleanField(default=False, help_text='Has the customer been notified?')
    notified_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Back in Stock Notification Request'
        verbose_name_plural = 'Back in Stock Notification Requests'
        # Ensure uniqueness of request (product/variant/user or product/variant/email)
        constraints = [
            models.UniqueConstraint(
                fields=['product', 'variant', 'user'],
                condition=Q(user__isnull=False),
                name='unique_back_in_stock_user'
            ),
            models.UniqueConstraint(
                fields=['product', 'variant', 'email'],
                condition=Q(user__isnull=True),
                name='unique_back_in_stock_email'
            ),
        ]

    def clean(self):
        if not self.user and not self.email:
            raise ValidationError("Either a user or an email address must be provided.")
        if self.user and self.email:
            raise ValidationError("Cannot provide both a user and an email address.")
        if self.variant and self.variant.product != self.product:
            raise ValidationError("The selected variant does not belong to the specified product.")

    def __str__(self):
        target = self.product.name
        if self.variant:
            target += f" ({self.variant})"
        requester = self.user.get_full_name() if self.user else self.email
        return f"Back in stock request for {target} by {requester}"


class PushToken(models.Model):
    """Push notification device tokens."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='push_tokens'
    )
    token = models.CharField(max_length=500, unique=True)
    device_type = models.CharField(max_length=20)  # ios, android, web, fcm_web
    device_name = models.CharField(max_length=100, blank=True, null=True)
    platform = models.CharField(max_length=50, blank=True, null=True)
    app_version = models.CharField(max_length=50, blank=True, null=True)
    locale = models.CharField(max_length=10, blank=True, null=True)
    timezone = models.CharField(max_length=50, blank=True, null=True)
    browser = models.CharField(max_length=100, blank=True, null=True)
    user_agent = models.CharField(max_length=500, blank=True, null=True)
    last_ip = models.GenericIPAddressField(null=True, blank=True)
    
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-last_used_at']
    
    def __str__(self):
        return f"{self.user.email} - {self.device_type}"
