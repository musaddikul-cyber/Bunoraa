"""
Payments models
"""
import logging
import uuid
from django.db import models
from django.core.validators import FileExtensionValidator
from django.conf import settings

logger = logging.getLogger(__name__)


class PaymentGateway(models.Model):
    """
    Payment gateway configuration - managed from admin panel.
    Stores enabled payment methods and their settings.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Gateway identification
    CODE_STRIPE = 'stripe'
    CODE_BKASH = 'bkash'
    CODE_NAGAD = 'nagad'
    CODE_COD = 'cod'
    CODE_BANK = 'bank_transfer'
    CODE_CHOICES = [
        (CODE_STRIPE, 'Credit/Debit Card (Stripe)'),
        (CODE_BKASH, 'bKash'),
        (CODE_NAGAD, 'Nagad'),
        (CODE_COD, 'Cash on Delivery'),
        (CODE_BANK, 'Bank Transfer'),
    ]
    code = models.CharField(max_length=50, choices=CODE_CHOICES, unique=True)
    
    # Display settings
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=255, blank=True)
    icon = models.FileField(
        upload_to='payment-icons/',
        blank=True,
        null=True,
        validators=[FileExtensionValidator(['svg', 'png', 'jpg', 'jpeg', 'webp', 'gif'])]
    )
    icon_class = models.CharField(max_length=50, blank=True, help_text="CSS class for icon (e.g., 'card', 'bkash')")
    color = models.CharField(max_length=20, default='gray', help_text="Color theme: blue, pink, orange, green, gray")
    
    # Fees
    fee_type = models.CharField(
        max_length=20,
        choices=[
            ('none', 'No Fee'),
            ('flat', 'Flat Fee'),
            ('percent', 'Percentage'),
        ],
        default='none'
    )
    fee_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    fee_text = models.CharField(max_length=50, blank=True, help_text="Display text for fee (e.g., '৳20 fee')")
    
    # Availability
    is_active = models.BooleanField(default=True)
    currencies = models.JSONField(
        default=list,
        blank=True,
        help_text="List of supported currency codes. Empty means all currencies."
    )
    countries = models.JSONField(
        default=list,
        blank=True,
        help_text="List of supported country codes. Empty means all countries."
    )
    min_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    max_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # API Configuration (encrypted in production)
    api_key = models.CharField(max_length=255, blank=True, help_text="Public/publishable key where applicable (store secrets securely)")
    api_secret = models.CharField(max_length=255, blank=True, help_text="Secret key (store encrypted in production, e.g., a secrets manager)")
    merchant_id = models.CharField(max_length=100, blank=True)
    webhook_secret = models.CharField(max_length=255, blank=True)
    is_sandbox = models.BooleanField(default=True, help_text="Use sandbox/test mode")

    # Bangladesh gateway specific configuration (inline fields for admin configuration)
    ssl_store_id = models.CharField(max_length=200, blank=True, help_text='SSLCommerz store id')
    ssl_store_passwd = models.CharField(max_length=200, blank=True, help_text='SSLCommerz store password (treat as secret)')

    bkash_mode = models.CharField(max_length=20, blank=True, db_index=True, help_text='bKash mode (sandbox/live)')
    bkash_app_key = models.CharField(max_length=255, blank=True, help_text='bKash app key (store securely)')
    bkash_app_secret = models.CharField(max_length=255, blank=True, help_text='bKash app secret (store securely)')
    bkash_username = models.CharField(max_length=255, blank=True, help_text='bKash API username')
    bkash_password = models.CharField(max_length=255, blank=True, help_text='bKash API password or secret')

    nagad_merchant_id = models.CharField(max_length=255, blank=True, help_text='Nagad merchant id')
    nagad_public_key = models.TextField(blank=True, help_text='Nagad public key (PEM)')
    nagad_private_key = models.TextField(blank=True, help_text='Nagad private key (PEM)')

    # Capability flags (index for fast filtering)
    supports_partial = models.BooleanField(default=False, db_index=True, help_text='Supports partial payments / splits')
    supports_recurring = models.BooleanField(default=False, db_index=True, help_text='Supports recurring payments/subscriptions')
    supports_bnpl = models.BooleanField(default=False, db_index=True, help_text='Supports BNPL provider integration')
    
    # Instructions for customers
    instructions = models.TextField(blank=True, help_text="Instructions shown to customer after selecting this payment method")
    
    # Bank details (for bank transfer)
    bank_name = models.CharField(max_length=100, blank=True)
    bank_account_name = models.CharField(max_length=100, blank=True)
    bank_account_number = models.CharField(max_length=50, blank=True)
    bank_routing_number = models.CharField(max_length=50, blank=True)
    bank_branch = models.CharField(max_length=100, blank=True)
    
    # Ordering
    sort_order = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        verbose_name = 'Payment Gateway'
        verbose_name_plural = 'Payment Gateways'
    
    def __str__(self):
        return self.name
    
    def is_available_for(self, currency=None, country=None, amount=None):
        """Check if this gateway is available for given parameters."""
        if not self.is_active:
            return False
        
        if currency and self.currencies:
            # Normalize codes for comparison
            try:
                cur = currency.upper()
            except Exception:
                cur = currency
            if cur not in [c.upper() for c in self.currencies]:
                return False

        if country and self.countries:
            try:
                ct = country.upper()
            except Exception:
                ct = country
            if ct not in [c.upper() for c in self.countries]:
                return False
        
        if amount:
            if self.min_amount and amount < self.min_amount:
                return False
            if self.max_amount and amount > self.max_amount:
                return False
        
        return True

    @property
    def icon_url(self):
        """Return a safe URL for the gateway icon (if any)."""
        try:
            return self.icon.url if self.icon else None
        except Exception:
            return None
    
    def calculate_fee(self, amount):
        """Calculate the fee for a given amount."""
        if self.fee_type == 'none':
            return 0
        elif self.fee_type == 'flat':
            return self.fee_amount
        elif self.fee_type == 'percent':
            return (amount * self.fee_amount) / 100
        return 0
    
    def to_dict(self):
        """Convert to dictionary for frontend."""
        return {
            'code': self.code,
            'name': self.name,
            'description': self.description,
            'icon_url': self.icon.url if self.icon else None,
            'icon_class': self.icon_class,
            'color': self.color,
            'fee_text': self.fee_text,
            'fee_type': self.fee_type,
            'fee_amount': float(self.fee_amount),
            'instructions': self.instructions,
            'is_sandbox': self.is_sandbox,
        }
    
    @classmethod
    def get_active_gateways(cls, currency=None, country=None, amount=None):
        """Get all active gateways filtered by parameters."""
        gateways = cls.objects.filter(is_active=True)
        try:
            total = gateways.count()
        except Exception:
            total = None
        filtered = [g for g in gateways if g.is_available_for(currency, country, amount)]
        try:
            logger.debug('PaymentGateway.get_active_gateways: total=%s, filtered=%s, currency=%s, country=%s, amount=%s', total, len(filtered), currency, country, amount)
        except Exception:
            pass
        return filtered


class PaymentMethod(models.Model):
    """Saved payment methods for users."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='payment_methods'
    )
    
    # Type
    TYPE_CARD = 'card'
    TYPE_BANK = 'bank'
    TYPE_PAYPAL = 'paypal'
    TYPE_CHOICES = [
        (TYPE_CARD, 'Credit/Debit Card'),
        (TYPE_BANK, 'Bank Account'),
        (TYPE_PAYPAL, 'PayPal'),
    ]
    type = models.CharField(max_length=20, choices=TYPE_CHOICES, default=TYPE_CARD)
    
    # Card details (masked)
    card_brand = models.CharField(max_length=20, blank=True, null=True)  # visa, mastercard, etc
    card_last_four = models.CharField(max_length=4, blank=True, null=True)
    card_exp_month = models.IntegerField(blank=True, null=True)
    card_exp_year = models.IntegerField(blank=True, null=True)
    
    # PayPal
    paypal_email = models.EmailField(blank=True, null=True)
    
    # Stripe
    stripe_payment_method_id = models.CharField(max_length=100, blank=True, null=True)
    
    # Status
    is_default = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    
    # Billing address
    billing_name = models.CharField(max_length=100, blank=True, null=True)
    billing_address = models.TextField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-is_default', '-created_at']
    
    def __str__(self):
        if self.type == self.TYPE_CARD:
            return f"{self.card_brand} ****{self.card_last_four}"
        elif self.type == self.TYPE_PAYPAL:
            return f"PayPal - {self.paypal_email}"
        return f"{self.get_type_display()}"
    
    def save(self, *args, **kwargs):
        # Ensure only one default per user
        if self.is_default:
            PaymentMethod.objects.filter(
                user=self.user,
                is_default=True
            ).exclude(pk=self.pk).update(is_default=False)
        super().save(*args, **kwargs)


class Payment(models.Model):
    """Payment transaction records."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    order = models.ForeignKey(
        'orders.Order',
        on_delete=models.CASCADE,
        related_name='payments'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='payments'
    )
    
    # Amount
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='BDT', help_text='Primary currency is BDT (Bangladeshi Taka)')
    
    # Payment method
    payment_method = models.ForeignKey(
        PaymentMethod,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    method_type = models.CharField(max_length=20, blank=True, null=True)
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_PROCESSING = 'processing'
    STATUS_SUCCEEDED = 'succeeded'
    STATUS_FAILED = 'failed'
    STATUS_CANCELLED = 'cancelled'
    STATUS_REFUNDED = 'refunded'
    STATUS_PARTIALLY_REFUNDED = 'partially_refunded'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_SUCCEEDED, 'Succeeded'),
        (STATUS_FAILED, 'Failed'),
        (STATUS_CANCELLED, 'Cancelled'),
        (STATUS_REFUNDED, 'Refunded'),
        (STATUS_PARTIALLY_REFUNDED, 'Partially Refunded'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    
    # Stripe
    stripe_payment_intent_id = models.CharField(max_length=100, blank=True, null=True, unique=True)
    stripe_charge_id = models.CharField(max_length=100, blank=True, null=True)
    
    # Response data
    gateway_response = models.JSONField(default=dict, blank=True)
    failure_reason = models.TextField(blank=True, null=True)
    
    # Refund tracking
    refunded_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['stripe_payment_intent_id']),
            models.Index(fields=['order', '-created_at']),
        ]
    
    def __str__(self):
        return f"Payment {self.id} - {self.amount} {self.currency} ({self.status})"


class Refund(models.Model):
    """Refund records."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    payment = models.ForeignKey(
        Payment,
        on_delete=models.CASCADE,
        related_name='refunds'
    )
    
    # Amount
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Reason
    REASON_DUPLICATE = 'duplicate'
    REASON_FRAUDULENT = 'fraudulent'
    REASON_CUSTOMER_REQUEST = 'requested_by_customer'
    REASON_PRODUCT_ISSUE = 'product_issue'
    REASON_OTHER = 'other'
    REASON_CHOICES = [
        (REASON_DUPLICATE, 'Duplicate'),
        (REASON_FRAUDULENT, 'Fraudulent'),
        (REASON_CUSTOMER_REQUEST, 'Requested by Customer'),
        (REASON_PRODUCT_ISSUE, 'Product Issue'),
        (REASON_OTHER, 'Other'),
    ]
    reason = models.CharField(max_length=30, choices=REASON_CHOICES)
    notes = models.TextField(blank=True, null=True)
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_SUCCEEDED = 'succeeded'
    STATUS_FAILED = 'failed'
    STATUS_CANCELLED = 'cancelled'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_SUCCEEDED, 'Succeeded'),
        (STATUS_FAILED, 'Failed'),
        (STATUS_CANCELLED, 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    
    # Stripe
    stripe_refund_id = models.CharField(max_length=100, blank=True, null=True)
    
    # Admin
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='refunds_created'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Refund {self.id} - {self.amount} ({self.status})"


# ------------------------- Advanced Payments & Gateway Extensions -------------------------
class PaymentTransaction(models.Model):
    """Generic transaction log for analytics and troubleshooting.

    Stores raw gateway payloads and normalized metadata for reporting and
    reconciliation.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    gateway = models.ForeignKey(PaymentGateway, on_delete=models.SET_NULL, null=True, related_name='transactions')
    payment = models.ForeignKey(Payment, on_delete=models.SET_NULL, null=True, blank=True, related_name='transactions')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='payment_transactions'
    )
    order = models.ForeignKey('orders.Order', on_delete=models.SET_NULL, null=True, blank=True)

    event_type = models.CharField(max_length=100, help_text='Event type reported by gateway (e.g., payment_success, refund)')
    reference = models.CharField(max_length=255, blank=True, null=True, db_index=True, help_text='Gateway reference id (transaction id)')
    payload = models.JSONField(default=dict, blank=True)
    fee_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (("gateway", "reference"),)
        indexes = [
            models.Index(fields=['gateway', '-created_at']),
            models.Index(fields=['reference']),
            models.Index(fields=['order']),
            models.Index(fields=['payment']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"Txn {self.reference or self.id} - {self.event_type}"


class PaymentLink(models.Model):
    """Shareable payment link for invoices or one-off payments."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    order = models.ForeignKey('orders.Order', on_delete=models.CASCADE, related_name='payment_links')
    gateway = models.ForeignKey(PaymentGateway, on_delete=models.SET_NULL, null=True, blank=True)
    code = models.CharField(max_length=64, unique=True, db_index=True, help_text='Auto-generated unique code')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='BDT')
    expires_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Payment Links"
        indexes = [
            models.Index(fields=['code']),
            models.Index(fields=['is_active']),
            models.Index(fields=['expires_at']),
            models.Index(fields=['order']),
        ]

    def __str__(self):
        return f"PaymentLink {self.code} for Order {self.order_id}"


class BkashCredential(models.Model):
    """Store bKash credentials and token cache for server-side integration.

    Tokens should be rotated and stored securely (use a secrets manager in prod).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    gateway = models.OneToOneField(PaymentGateway, on_delete=models.CASCADE, related_name='bkash_credentials')
    app_key = models.CharField(max_length=255, blank=True)
    app_secret = models.CharField(max_length=255, blank=True)
    username = models.CharField(max_length=255, blank=True)
    password = models.CharField(max_length=255, blank=True)
    auth_token = models.CharField(max_length=1024, blank=True)
    token_expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [models.Index(fields=['token_expires_at'])]

    def __str__(self):
        return f"bKash credentials for {self.gateway.name}"


class BNPLProvider(models.Model):
    """BNPL provider configuration (e.g., provider code, merchant keys)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    is_active = models.BooleanField(default=True, db_index=True)
    config = models.JSONField(default=dict, blank=True, help_text='Provider-specific configuration')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['code']), models.Index(fields=['is_active'])]

    def __str__(self):
        return self.name


class BNPLAgreement(models.Model):
    """Record of a customer's BNPL agreement/approval with a provider."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='bnpl_agreements')
    provider = models.ForeignKey(BNPLProvider, on_delete=models.CASCADE, related_name='agreements')
    provider_reference = models.CharField(max_length=255, blank=True, db_index=True)
    approved = models.BooleanField(default=False, db_index=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['user']), models.Index(fields=['provider_reference'])]

    def __str__(self):
        return f"BNPL {self.provider} for {self.user} ({'approved' if self.approved else 'pending'})"


class AuditLog(models.Model):
    """Simple audit log for tracking sensitive payment actions."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    action = models.CharField(max_length=255)
    object_type = models.CharField(max_length=100, blank=True)
    object_id = models.CharField(max_length=255, blank=True)
    data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['user']), models.Index(fields=['action']), models.Index(fields=['created_at'])]

    def __str__(self):
        return f"Audit {self.action} by {self.user} at {self.created_at}"


class RecurringCharge(models.Model):
    """Track recurring billing attempts for subscriptions (invoices created, attempts, failures)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    subscription = models.ForeignKey('subscriptions.Subscription', on_delete=models.CASCADE, related_name='recurring_charges')
    payment = models.ForeignKey(Payment, on_delete=models.SET_NULL, null=True, blank=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='BDT')
    status = models.CharField(max_length=30, default='pending')
    attempt_at = models.DateTimeField(null=True, blank=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    stripe_subscription_id = models.CharField(max_length=255, blank=True, null=True, db_index=True, help_text='Optional external subscription id reference')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['subscription']),
            models.Index(fields=['status']),
            models.Index(fields=['attempt_at']),
            models.Index(fields=['payment']),
            models.Index(fields=['stripe_subscription_id']),
        ]

    def __str__(self):
        return f"RecurringCharge {self.subscription} - {self.amount} {self.currency} ({self.status})"


# Helper property on PaymentGateway
def gateway_requires_client_js(self):
    return self.code in (self.CODE_STRIPE, self.CODE_BKASH)

PaymentGateway.requires_client_js = property(gateway_requires_client_js)
