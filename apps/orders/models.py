"""
Orders models
"""
import uuid
import secrets
from decimal import Decimal
from django.db import models, IntegrityError, transaction
from django.conf import settings
from django.utils import timezone


class Order(models.Model):
    """
    Order model representing a customer purchase.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Order number for display
    order_number = models.CharField(max_length=50, unique=True, db_index=True)
    
    # Customer
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name='orders',
        null=True,
        blank=True
    )
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True)
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_CONFIRMED = 'confirmed'
    STATUS_PROCESSING = 'processing'
    STATUS_SHIPPED = 'shipped'
    STATUS_DELIVERED = 'delivered'
    STATUS_CANCELLED = 'cancelled'
    STATUS_REFUNDED = 'refunded'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_CONFIRMED, 'Confirmed'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_SHIPPED, 'Shipped'),
        (STATUS_DELIVERED, 'Delivered'),
        (STATUS_CANCELLED, 'Cancelled'),
        (STATUS_REFUNDED, 'Refunded'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True
    )
    
    # Shipping address
    shipping_first_name = models.CharField(max_length=100)
    shipping_last_name = models.CharField(max_length=100)
    shipping_address_line_1 = models.CharField(max_length=255)
    shipping_address_line_2 = models.CharField(max_length=255, blank=True)
    shipping_city = models.CharField(max_length=100)
    shipping_state = models.CharField(max_length=100, blank=True)
    shipping_postal_code = models.CharField(max_length=20)
    shipping_country = models.CharField(max_length=100)
    
    # Billing address
    billing_first_name = models.CharField(max_length=100)
    billing_last_name = models.CharField(max_length=100)
    billing_address_line_1 = models.CharField(max_length=255)
    billing_address_line_2 = models.CharField(max_length=255, blank=True)
    billing_city = models.CharField(max_length=100)
    billing_state = models.CharField(max_length=100, blank=True)
    billing_postal_code = models.CharField(max_length=20)
    billing_country = models.CharField(max_length=100)
    
    # Shipping
    SHIPPING_STANDARD = 'standard'
    SHIPPING_EXPRESS = 'express'
    SHIPPING_OVERNIGHT = 'overnight'
    SHIPPING_PICKUP = 'pickup'
    SHIPPING_CHOICES = [
        (SHIPPING_STANDARD, 'Standard Shipping'),
        (SHIPPING_EXPRESS, 'Express Shipping'),
        (SHIPPING_OVERNIGHT, 'Overnight Shipping'),
        (SHIPPING_PICKUP, 'Store Pickup'),
    ]
    shipping_method = models.CharField(
        max_length=20,
        choices=SHIPPING_CHOICES,
        default=SHIPPING_STANDARD,
        db_index=True,
        help_text='Shipping method chosen for this order'
    )
    shipping_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    pickup_location = models.ForeignKey(
        'contacts.StoreLocation',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='order_pickups'
    )
    tracking_number = models.CharField(max_length=100, blank=True)
    tracking_url = models.URLField(blank=True)
    
    # Payment
    PAYMENT_STRIPE = 'stripe'
    PAYMENT_PAYPAL = 'paypal'
    PAYMENT_CHOICES = [
        (PAYMENT_STRIPE, 'Credit Card (Stripe)'),
        (PAYMENT_PAYPAL, 'PayPal'),
    ]
    payment_method = models.CharField(
        max_length=20,
        choices=PAYMENT_CHOICES,
        default=PAYMENT_STRIPE,
        db_index=True,
        help_text='Payment method used for this order'
    )

    # Payment status
    PAYMENT_PENDING = 'pending'
    PAYMENT_SUCCEEDED = 'succeeded'
    PAYMENT_FAILED = 'failed'
    PAYMENT_REFUNDED = 'refunded'
    PAYMENT_CHOICES = [
        (PAYMENT_PENDING, 'Pending'),
        (PAYMENT_SUCCEEDED, 'Succeeded'),
        (PAYMENT_FAILED, 'Failed'),
        (PAYMENT_REFUNDED, 'Refunded'),
    ]
    payment_status = models.CharField(max_length=20, choices=PAYMENT_CHOICES, default=PAYMENT_PENDING, db_index=True)
    stripe_payment_intent_id = models.CharField(max_length=255, blank=True)
    
    # Amounts
    subtotal = models.DecimalField(max_digits=10, decimal_places=2, help_text='Order subtotal in BDT')
    discount = models.DecimalField(max_digits=10, decimal_places=2, default=0, help_text='Discount amount applied')
    tax = models.DecimalField(max_digits=10, decimal_places=2, default=0, help_text='Tax amount applied')
    total = models.DecimalField(max_digits=10, decimal_places=2, help_text='Order total in BDT')

    # Currency snapshot
    currency = models.CharField(max_length=3, default='BDT')
    exchange_rate = models.DecimalField(max_digits=10, decimal_places=6, default=1)

    # Payment fee snapshot
    payment_fee_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    payment_fee_label = models.CharField(max_length=100, blank=True)
    
    # Coupon
    coupon = models.ForeignKey(
        'promotions.Coupon',
        on_delete=models.SET_NULL,
        related_name='orders',
        null=True,
        blank=True
    )
    coupon_code = models.CharField(max_length=50, blank=True)
    
    # Notes
    customer_notes = models.TextField(blank=True)
    admin_notes = models.TextField(blank=True)
    
    # Gift Options
    is_gift = models.BooleanField(default=False)
    gift_message = models.TextField(blank=True, max_length=500)
    gift_wrap = models.BooleanField(default=False)
    gift_wrap_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    confirmed_at = models.DateTimeField(null=True, blank=True)
    shipped_at = models.DateTimeField(null=True, blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    
    # Soft delete
    is_deleted = models.BooleanField(default=False, db_index=True, help_text='Soft-delete flag; delete is logical')
    deleted_at = models.DateTimeField(null=True, blank=True, help_text='Timestamp when the order was soft-deleted')

    def soft_delete(self):
        """Soft-delete the order (logical delete)."""
        if not self.is_deleted:
            self.is_deleted = True
            self.deleted_at = timezone.now()
            self.save(update_fields=['is_deleted', 'deleted_at'])
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Order'
        verbose_name_plural = 'Orders'
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status', '-created_at']),
            models.Index(fields=['order_number']),
            models.Index(fields=['email']),
        ]
    
    def __str__(self):
        return f"Order {self.order_number}"
    
    def save(self, *args, **kwargs):
        if self.order_number:
            super().save(*args, **kwargs)
            return

        for attempt in range(5):
            self.order_number = self.generate_order_number()
            try:
                with transaction.atomic():
                    super().save(*args, **kwargs)
                return
            except IntegrityError:
                if attempt >= 4:
                    raise
                self.order_number = None
    
    def generate_order_number(self):
        """Generate unique order number."""
        from datetime import datetime
        prefix = datetime.now().strftime('%Y%m%d')
        # Get count of orders today
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        random_suffix = secrets.token_hex(2).upper()
        return f"ORD-{prefix}-{random_suffix}"
    
    @property
    def shipping_full_name(self):
        return f"{self.shipping_first_name} {self.shipping_last_name}"

    @property
    def billing_full_name(self):
        return f"{self.billing_first_name} {self.billing_last_name}"

    @property
    def total_amount(self):
        return self.total

    @property
    def discount_amount(self):
        return self.discount

    @property
    def tax_amount(self):
        return self.tax
    
    @property
    def item_count(self):
        return sum(item.quantity for item in self.items.all())
    
    @property
    def is_paid(self):
        return self.payment_status == 'succeeded'
    
    @property
    def can_cancel(self):
        return self.status in [self.STATUS_PENDING, self.STATUS_CONFIRMED]
    
    def get_shipping_address_display(self):
        """Get formatted shipping address."""
        lines = [
            self.shipping_full_name,
            self.shipping_address_line_1,
        ]
        if self.shipping_address_line_2:
            lines.append(self.shipping_address_line_2)
        lines.append(f"{self.shipping_city}, {self.shipping_state} {self.shipping_postal_code}")
        lines.append(self.shipping_country)
        return '\n'.join(lines)
    
    def get_billing_address_display(self):
        """Get formatted billing address."""
        lines = [
            self.billing_full_name,
            self.billing_address_line_1,
        ]
        if self.billing_address_line_2:
            lines.append(self.billing_address_line_2)
        lines.append(f"{self.billing_city}, {self.billing_state} {self.billing_postal_code}")
        lines.append(self.billing_country)
        return '\n'.join(lines)


class OrderItem(models.Model):
    """
    Order item representing a product in an order.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name='items',
        null=True,
        blank=True
    )
    
    # Product reference (keep even if product deleted)
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.SET_NULL,
        related_name='order_items',
        null=True
    )
    variant = models.ForeignKey(
        'catalog.ProductVariant',
        on_delete=models.SET_NULL,
        related_name='order_items',
        null=True,
        blank=True
    )
    
    # Snapshot of product at time of order
    product_name = models.CharField(max_length=255)
    product_sku = models.CharField(max_length=100, blank=True)
    variant_name = models.CharField(max_length=255, blank=True)
    product_image = models.URLField(blank=True)
    
    # Pricing
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.PositiveIntegerField(default=1)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        verbose_name = 'Order Item'
        verbose_name_plural = 'Order Items'
    
    def __str__(self):
        return f"{self.product_name} x {self.quantity}"
    
    @property
    def line_total(self):
        """Return line total as Decimal, treating missing price or quantity as zero."""
        unit = self.unit_price if self.unit_price is not None else Decimal('0.00')
        qty = self.quantity if self.quantity is not None else 0
        return unit * qty


class OrderStatusHistory(models.Model):
    """
    Track order status changes.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name='status_history',
        null=True,
        blank=True
    )
    
    old_status = models.CharField(max_length=20, blank=True)
    new_status = models.CharField(max_length=20)
    
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Order Status History'
        verbose_name_plural = 'Order Status Histories'
    
    def __str__(self):
        return f"{self.order.order_number}: {self.old_status} → {self.new_status}"
