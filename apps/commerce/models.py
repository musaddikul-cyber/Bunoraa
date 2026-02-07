"""
Commerce models - Unified cart, checkout, and wishlist models
"""
import uuid
import hashlib
from decimal import Decimal
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


# =============================================================================
# Cart Models
# =============================================================================

class Cart(models.Model):
    """Shopping cart - persistent for both guests and users."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Can be linked to user or session
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='commerce_cart'
    )
    session_key = models.CharField(max_length=40, blank=True, null=True, db_index=True)
    
    # Coupon
    coupon = models.ForeignKey(
        'promotions.Coupon',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='commerce_carts'
    )
    
    # Currency
    currency = models.CharField(max_length=3, default='BDT')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user']),
            models.Index(fields=['session_key']),
            models.Index(fields=['updated_at']),
        ]
    
    def __str__(self):
        if self.user:
            return f"Cart for {self.user.email}"
        return f"Guest Cart {self.session_key}"
    
    @property
    def item_count(self):
        """Get total number of items in cart."""
        return sum(item.quantity for item in self.items.all())
    
    @property
    def subtotal(self):
        """Calculate cart subtotal (before discounts)."""
        return sum(item.total for item in self.items.all())
    
    @property
    def discount_amount(self):
        """Calculate discount from coupon."""
        if not self.coupon:
            return Decimal('0.00')
        
        try:
            return self.coupon.calculate_discount(self.subtotal)
        except Exception:
            return Decimal('0.00')
    
    @property
    def total(self):
        """Calculate cart total (after discounts)."""
        return max(Decimal('0.00'), self.subtotal - self.discount_amount)
    
    def clear(self):
        """Remove all items from cart."""
        self.items.all().delete()
        self.coupon = None
        self.save()
    
    def merge_from_session(self, session_cart):
        """Merge items from a session cart into this cart."""
        for item in session_cart.items.all():
            existing_item = self.items.filter(
                product=item.product,
                variant=item.variant
            ).first()
            
            if existing_item:
                existing_item.quantity += item.quantity
                existing_item.save()
            else:
                item.cart = self
                item.save()
        
        session_cart.delete()


class CartItem(models.Model):
    """Items in a shopping cart."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='commerce_cart_items'
    )
    variant = models.ForeignKey(
        'catalog.ProductVariant',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='commerce_cart_items'
    )
    quantity = models.PositiveIntegerField(default=1, validators=[MinValueValidator(1)])
    
    # Store price at time of adding (for price protection)
    price_at_add = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0.00'))]
    )
    
    # Price lock feature - lock price for limited time
    price_locked_until = models.DateTimeField(null=True, blank=True, help_text="Price lock expiry")
    locked_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Gift wrap option
    gift_wrap = models.BooleanField(default=False)
    gift_message = models.TextField(blank=True, max_length=500)
    gift_recipient_name = models.CharField(max_length=200, blank=True)
    gift_recipient_email = models.EmailField(blank=True)
    
    # Item-level notes
    customer_note = models.TextField(blank=True, max_length=500, help_text="Customer notes for this item")
    internal_note = models.TextField(blank=True, max_length=500, help_text="Internal notes")
    
    # Priority/ordering
    priority = models.PositiveSmallIntegerField(default=0, help_text="Item priority for display ordering")
    
    # Recurring purchase
    is_recurring = models.BooleanField(default=False)
    recurring_interval = models.CharField(
        max_length=20,
        choices=[
            ('weekly', 'Weekly'),
            ('biweekly', 'Every 2 Weeks'),
            ('monthly', 'Monthly'),
            ('bimonthly', 'Every 2 Months'),
            ('quarterly', 'Quarterly'),
        ],
        blank=True
    )
    
    # Source tracking
    added_from = models.CharField(
        max_length=50,
        choices=[
            ('product_page', 'Product Page'),
            ('category', 'Category Page'),
            ('search', 'Search Results'),
            ('wishlist', 'Wishlist'),
            ('recommendation', 'Recommendation'),
            ('quick_view', 'Quick View'),
            ('buy_again', 'Buy Again'),
            ('cart_drawer', 'Cart Drawer'),
            ('api', 'API'),
            ('other', 'Other'),
        ],
        default='product_page'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['cart', 'product', 'variant']
        ordering = ['priority', '-created_at']
    
    def __str__(self):
        name = self.product.name
        if self.variant:
            name += f" - {self.variant}"
        return f"{name} x {self.quantity}"
    
    @property
    def unit_price(self):
        """Get current unit price, respecting price lock if valid."""
        # Check if price is locked and still valid
        if self.price_locked_until and self.locked_price:
            if timezone.now() <= self.price_locked_until:
                return self.locked_price
        
        if self.variant and self.variant.price:
            return self.variant.price
        return self.product.current_price
    
    @property
    def original_unit_price(self):
        """Get original price without sale/discount."""
        if self.variant and self.variant.price:
            return self.variant.compare_at_price or self.variant.price
        return self.product.compare_at_price or self.product.price
    
    @property
    def total(self):
        """Calculate line item total."""
        return self.unit_price * self.quantity
    
    @property
    def savings(self):
        """Calculate savings if on sale."""
        original = self.original_unit_price * self.quantity
        current = self.total
        return max(Decimal('0'), original - current)
    
    @property
    def is_price_locked(self):
        """Check if price lock is active."""
        if self.price_locked_until and self.locked_price:
            return timezone.now() <= self.price_locked_until
        return False
    
    @property
    def price_lock_remaining(self):
        """Get remaining time on price lock in seconds."""
        if self.is_price_locked:
            remaining = self.price_locked_until - timezone.now()
            return max(0, int(remaining.total_seconds()))
        return 0
    
    def lock_price(self, duration_hours=24):
        """Lock current price for specified duration."""
        self.locked_price = self.unit_price
        self.price_locked_until = timezone.now() + timezone.timedelta(hours=duration_hours)
        self.save(update_fields=['locked_price', 'price_locked_until'])


class CartSettings(models.Model):
    """Global cart settings (singleton)."""
    gift_wrap_enabled = models.BooleanField(default=True)
    gift_wrap_amount = models.DecimalField(max_digits=10, decimal_places=2, default=50.00)
    gift_wrap_label = models.CharField(max_length=100, blank=True, default='Gift Wrap')
    
    # Cart expiry
    cart_expiry_days = models.PositiveIntegerField(default=30)
    
    # Price lock settings
    price_lock_enabled = models.BooleanField(default=True, help_text="Enable price lock feature")
    price_lock_duration_hours = models.PositiveIntegerField(default=24, help_text="Default price lock duration")
    price_lock_auto_apply = models.BooleanField(default=False, help_text="Auto-apply price lock on add")
    
    # Stock reservation
    stock_reservation_enabled = models.BooleanField(default=True)
    stock_reservation_minutes = models.PositiveIntegerField(default=30)
    
    # Minimum order
    minimum_order_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    minimum_order_message = models.CharField(max_length=255, blank=True, default='Minimum order amount not met')
    
    # Free shipping threshold (Bangladesh default)
    free_shipping_threshold = models.DecimalField(max_digits=10, decimal_places=2, default=3000.00)
    free_shipping_message = models.CharField(max_length=255, blank=True, default='Free delivery on orders over ৳2,000')
    
    # Cart item limits
    max_items_per_cart = models.PositiveIntegerField(default=50)
    max_quantity_per_item = models.PositiveIntegerField(default=10)
    
    # Abandoned cart settings
    abandoned_cart_threshold_hours = models.PositiveIntegerField(default=1)
    abandoned_cart_reminder_delay_hours = models.PositiveIntegerField(default=24)
    abandoned_cart_max_reminders = models.PositiveIntegerField(default=3)
    abandoned_cart_discount_enabled = models.BooleanField(default=True)
    abandoned_cart_discount_percent = models.PositiveIntegerField(default=10)
    
    # Currency (Bangladesh default)
    default_currency = models.CharField(max_length=3, default='BDT')
    
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Cart Settings'
        verbose_name_plural = 'Cart Settings'
    
    def __str__(self):
        return 'Cart Settings'
    
    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)
    
    def delete(self, *args, **kwargs):
        pass
    
    @classmethod
    def get_settings(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj


# =============================================================================
# Wishlist Models
# =============================================================================

class Wishlist(models.Model):
    """User's wishlist container."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='commerce_wishlist'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Wishlist for {self.user.email}"
    
    @property
    def item_count(self):
        return self.items.count()
    
    @property
    def total_value(self):
        """Calculate total value of wishlist items."""
        return sum(
            item.product.current_price
            for item in self.items.filter(product__is_active=True)
        )


class WishlistItem(models.Model):
    """Item in a wishlist."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    wishlist = models.ForeignKey(Wishlist, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='commerce_wishlist_items'
    )
    variant = models.ForeignKey(
        'catalog.ProductVariant',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='commerce_wishlist_items'
    )
    added_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)
    
    # Price tracking
    price_at_add = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    lowest_price_seen = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    highest_price_seen = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Priority for sorting
    PRIORITY_LOW = 1
    PRIORITY_NORMAL = 2
    PRIORITY_HIGH = 3
    PRIORITY_URGENT = 4
    PRIORITY_CHOICES = [
        (PRIORITY_LOW, 'Low'),
        (PRIORITY_NORMAL, 'Normal'),
        (PRIORITY_HIGH, 'High'),
        (PRIORITY_URGENT, 'Must Have'),
    ]
    priority = models.PositiveSmallIntegerField(choices=PRIORITY_CHOICES, default=PRIORITY_NORMAL)
    
    # Reminder feature
    reminder_date = models.DateField(null=True, blank=True, help_text="When to remind about this item")
    reminder_sent = models.BooleanField(default=False)
    reminder_message = models.CharField(max_length=255, blank=True)
    
    # Desired quantity for when moving to cart
    desired_quantity = models.PositiveIntegerField(default=1)
    
    # Notification preferences
    notify_on_sale = models.BooleanField(default=True)
    notify_on_restock = models.BooleanField(default=True)
    notify_on_price_drop = models.BooleanField(default=True)
    target_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Notify when price drops to this")
    
    # Tracking
    last_notified_at = models.DateTimeField(null=True, blank=True)
    view_count = models.PositiveIntegerField(default=0, help_text="Times product viewed from wishlist")
    last_viewed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-priority', '-added_at']
        unique_together = ['wishlist', 'product', 'variant']
    
    def __str__(self):
        return f"{self.product.name} in {self.wishlist}"
    
    def save(self, *args, **kwargs):
        if not self.price_at_add and self.product:
            self.price_at_add = self.product.current_price
        
        # Track price history
        current_price = self.product.current_price if self.product else None
        if current_price:
            if self.lowest_price_seen is None or current_price < self.lowest_price_seen:
                self.lowest_price_seen = current_price
            if self.highest_price_seen is None or current_price > self.highest_price_seen:
                self.highest_price_seen = current_price
        
        super().save(*args, **kwargs)
    
    @property
    def current_price(self):
        if self.variant and self.variant.price:
            return self.variant.price
        return self.product.current_price
    
    @property
    def price_change(self):
        if self.price_at_add:
            return self.current_price - self.price_at_add
        return Decimal('0')
    
    @property
    def price_change_percent(self):
        if self.price_at_add and self.price_at_add > 0:
            change = self.price_change
            return (change / self.price_at_add) * 100
        return Decimal('0')
    
    @property
    def is_on_sale(self):
        return self.product.is_on_sale
    
    @property
    def is_at_target_price(self):
        """Check if current price is at or below target price."""
        if self.target_price:
            return self.current_price <= self.target_price
        return False
    
    @property
    def savings_from_highest(self):
        """Calculate savings from highest price seen."""
        if self.highest_price_seen:
            return max(Decimal('0'), self.highest_price_seen - self.current_price)
        return Decimal('0')
    
    def record_view(self):
        """Record that user viewed product from wishlist."""
        self.view_count += 1
        self.last_viewed_at = timezone.now()
        self.save(update_fields=['view_count', 'last_viewed_at'])
    
    def set_reminder(self, date, message=''):
        """Set reminder for this item."""
        self.reminder_date = date
        self.reminder_message = message
        self.reminder_sent = False
        self.save(update_fields=['reminder_date', 'reminder_message', 'reminder_sent'])


class WishlistShare(models.Model):
    """
    Share wishlists with friends and family.
    Allow comments and suggestions on shared wishlists.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    wishlist = models.OneToOneField(
        'commerce.Wishlist',
        on_delete=models.CASCADE,
        related_name='share'
    )
    
    # Share token for public/private access
    share_token = models.CharField(
        max_length=50,
        unique=True,
        editable=False
    )
    
    # Settings
    is_public = models.BooleanField(
        default=False,
        help_text='Anyone with link can view'
    )
    allow_comments = models.BooleanField(default=True)
    allow_suggestions = models.BooleanField(
        default=True,
        help_text='Friends can suggest items to add'
    )
    
    # Tracking
    view_count = models.PositiveIntegerField(default=0)
    shared_with = models.ManyToManyField(
        'accounts.User',
        blank=True,
        related_name='shared_wishlists'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'wishlist share'
        verbose_name_plural = 'wishlist shares'
    
    def __str__(self):
        return f"Share of {self.wishlist.name}"
    
    def save(self, *args, **kwargs):
        if not self.share_token:
            import secrets
            self.share_token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)
    
    def get_share_url(self):
        from django.urls import reverse
        return reverse('wishlist:shared_view', kwargs={'token': self.share_token})


class WishlistComment(models.Model):
    """
    Comments on shared wishlists.
    Friends can discuss items and make suggestions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    wishlist_share = models.ForeignKey(
        WishlistShare,
        on_delete=models.CASCADE,
        related_name='comments'
    )
    
    author = models.ForeignKey(
        'accounts.User',
        on_delete=models.CASCADE,
        related_name='wishlist_comments'
    )
    
    # Comment content
    content = models.TextField()
    
    # Reply to another comment (threaded)
    reply_to = models.ForeignKey(
        'self',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='replies'
    )
    
    # Moderation
    is_approved = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'wishlist comment'
        verbose_name_plural = 'wishlist comments'
        ordering = ['created_at']
    
    def __str__(self):
        return f"Comment by {self.author.email} on {self.wishlist_share}"


class WishlistSuggestion(models.Model):
    """
    Suggested items to add to wishlist from friends.
    Owner can approve/reject suggestions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    wishlist_share = models.ForeignKey(
        WishlistShare,
        on_delete=models.CASCADE,
        related_name='suggestions'
    )
    
    suggested_by = models.ForeignKey(
        'accounts.User',
        on_delete=models.CASCADE,
        related_name='wishlist_suggestions'
    )
    
    # Suggested product
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='wishlist_suggestions'
    )
    
    # Reason for suggestion
    reason = models.TextField(blank=True)
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_ADDED = 'added'
    STATUS_REJECTED = 'rejected'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_ADDED, 'Added to Wishlist'),
        (STATUS_REJECTED, 'Rejected'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = 'wishlist suggestion'
        verbose_name_plural = 'wishlist suggestions'
        ordering = ['-created_at']
        unique_together = ['wishlist_share', 'suggested_by', 'product']
    
    def __str__(self):
        return f"{self.suggested_by.email} suggested {self.product.name}"

        if not self.share_token:
            import secrets
            self.share_token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)
    
    @property
    def is_valid(self):
        if not self.is_active:
            return False
        if self.expires_at and timezone.now() > self.expires_at:
            return False
        return True


# =============================================================================
# Checkout Models
# =============================================================================

class CheckoutSession(models.Model):
    """Checkout session holding state before order creation."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Security token for guest access
    security_token = models.CharField(max_length=64, blank=True, db_index=True)
    
    # User or session
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='commerce_checkout_sessions',
        null=True,
        blank=True
    )
    session_key = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    
    # Cart reference
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='checkout_sessions')
    
    # Email for guest checkout
    email = models.EmailField(blank=True)
    
    # Shipping address
    shipping_first_name = models.CharField(max_length=100, blank=True)
    shipping_last_name = models.CharField(max_length=100, blank=True)
    shipping_company = models.CharField(max_length=200, blank=True)
    shipping_email = models.EmailField(blank=True)
    shipping_phone = models.CharField(max_length=20, blank=True)
    shipping_address_line_1 = models.CharField(max_length=255, blank=True)
    shipping_address_line_2 = models.CharField(max_length=255, blank=True)
    shipping_city = models.CharField(max_length=100, blank=True)
    shipping_state = models.CharField(max_length=100, blank=True)
    shipping_postal_code = models.CharField(max_length=20, blank=True)
    shipping_country = models.CharField(max_length=100, default='Bangladesh')
    
    # Saved address reference
    saved_shipping_address = models.ForeignKey(
        'accounts.Address',
        on_delete=models.SET_NULL,
        related_name='commerce_checkout_shipping_uses',
        null=True,
        blank=True
    )
    
    # Billing address (if different)
    billing_same_as_shipping = models.BooleanField(default=True)
    billing_first_name = models.CharField(max_length=100, blank=True)
    billing_last_name = models.CharField(max_length=100, blank=True)
    billing_company = models.CharField(max_length=200, blank=True)
    billing_address_line_1 = models.CharField(max_length=255, blank=True)
    billing_address_line_2 = models.CharField(max_length=255, blank=True)
    billing_city = models.CharField(max_length=100, blank=True)
    billing_state = models.CharField(max_length=100, blank=True)
    billing_postal_code = models.CharField(max_length=20, blank=True)
    billing_country = models.CharField(max_length=100, default='Bangladesh')
    
    saved_billing_address = models.ForeignKey(
        'accounts.Address',
        on_delete=models.SET_NULL,
        related_name='commerce_checkout_billing_uses',
        null=True,
        blank=True
    )
    
    # Shipping method
    SHIPPING_STANDARD = 'standard'
    SHIPPING_EXPRESS = 'express'
    SHIPPING_OVERNIGHT = 'overnight'
    SHIPPING_PICKUP = 'pickup'
    SHIPPING_FREE = 'free'
    SHIPPING_CHOICES = [
        (SHIPPING_STANDARD, 'Standard Shipping (5-7 days)'),
        (SHIPPING_EXPRESS, 'Express Shipping (2-3 days)'),
        (SHIPPING_OVERNIGHT, 'Overnight Shipping'),
        (SHIPPING_PICKUP, 'Store Pickup'),
        (SHIPPING_FREE, 'Free Shipping'),
    ]
    shipping_method = models.CharField(max_length=20, choices=SHIPPING_CHOICES, default=SHIPPING_STANDARD)
    shipping_rate = models.ForeignKey(
        'shipping.ShippingRate',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='commerce_checkout_sessions'
    )
    shipping_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Store pickup location
    pickup_location = models.ForeignKey(
        'contacts.StoreLocation',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='commerce_checkout_pickups'
    )
    
    # Payment methods
    PAYMENT_STRIPE = 'stripe'
    PAYMENT_PAYPAL = 'paypal'
    PAYMENT_COD = 'cod'
    PAYMENT_BKASH = 'bkash'
    PAYMENT_NAGAD = 'nagad'
    PAYMENT_BANK = 'bank_transfer'
    PAYMENT_CHOICES = [
        (PAYMENT_STRIPE, 'Credit/Debit Card (Stripe)'),
        (PAYMENT_PAYPAL, 'PayPal'),
        (PAYMENT_COD, 'Cash on Delivery'),
        (PAYMENT_BKASH, 'bKash'),
        (PAYMENT_NAGAD, 'Nagad'),
        (PAYMENT_BANK, 'Bank Transfer'),
    ]
    payment_method = models.CharField(max_length=20, choices=PAYMENT_CHOICES, default=PAYMENT_COD)

    # Payment fee snapshot
    payment_fee_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    payment_fee_label = models.CharField(max_length=100, blank=True)
    
    # Payment gateway references
    stripe_payment_intent_id = models.CharField(max_length=255, blank=True)
    stripe_client_secret = models.CharField(max_length=255, blank=True)
    paypal_order_id = models.CharField(max_length=255, blank=True)
    bkash_payment_id = models.CharField(max_length=255, blank=True)
    nagad_payment_id = models.CharField(max_length=255, blank=True)
    
    # Coupon/Discount
    coupon = models.ForeignKey(
        'promotions.Coupon',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='commerce_checkout_sessions'
    )
    coupon_code = models.CharField(max_length=50, blank=True)
    discount_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Gift options
    is_gift = models.BooleanField(default=False)
    gift_message = models.TextField(blank=True, max_length=500)
    gift_wrap = models.BooleanField(default=False)
    gift_wrap_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Order notes
    order_notes = models.TextField(blank=True)
    delivery_instructions = models.TextField(blank=True)
    
    # Tax calculation
    tax_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    tax_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    tax_included = models.BooleanField(default=False)
    
    # Pricing snapshot
    subtotal = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Currency
    currency = models.CharField(max_length=3, default='BDT')
    exchange_rate = models.DecimalField(max_digits=10, decimal_places=6, default=1)
    
    # Status
    STEP_CART = 'cart'
    STEP_INFORMATION = 'information'
    STEP_SHIPPING = 'shipping'
    STEP_PAYMENT = 'payment'
    STEP_REVIEW = 'review'
    STEP_PROCESSING = 'processing'
    STEP_COMPLETED = 'completed'
    STEP_ABANDONED = 'abandoned'
    STEP_CHOICES = [
        (STEP_CART, 'Cart'),
        (STEP_INFORMATION, 'Information'),
        (STEP_SHIPPING, 'Shipping'),
        (STEP_PAYMENT, 'Payment'),
        (STEP_REVIEW, 'Review'),
        (STEP_PROCESSING, 'Processing'),
        (STEP_COMPLETED, 'Completed'),
        (STEP_ABANDONED, 'Abandoned'),
    ]
    current_step = models.CharField(max_length=20, choices=STEP_CHOICES, default=STEP_CART)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Analytics
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    referrer = models.URLField(blank=True)
    utm_source = models.CharField(max_length=100, blank=True)
    utm_medium = models.CharField(max_length=100, blank=True)
    utm_campaign = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'current_step']),
            models.Index(fields=['session_key']),
            models.Index(fields=['expires_at']),
            models.Index(fields=['current_step', 'created_at']),
        ]
    
    def __str__(self):
        return f"Checkout {self.id} - {self.current_step}"
    
    def save(self, *args, **kwargs):
        if not self.security_token:
            self.security_token = hashlib.sha256(
                f"{self.id}{timezone.now().isoformat()}".encode()
            ).hexdigest()[:64]
        super().save(*args, **kwargs)
    
    @property
    def is_expired(self):
        if self.expires_at:
            return timezone.now() > self.expires_at
        return False
    
    def extend_expiry(self, hours=48):
        self.expires_at = timezone.now() + timezone.timedelta(hours=hours)
        self.save(update_fields=['expires_at'])
    
    def calculate_totals(self):
        """Recalculate all totals."""
        self.subtotal = self.cart.subtotal
        self.discount_amount = self.cart.discount_amount
        self.total = (
            self.subtotal 
            - self.discount_amount 
            + self.shipping_cost 
            + self.tax_amount 
            + self.gift_wrap_cost
        )
        self.save(update_fields=['subtotal', 'discount_amount', 'total'])

    def get_shipping_address_dict(self):
        """Return shipping address data as a dict."""
        return {
            'first_name': self.shipping_first_name or '',
            'last_name': self.shipping_last_name or '',
            'address_line_1': self.shipping_address_line_1 or '',
            'address_line_2': self.shipping_address_line_2 or '',
            'city': self.shipping_city or '',
            'state': self.shipping_state or '',
            'postal_code': self.shipping_postal_code or '',
            'country': self.shipping_country or '',
        }

    def get_billing_address_dict(self):
        """Return billing address data as a dict (falls back to shipping if needed)."""
        if self.billing_same_as_shipping:
            return self.get_shipping_address_dict()

        data = {
            'first_name': self.billing_first_name or '',
            'last_name': self.billing_last_name or '',
            'address_line_1': self.billing_address_line_1 or '',
            'address_line_2': self.billing_address_line_2 or '',
            'city': self.billing_city or '',
            'state': self.billing_state or '',
            'postal_code': self.billing_postal_code or '',
            'country': self.billing_country or '',
        }

        if not data['first_name'] or not data['address_line_1'] or not data['city'] or not data['postal_code'] or not data['country']:
            return self.get_shipping_address_dict()

        return data


class CheckoutEvent(models.Model):
    """Track checkout events for analytics."""
    
    EVENT_STARTED = 'started'
    EVENT_INFO_SUBMITTED = 'info_submitted'
    EVENT_SHIPPING_SELECTED = 'shipping_selected'
    EVENT_PAYMENT_INITIATED = 'payment_initiated'
    EVENT_PAYMENT_COMPLETED = 'payment_completed'
    EVENT_PAYMENT_FAILED = 'payment_failed'
    EVENT_ORDER_CREATED = 'order_created'
    EVENT_ABANDONED = 'abandoned'
    EVENT_CHOICES = [
        (EVENT_STARTED, 'Checkout Started'),
        (EVENT_INFO_SUBMITTED, 'Information Submitted'),
        (EVENT_SHIPPING_SELECTED, 'Shipping Selected'),
        (EVENT_PAYMENT_INITIATED, 'Payment Initiated'),
        (EVENT_PAYMENT_COMPLETED, 'Payment Completed'),
        (EVENT_PAYMENT_FAILED, 'Payment Failed'),
        (EVENT_ORDER_CREATED, 'Order Created'),
        (EVENT_ABANDONED, 'Checkout Abandoned'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    checkout_session = models.ForeignKey(
        CheckoutSession,
        on_delete=models.CASCADE,
        related_name='events'
    )
    event_type = models.CharField(max_length=30, choices=EVENT_CHOICES)
    data = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.event_type} - {self.checkout_session_id}"


# =============================================================================
# Saved For Later (Items moved out of cart temporarily)
# =============================================================================

class SavedForLater(models.Model):
    """Items saved for later from cart."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Can be user-based or session-based
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='saved_for_later_items'
    )
    session_key = models.CharField(max_length=40, blank=True, null=True, db_index=True)
    
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='saved_for_later_items'
    )
    variant = models.ForeignKey(
        'catalog.ProductVariant',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    quantity = models.PositiveIntegerField(default=1)
    
    # Price tracking
    price_at_save = models.DecimalField(max_digits=12, decimal_places=2)
    notify_on_price_drop = models.BooleanField(default=True)
    notify_on_restock = models.BooleanField(default=True)
    
    # Timestamps
    saved_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-saved_at']
        indexes = [
            models.Index(fields=['user', 'saved_at']),
            models.Index(fields=['session_key']),
        ]
    
    def __str__(self):
        return f"Saved: {self.product.name}"
    
    @property
    def current_price(self):
        if self.variant and self.variant.price:
            return self.variant.price
        return self.product.current_price
    
    @property
    def price_change(self):
        return self.current_price - self.price_at_save
    
    @property
    def is_expired(self):
        if self.expires_at and timezone.now() > self.expires_at:
            return True
        return False


# =============================================================================
# Cart Sharing
# =============================================================================

class SharedCart(models.Model):
    """Shareable cart for collaboration or gift registries."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='shares')
    
    # Share configuration
    share_token = models.CharField(max_length=64, unique=True, db_index=True)
    name = models.CharField(max_length=200, blank=True, help_text="Name for this shared cart")
    description = models.TextField(blank=True)
    
    # Permissions
    PERMISSION_VIEW = 'view'
    PERMISSION_EDIT = 'edit'
    PERMISSION_PURCHASE = 'purchase'
    PERMISSION_CHOICES = [
        (PERMISSION_VIEW, 'View only'),
        (PERMISSION_EDIT, 'Can edit items'),
        (PERMISSION_PURCHASE, 'Can purchase items'),
    ]
    permission = models.CharField(max_length=20, choices=PERMISSION_CHOICES, default=PERMISSION_VIEW)
    
    # Access control
    password_hash = models.CharField(max_length=128, blank=True, help_text="Optional password protection")
    allowed_emails = models.JSONField(default=list, blank=True, help_text="List of allowed email addresses")
    max_uses = models.PositiveIntegerField(null=True, blank=True, help_text="Maximum number of uses")
    
    # Status
    is_active = models.BooleanField(default=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    # Analytics
    view_count = models.PositiveIntegerField(default=0)
    last_viewed_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_shared_carts'
    )
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Shared Cart: {self.name or self.share_token[:8]}"
    
    def save(self, *args, **kwargs):
        if not self.share_token:
            import secrets
            self.share_token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)
    
    @property
    def is_valid(self):
        if not self.is_active:
            return False
        if self.expires_at and timezone.now() > self.expires_at:
            return False
        if self.max_uses and self.view_count >= self.max_uses:
            return False
        return True
    
    def record_view(self):
        self.view_count += 1
        self.last_viewed_at = timezone.now()
        self.save(update_fields=['view_count', 'last_viewed_at'])


# =============================================================================
# Session-Based Wishlist (for guests)
# =============================================================================

class SessionWishlist(models.Model):
    """Wishlist for non-authenticated users (guests)."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_key = models.CharField(max_length=40, unique=True, db_index=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Session Wishlist: {self.session_key}"
    
    @property
    def item_count(self):
        return self.items.count()
    
    def merge_into_user_wishlist(self, user_wishlist):
        """Merge session wishlist into user's wishlist."""
        for item in self.items.all():
            WishlistItem.objects.get_or_create(
                wishlist=user_wishlist,
                product=item.product,
                variant=item.variant,
                defaults={
                    'notes': item.notes,
                    'price_at_add': item.price_at_add,
                    'notify_on_sale': item.notify_on_sale,
                    'notify_on_restock': item.notify_on_restock,
                    'notify_on_price_drop': item.notify_on_price_drop,
                }
            )
        self.delete()


class SessionWishlistItem(models.Model):
    """Item in a session wishlist."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    wishlist = models.ForeignKey(SessionWishlist, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey('catalog.Product', on_delete=models.CASCADE)
    variant = models.ForeignKey('catalog.ProductVariant', on_delete=models.SET_NULL, null=True, blank=True)
    
    # Metadata
    notes = models.TextField(blank=True)
    price_at_add = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    lowest_price_seen = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Priority
    priority = models.PositiveSmallIntegerField(default=2)  # Same as PRIORITY_NORMAL
    
    # Desired quantity
    desired_quantity = models.PositiveIntegerField(default=1)
    
    # Notification preferences (stored for when user signs up)
    notify_on_sale = models.BooleanField(default=True)
    notify_on_restock = models.BooleanField(default=True)
    notify_on_price_drop = models.BooleanField(default=True)
    target_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    added_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-priority', '-added_at']
        unique_together = ['wishlist', 'product', 'variant']
    
    def __str__(self):
        return f"{self.product.name} in Session Wishlist"
    
    def save(self, *args, **kwargs):
        if not self.price_at_add and self.product:
            self.price_at_add = self.product.current_price
        
        # Track lowest price
        current_price = self.product.current_price if self.product else None
        if current_price:
            if self.lowest_price_seen is None or current_price < self.lowest_price_seen:
                self.lowest_price_seen = current_price
        
        super().save(*args, **kwargs)
    
    @property
    def current_price(self):
        if self.variant and self.variant.price:
            return self.variant.price
        return self.product.current_price
    
    @property
    def price_change(self):
        if self.price_at_add:
            return self.current_price - self.price_at_add
        return Decimal('0')


# =============================================================================
# Wishlist Collections
# =============================================================================

class WishlistCollection(models.Model):
    """Collection/folder for organizing wishlist items."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    wishlist = models.ForeignKey(Wishlist, on_delete=models.CASCADE, related_name='collections')
    
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    emoji = models.CharField(max_length=10, blank=True, help_text="Emoji icon for collection")
    color = models.CharField(max_length=7, blank=True, help_text="Hex color code")
    
    # Privacy
    is_public = models.BooleanField(default=False)
    
    # Ordering
    sort_order = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        unique_together = ['wishlist', 'name']
    
    def __str__(self):
        return f"{self.emoji} {self.name}" if self.emoji else self.name
    
    @property
    def item_count(self):
        return self.items.count()


class WishlistCollectionItem(models.Model):
    """Link between wishlist items and collections."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    collection = models.ForeignKey(WishlistCollection, on_delete=models.CASCADE, related_name='items')
    wishlist_item = models.ForeignKey(WishlistItem, on_delete=models.CASCADE, related_name='collection_memberships')
    
    added_at = models.DateTimeField(auto_now_add=True)
    sort_order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['sort_order', '-added_at']
        unique_together = ['collection', 'wishlist_item']
    
    def __str__(self):
        return f"{self.wishlist_item.product.name} in {self.collection.name}"


# =============================================================================
# Price Drop Alerts
# =============================================================================

class PriceAlert(models.Model):
    """Price drop alert for products."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # User or email for guests
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='price_alerts'
    )
    email = models.EmailField(blank=True, help_text="For guest alerts")
    
    product = models.ForeignKey('catalog.Product', on_delete=models.CASCADE, related_name='price_alerts')
    variant = models.ForeignKey('catalog.ProductVariant', on_delete=models.SET_NULL, null=True, blank=True)
    
    # Alert configuration
    ALERT_ANY_DROP = 'any'
    ALERT_THRESHOLD = 'threshold'
    ALERT_PERCENTAGE = 'percentage'
    ALERT_TYPE_CHOICES = [
        (ALERT_ANY_DROP, 'Any price drop'),
        (ALERT_THRESHOLD, 'When price drops below threshold'),
        (ALERT_PERCENTAGE, 'When price drops by percentage'),
    ]
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPE_CHOICES, default=ALERT_ANY_DROP)
    
    # Thresholds
    target_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    percentage_drop = models.PositiveIntegerField(null=True, blank=True, help_text="Minimum percentage drop to alert")
    
    # Tracking
    price_at_creation = models.DecimalField(max_digits=12, decimal_places=2)
    lowest_price_seen = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_triggered = models.BooleanField(default=False)
    triggered_at = models.DateTimeField(null=True, blank=True)
    triggered_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Notification tracking
    last_notified_at = models.DateTimeField(null=True, blank=True)
    notification_count = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['product', 'is_active']),
            models.Index(fields=['user', 'is_active']),
        ]
    
    def __str__(self):
        return f"Price Alert for {self.product.name}"
    
    def check_and_trigger(self, new_price):
        """Check if alert should be triggered."""
        if not self.is_active:
            return False
        
        if self.expires_at and timezone.now() > self.expires_at:
            self.is_active = False
            self.save(update_fields=['is_active'])
            return False
        
        # Update lowest price
        if self.lowest_price_seen is None or new_price < self.lowest_price_seen:
            self.lowest_price_seen = new_price
            self.save(update_fields=['lowest_price_seen'])
        
        should_trigger = False
        
        if self.alert_type == self.ALERT_ANY_DROP:
            should_trigger = new_price < self.price_at_creation
        
        elif self.alert_type == self.ALERT_THRESHOLD and self.target_price:
            should_trigger = new_price <= self.target_price
        
        elif self.alert_type == self.ALERT_PERCENTAGE and self.percentage_drop:
            drop_percentage = ((self.price_at_creation - new_price) / self.price_at_creation) * 100
            should_trigger = drop_percentage >= self.percentage_drop
        
        if should_trigger and not self.is_triggered:
            self.is_triggered = True
            self.triggered_at = timezone.now()
            self.triggered_price = new_price
            self.save(update_fields=['is_triggered', 'triggered_at', 'triggered_price'])
            return True
        
        return False


# =============================================================================
# Abandoned Cart Tracking
# =============================================================================

class AbandonedCart(models.Model):
    """Track and recover abandoned carts."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cart = models.OneToOneField(Cart, on_delete=models.CASCADE, related_name='abandoned_tracking')
    
    # User info (captured even if not logged in)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    email = models.EmailField(blank=True)
    
    # Cart snapshot
    cart_value = models.DecimalField(max_digits=12, decimal_places=2)
    item_count = models.PositiveIntegerField()
    items_snapshot = models.JSONField(default=list)
    
    # Recovery
    recovery_token = models.CharField(max_length=64, unique=True, db_index=True)
    recovery_url = models.URLField(blank=True)
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_REMINDED = 'reminded'
    STATUS_RECOVERED = 'recovered'
    STATUS_EXPIRED = 'expired'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_REMINDED, 'Reminder Sent'),
        (STATUS_RECOVERED, 'Recovered'),
        (STATUS_EXPIRED, 'Expired'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    
    # Reminder tracking
    reminder_count = models.PositiveIntegerField(default=0)
    last_reminder_at = models.DateTimeField(null=True, blank=True)
    next_reminder_at = models.DateTimeField(null=True, blank=True)
    
    # Recovery incentive
    recovery_discount_code = models.CharField(max_length=50, blank=True)
    recovery_discount_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Analytics
    abandoned_at = models.DateTimeField(auto_now_add=True)
    recovered_at = models.DateTimeField(null=True, blank=True)
    last_activity = models.DateTimeField(auto_now=True)
    
    # Source tracking
    referrer = models.URLField(blank=True)
    utm_source = models.CharField(max_length=100, blank=True)
    utm_medium = models.CharField(max_length=100, blank=True)
    utm_campaign = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['-abandoned_at']
        indexes = [
            models.Index(fields=['status', 'next_reminder_at']),
            models.Index(fields=['email', 'status']),
        ]
    
    def __str__(self):
        return f"Abandoned Cart {self.recovery_token[:8]}"
    
    def save(self, *args, **kwargs):
        if not self.recovery_token:
            import secrets
            self.recovery_token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)
    
    def mark_recovered(self, order=None):
        """Mark cart as recovered."""
        self.status = self.STATUS_RECOVERED
        self.recovered_at = timezone.now()
        self.save(update_fields=['status', 'recovered_at'])


# =============================================================================
# Item Reservation
# =============================================================================

class ItemReservation(models.Model):
    """Temporary stock reservation for cart items."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='reservations')
    cart_item = models.OneToOneField(CartItem, on_delete=models.CASCADE, related_name='reservation')
    
    product = models.ForeignKey('catalog.Product', on_delete=models.CASCADE)
    variant = models.ForeignKey('catalog.ProductVariant', on_delete=models.SET_NULL, null=True, blank=True)
    
    quantity = models.PositiveIntegerField()
    
    # Timing
    reserved_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    
    # Status
    is_active = models.BooleanField(default=True)
    released_at = models.DateTimeField(null=True, blank=True)
    converted_to_order = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-reserved_at']
        indexes = [
            models.Index(fields=['product', 'is_active', 'expires_at']),
            models.Index(fields=['expires_at', 'is_active']),
        ]
    
    def __str__(self):
        return f"Reservation: {self.quantity}x {self.product.name}"
    
    @property
    def is_expired(self):
        return timezone.now() > self.expires_at
    
    def release(self):
        """Release the reservation."""
        if self.is_active:
            self.is_active = False
            self.released_at = timezone.now()
            self.save(update_fields=['is_active', 'released_at'])
            return True
        return False
    
    def extend(self, minutes=30):
        """Extend reservation time."""
        from datetime import timedelta
        self.expires_at = timezone.now() + timedelta(minutes=minutes)
        self.save(update_fields=['expires_at'])


# =============================================================================
# Cart Analytics
# =============================================================================

class CartAnalytics(models.Model):
    """Analytics for cart behavior."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cart = models.OneToOneField(Cart, on_delete=models.CASCADE, related_name='analytics')
    
    # Engagement metrics
    page_views = models.PositiveIntegerField(default=0)
    time_on_cart = models.PositiveIntegerField(default=0, help_text="Seconds spent on cart page")
    checkout_attempts = models.PositiveIntegerField(default=0)
    
    # Item interaction
    items_added = models.PositiveIntegerField(default=0)
    items_removed = models.PositiveIntegerField(default=0)
    items_saved_for_later = models.PositiveIntegerField(default=0)
    quantity_changes = models.PositiveIntegerField(default=0)
    
    # Coupon behavior
    coupons_tried = models.PositiveIntegerField(default=0)
    coupons_applied = models.PositiveIntegerField(default=0)
    coupons_failed = models.PositiveIntegerField(default=0)
    
    # Source tracking
    first_source = models.CharField(max_length=100, blank=True)
    last_source = models.CharField(max_length=100, blank=True)
    
    # Timestamps
    first_item_added = models.DateTimeField(null=True, blank=True)
    last_interaction = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = 'Cart Analytics'
    
    def __str__(self):
        return f"Analytics for Cart {self.cart_id}"
    
    def record_event(self, event_type):
        """Record a cart event."""
        events = {
            'view': 'page_views',
            'add': 'items_added',
            'remove': 'items_removed',
            'save_later': 'items_saved_for_later',
            'quantity': 'quantity_changes',
            'checkout': 'checkout_attempts',
            'coupon_try': 'coupons_tried',
            'coupon_apply': 'coupons_applied',
            'coupon_fail': 'coupons_failed',
        }
        
        if event_type in events:
            field = events[event_type]
            setattr(self, field, getattr(self, field) + 1)
            self.save(update_fields=[field, 'last_interaction'])
