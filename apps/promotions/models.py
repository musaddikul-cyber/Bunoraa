"""
Promotions models
"""
import uuid
from decimal import Decimal
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.core.validators import MinValueValidator, MaxValueValidator


class Bundle(models.Model):
    """
    Product bundle - Group related products with bundle pricing.
    Perfect for embroidery sets, gift packages, etc.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Basic info
    name = models.CharField(_('name'), max_length=255)
    slug = models.SlugField(_('slug'), max_length=255, unique=True)
    description = models.TextField(_('description'), blank=True)
    image = models.ImageField(_('image'), upload_to='bundles/', blank=True, null=True)
    
    # Products in bundle
    products = models.ManyToManyField(
        'catalog.Product',
        through='BundleItem',
        related_name='bundles'
    )
    
    # Pricing
    regular_price = models.DecimalField(
        _('regular price'),
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0.00'))]
    )
    bundle_price = models.DecimalField(
        _('bundle price'),
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0.00'))]
    )
    
    @property
    def savings(self):
        """Calculate savings compared to buying individually."""
        return self.regular_price - self.bundle_price
    
    @property
    def discount_percentage(self):
        """Calculate discount percentage."""
        if self.regular_price == 0:
            return Decimal('0')
        return round((self.savings / self.regular_price) * 100, 2)
    
    # Display
    is_featured = models.BooleanField(_('featured'), default=False, db_index=True)
    is_active = models.BooleanField(_('active'), default=True, db_index=True)
    
    # SEO
    meta_title = models.CharField(_('meta title'), max_length=255, blank=True)
    meta_description = models.CharField(_('meta description'), max_length=500, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('bundle')
        verbose_name_plural = _('bundles')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_active', '-created_at']),
            models.Index(fields=['is_featured']),
        ]
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('bundles:detail', kwargs={'slug': self.slug})


class BundleItem(models.Model):
    """
    Item in a bundle - Links products to bundles with ordering.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    bundle = models.ForeignKey(
        Bundle,
        on_delete=models.CASCADE,
        related_name='items'
    )
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='bundle_items'
    )
    
    # Quantity in bundle
    quantity = models.PositiveIntegerField(default=1)
    
    # Display order
    display_order = models.PositiveIntegerField(default=0)
    
    # Optional description
    description = models.CharField(max_length=255, blank=True)
    
    class Meta:
        verbose_name = _('bundle item')
        verbose_name_plural = _('bundle items')
        ordering = ['display_order']
        unique_together = ['bundle', 'product']
    
    def __str__(self):
        return f"{self.bundle.name} - {self.product.name} (x{self.quantity})"


class Coupon(models.Model):
    """
    Coupon/discount code model.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    code = models.CharField(max_length=50, unique=True, db_index=True)
    description = models.TextField(blank=True)
    
    # Discount type
    DISCOUNT_PERCENTAGE = 'percentage'
    DISCOUNT_FIXED = 'fixed'
    DISCOUNT_CHOICES = [
        (DISCOUNT_PERCENTAGE, 'Percentage'),
        (DISCOUNT_FIXED, 'Fixed Amount'),
    ]
    discount_type = models.CharField(
        max_length=20,
        choices=DISCOUNT_CHOICES,
        default=DISCOUNT_PERCENTAGE
    )
    discount_value = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Limits
    minimum_order_amount = models.DecimalField(
        max_digits=10, decimal_places=2,
        null=True, blank=True,
        help_text="Minimum order amount required"
    )
    maximum_discount = models.DecimalField(
        max_digits=10, decimal_places=2,
        null=True, blank=True,
        help_text="Maximum discount amount (for percentage coupons)"
    )
    
    # Usage limits
    usage_limit = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Total number of times this coupon can be used"
    )
    usage_limit_per_user = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Number of times each user can use this coupon"
    )
    times_used = models.PositiveIntegerField(default=0)
    
    # Validity period
    valid_from = models.DateTimeField(null=True, blank=True)
    valid_until = models.DateTimeField(null=True, blank=True)
    
    # Restrictions
    categories = models.ManyToManyField(
        'catalog.Category',
        blank=True,
        related_name='coupons',
        help_text="If set, coupon only applies to these categories"
    )
    products = models.ManyToManyField(
        'catalog.Product',
        blank=True,
        related_name='coupons',
        help_text="If set, coupon only applies to these products"
    )
    
    # For specific users
    users = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        blank=True,
        related_name='available_coupons',
        help_text="If set, coupon only available to these users"
    )
    
    # First order only
    first_order_only = models.BooleanField(default=False)
    
    # Status
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Coupon'
        verbose_name_plural = 'Coupons'
    
    def __str__(self):
        return self.code
    
    @property
    def is_valid(self):
        """Check if coupon is currently valid."""
        if not self.is_active:
            return False
        
        now = timezone.now()
        
        if self.valid_from and now < self.valid_from:
            return False
        
        if self.valid_until and now > self.valid_until:
            return False
        
        if self.usage_limit and self.times_used >= self.usage_limit:
            return False
        
        return True
    
    def calculate_discount(self, subtotal):
        """Calculate discount amount for given subtotal."""
        if self.discount_type == self.DISCOUNT_PERCENTAGE:
            discount = subtotal * (self.discount_value / Decimal('100'))
            if self.maximum_discount:
                discount = min(discount, self.maximum_discount)
        else:
            discount = self.discount_value
        
        return min(discount, subtotal)  # Don't exceed subtotal
    
    def can_use(self, user=None, subtotal=Decimal('0')):
        """Check if coupon can be used by user for given subtotal."""
        if not self.is_valid:
            return False, "Coupon is not valid"
        
        # Check minimum order
        if self.minimum_order_amount and subtotal < self.minimum_order_amount:
            return False, f"Minimum order amount is ${self.minimum_order_amount}"
        
        # Check user restrictions
        if self.users.exists() and user:
            if not self.users.filter(id=user.id).exists():
                return False, "Coupon not available for your account"
        
        # Check per-user usage
        if self.usage_limit_per_user and user:
            from apps.orders.models import Order
            user_usage = Order.objects.filter(
                user=user,
                coupon=self
            ).exclude(
                status__in=[Order.STATUS_CANCELLED, Order.STATUS_REFUNDED]
            ).count()
            
            if user_usage >= self.usage_limit_per_user:
                return False, "You have already used this coupon"
        
        # Check first order only
        if self.first_order_only and user:
            from apps.orders.models import Order
            has_orders = Order.objects.filter(
                user=user
            ).exclude(
                status__in=[Order.STATUS_CANCELLED, Order.STATUS_REFUNDED]
            ).exists()
            
            if has_orders:
                return False, "This coupon is for first orders only"
        
        return True, "Coupon is valid"


class CouponUsage(models.Model):
    """Track coupon usage per user."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    coupon = models.ForeignKey(
        Coupon,
        on_delete=models.CASCADE,
        related_name='usage_records'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='coupon_usages'
    )
    order = models.ForeignKey(
        'orders.Order',
        on_delete=models.CASCADE,
        related_name='coupon_usage'
    )
    
    discount_applied = models.DecimalField(max_digits=10, decimal_places=2)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Coupon Usage'
        verbose_name_plural = 'Coupon Usages'


class Banner(models.Model):
    """
    Promotional banner for homepage/pages.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    title = models.CharField(max_length=200)
    subtitle = models.CharField(max_length=500, blank=True)
    
    image = models.ImageField(upload_to='banners/')
    image_mobile = models.ImageField(upload_to='banners/mobile/', blank=True, null=True)
    
    # Link
    link_url = models.URLField(blank=True)
    link_text = models.CharField(max_length=100, blank=True)
    
    # Positioning
    POSITION_HOME_HERO = 'home_hero'
    POSITION_HOME_SECONDARY = 'home_secondary'
    POSITION_CATEGORY = 'category'
    POSITION_CHOICES = [
        (POSITION_HOME_HERO, 'Home Hero'),
        (POSITION_HOME_SECONDARY, 'Home Secondary'),
        (POSITION_CATEGORY, 'Category Page'),
    ]
    position = models.CharField(
        max_length=20,
        choices=POSITION_CHOICES,
        default=POSITION_HOME_HERO
    )
    sort_order = models.PositiveIntegerField(default=0)
    
    # Validity
    start_date = models.DateTimeField(null=True, blank=True)
    end_date = models.DateTimeField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', '-created_at']
        verbose_name = 'Banner'
        verbose_name_plural = 'Banners'
    
    def __str__(self):
        return self.title
    
    @property
    def is_visible(self):
        """Check if banner should be visible."""
        if not self.is_active:
            return False
        
        now = timezone.now()
        
        if self.start_date and now < self.start_date:
            return False
        
        if self.end_date and now > self.end_date:
            return False
        
        return True


class Sale(models.Model):
    """
    Sale/promotion event model.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True)
    description = models.TextField(blank=True)
    
    # Discount
    DISCOUNT_PERCENTAGE = 'percentage'
    DISCOUNT_FIXED = 'fixed'
    DISCOUNT_CHOICES = [
        (DISCOUNT_PERCENTAGE, 'Percentage'),
        (DISCOUNT_FIXED, 'Fixed Amount'),
    ]
    discount_type = models.CharField(
        max_length=20,
        choices=DISCOUNT_CHOICES,
        default=DISCOUNT_PERCENTAGE
    )
    discount_value = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Products in sale
    products = models.ManyToManyField(
        'catalog.Product',
        related_name='sales',
        blank=True
    )
    categories = models.ManyToManyField(
        'catalog.Category',
        related_name='sales',
        blank=True,
        help_text="All products in these categories are on sale"
    )
    
    # Banner
    banner_image = models.ImageField(upload_to='sales/', blank=True, null=True)
    
    # Validity
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    
    # Status
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-start_date']
        verbose_name = 'Sale'
        verbose_name_plural = 'Sales'
    
    def __str__(self):
        return self.name
    
    @property
    def is_running(self):
        """Check if sale is currently running."""
        if not self.is_active:
            return False
        
        now = timezone.now()
        return self.start_date <= now <= self.end_date
    
    def get_sale_price(self, original_price):
        """Calculate sale price for given original price."""
        if self.discount_type == self.DISCOUNT_PERCENTAGE:
            discount = original_price * (self.discount_value / Decimal('100'))
        else:
            discount = self.discount_value
        
        return max(original_price - discount, Decimal('0'))
