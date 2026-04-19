"""
Promotions models
"""
import uuid
from decimal import Decimal
from django.db import models
from colorfield.fields import ColorField
from django.conf import settings
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.core.validators import MinValueValidator, MaxValueValidator


def normalize_currency_code(value):
    """Normalize to a 3-letter ISO currency code with BDT fallback."""
    if hasattr(value, "code"):
        value = getattr(value, "code", None)
    code = str(value or "").strip().upper()
    if len(code) != 3:
        return "BDT"
    return code


def get_site_default_currency_code():
    """Default promotions currency from SiteSettings, fallback to BDT."""
    try:
        from apps.pages.models import SiteSettings

        settings_obj = SiteSettings.get_settings()
        return normalize_currency_code(getattr(settings_obj, "currency", None))
    except Exception:
        return "BDT"


def convert_amount_by_code(amount, from_code, to_code):
    """Convert amount between currencies, returning original amount on failure."""
    amount_decimal = Decimal(str(amount or 0))
    source = normalize_currency_code(from_code)
    target = normalize_currency_code(to_code)
    if source == target:
        return amount_decimal
    try:
        from apps.i18n.services import CurrencyConversionService

        converted = CurrencyConversionService.convert_by_code(
            amount_decimal,
            source,
            target,
            round_result=False,
        )
        return Decimal(str(converted))
    except Exception:
        return amount_decimal


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
    currency = models.ForeignKey(
        'i18n.Currency',
        on_delete=models.PROTECT,
        to_field='code',
        db_column='currency',
        related_name='promotion_bundles',
        default=get_site_default_currency_code,
        verbose_name=_('currency'),
        help_text=_('Currency for bundle prices (e.g. BDT, USD).')
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

    def save(self, *args, **kwargs):
        self.currency_id = normalize_currency_code(self.currency_id or get_site_default_currency_code())
        super().save(*args, **kwargs)


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
    currency = models.ForeignKey(
        'i18n.Currency',
        on_delete=models.PROTECT,
        to_field='code',
        db_column='currency',
        related_name='promotion_coupons',
        default=get_site_default_currency_code,
        help_text="Currency for fixed/amount-based coupon values"
    )
    
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
    
    def calculate_discount(self, subtotal, subtotal_currency=None):
        """Calculate discount amount for given subtotal."""
        subtotal = Decimal(str(subtotal or 0))
        target_currency = normalize_currency_code(subtotal_currency or self.currency_id)
        coupon_currency = normalize_currency_code(self.currency_id)

        if self.discount_type == self.DISCOUNT_PERCENTAGE:
            discount = subtotal * (self.discount_value / Decimal('100'))
            if self.maximum_discount:
                max_discount = convert_amount_by_code(
                    self.maximum_discount,
                    coupon_currency,
                    target_currency,
                )
                discount = min(discount, max_discount)
        else:
            discount = convert_amount_by_code(
                self.discount_value,
                coupon_currency,
                target_currency,
            )

        return min(discount, subtotal)  # Don't exceed subtotal

    def can_use(self, user=None, subtotal=Decimal('0'), subtotal_currency=None):
        """Check if coupon can be used by user for given subtotal."""
        subtotal = Decimal(str(subtotal or 0))
        target_currency = normalize_currency_code(subtotal_currency or self.currency_id)
        coupon_currency = normalize_currency_code(self.currency_id)

        if not self.is_valid:
            return False, "Coupon is not valid"
        
        # Check minimum order
        if self.minimum_order_amount:
            required_minimum = convert_amount_by_code(
                self.minimum_order_amount,
                coupon_currency,
                target_currency,
            )
            if subtotal < required_minimum:
                return False, f"Minimum order amount is {required_minimum:.2f} {target_currency}"
        
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

    def save(self, *args, **kwargs):
        self.currency_id = normalize_currency_code(self.currency_id or get_site_default_currency_code())
        super().save(*args, **kwargs)


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
    currency = models.ForeignKey(
        'i18n.Currency',
        on_delete=models.PROTECT,
        to_field='code',
        db_column='currency',
        related_name='promotion_coupon_usages',
        default=get_site_default_currency_code,
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Coupon Usage'
        verbose_name_plural = 'Coupon Usages'

    def save(self, *args, **kwargs):
        if self.coupon and self.coupon.currency_id:
            self.currency_id = normalize_currency_code(self.coupon.currency_id)
        else:
            self.currency_id = normalize_currency_code(self.currency_id or get_site_default_currency_code())
        super().save(*args, **kwargs)


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

    # Style controls (optional)
    style_height = models.CharField(
        max_length=20, blank=True,
        help_text="CSS height value for the banner (e.g. 420px, 60vh)."
    )
    style_width = models.CharField(
        max_length=20, blank=True,
        help_text="CSS width value for the banner (e.g. 100%, 1200px)."
    )
    style_max_width = models.CharField(
        max_length=20, blank=True,
        help_text="CSS max-width value for the banner (e.g. 1200px)."
    )
    style_border_radius = models.CharField(
        max_length=20, blank=True,
        help_text="CSS border-radius value (e.g. 24px)."
    )
    style_border_width = models.CharField(
        max_length=10, blank=True,
        help_text="CSS border width (e.g. 1px)."
    )
    style_border_color = ColorField(blank=True, null=True)
    style_background_color = ColorField(blank=True, null=True)
    overlay_color = ColorField(blank=True, null=True)
    overlay_opacity = models.DecimalField(
        max_digits=3, decimal_places=2, null=True, blank=True,
        help_text="Overlay opacity from 0 to 1 (e.g. 0.6)."
    )
    text_color = ColorField(blank=True, null=True)
    CONTENT_VERTICAL_TOP = 'top'
    CONTENT_VERTICAL_CENTER = 'center'
    CONTENT_VERTICAL_BOTTOM = 'bottom'
    CONTENT_VERTICAL_CHOICES = [
        (CONTENT_VERTICAL_TOP, 'Top'),
        (CONTENT_VERTICAL_CENTER, 'Center'),
        (CONTENT_VERTICAL_BOTTOM, 'Bottom'),
    ]
    content_vertical_position = models.CharField(
        max_length=10,
        choices=CONTENT_VERTICAL_CHOICES,
        default=CONTENT_VERTICAL_BOTTOM,
        help_text="Vertical placement of title/subtitle/button block."
    )
    CONTENT_ALIGN_LEFT = 'left'
    CONTENT_ALIGN_CENTER = 'center'
    CONTENT_ALIGN_RIGHT = 'right'
    CONTENT_ALIGN_CHOICES = [
        (CONTENT_ALIGN_LEFT, 'Left'),
        (CONTENT_ALIGN_CENTER, 'Center'),
        (CONTENT_ALIGN_RIGHT, 'Right'),
    ]
    content_horizontal_alignment = models.CharField(
        max_length=10,
        choices=CONTENT_ALIGN_CHOICES,
        default=CONTENT_ALIGN_LEFT,
        help_text="Horizontal alignment for title/subtitle content."
    )
    button_alignment = models.CharField(
        max_length=10,
        choices=CONTENT_ALIGN_CHOICES,
        default=CONTENT_ALIGN_LEFT,
        help_text="Horizontal alignment for button."
    )
    title_font_size = models.CharField(
        max_length=20, blank=True,
        help_text="CSS font-size for title (e.g. 32px, 2rem)."
    )
    subtitle_font_size = models.CharField(
        max_length=20, blank=True,
        help_text="CSS font-size for subtitle (e.g. 16px, 1rem)."
    )
    button_font_size = models.CharField(
        max_length=20, blank=True,
        help_text="CSS font-size for button text (e.g. 12px, 0.875rem)."
    )
    button_padding = models.CharField(
        max_length=30, blank=True,
        help_text="CSS padding for button (e.g. 6px 16px)."
    )
    button_min_height = models.CharField(
        max_length=20, blank=True,
        help_text="CSS min-height for button (e.g. 40px)."
    )

    # Text colors
    title_color = ColorField(blank=True, null=True, verbose_name='title color')
    subtitle_color = ColorField(blank=True, null=True, verbose_name='subtitle color')

    # Font families
    FONT_FAMILY_CHOICES = [
        ('', 'Default'),
        ('system-ui, -apple-system, sans-serif', 'System'),
        ('Georgia, serif', 'Georgia (Serif)'),
        ('Times New Roman, serif', 'Times (Serif)'),
        ('Helvetica, Arial, sans-serif', 'Helvetica (Sans)'),
        ('Roboto, sans-serif', 'Roboto'),
        ('Open Sans, sans-serif', 'Open Sans'),
        ('Lato, sans-serif', 'Lato'),
        ('Montserrat, sans-serif', 'Montserrat'),
        ('Poppins, sans-serif', 'Poppins'),
        ('Playfair Display, serif', 'Playfair (Elegant)'),
    ]
    title_font_family = models.CharField(
        max_length=100, blank=True,
        choices=FONT_FAMILY_CHOICES,
        default='',
        verbose_name='title font family'
    )
    subtitle_font_family = models.CharField(
        max_length=100, blank=True,
        choices=FONT_FAMILY_CHOICES,
        default='',
        verbose_name='subtitle font family'
    )

    # Button colors
    button_background_color = ColorField(blank=True, null=True, verbose_name='button background color')
    button_text_color = ColorField(blank=True, null=True, verbose_name='button text color')
    button_hover_background_color = ColorField(blank=True, null=True, verbose_name='button hover background color')
    button_hover_text_color = ColorField(blank=True, null=True, verbose_name='button hover text color')

    # Animations and transitions
    ANIMATION_TYPE_CHOICES = [
        ('', 'None'),
        ('fade', 'Fade In'),
        ('slide-up', 'Slide Up'),
        ('slide-down', 'Slide Down'),
        ('slide-left', 'Slide Left'),
        ('slide-right', 'Slide Right'),
        ('zoom', 'Zoom In'),
        ('bounce', 'Bounce'),
        ('flip', 'Flip'),
    ]
    animation_type = models.CharField(
        max_length=50, blank=True,
        choices=ANIMATION_TYPE_CHOICES,
        default='fade',
        verbose_name='animation type'
    )
    transition_duration = models.DecimalField(
        max_digits=4, decimal_places=2, blank=True, null=True,
        verbose_name='transition duration (seconds)',
        help_text='Duration of the transition animation (e.g., 0.5)'
    )

    # Banner timing for carousel
    autoplay_delay = models.PositiveIntegerField(
        blank=True, null=True,
        verbose_name='autoplay delay (seconds)',
        help_text='Time before auto-rotating to next banner (leave empty to use default)'
    )

    # Banner size presets
    SIZE_PRESET_CHOICES = [
        ('', 'Default'),
        ('compact', 'Compact (280px)'),
        ('small', 'Small (350px)'),
        ('medium', 'Medium (420px)'),
        ('large', 'Large (520px)'),
        ('hero', 'Hero (600px)'),
        ('fullscreen', 'Fullscreen (100vh)'),
        ('custom', 'Custom (use height field)'),
    ]
    size_preset = models.CharField(
        max_length=30, blank=True,
        choices=SIZE_PRESET_CHOICES,
        default='medium',
        verbose_name='size preset'
    )

    # Opacity/Transparency
    container_opacity = models.DecimalField(
        max_digits=3, decimal_places=2, blank=True, null=True,
        verbose_name='container opacity',
        help_text='Overall banner opacity from 0 (transparent) to 1 (opaque)'
    )

    # Background image settings
    BACKGROUND_SIZE_CHOICES = [
        ('cover', 'Cover (default)'),
        ('contain', 'Contain'),
        ('auto', 'Auto'),
        ('100% 100%', 'Stretch'),
    ]
    background_size = models.CharField(
        max_length=30, blank=True,
        choices=BACKGROUND_SIZE_CHOICES,
        default='cover',
        verbose_name='background size'
    )
    BACKGROUND_POSITION_CHOICES = [
        ('center', 'Center'),
        ('top', 'Top'),
        ('bottom', 'Bottom'),
        ('left', 'Left'),
        ('right', 'Right'),
        ('top left', 'Top Left'),
        ('top right', 'Top Right'),
        ('bottom left', 'Bottom Left'),
        ('bottom right', 'Bottom Right'),
    ]
    background_position = models.CharField(
        max_length=30, blank=True,
        choices=BACKGROUND_POSITION_CHOICES,
        default='center',
        verbose_name='background position'
    )

    # Mobile-specific settings
    mobile_height = models.CharField(
        max_length=20, blank=True,
        verbose_name='mobile height',
        help_text='CSS height for mobile devices (e.g., 280px, 50vh)'
    )
    hide_on_mobile = models.BooleanField(default=False, verbose_name='hide on mobile')
    hide_on_desktop = models.BooleanField(default=False, verbose_name='hide on desktop')

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
    currency = models.ForeignKey(
        'i18n.Currency',
        on_delete=models.PROTECT,
        to_field='code',
        db_column='currency',
        related_name='promotion_sales',
        default=get_site_default_currency_code,
        help_text="Currency for fixed discount values"
    )
    
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
    
    def get_sale_price(self, original_price, price_currency=None):
        """Calculate sale price for given original price."""
        original_price = Decimal(str(original_price or 0))
        target_currency = normalize_currency_code(price_currency or self.currency_id)

        if self.discount_type == self.DISCOUNT_PERCENTAGE:
            discount = original_price * (self.discount_value / Decimal('100'))
        else:
            discount = convert_amount_by_code(
                self.discount_value,
                self.currency_id,
                target_currency,
            )
        
        return max(original_price - discount, Decimal('0'))

    def save(self, *args, **kwargs):
        self.currency_id = normalize_currency_code(self.currency_id or get_site_default_currency_code())
        super().save(*args, **kwargs)
