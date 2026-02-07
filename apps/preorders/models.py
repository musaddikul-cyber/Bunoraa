"""
Pre-order models - Comprehensive custom pre-order system
"""
import uuid
from decimal import Decimal
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.core.validators import MinValueValidator, MaxValueValidator, FileExtensionValidator
from django.core.exceptions import ValidationError


def preorder_design_path(instance, filename):
    """Generate upload path for pre-order design files."""
    return f'preorders/{instance.preorder.preorder_number}/designs/{filename}'


def preorder_reference_path(instance, filename):
    """Generate upload path for pre-order reference images."""
    return f'preorders/{instance.preorder.preorder_number}/references/{filename}'


def preorder_item_design_path(instance, filename):
    """Generate upload path for pre-order item design files."""
    return f'preorders/{instance.preorder_item.preorder.preorder_number}/items/{instance.preorder_item.id}/{filename}'


class PreOrderCategory(models.Model):
    """
    Categories for pre-orders to organize different types of custom orders.
    Examples: Custom Jewelry, Personalized Gifts, Bulk Orders, Special Events, etc.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(_('name'), max_length=100)
    slug = models.SlugField(_('slug'), max_length=120, unique=True)
    description = models.TextField(_('description'), blank=True)
    icon = models.CharField(_('icon class'), max_length=50, blank=True, help_text=_('CSS icon class'))
    image = models.ImageField(
        _('image'),
        upload_to='preorder_categories/',
        blank=True,
        null=True
    )
    
    # Pricing
    base_price = models.DecimalField(
        _('base price'),
        max_digits=12,
        decimal_places=2,
        default=Decimal('0.00'),
        validators=[MinValueValidator(Decimal('0.00'))],
        help_text=_('Minimum base price for this category')
    )
    deposit_percentage = models.PositiveIntegerField(
        _('deposit percentage'),
        default=30,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text=_('Required deposit percentage for this category')
    )
    
    # Timing
    min_production_days = models.PositiveIntegerField(
        _('minimum production days'),
        default=7,
        help_text=_('Minimum days required to produce items in this category')
    )
    max_production_days = models.PositiveIntegerField(
        _('maximum production days'),
        default=30,
        help_text=_('Maximum days to produce items in this category')
    )
    
    # Settings
    requires_design = models.BooleanField(
        _('requires design file'),
        default=True,
        help_text=_('Whether customers must upload a design file')
    )
    requires_approval = models.BooleanField(
        _('requires approval'),
        default=True,
        help_text=_('Whether pre-orders need admin approval before production')
    )
    allow_rush_order = models.BooleanField(
        _('allow rush order'),
        default=True,
        help_text=_('Whether rush production is available')
    )
    rush_order_fee_percentage = models.PositiveIntegerField(
        _('rush order fee percentage'),
        default=25,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text=_('Additional percentage charged for rush orders')
    )
    
    # Limits
    min_quantity = models.PositiveIntegerField(_('minimum quantity'), default=1)
    max_quantity = models.PositiveIntegerField(_('maximum quantity'), default=1000)
    
    # Status
    is_active = models.BooleanField(_('active'), default=True)
    order = models.PositiveIntegerField(_('display order'), default=0)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('pre-order category')
        verbose_name_plural = _('pre-order categories')
        ordering = ['order', 'name']
    
    def __str__(self):
        return self.name


class PreOrderOption(models.Model):
    """
    Customization options available for pre-orders within a category.
    Examples: Material type, Color, Size, Engraving, etc.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    category = models.ForeignKey(
        PreOrderCategory,
        on_delete=models.CASCADE,
        related_name='options',
        verbose_name=_('category')
    )
    
    # Option details
    name = models.CharField(_('name'), max_length=100)
    description = models.TextField(_('description'), blank=True)
    
    # Type
    OPTION_TEXT = 'text'
    OPTION_TEXTAREA = 'textarea'
    OPTION_NUMBER = 'number'
    OPTION_SELECT = 'select'
    OPTION_MULTISELECT = 'multiselect'
    OPTION_CHECKBOX = 'checkbox'
    OPTION_COLOR = 'color'
    OPTION_FILE = 'file'
    OPTION_DATE = 'date'
    OPTION_TYPE_CHOICES = [
        (OPTION_TEXT, _('Single Line Text')),
        (OPTION_TEXTAREA, _('Multi-line Text')),
        (OPTION_NUMBER, _('Number')),
        (OPTION_SELECT, _('Dropdown Select')),
        (OPTION_MULTISELECT, _('Multi-Select')),
        (OPTION_CHECKBOX, _('Checkbox')),
        (OPTION_COLOR, _('Color Picker')),
        (OPTION_FILE, _('File Upload')),
        (OPTION_DATE, _('Date Picker')),
    ]
    option_type = models.CharField(
        _('option type'),
        max_length=20,
        choices=OPTION_TYPE_CHOICES,
        default=OPTION_TEXT
    )
    
    # Validation
    is_required = models.BooleanField(_('required'), default=False)
    min_length = models.PositiveIntegerField(_('minimum length'), null=True, blank=True)
    max_length = models.PositiveIntegerField(_('maximum length'), null=True, blank=True)
    
    # Pricing
    price_modifier = models.DecimalField(
        _('price modifier'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00'),
        help_text=_('Additional cost for this option')
    )
    
    # Display
    placeholder = models.CharField(_('placeholder text'), max_length=200, blank=True)
    help_text = models.CharField(_('help text'), max_length=500, blank=True)
    order = models.PositiveIntegerField(_('display order'), default=0)
    is_active = models.BooleanField(_('active'), default=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('pre-order option')
        verbose_name_plural = _('pre-order options')
        ordering = ['category', 'order', 'name']
    
    def __str__(self):
        return f"{self.category.name} - {self.name}"


class PreOrderOptionChoice(models.Model):
    """
    Choices for select/multiselect pre-order options.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    option = models.ForeignKey(
        PreOrderOption,
        on_delete=models.CASCADE,
        related_name='choices',
        verbose_name=_('option')
    )
    
    value = models.CharField(_('value'), max_length=200)
    display_name = models.CharField(_('display name'), max_length=200)
    price_modifier = models.DecimalField(
        _('price modifier'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    color_code = models.CharField(
        _('color code'),
        max_length=7,
        blank=True,
        help_text=_('Hex color code for color options')
    )
    image = models.ImageField(
        _('image'),
        upload_to='preorder_option_choices/',
        blank=True,
        null=True
    )
    
    order = models.PositiveIntegerField(_('display order'), default=0)
    is_active = models.BooleanField(_('active'), default=True)
    
    class Meta:
        verbose_name = _('option choice')
        verbose_name_plural = _('option choices')
        ordering = ['option', 'order', 'display_name']
    
    def __str__(self):
        return f"{self.option.name} - {self.display_name}"


class PreOrder(models.Model):
    """
    Main pre-order model representing a custom order request.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Pre-order number for display
    preorder_number = models.CharField(
        _('pre-order number'),
        max_length=50,
        unique=True,
        db_index=True
    )
    
    # Customer
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name='preorders',
        null=True,
        blank=True,
        verbose_name=_('user')
    )
    email = models.EmailField(_('email'))
    phone = models.CharField(_('phone'), max_length=20, blank=True)
    full_name = models.CharField(_('full name'), max_length=200)
    
    # Category
    category = models.ForeignKey(
        PreOrderCategory,
        on_delete=models.PROTECT,
        related_name='preorders',
        verbose_name=_('category')
    )
    
    # Linked product (optional - if based on existing product)
    base_product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.SET_NULL,
        related_name='preorders',
        null=True,
        blank=True,
        verbose_name=_('base product'),
        help_text=_('Optional product to base the custom order on')
    )
    
    # Order details
    title = models.CharField(
        _('order title'),
        max_length=300,
        help_text=_('Brief title describing the order')
    )
    description = models.TextField(
        _('detailed description'),
        help_text=_('Detailed description of what you want')
    )
    quantity = models.PositiveIntegerField(
        _('quantity'),
        default=1,
        validators=[MinValueValidator(1)]
    )
    
    # Special requests
    special_instructions = models.TextField(
        _('special instructions'),
        blank=True,
        help_text=_('Any special requirements or instructions')
    )
    gift_message = models.TextField(
        _('gift message'),
        blank=True,
        help_text=_('Message to include if this is a gift')
    )
    is_gift = models.BooleanField(_('is gift'), default=False)
    gift_wrap = models.BooleanField(_('gift wrap'), default=False)
    
    # Status
    STATUS_DRAFT = 'draft'
    STATUS_SUBMITTED = 'submitted'
    STATUS_UNDER_REVIEW = 'under_review'
    STATUS_QUOTED = 'quoted'
    STATUS_QUOTE_ACCEPTED = 'quote_accepted'
    STATUS_QUOTE_REJECTED = 'quote_rejected'
    STATUS_DEPOSIT_PENDING = 'deposit_pending'
    STATUS_DEPOSIT_PAID = 'deposit_paid'
    STATUS_IN_PRODUCTION = 'in_production'
    STATUS_QUALITY_CHECK = 'quality_check'
    STATUS_AWAITING_APPROVAL = 'awaiting_approval'
    STATUS_REVISION_REQUESTED = 'revision_requested'
    STATUS_FINAL_PAYMENT_PENDING = 'final_payment_pending'
    STATUS_COMPLETED = 'completed'
    STATUS_READY_TO_SHIP = 'ready_to_ship'
    STATUS_SHIPPED = 'shipped'
    STATUS_DELIVERED = 'delivered'
    STATUS_CANCELLED = 'cancelled'
    STATUS_REFUNDED = 'refunded'
    STATUS_ON_HOLD = 'on_hold'
    
    STATUS_CHOICES = [
        (STATUS_DRAFT, _('Draft')),
        (STATUS_SUBMITTED, _('Submitted')),
        (STATUS_UNDER_REVIEW, _('Under Review')),
        (STATUS_QUOTED, _('Quote Provided')),
        (STATUS_QUOTE_ACCEPTED, _('Quote Accepted')),
        (STATUS_QUOTE_REJECTED, _('Quote Rejected')),
        (STATUS_DEPOSIT_PENDING, _('Deposit Pending')),
        (STATUS_DEPOSIT_PAID, _('Deposit Paid')),
        (STATUS_IN_PRODUCTION, _('In Production')),
        (STATUS_QUALITY_CHECK, _('Quality Check')),
        (STATUS_AWAITING_APPROVAL, _('Awaiting Customer Approval')),
        (STATUS_REVISION_REQUESTED, _('Revision Requested')),
        (STATUS_FINAL_PAYMENT_PENDING, _('Final Payment Pending')),
        (STATUS_COMPLETED, _('Completed')),
        (STATUS_READY_TO_SHIP, _('Ready to Ship')),
        (STATUS_SHIPPED, _('Shipped')),
        (STATUS_DELIVERED, _('Delivered')),
        (STATUS_CANCELLED, _('Cancelled')),
        (STATUS_REFUNDED, _('Refunded')),
        (STATUS_ON_HOLD, _('On Hold')),
    ]
    status = models.CharField(
        _('status'),
        max_length=30,
        choices=STATUS_CHOICES,
        default=STATUS_DRAFT,
        db_index=True
    )
    
    # Priority
    PRIORITY_LOW = 'low'
    PRIORITY_NORMAL = 'normal'
    PRIORITY_HIGH = 'high'
    PRIORITY_URGENT = 'urgent'
    PRIORITY_CHOICES = [
        (PRIORITY_LOW, _('Low')),
        (PRIORITY_NORMAL, _('Normal')),
        (PRIORITY_HIGH, _('High')),
        (PRIORITY_URGENT, _('Urgent')),
    ]
    priority = models.CharField(
        _('priority'),
        max_length=10,
        choices=PRIORITY_CHOICES,
        default=PRIORITY_NORMAL
    )
    
    # Rush order
    is_rush_order = models.BooleanField(_('rush order'), default=False)
    rush_order_fee = models.DecimalField(
        _('rush order fee'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    
    # Pricing
    estimated_price = models.DecimalField(
        _('estimated price'),
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True
    )
    final_price = models.DecimalField(
        _('final price'),
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True
    )
    discount_amount = models.DecimalField(
        _('discount amount'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    tax_amount = models.DecimalField(
        _('tax amount'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    shipping_cost = models.DecimalField(
        _('shipping cost'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    total_amount = models.DecimalField(
        _('total amount'),
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True
    )
    
    # Deposits and payments
    deposit_required = models.DecimalField(
        _('deposit required'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    deposit_paid = models.DecimalField(
        _('deposit paid'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    amount_paid = models.DecimalField(
        _('total amount paid'),
        max_digits=12,
        decimal_places=2,
        default=Decimal('0.00')
    )
    amount_remaining = models.DecimalField(
        _('amount remaining'),
        max_digits=12,
        decimal_places=2,
        default=Decimal('0.00')
    )
    
    # Currency
    currency = models.CharField(_('currency'), max_length=3, default='BDT')
    
    # Coupon
    coupon = models.ForeignKey(
        'promotions.Coupon',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='preorders',
        verbose_name=_('coupon')
    )
    
    # Shipping address
    shipping_first_name = models.CharField(_('shipping first name'), max_length=100, blank=True)
    shipping_last_name = models.CharField(_('shipping last name'), max_length=100, blank=True)
    shipping_address_line_1 = models.CharField(_('shipping address line 1'), max_length=255, blank=True)
    shipping_address_line_2 = models.CharField(_('shipping address line 2'), max_length=255, blank=True)
    shipping_city = models.CharField(_('shipping city'), max_length=100, blank=True)
    shipping_state = models.CharField(_('shipping state'), max_length=100, blank=True)
    shipping_postal_code = models.CharField(_('shipping postal code'), max_length=20, blank=True)
    shipping_country = models.CharField(_('shipping country'), max_length=100, blank=True)
    
    # Shipping
    SHIPPING_STANDARD = 'standard'
    SHIPPING_EXPRESS = 'express'
    SHIPPING_OVERNIGHT = 'overnight'
    SHIPPING_PICKUP = 'pickup'
    SHIPPING_CHOICES = [
        (SHIPPING_STANDARD, _('Standard Shipping')),
        (SHIPPING_EXPRESS, _('Express Shipping')),
        (SHIPPING_OVERNIGHT, _('Overnight Shipping')),
        (SHIPPING_PICKUP, _('Store Pickup')),
    ]
    shipping_method = models.CharField(
        _('shipping method'),
        max_length=20,
        choices=SHIPPING_CHOICES,
        default=SHIPPING_STANDARD
    )
    tracking_number = models.CharField(_('tracking number'), max_length=100, blank=True)
    tracking_url = models.URLField(_('tracking URL'), blank=True)
    
    # Dates
    requested_delivery_date = models.DateField(
        _('requested delivery date'),
        null=True,
        blank=True
    )
    estimated_completion_date = models.DateField(
        _('estimated completion date'),
        null=True,
        blank=True
    )
    actual_completion_date = models.DateField(
        _('actual completion date'),
        null=True,
        blank=True
    )
    production_start_date = models.DateField(
        _('production start date'),
        null=True,
        blank=True
    )
    
    # Assignment
    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name='assigned_preorders',
        null=True,
        blank=True,
        verbose_name=_('assigned to'),
        help_text=_('Staff member assigned to handle this pre-order')
    )
    
    # Notes
    customer_notes = models.TextField(_('customer notes'), blank=True)
    admin_notes = models.TextField(_('admin notes'), blank=True)
    production_notes = models.TextField(_('production notes'), blank=True)
    
    # Quote details
    quote_valid_until = models.DateTimeField(_('quote valid until'), null=True, blank=True)
    quote_notes = models.TextField(_('quote notes'), blank=True)
    
    # Revision tracking
    revision_count = models.PositiveIntegerField(_('revision count'), default=0)
    max_revisions = models.PositiveIntegerField(_('max revisions allowed'), default=3)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    submitted_at = models.DateTimeField(_('submitted at'), null=True, blank=True)
    quoted_at = models.DateTimeField(_('quoted at'), null=True, blank=True)
    approved_at = models.DateTimeField(_('approved at'), null=True, blank=True)
    production_started_at = models.DateTimeField(_('production started at'), null=True, blank=True)
    completed_at = models.DateTimeField(_('completed at'), null=True, blank=True)
    shipped_at = models.DateTimeField(_('shipped at'), null=True, blank=True)
    delivered_at = models.DateTimeField(_('delivered at'), null=True, blank=True)
    cancelled_at = models.DateTimeField(_('cancelled at'), null=True, blank=True)
    
    # Soft delete
    is_deleted = models.BooleanField(_('deleted'), default=False)
    deleted_at = models.DateTimeField(_('deleted at'), null=True, blank=True)
    
    # Source tracking
    source = models.CharField(
        _('source'),
        max_length=50,
        blank=True,
        help_text=_('How the customer found us (referral, social, etc.)')
    )
    utm_source = models.CharField(_('UTM source'), max_length=100, blank=True)
    utm_medium = models.CharField(_('UTM medium'), max_length=100, blank=True)
    utm_campaign = models.CharField(_('UTM campaign'), max_length=100, blank=True)
    
    class Meta:
        verbose_name = _('pre-order')
        verbose_name_plural = _('pre-orders')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['preorder_number']),
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['user', 'status']),
            models.Index(fields=['assigned_to', 'status']),
        ]
    
    def __str__(self):
        return f"{self.preorder_number} - {self.title}"
    
    def save(self, *args, **kwargs):
        if not self.preorder_number:
            self.preorder_number = self.generate_preorder_number()
        self.calculate_amounts()
        super().save(*args, **kwargs)
    
    def generate_preorder_number(self):
        """Generate unique pre-order number."""
        from django.utils import timezone
        import random
        prefix = "PRE"
        date_str = timezone.now().strftime("%Y%m%d")
        random_str = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        return f"{prefix}-{date_str}-{random_str}"
    
    def calculate_amounts(self):
        """Calculate total amounts."""
        if self.final_price:
            base = self.final_price
        elif self.estimated_price:
            base = self.estimated_price
        else:
            base = Decimal('0.00')
        
        base += self.rush_order_fee
        base -= self.discount_amount
        base += self.tax_amount
        base += self.shipping_cost
        self.total_amount = base
        self.amount_remaining = base - self.amount_paid
    
    @property
    def is_fully_paid(self):
        """Check if the pre-order is fully paid."""
        return self.amount_remaining <= Decimal('0.00')
    
    @property
    def deposit_is_paid(self):
        """Check if the required deposit has been paid."""
        return self.deposit_paid >= self.deposit_required
    
    @property
    def can_start_production(self):
        """Check if production can start."""
        return (
            self.status in [self.STATUS_DEPOSIT_PAID, self.STATUS_QUOTE_ACCEPTED] and
            self.deposit_is_paid
        )
    
    @property
    def days_until_deadline(self):
        """Calculate days until requested delivery date."""
        if not self.requested_delivery_date:
            return None
        delta = self.requested_delivery_date - timezone.now().date()
        return delta.days
    
    @property
    def is_overdue(self):
        """Check if the pre-order is overdue."""
        if not self.estimated_completion_date:
            return False
        if self.status in [self.STATUS_COMPLETED, self.STATUS_DELIVERED, 
                          self.STATUS_CANCELLED, self.STATUS_REFUNDED]:
            return False
        return timezone.now().date() > self.estimated_completion_date


class PreOrderItem(models.Model):
    """
    Individual items within a pre-order (for multi-item pre-orders).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='items',
        verbose_name=_('pre-order')
    )
    
    # Item details
    name = models.CharField(_('item name'), max_length=300)
    description = models.TextField(_('description'), blank=True)
    quantity = models.PositiveIntegerField(_('quantity'), default=1)
    
    # Pricing
    unit_price = models.DecimalField(
        _('unit price'),
        max_digits=12,
        decimal_places=2,
        default=Decimal('0.00')
    )
    total_price = models.DecimalField(
        _('total price'),
        max_digits=12,
        decimal_places=2,
        default=Decimal('0.00')
    )
    
    # Customization
    customization_details = models.JSONField(
        _('customization details'),
        default=dict,
        blank=True
    )
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_IN_PROGRESS = 'in_progress'
    STATUS_COMPLETED = 'completed'
    STATUS_CANCELLED = 'cancelled'
    STATUS_CHOICES = [
        (STATUS_PENDING, _('Pending')),
        (STATUS_IN_PROGRESS, _('In Progress')),
        (STATUS_COMPLETED, _('Completed')),
        (STATUS_CANCELLED, _('Cancelled')),
    ]
    status = models.CharField(
        _('status'),
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
    # Notes
    notes = models.TextField(_('notes'), blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('pre-order item')
        verbose_name_plural = _('pre-order items')
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.preorder.preorder_number} - {self.name}"
    
    def save(self, *args, **kwargs):
        self.total_price = self.unit_price * self.quantity
        super().save(*args, **kwargs)


class PreOrderOptionValue(models.Model):
    """
    Stores the selected option values for a pre-order.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='option_values',
        verbose_name=_('pre-order')
    )
    option = models.ForeignKey(
        PreOrderOption,
        on_delete=models.CASCADE,
        related_name='values_submitted',
        verbose_name=_('option')
    )
    
    # Value storage (flexible for different option types)
    text_value = models.TextField(_('text value'), blank=True)
    number_value = models.DecimalField(
        _('number value'),
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True
    )
    choice_value = models.ForeignKey(
        PreOrderOptionChoice,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='submitted_values',
        verbose_name=_('selected choice')
    )
    multi_choice_values = models.ManyToManyField(
        PreOrderOptionChoice,
        blank=True,
        related_name='multi_submitted_values',
        verbose_name=_('selected choices')
    )
    boolean_value = models.BooleanField(_('boolean value'), null=True, blank=True)
    date_value = models.DateField(_('date value'), null=True, blank=True)
    file_value = models.FileField(
        _('file value'),
        upload_to='preorders/option_files/',
        blank=True,
        null=True
    )
    
    # Price modifier applied
    price_modifier_applied = models.DecimalField(
        _('price modifier applied'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('pre-order option value')
        verbose_name_plural = _('pre-order option values')
        unique_together = ['preorder', 'option']
    
    def __str__(self):
        return f"{self.preorder.preorder_number} - {self.option.name}"
    
    def get_value(self):
        """Get the appropriate value based on option type."""
        option_type = self.option.option_type
        if option_type in [PreOrderOption.OPTION_TEXT, PreOrderOption.OPTION_TEXTAREA]:
            return self.text_value
        elif option_type == PreOrderOption.OPTION_NUMBER:
            return self.number_value
        elif option_type == PreOrderOption.OPTION_SELECT:
            return self.choice_value
        elif option_type == PreOrderOption.OPTION_MULTISELECT:
            return list(self.multi_choice_values.all())
        elif option_type == PreOrderOption.OPTION_CHECKBOX:
            return self.boolean_value
        elif option_type == PreOrderOption.OPTION_COLOR:
            return self.text_value
        elif option_type == PreOrderOption.OPTION_FILE:
            return self.file_value
        elif option_type == PreOrderOption.OPTION_DATE:
            return self.date_value
        return None


class PreOrderDesign(models.Model):
    """
    Design files uploaded for a pre-order.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='designs',
        verbose_name=_('pre-order')
    )
    
    # File
    file = models.FileField(
        _('design file'),
        upload_to=preorder_design_path,
        validators=[
            FileExtensionValidator(
                allowed_extensions=['pdf', 'png', 'jpg', 'jpeg', 'ai', 'psd', 
                                   'svg', 'eps', 'cdr', 'zip', 'rar']
            )
        ]
    )
    original_filename = models.CharField(_('original filename'), max_length=255)
    file_size = models.PositiveIntegerField(_('file size (bytes)'), default=0)
    
    # Type
    DESIGN_CUSTOMER = 'customer'
    DESIGN_ADMIN = 'admin'
    DESIGN_FINAL = 'final'
    DESIGN_PROOF = 'proof'
    DESIGN_TYPE_CHOICES = [
        (DESIGN_CUSTOMER, _('Customer Upload')),
        (DESIGN_ADMIN, _('Admin Upload')),
        (DESIGN_FINAL, _('Final Design')),
        (DESIGN_PROOF, _('Proof')),
    ]
    design_type = models.CharField(
        _('design type'),
        max_length=20,
        choices=DESIGN_TYPE_CHOICES,
        default=DESIGN_CUSTOMER
    )
    
    # Approval
    is_approved = models.BooleanField(_('approved'), default=False)
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='approved_designs',
        verbose_name=_('approved by')
    )
    approved_at = models.DateTimeField(_('approved at'), null=True, blank=True)
    
    # Version tracking
    version = models.PositiveIntegerField(_('version'), default=1)
    is_current = models.BooleanField(_('current version'), default=True)
    
    # Notes
    notes = models.TextField(_('notes'), blank=True)
    
    # Uploader
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='uploaded_designs',
        verbose_name=_('uploaded by')
    )
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('pre-order design')
        verbose_name_plural = _('pre-order designs')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.preorder.preorder_number} - {self.original_filename}"
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
            if not self.original_filename:
                self.original_filename = self.file.name
        super().save(*args, **kwargs)


class PreOrderReference(models.Model):
    """
    Reference images/files uploaded by customer for inspiration.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='references',
        verbose_name=_('pre-order')
    )
    
    file = models.FileField(
        _('reference file'),
        upload_to=preorder_reference_path
    )
    original_filename = models.CharField(_('original filename'), max_length=255)
    description = models.TextField(_('description'), blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    
    class Meta:
        verbose_name = _('pre-order reference')
        verbose_name_plural = _('pre-order references')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.preorder.preorder_number} - {self.original_filename}"


class PreOrderStatusHistory(models.Model):
    """
    Track status changes for pre-orders.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='status_history',
        verbose_name=_('pre-order')
    )
    
    # Status change
    from_status = models.CharField(_('from status'), max_length=30, blank=True)
    to_status = models.CharField(_('to status'), max_length=30)
    
    # Who made the change
    changed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='preorder_status_changes',
        verbose_name=_('changed by')
    )
    
    # Notes
    notes = models.TextField(_('notes'), blank=True)
    is_system = models.BooleanField(_('system generated'), default=False)
    
    # Notification tracking
    notification_sent = models.BooleanField(_('notification sent'), default=False)
    notification_sent_at = models.DateTimeField(_('notification sent at'), null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    
    class Meta:
        verbose_name = _('pre-order status history')
        verbose_name_plural = _('pre-order status histories')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.preorder.preorder_number}: {self.from_status} → {self.to_status}"


class PreOrderPayment(models.Model):
    """
    Track payments for pre-orders.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='payments',
        verbose_name=_('pre-order')
    )
    
    # Payment type
    PAYMENT_DEPOSIT = 'deposit'
    PAYMENT_PARTIAL = 'partial'
    PAYMENT_FINAL = 'final'
    PAYMENT_REFUND = 'refund'
    PAYMENT_TYPE_CHOICES = [
        (PAYMENT_DEPOSIT, _('Deposit')),
        (PAYMENT_PARTIAL, _('Partial Payment')),
        (PAYMENT_FINAL, _('Final Payment')),
        (PAYMENT_REFUND, _('Refund')),
    ]
    payment_type = models.CharField(
        _('payment type'),
        max_length=20,
        choices=PAYMENT_TYPE_CHOICES,
        default=PAYMENT_DEPOSIT
    )
    
    # Amount
    amount = models.DecimalField(
        _('amount'),
        max_digits=12,
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0.01'))]
    )
    currency = models.CharField(_('currency'), max_length=3, default='BDT')
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'
    STATUS_CANCELLED = 'cancelled'
    STATUS_REFUNDED = 'refunded'
    STATUS_CHOICES = [
        (STATUS_PENDING, _('Pending')),
        (STATUS_PROCESSING, _('Processing')),
        (STATUS_COMPLETED, _('Completed')),
        (STATUS_FAILED, _('Failed')),
        (STATUS_CANCELLED, _('Cancelled')),
        (STATUS_REFUNDED, _('Refunded')),
    ]
    status = models.CharField(
        _('status'),
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
    # Payment method
    PAYMENT_METHOD_STRIPE = 'stripe'
    PAYMENT_METHOD_PAYPAL = 'paypal'
    PAYMENT_METHOD_BANK = 'bank_transfer'
    PAYMENT_METHOD_CASH = 'cash'
    PAYMENT_METHOD_BKASH = 'bkash'
    PAYMENT_METHOD_NAGAD = 'nagad'
    PAYMENT_METHOD_CHOICES = [
        (PAYMENT_METHOD_STRIPE, _('Credit Card (Stripe)')),
        (PAYMENT_METHOD_PAYPAL, _('PayPal')),
        (PAYMENT_METHOD_BANK, _('Bank Transfer')),
        (PAYMENT_METHOD_CASH, _('Cash')),
        (PAYMENT_METHOD_BKASH, _('bKash')),
        (PAYMENT_METHOD_NAGAD, _('Nagad')),
    ]
    payment_method = models.CharField(
        _('payment method'),
        max_length=20,
        choices=PAYMENT_METHOD_CHOICES,
        default=PAYMENT_METHOD_STRIPE
    )
    
    # External references
    transaction_id = models.CharField(_('transaction ID'), max_length=200, blank=True)
    stripe_payment_intent_id = models.CharField(_('Stripe payment intent ID'), max_length=200, blank=True)
    stripe_charge_id = models.CharField(_('Stripe charge ID'), max_length=200, blank=True)
    
    # Gateway response
    gateway_response = models.JSONField(_('gateway response'), default=dict, blank=True)
    
    # Notes
    notes = models.TextField(_('notes'), blank=True)
    
    # Receipt
    receipt_url = models.URLField(_('receipt URL'), blank=True)
    receipt_sent = models.BooleanField(_('receipt sent'), default=False)
    receipt_sent_at = models.DateTimeField(_('receipt sent at'), null=True, blank=True)
    
    # Recorded by
    recorded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='recorded_preorder_payments',
        verbose_name=_('recorded by')
    )
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    paid_at = models.DateTimeField(_('paid at'), null=True, blank=True)
    
    class Meta:
        verbose_name = _('pre-order payment')
        verbose_name_plural = _('pre-order payments')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.preorder.preorder_number} - {self.payment_type} - {self.amount}"


class PreOrderMessage(models.Model):
    """
    Messages between customer and admin for a pre-order.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name=_('pre-order')
    )
    
    # Sender
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sent_preorder_messages',
        verbose_name=_('sender')
    )
    is_from_customer = models.BooleanField(_('from customer'), default=True)
    is_from_system = models.BooleanField(_('system message'), default=False)
    
    # Message
    subject = models.CharField(_('subject'), max_length=200, blank=True)
    message = models.TextField(_('message'))
    
    # Attachments
    attachment = models.FileField(
        _('attachment'),
        upload_to='preorders/messages/',
        blank=True,
        null=True
    )
    
    # Read status
    is_read = models.BooleanField(_('read'), default=False)
    read_at = models.DateTimeField(_('read at'), null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    
    class Meta:
        verbose_name = _('pre-order message')
        verbose_name_plural = _('pre-order messages')
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.preorder.preorder_number} - {self.subject or 'Message'}"


class PreOrderRevision(models.Model):
    """
    Track revision requests from customers.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='revisions',
        verbose_name=_('pre-order')
    )
    
    # Revision details
    revision_number = models.PositiveIntegerField(_('revision number'))
    description = models.TextField(_('revision description'))
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_IN_PROGRESS = 'in_progress'
    STATUS_COMPLETED = 'completed'
    STATUS_REJECTED = 'rejected'
    STATUS_CHOICES = [
        (STATUS_PENDING, _('Pending')),
        (STATUS_IN_PROGRESS, _('In Progress')),
        (STATUS_COMPLETED, _('Completed')),
        (STATUS_REJECTED, _('Rejected')),
    ]
    status = models.CharField(
        _('status'),
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
    # Additional cost
    additional_cost = models.DecimalField(
        _('additional cost'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00'),
        help_text=_('Additional cost for this revision (if beyond free revisions)')
    )
    
    # Requested by
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='requested_revisions',
        verbose_name=_('requested by')
    )
    
    # Admin response
    admin_response = models.TextField(_('admin response'), blank=True)
    responded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='responded_revisions',
        verbose_name=_('responded by')
    )
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    completed_at = models.DateTimeField(_('completed at'), null=True, blank=True)
    
    class Meta:
        verbose_name = _('pre-order revision')
        verbose_name_plural = _('pre-order revisions')
        ordering = ['-created_at']
        unique_together = ['preorder', 'revision_number']
    
    def __str__(self):
        return f"{self.preorder.preorder_number} - Revision {self.revision_number}"


class PreOrderQuote(models.Model):
    """
    Store quote history for a pre-order (if multiple quotes are generated).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    preorder = models.ForeignKey(
        PreOrder,
        on_delete=models.CASCADE,
        related_name='quotes',
        verbose_name=_('pre-order')
    )
    
    # Quote number
    quote_number = models.CharField(_('quote number'), max_length=50, unique=True)
    
    # Pricing breakdown
    base_price = models.DecimalField(_('base price'), max_digits=12, decimal_places=2)
    customization_cost = models.DecimalField(
        _('customization cost'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    rush_fee = models.DecimalField(
        _('rush fee'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    discount = models.DecimalField(
        _('discount'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    shipping = models.DecimalField(
        _('shipping'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    tax = models.DecimalField(
        _('tax'),
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    total = models.DecimalField(_('total'), max_digits=12, decimal_places=2)
    
    # Validity
    valid_from = models.DateTimeField(_('valid from'), default=timezone.now)
    valid_until = models.DateTimeField(_('valid until'))
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_SENT = 'sent'
    STATUS_VIEWED = 'viewed'
    STATUS_ACCEPTED = 'accepted'
    STATUS_REJECTED = 'rejected'
    STATUS_EXPIRED = 'expired'
    STATUS_SUPERSEDED = 'superseded'
    STATUS_CHOICES = [
        (STATUS_PENDING, _('Pending')),
        (STATUS_SENT, _('Sent')),
        (STATUS_VIEWED, _('Viewed')),
        (STATUS_ACCEPTED, _('Accepted')),
        (STATUS_REJECTED, _('Rejected')),
        (STATUS_EXPIRED, _('Expired')),
        (STATUS_SUPERSEDED, _('Superseded')),
    ]
    status = models.CharField(
        _('status'),
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
    # Terms and notes
    terms = models.TextField(_('terms and conditions'), blank=True)
    notes = models.TextField(_('notes'), blank=True)
    
    # Production timeline
    estimated_production_days = models.PositiveIntegerField(
        _('estimated production days'),
        default=14
    )
    estimated_delivery_date = models.DateField(
        _('estimated delivery date'),
        null=True,
        blank=True
    )
    
    # Created by
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_quotes',
        verbose_name=_('created by')
    )
    
    # Customer response
    customer_response_notes = models.TextField(_('customer response notes'), blank=True)
    responded_at = models.DateTimeField(_('responded at'), null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    sent_at = models.DateTimeField(_('sent at'), null=True, blank=True)
    
    class Meta:
        verbose_name = _('pre-order quote')
        verbose_name_plural = _('pre-order quotes')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.quote_number} - {self.preorder.preorder_number}"
    
    def save(self, *args, **kwargs):
        if not self.quote_number:
            self.quote_number = self.generate_quote_number()
        super().save(*args, **kwargs)
    
    def generate_quote_number(self):
        """Generate unique quote number."""
        import random
        prefix = "QUO"
        date_str = timezone.now().strftime("%Y%m%d")
        random_str = ''.join([str(random.randint(0, 9)) for _ in range(4)])
        return f"{prefix}-{date_str}-{random_str}"
    
    @property
    def is_expired(self):
        """Check if the quote has expired."""
        return timezone.now() > self.valid_until


class PreOrderTemplate(models.Model):
    """
    Templates for common pre-order types to speed up ordering.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    name = models.CharField(_('template name'), max_length=200)
    slug = models.SlugField(_('slug'), max_length=220, unique=True)
    description = models.TextField(_('description'))
    
    # Category
    category = models.ForeignKey(
        PreOrderCategory,
        on_delete=models.CASCADE,
        related_name='templates',
        verbose_name=_('category')
    )
    
    # Template image
    image = models.ImageField(
        _('template image'),
        upload_to='preorder_templates/',
        blank=True,
        null=True
    )
    
    # Default values
    default_quantity = models.PositiveIntegerField(_('default quantity'), default=1)
    base_price = models.DecimalField(
        _('base price'),
        max_digits=12,
        decimal_places=2,
        default=Decimal('0.00')
    )
    estimated_days = models.PositiveIntegerField(_('estimated production days'), default=14)
    
    # Pre-filled options (JSON)
    default_options = models.JSONField(
        _('default options'),
        default=dict,
        blank=True,
        help_text=_('Default option values as JSON')
    )
    
    # Status
    is_active = models.BooleanField(_('active'), default=True)
    is_featured = models.BooleanField(_('featured'), default=False)
    order = models.PositiveIntegerField(_('display order'), default=0)
    
    # Statistics
    use_count = models.PositiveIntegerField(_('times used'), default=0)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('pre-order template')
        verbose_name_plural = _('pre-order templates')
        ordering = ['order', 'name']
    
    def __str__(self):
        return self.name
