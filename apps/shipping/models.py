"""
Shipping Models
Comprehensive shipping system with zones, methods, rates, and carrier integration.
"""
import uuid
from decimal import Decimal
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator


class ShippingZone(models.Model):
    """
    Shipping zone representing a geographic region.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    
    # Zone can be based on countries, states, cities, or postal codes
    countries = models.JSONField(
        default=list,
        help_text="List of ISO country codes (e.g., ['US', 'BD', 'GB'])"
    )
    states = models.JSONField(
        default=list,
        help_text="List of state/division names (e.g., ['Dhaka', 'Rangpur'])"
    )
    cities = models.JSONField(
        default=list,
        help_text="List of city/district names (e.g., ['Dhaka', 'Mirpur', 'Gulshan'])"
    )
    postal_codes = models.JSONField(
        default=list,
        help_text="List of postal code patterns (e.g., ['1200', '1205', '56*'])"
    )
    
    # Settings
    is_active = models.BooleanField(default=True)
    is_default = models.BooleanField(
        default=False,
        help_text="Default zone for unmatched locations"
    )
    priority = models.PositiveIntegerField(
        default=0,
        help_text="Higher priority zones are matched first"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-priority', 'name']
        verbose_name = 'Shipping Zone'
        verbose_name_plural = 'Shipping Zones'
    
    def __str__(self):
        return self.name
    
    def matches_location(self, country=None, state=None, city=None, postal_code=None):
        """
        Check if a location matches this zone. 
        Postal code is the PRIMARY matching key - if provided and matches, that's definitive.
        Fallback hierarchy: postal_code > city > state > country.
        """
        # If this is a default zone, match if country matches (or any location for catch-all)
        if self.is_default:
            if self.countries and country:
                country_upper = country.upper().strip() if isinstance(country, str) else country
                return country_upper in self.countries or country in self.countries
            return True
        
        # POSTAL CODE IS PRIMARY - if provided, use it as the main matching criterion
        if postal_code:
            postal_code_str = str(postal_code).strip()
            
            # If zone has postal codes defined, check if this postal code matches
            if self.postal_codes:
                for pattern in self.postal_codes:
                    pattern_str = str(pattern).strip()
                    if pattern_str.endswith('*'):
                        # Wildcard match (e.g., "56*" matches "5621")
                        if postal_code_str.startswith(pattern_str[:-1]):
                            return True
                    elif postal_code_str == pattern_str:
                        # Exact match
                        return True
                # Postal code doesn't match this zone's patterns
                return False
            else:
                # Zone has no postal codes defined - can't match by postal code
                # Fall through to check other criteria, but this zone is unlikely to match
                pass
        
        # If no postal code provided, fall back to city/state matching
        # Check cities/districts - if provided and zone has cities
        if city and self.cities:
            city_lower = city.lower().strip()
            for zone_city in self.cities:
                zone_city_lower = zone_city.lower().strip()
                if city_lower == zone_city_lower:
                    return True
                # Also check if zone city is contained in the provided city
                if zone_city_lower in city_lower:
                    return True
            # Zone has cities but none matched
            return False
        
        # Check states/divisions - if provided and zone has states
        if state and self.states:
            state_lower = state.lower().strip()
            for zone_state in self.states:
                zone_state_lower = zone_state.lower().strip()
                if state_lower == zone_state_lower:
                    return True
                # Also allow partial match (e.g., "Dhaka" matches "Dhaka Division")
                if zone_state_lower in state_lower or state_lower in zone_state_lower:
                    return True
            return False
        
        # Check countries (least specific) - only if no other criteria defined
        if country and self.countries:
            country_upper = country.upper().strip() if isinstance(country, str) else country
            if country_upper in self.countries or country in self.countries:
                if not self.postal_codes and not self.cities and not self.states:
                    return True
        
        return False


class ShippingCarrier(models.Model):
    """
    Shipping carrier (e.g., UPS, FedEx, USPS, DHL).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=50, unique=True)
    logo = models.ImageField(upload_to='carriers/', blank=True, null=True)
    website = models.URLField(blank=True)
    tracking_url_template = models.CharField(
        max_length=500,
        blank=True,
        help_text="URL template for tracking. Use {tracking_number} as placeholder."
    )
    
    # API Configuration
    api_enabled = models.BooleanField(default=False)
    api_key = models.CharField(max_length=255, blank=True)
    api_secret = models.CharField(max_length=255, blank=True)
    api_account_number = models.CharField(max_length=100, blank=True)
    api_endpoint = models.URLField(blank=True)
    api_sandbox = models.BooleanField(
        default=True,
        help_text="Use sandbox/test mode"
    )
    
    # Settings
    is_active = models.BooleanField(default=True)
    supports_real_time_rates = models.BooleanField(default=False)
    supports_tracking = models.BooleanField(default=True)
    supports_label_generation = models.BooleanField(default=False)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = 'Shipping Carrier'
        verbose_name_plural = 'Shipping Carriers'
    
    def __str__(self):
        return self.name
    
    def get_tracking_url(self, tracking_number):
        """Get the tracking URL for a shipment."""
        if self.tracking_url_template and tracking_number:
            return self.tracking_url_template.replace('{tracking_number}', tracking_number)
        return None


class ShippingMethod(models.Model):
    """
    Shipping method (e.g., Standard, Express, Overnight).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)


    
    # Carrier association
    carrier = models.ForeignKey(
        ShippingCarrier,
        on_delete=models.SET_NULL,
        related_name='methods',
        null=True,
        blank=True
    )
    carrier_service_code = models.CharField(
        max_length=50,
        blank=True,
        help_text="Carrier-specific service code for API calls"
    )
    
    # Delivery time
    min_delivery_days = models.PositiveIntegerField(default=3)
    max_delivery_days = models.PositiveIntegerField(default=7)
    delivery_time_text = models.CharField(
        max_length=100,
        blank=True,
        help_text="Display text (e.g., '3-7 business days')"
    )
    
    # Settings
    is_active = models.BooleanField(default=True)
    requires_signature = models.BooleanField(default=False)
    is_express = models.BooleanField(default=False)
    sort_order = models.PositiveIntegerField(default=0)
    
    # Restrictions
    max_weight = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Maximum weight in kg"
    )
    max_dimensions = models.JSONField(
        default=dict,
        blank=True,
        help_text="Max dimensions: {length, width, height} in cm"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'name']
        verbose_name = 'Shipping Method'
        verbose_name_plural = 'Shipping Methods'
    
    def __str__(self):
        return self.name
    
    @property
    def delivery_estimate(self):
        """Get delivery estimate text."""
        if self.delivery_time_text:
            return self.delivery_time_text
        if self.min_delivery_days == self.max_delivery_days:
            return f"{self.min_delivery_days} business day{'s' if self.min_delivery_days > 1 else ''}"
        return f"{self.min_delivery_days}-{self.max_delivery_days} business days"


class ShippingRate(models.Model):
    """
    Shipping rate configuration for zone/method combinations.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Zone and method association
    zone = models.ForeignKey(
        ShippingZone,
        on_delete=models.CASCADE,
        related_name='rates'
    )
    method = models.ForeignKey(
        ShippingMethod,
        on_delete=models.CASCADE,
        related_name='rates'
    )
    
    # Rate type
    RATE_TYPE_FLAT = 'flat'
    RATE_TYPE_WEIGHT = 'weight'
    RATE_TYPE_PRICE = 'price'
    RATE_TYPE_ITEM = 'item'
    RATE_TYPE_FREE = 'free'
    RATE_TYPE_CHOICES = [
        (RATE_TYPE_FLAT, 'Flat Rate'),
        (RATE_TYPE_WEIGHT, 'Weight Based'),
        (RATE_TYPE_PRICE, 'Price Based'),
        (RATE_TYPE_ITEM, 'Per Item'),
        (RATE_TYPE_FREE, 'Free Shipping'),
    ]
    rate_type = models.CharField(
        max_length=20,
        choices=RATE_TYPE_CHOICES,
        default=RATE_TYPE_FLAT
    )
    
    # Pricing
    base_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00'),
        help_text="Base shipping cost"
    )
    per_kg_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00'),
        help_text="Additional cost per kg (for weight-based)"
    )
    per_item_rate = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00'),
        help_text="Additional cost per item (for item-based)"
    )
    
    # Conditions for free shipping
    free_shipping_threshold = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Order subtotal for free shipping"
    )
    
    # Weight tiers (for weight-based rates)
    weight_tiers = models.JSONField(
        default=list,
        blank=True,
        help_text="Weight tiers: [{max_weight, rate}, ...]"
    )
    
    # Price tiers (for price-based rates)
    price_tiers = models.JSONField(
        default=list,
        blank=True,
        help_text="Price tiers: [{max_price, rate}, ...]"
    )
    
    # Settings
    is_active = models.BooleanField(default=True)
    # Currency for this shipping rate (location-based). Use a FK to Currency for consistency.
    currency = models.ForeignKey(
        'i18n.Currency',
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name='shipping_rates'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['zone', 'method', 'base_rate']
        unique_together = ['zone', 'method']
        verbose_name = 'Shipping Rate'
        verbose_name_plural = 'Shipping Rates'
    
    def __str__(self):
        return f"{self.zone.name} - {self.method.name}: {self.base_rate}"
    
    def calculate_rate(self, subtotal=Decimal('0'), weight=Decimal('0'), item_count=1):
        """Calculate shipping rate based on type."""
        # Check global free shipping settings first
        try:
            from .models import ShippingSettings as _SS
            s = _SS.get_settings()
            if getattr(s, 'enable_free_shipping', False) and getattr(s, 'free_shipping_threshold', None) is not None:
                if s.free_shipping_threshold and subtotal >= s.free_shipping_threshold:
                    return Decimal('0.00')
        except Exception:
            pass

        # Check rate-specific free shipping threshold
        if self.free_shipping_threshold and subtotal >= self.free_shipping_threshold:
            return Decimal('0.00')
        
        if self.rate_type == self.RATE_TYPE_FREE:
            return Decimal('0.00')
        
        if self.rate_type == self.RATE_TYPE_FLAT:
            return self.base_rate
        
        if self.rate_type == self.RATE_TYPE_WEIGHT:
            # Check weight tiers
            if self.weight_tiers:
                for tier in sorted(self.weight_tiers, key=lambda x: x.get('max_weight', 0)):
                    if weight <= Decimal(str(tier.get('max_weight', 0))):
                        return Decimal(str(tier.get('rate', 0)))
            # Use per kg rate
            return self.base_rate + (weight * self.per_kg_rate)
        
        if self.rate_type == self.RATE_TYPE_PRICE:
            # Check price tiers
            if self.price_tiers:
                for tier in sorted(self.price_tiers, key=lambda x: x.get('max_price', 0)):
                    if subtotal <= Decimal(str(tier.get('max_price', 0))):
                        return Decimal(str(tier.get('rate', 0)))
            return self.base_rate
        
        if self.rate_type == self.RATE_TYPE_ITEM:
            return self.base_rate + (item_count * self.per_item_rate)
        
        return self.base_rate


class ShippingRestriction(models.Model):
    """
    Shipping restrictions for products or zones.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Can restrict by zone, method, or product
    zone = models.ForeignKey(
        ShippingZone,
        on_delete=models.CASCADE,
        related_name='restrictions',
        null=True,
        blank=True
    )
    method = models.ForeignKey(
        ShippingMethod,
        on_delete=models.CASCADE,
        related_name='restrictions',
        null=True,
        blank=True
    )
    
    # Restriction type
    RESTRICTION_PRODUCT = 'product'
    RESTRICTION_CATEGORY = 'category'
    RESTRICTION_HAZMAT = 'hazmat'
    RESTRICTION_OVERSIZED = 'oversized'
    RESTRICTION_FRAGILE = 'fragile'
    RESTRICTION_TYPE_CHOICES = [
        (RESTRICTION_PRODUCT, 'Specific Product'),
        (RESTRICTION_CATEGORY, 'Category'),
        (RESTRICTION_HAZMAT, 'Hazardous Materials'),
        (RESTRICTION_OVERSIZED, 'Oversized Items'),
        (RESTRICTION_FRAGILE, 'Fragile Items'),
    ]
    restriction_type = models.CharField(
        max_length=20,
        choices=RESTRICTION_TYPE_CHOICES
    )
    
    # Reference IDs
    product_ids = models.JSONField(default=list, blank=True)
    category_ids = models.JSONField(default=list, blank=True)
    
    # Action
    ACTION_BLOCK = 'block'
    ACTION_SURCHARGE = 'surcharge'
    ACTION_REQUIRE_SIGNATURE = 'signature'
    ACTION_CHOICES = [
        (ACTION_BLOCK, 'Block Shipping'),
        (ACTION_SURCHARGE, 'Add Surcharge'),
        (ACTION_REQUIRE_SIGNATURE, 'Require Signature'),
    ]
    action = models.CharField(
        max_length=20,
        choices=ACTION_CHOICES,
        default=ACTION_BLOCK
    )
    surcharge_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    
    message = models.TextField(
        blank=True,
        help_text="Message to display when restriction applies"
    )
    is_active = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Shipping Restriction'
        verbose_name_plural = 'Shipping Restrictions'
    
    def __str__(self):
        return f"{self.get_restriction_type_display()} - {self.get_action_display()}"


class Shipment(models.Model):
    """
    Shipment tracking for orders.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Order association
    order = models.ForeignKey(
        'orders.Order',
        on_delete=models.CASCADE,
        related_name='shipments'
    )
    
    # Carrier and method
    carrier = models.ForeignKey(
        ShippingCarrier,
        on_delete=models.SET_NULL,
        null=True
    )
    method = models.ForeignKey(
        ShippingMethod,
        on_delete=models.SET_NULL,
        null=True
    )
    
    # Tracking
    tracking_number = models.CharField(max_length=100, blank=True)
    tracking_url = models.URLField(blank=True)
    
    # Status
    STATUS_PENDING = 'pending'
    STATUS_LABEL_CREATED = 'label_created'
    STATUS_PICKED_UP = 'picked_up'
    STATUS_IN_TRANSIT = 'in_transit'
    STATUS_OUT_FOR_DELIVERY = 'out_for_delivery'
    STATUS_DELIVERED = 'delivered'
    STATUS_EXCEPTION = 'exception'
    STATUS_RETURNED = 'returned'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_LABEL_CREATED, 'Label Created'),
        (STATUS_PICKED_UP, 'Picked Up'),
        (STATUS_IN_TRANSIT, 'In Transit'),
        (STATUS_OUT_FOR_DELIVERY, 'Out for Delivery'),
        (STATUS_DELIVERED, 'Delivered'),
        (STATUS_EXCEPTION, 'Exception'),
        (STATUS_RETURNED, 'Returned'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    
    # Shipping details
    weight = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True
    )
    dimensions = models.JSONField(
        default=dict,
        blank=True,
        help_text="{length, width, height} in cm"
    )
    shipping_cost = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00')
    )
    
    # Label
    label_url = models.URLField(blank=True)
    label_format = models.CharField(max_length=10, blank=True)  # PDF, ZPL, PNG
    
    # Dates
    shipped_at = models.DateTimeField(null=True, blank=True)
    estimated_delivery = models.DateField(null=True, blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    
    # Signature
    signature_required = models.BooleanField(default=False)
    signature_image = models.ImageField(
        upload_to='signatures/',
        blank=True,
        null=True
    )
    signed_by = models.CharField(max_length=100, blank=True)
    
    # Notes
    notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Shipment'
        verbose_name_plural = 'Shipments'
    
    def __str__(self):
        return f"Shipment {self.tracking_number or self.id}"
    
    def save(self, *args, **kwargs):
        # Auto-generate tracking URL
        if self.tracking_number and self.carrier and not self.tracking_url:
            self.tracking_url = self.carrier.get_tracking_url(self.tracking_number) or ''
        super().save(*args, **kwargs)


class ShipmentEvent(models.Model):
    """
    Tracking events for a shipment.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    shipment = models.ForeignKey(
        Shipment,
        on_delete=models.CASCADE,
        related_name='events'
    )
    
    # Event details
    status = models.CharField(max_length=50)
    description = models.TextField()
    location = models.CharField(max_length=255, blank=True)
    
    # Timestamps
    occurred_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-occurred_at']
        verbose_name = 'Shipment Event'
        verbose_name_plural = 'Shipment Events'
    
    def __str__(self):
        return f"{self.status} - {self.occurred_at}"


class ShippingSettings(models.Model):
    """
    Global shipping settings (singleton).
    """
    # Origin address
    origin_address_line1 = models.CharField(max_length=255)
    origin_address_line2 = models.CharField(max_length=255, blank=True)
    origin_city = models.CharField(max_length=100)
    origin_state = models.CharField(max_length=100)
    origin_postal_code = models.CharField(max_length=20)
    origin_country = models.CharField(max_length=2, default='US')
    origin_phone = models.CharField(max_length=20, blank=True)
    
    # Default settings
    default_weight_unit = models.CharField(
        max_length=5,
        choices=[('kg', 'Kilograms'), ('lb', 'Pounds')],
        default='kg'
    )
    default_dimension_unit = models.CharField(
        max_length=5,
        choices=[('cm', 'Centimeters'), ('in', 'Inches')],
        default='cm'
    )
    default_package_weight = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=Decimal('0.50'),
        help_text="Default package weight when not specified"
    )
    
    # Display settings
    show_delivery_estimates = models.BooleanField(default=True)
    show_carrier_logos = models.BooleanField(default=True)
    
    # Free shipping
    enable_free_shipping = models.BooleanField(default=False)
    free_shipping_threshold = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True
    )
    free_shipping_countries = models.JSONField(
        default=list,
        blank=True,
        help_text="Countries eligible for free shipping"
    )
    
    # Processing time
    handling_days = models.PositiveIntegerField(
        default=1,
        help_text="Business days to process order before shipping"
    )
    cutoff_time = models.TimeField(
        null=True,
        blank=True,
        help_text="Orders after this time processed next business day"
    )
    
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Shipping Settings'
        verbose_name_plural = 'Shipping Settings'
    
    def __str__(self):
        return "Shipping Settings"
    
    @classmethod
    def get_settings(cls):
        """Get or create singleton settings."""
        settings, _ = cls.objects.get_or_create(pk=1)
        return settings
    
    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)
