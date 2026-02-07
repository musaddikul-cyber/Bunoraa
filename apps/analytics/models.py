"""
Analytics models
"""
import uuid
from django.db import models
from django.conf import settings


class PageView(models.Model):
    """Track page views."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='page_views'
    )
    session_key = models.CharField(max_length=40, blank=True, null=True)
    
    # Page info
    path = models.CharField(max_length=500)
    query_string = models.CharField(max_length=1000, blank=True, null=True)
    referrer = models.URLField(max_length=1000, blank=True, null=True)
    
    # Device info
    user_agent = models.CharField(max_length=500, blank=True, null=True)
    device_type = models.CharField(max_length=20, blank=True, null=True)  # mobile, tablet, desktop
    browser = models.CharField(max_length=50, blank=True, null=True)
    os = models.CharField(max_length=50, blank=True, null=True)
    
    # Location
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    
    # Timing
    time_on_page = models.IntegerField(default=0, help_text='Time in seconds')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['path', '-created_at']),
            models.Index(fields=['session_key']),
        ]
    
    def __str__(self):
        return f"{self.path} - {self.created_at}"


class ProductView(models.Model):
    """Track product views."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='analytics_views'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    session_key = models.CharField(max_length=40, blank=True, null=True)
    
    # Source
    source = models.CharField(max_length=50, blank=True, null=True)  # direct, search, category, related
    referrer = models.URLField(max_length=1000, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['product', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.product.name} view - {self.created_at}"


class SearchQuery(models.Model):
    """Track search queries."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query = models.CharField(max_length=500)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    session_key = models.CharField(max_length=40, blank=True, null=True)
    
    results_count = models.IntegerField(default=0)
    clicked_product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='search_clicks'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Search Queries'
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['query']),
        ]
    
    def __str__(self):
        return f'"{self.query}" - {self.created_at}'


class CartEvent(models.Model):
    """Track cart events."""
    EVENT_ADD = 'add'
    EVENT_REMOVE = 'remove'
    EVENT_UPDATE = 'update'
    EVENT_CHECKOUT_START = 'checkout_start'
    EVENT_CHECKOUT_COMPLETE = 'checkout_complete'
    EVENT_ABANDONED = 'abandoned'
    EVENT_CHOICES = [
        (EVENT_ADD, 'Add to Cart'),
        (EVENT_REMOVE, 'Remove from Cart'),
        (EVENT_UPDATE, 'Update Quantity'),
        (EVENT_CHECKOUT_START, 'Start Checkout'),
        (EVENT_CHECKOUT_COMPLETE, 'Complete Checkout'),
        (EVENT_ABANDONED, 'Abandoned Cart'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    event_type = models.CharField(max_length=20, choices=EVENT_CHOICES)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    session_key = models.CharField(max_length=40, blank=True, null=True)
    
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    quantity = models.IntegerField(default=1)
    cart_value = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['event_type', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.event_type} - {self.created_at}"


class DailyStat(models.Model):
    """Daily aggregated statistics."""
    date = models.DateField(unique=True)
    
    # Traffic
    page_views = models.IntegerField(default=0)
    unique_visitors = models.IntegerField(default=0)
    new_visitors = models.IntegerField(default=0)
    returning_visitors = models.IntegerField(default=0)
    
    # Products
    product_views = models.IntegerField(default=0)
    products_added_to_cart = models.IntegerField(default=0)
    
    # Orders
    orders_count = models.IntegerField(default=0)
    orders_revenue = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    average_order_value = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Conversion
    checkout_starts = models.IntegerField(default=0)
    checkout_completions = models.IntegerField(default=0)
    conversion_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    cart_abandonment_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    
    # Users
    new_registrations = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date']
        verbose_name_plural = 'Daily Stats'
    
    def __str__(self):
        return f"Stats for {self.date}"


class ProductStat(models.Model):
    """Product-level statistics."""
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='stats'
    )
    date = models.DateField()
    
    views = models.IntegerField(default=0)
    add_to_cart_count = models.IntegerField(default=0)
    orders_count = models.IntegerField(default=0)
    revenue = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    conversion_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['product', 'date']
        verbose_name_plural = 'Product Stats'
    
    def __str__(self):
        return f"{self.product.name} - {self.date}"


class CategoryStat(models.Model):
    """Category-level statistics."""
    category = models.ForeignKey(
        'catalog.Category',
        on_delete=models.CASCADE,
        related_name='stats'
    )
    date = models.DateField()
    
    views = models.IntegerField(default=0)
    product_views = models.IntegerField(default=0)
    orders_count = models.IntegerField(default=0)
    revenue = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['category', 'date']
        verbose_name_plural = 'Category Stats'
    
    def __str__(self):
        return f"{self.category.name} - {self.date}"
