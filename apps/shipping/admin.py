"""
Shipping Admin Configuration
"""
from django.contrib import admin
from .models import (
    ShippingZone, ShippingCarrier, ShippingMethod, ShippingRate,
    ShippingRestriction, Shipment, ShipmentEvent, ShippingSettings
)


@admin.register(ShippingZone)
class ShippingZoneAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active', 'is_default', 'priority', 'country_count', 'created_at']
    list_filter = ['is_active', 'is_default']
    search_fields = ['name', 'description']
    ordering = ['-priority', 'name']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'description')
        }),
        ('Geographic Coverage', {
            'fields': ('countries', 'states', 'cities', 'postal_codes'),
            'description': 'Define the geographic areas this zone covers.'
        }),
        ('Settings', {
            'fields': ('is_active', 'is_default', 'priority')
        }),
    )
    
    def country_count(self, obj):
        return len(obj.countries)
    country_count.short_description = 'Countries'


@admin.register(ShippingCarrier)
class ShippingCarrierAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'is_active', 'api_enabled', 'supports_tracking']
    list_filter = ['is_active', 'api_enabled', 'supports_tracking', 'supports_label_generation']
    search_fields = ['name', 'code']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'code', 'logo', 'website')
        }),
        ('Tracking', {
            'fields': ('tracking_url_template',),
            'description': 'Use {tracking_number} as placeholder in URL template.'
        }),
        ('API Configuration', {
            'fields': ('api_enabled', 'api_key', 'api_secret', 'api_account_number', 
                      'api_endpoint', 'api_sandbox'),
            'classes': ('collapse',)
        }),
        ('Capabilities', {
            'fields': ('is_active', 'supports_real_time_rates', 'supports_tracking', 
                      'supports_label_generation')
        }),
    )


@admin.register(ShippingMethod)
class ShippingMethodAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'carrier', 'delivery_estimate', 'is_active', 'is_express', 'sort_order']
    list_filter = ['is_active', 'is_express', 'carrier']
    search_fields = ['name', 'code']
    ordering = ['sort_order', 'name']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'code', 'description')
        }),
        ('Carrier', {
            'fields': ('carrier', 'carrier_service_code')
        }),
        ('Delivery Time', {
            'fields': ('min_delivery_days', 'max_delivery_days', 'delivery_time_text')
        }),
        ('Settings', {
            'fields': ('is_active', 'requires_signature', 'is_express', 'sort_order')
        }),
        ('Restrictions', {
            'fields': ('max_weight', 'max_dimensions'),
            'classes': ('collapse',)
        }),
    )


class ShippingRateInline(admin.TabularInline):
    model = ShippingRate
    extra = 1
    fields = ['method', 'rate_type', 'base_rate', 'per_kg_rate', 'currency', 'free_shipping_threshold', 'is_active']


@admin.register(ShippingRate)
class ShippingRateAdmin(admin.ModelAdmin):
    list_display = ['zone', 'method', 'rate_type', 'base_rate', 'currency', 'free_shipping_threshold', 'is_active']
    list_filter = ['zone', 'method', 'rate_type', 'is_active']
    search_fields = ['zone__name', 'method__name']
    
    fieldsets = (
        (None, {
            'fields': ('zone', 'method', 'rate_type')
        }),
        ('Pricing', {
            'fields': ('base_rate', 'per_kg_rate', 'per_item_rate', 'currency')
        }),
        ('Free Shipping', {
            'fields': ('free_shipping_threshold',)
        }),
        ('Tiers', {
            'fields': ('weight_tiers', 'price_tiers'),
            'classes': ('collapse',),
            'description': 'Configure tiered pricing for weight or price based rates.'
        }),
        ('Settings', {
            'fields': ('is_active',)
        }),
    )


@admin.register(ShippingRestriction)
class ShippingRestrictionAdmin(admin.ModelAdmin):
    list_display = ['restriction_type', 'zone', 'method', 'action', 'is_active']
    list_filter = ['restriction_type', 'action', 'is_active']
    search_fields = ['message']


class ShipmentEventInline(admin.TabularInline):
    model = ShipmentEvent
    extra = 0
    readonly_fields = ['status', 'description', 'location', 'occurred_at']
    can_delete = False
    ordering = ['-occurred_at']


@admin.register(Shipment)
class ShipmentAdmin(admin.ModelAdmin):
    list_display = ['id', 'order', 'carrier', 'tracking_number', 'status', 'shipped_at', 'delivered_at']
    list_filter = ['status', 'carrier']
    search_fields = ['tracking_number', 'order__order_number']
    readonly_fields = ['created_at', 'updated_at']
    inlines = [ShipmentEventInline]
    
    fieldsets = (
        ('Order', {
            'fields': ('order',)
        }),
        ('Shipping', {
            'fields': ('carrier', 'method', 'status')
        }),
        ('Tracking', {
            'fields': ('tracking_number', 'tracking_url')
        }),
        ('Package Details', {
            'fields': ('weight', 'dimensions', 'shipping_cost')
        }),
        ('Label', {
            'fields': ('label_url', 'label_format'),
            'classes': ('collapse',)
        }),
        ('Dates', {
            'fields': ('shipped_at', 'estimated_delivery', 'delivered_at')
        }),
        ('Signature', {
            'fields': ('signature_required', 'signed_by', 'signature_image'),
            'classes': ('collapse',)
        }),
        ('Notes', {
            'fields': ('notes',),
            'classes': ('collapse',)
        }),
    )


@admin.register(ShippingSettings)
class ShippingSettingsAdmin(admin.ModelAdmin):
    fieldsets = (
        ('Origin Address', {
            'fields': ('origin_address_line1', 'origin_address_line2', 'origin_city',
                      'origin_state', 'origin_postal_code', 'origin_country', 'origin_phone')
        }),
        ('Units', {
            'fields': ('default_weight_unit', 'default_dimension_unit', 'default_package_weight')
        }),
        ('Display', {
            'fields': ('show_delivery_estimates', 'show_carrier_logos')
        }),
        ('Free Shipping', {
            'fields': ('enable_free_shipping', 'free_shipping_threshold', 'free_shipping_countries')
        }),
        ('Processing', {
            'fields': ('handling_days', 'cutoff_time')
        }),
    )
    
    def has_add_permission(self, request):
        return not ShippingSettings.objects.exists()
    
    def has_delete_permission(self, request, obj=None):
        return False
