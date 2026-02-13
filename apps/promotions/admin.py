"""
Promotions admin configuration
"""
from django.contrib import admin
from .models import Coupon, CouponUsage, Banner, Sale
from core.admin_mixins import ImportExportEnhancedModelAdmin


@admin.register(Coupon)
class CouponAdmin(ImportExportEnhancedModelAdmin):
    list_display = [
        'code', 'discount_type', 'discount_value', 'is_valid',
        'times_used', 'usage_limit', 'valid_from', 'valid_until', 'is_active'
    ]
    list_filter = ['discount_type', 'is_active', 'first_order_only', 'created_at']
    search_fields = ['code', 'description']
    filter_horizontal = ['categories', 'products', 'users']
    readonly_fields = ['times_used', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('code', 'description')
        }),
        ('Discount', {
            'fields': ('discount_type', 'discount_value', 'maximum_discount')
        }),
        ('Requirements', {
            'fields': ('minimum_order_amount', 'first_order_only')
        }),
        ('Usage Limits', {
            'fields': ('usage_limit', 'usage_limit_per_user', 'times_used')
        }),
        ('Validity', {
            'fields': ('valid_from', 'valid_until', 'is_active')
        }),
        ('Restrictions', {
            'fields': ('categories', 'products', 'users'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(CouponUsage)
class CouponUsageAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['coupon', 'user', 'order', 'discount_applied', 'created_at']
    list_filter = ['created_at']
    search_fields = ['coupon__code', 'user__email', 'order__order_number']
    readonly_fields = ['coupon', 'user', 'order', 'discount_applied', 'created_at']


@admin.register(Banner)
class BannerAdmin(ImportExportEnhancedModelAdmin):
    list_display = [
        'title', 'position', 'is_visible', 'sort_order',
        'start_date', 'end_date', 'is_active'
    ]
    list_filter = ['position', 'is_active', 'created_at']
    search_fields = ['title', 'subtitle']
    list_editable = ['sort_order', 'is_active']
    ordering = ['position', 'sort_order']
    
    fieldsets = (
        ('Content', {
            'fields': ('title', 'subtitle')
        }),
        ('Images', {
            'fields': ('image', 'image_mobile')
        }),
        ('Link', {
            'fields': ('link_url', 'link_text')
        }),
        ('Style', {
            'fields': (
                'style_height', 'style_width', 'style_max_width',
                'style_border_radius', 'style_border_width',
                'style_border_color', 'style_background_color',
                'overlay_color', 'overlay_opacity', 'text_color'
            ),
            'classes': ('collapse',),
        }),
        ('Display', {
            'fields': ('position', 'sort_order')
        }),
        ('Validity', {
            'fields': ('start_date', 'end_date', 'is_active')
        }),
    )


@admin.register(Sale)
class SaleAdmin(ImportExportEnhancedModelAdmin):
    list_display = [
        'name', 'slug', 'discount_type', 'discount_value',
        'is_running', 'start_date', 'end_date', 'is_active'
    ]
    list_filter = ['discount_type', 'is_active', 'start_date']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    filter_horizontal = ['products', 'categories']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'slug', 'description')
        }),
        ('Discount', {
            'fields': ('discount_type', 'discount_value')
        }),
        ('Products', {
            'fields': ('products', 'categories')
        }),
        ('Banner', {
            'fields': ('banner_image',)
        }),
        ('Schedule', {
            'fields': ('start_date', 'end_date', 'is_active')
        }),
    )
