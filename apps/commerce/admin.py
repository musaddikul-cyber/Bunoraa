"""
Commerce Admin Configuration
"""
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse

from .models import (
    Cart, CartItem, CartSettings,
    Wishlist, WishlistItem, WishlistShare,
    CheckoutSession, CheckoutEvent
)


# =============================================================================
# Inline Admins
# =============================================================================

class CartItemInline(admin.TabularInline):
    model = CartItem
    extra = 0
    readonly_fields = ['product', 'variant', 'quantity', 'price_at_add', 'unit_price', 'total', 'created_at']
    can_delete = True
    
    def unit_price(self, obj):
        return obj.unit_price
    
    def total(self, obj):
        return obj.total


class WishlistItemInline(admin.TabularInline):
    model = WishlistItem
    extra = 0
    readonly_fields = ['product', 'variant', 'price_at_add', 'added_at']
    can_delete = True


class CheckoutEventInline(admin.TabularInline):
    model = CheckoutEvent
    extra = 0
    readonly_fields = ['event_type', 'data', 'created_at']
    can_delete = False


# =============================================================================
# Cart Admin
# =============================================================================

@admin.register(Cart)
class CartAdmin(admin.ModelAdmin):
    list_display = ['id', 'user_display', 'session_key_short', 'item_count', 'total', 'coupon', 'updated_at']
    list_filter = ['updated_at', 'created_at']
    search_fields = ['user__email', 'session_key']
    readonly_fields = ['created_at', 'updated_at']
    inlines = [CartItemInline]
    
    def user_display(self, obj):
        return obj.user.email if obj.user else 'Guest'
    user_display.short_description = 'User'
    
    def session_key_short(self, obj):
        return obj.session_key[:8] + '...' if obj.session_key else '-'
    session_key_short.short_description = 'Session'


@admin.register(CartSettings)
class CartSettingsAdmin(admin.ModelAdmin):
    list_display = ['gift_wrap_enabled', 'gift_wrap_amount', 'cart_expiry_days', 'updated_at']
    fields = ['gift_wrap_enabled', 'gift_wrap_amount', 'gift_wrap_label', 'cart_expiry_days']


# =============================================================================
# Wishlist Admin
# =============================================================================

@admin.register(Wishlist)
class WishlistAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'item_count', 'updated_at']
    list_filter = ['updated_at', 'created_at']
    search_fields = ['user__email']
    readonly_fields = ['created_at', 'updated_at']
    inlines = [WishlistItemInline]


@admin.register(WishlistShare)
class WishlistShareAdmin(admin.ModelAdmin):
    list_display = ['id', 'wishlist', 'share_token_short', 'view_count', 'is_public', 'allow_comments', 'created_at']
    list_filter = ['is_public', 'allow_comments', 'created_at']
    search_fields = ['wishlist__user__email', 'share_token']
    readonly_fields = ['share_token', 'created_at', 'updated_at']
    
    def share_token_short(self, obj):
        return obj.share_token[:16] + '...'
    share_token_short.short_description = 'Token'


# =============================================================================
# Checkout Admin
# =============================================================================

@admin.register(CheckoutSession)
class CheckoutSessionAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'user_display', 'current_step', 'email', 'shipping_city',
        'payment_method', 'total', 'created_at', 'expires_at'
    ]
    list_filter = ['current_step', 'payment_method', 'shipping_method', 'created_at']
    search_fields = ['email', 'shipping_email', 'user__email', 'shipping_phone']
    readonly_fields = ['created_at', 'updated_at', 'completed_at']
    inlines = [CheckoutEventInline]
    
    fieldsets = (
        ('Session', {
            'fields': ('user', 'cart', 'session_key', 'current_step', 'expires_at')
        }),
        ('Contact', {
            'fields': ('email',)
        }),
        ('Shipping Address', {
            'fields': (
                'shipping_first_name', 'shipping_last_name', 'shipping_company',
                'shipping_email', 'shipping_phone',
                'shipping_address_line_1', 'shipping_address_line_2',
                'shipping_city', 'shipping_state', 'shipping_postal_code', 'shipping_country'
            )
        }),
        ('Billing Address', {
            'fields': (
                'billing_same_as_shipping',
                'billing_first_name', 'billing_last_name', 'billing_company',
                'billing_address_line_1', 'billing_address_line_2',
                'billing_city', 'billing_state', 'billing_postal_code', 'billing_country'
            ),
            'classes': ('collapse',)
        }),
        ('Shipping & Payment', {
            'fields': ('shipping_method', 'shipping_cost', 'payment_method')
        }),
        ('Totals', {
            'fields': ('subtotal', 'discount_amount', 'tax_amount', 'total', 'coupon', 'coupon_code')
        }),
        ('Options', {
            'fields': ('order_notes', 'delivery_instructions', 'is_gift', 'gift_message', 'gift_wrap', 'gift_wrap_cost'),
            'classes': ('collapse',)
        }),
        ('Analytics', {
            'fields': ('utm_source', 'utm_medium', 'utm_campaign', 'ip_address', 'user_agent', 'referrer'),
            'classes': ('collapse',)
        }),
        ('Recovery', {
            'fields': ('recovery_email_sent', 'recovery_email_sent_at'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'completed_at')
        }),
    )
    
    def user_display(self, obj):
        return obj.user.email if obj.user else 'Guest'
    user_display.short_description = 'User'
