"""
Payments admin configuration
"""
from django.contrib import admin
from .models import (
    Payment, PaymentMethod, Refund, PaymentGateway,
    PaymentTransaction, PaymentLink, BkashCredential, BNPLProvider, BNPLAgreement, RecurringCharge
)


@admin.register(PaymentGateway)
class PaymentGatewayAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'is_active', 'fee_text', 'is_sandbox', 'ssl_store_id', 'bkash_mode', 'nagad_merchant_id', 'supports_recurring', 'supports_bnpl', 'sort_order']
    list_filter = ['is_active', 'is_sandbox', 'fee_type']
    search_fields = ['name', 'code', 'description']
    list_editable = ['is_active', 'sort_order']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('code', 'name', 'description', 'is_active', 'sort_order')
        }),
        ('Appearance', {
            'fields': ('icon', 'icon_class', 'color')
        }),
        ('Fees', {
            'fields': ('fee_type', 'fee_amount', 'fee_text'),
            'description': 'Configure additional fees for this payment method'
        }),
        ('Availability', {
            'fields': ('currencies', 'countries', 'min_amount', 'max_amount'),
            'classes': ('collapse',),
            'description': 'Leave empty to allow all currencies/countries'
        }),
        ('API Configuration', {
            'fields': ('api_key', 'api_secret', 'merchant_id', 'webhook_secret', 'is_sandbox'),
            'classes': ('collapse',),
            'description': 'API credentials for this payment gateway'
        }),
        ('Bangladesh Gateways', {
            'fields': (
                'ssl_store_id', 'ssl_store_passwd',
                'bkash_mode', 'bkash_app_key', 'bkash_app_secret', 'bkash_username', 'bkash_password',
                'nagad_merchant_id', 'nagad_public_key', 'nagad_private_key'
            ),
            'classes': ('collapse',),
            'description': 'Configure SSLCommerz, bKash, and Nagad credentials for Bangladesh integrations'
        }),
        ('Capabilities', {
            'fields': ('supports_partial', 'supports_recurring', 'supports_bnpl'),
            'description': 'Toggle gateway capabilities for admin use'
        }),
        ('Bank Transfer Details', {
            'fields': ('bank_name', 'bank_account_name', 'bank_account_number', 'bank_routing_number', 'bank_branch'),
            'classes': ('collapse',),
            'description': 'Only for Bank Transfer payment method'
        }),
        ('Customer Instructions', {
            'fields': ('instructions',),
        }),
        ('Timestamps', {
            'fields': ('id', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(PaymentMethod)
class PaymentMethodAdmin(admin.ModelAdmin):
    list_display = ['user', 'type', 'card_brand', 'card_last_four', 'is_default', 'is_active', 'created_at']
    list_filter = ['type', 'card_brand', 'is_default', 'is_active']
    search_fields = ['user__email', 'card_last_four', 'paypal_email']
    readonly_fields = ['id', 'stripe_payment_method_id', 'created_at', 'updated_at']
    
    fieldsets = (
        (None, {
            'fields': ('id', 'user', 'type', 'is_default', 'is_active')
        }),
        ('Card Details', {
            'fields': ('card_brand', 'card_last_four', 'card_exp_month', 'card_exp_year')
        }),
        ('PayPal', {
            'fields': ('paypal_email',)
        }),
        ('Stripe', {
            'fields': ('stripe_payment_method_id',)
        }),
        ('Billing', {
            'fields': ('billing_name', 'billing_address')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['id', 'order', 'amount', 'currency', 'status', 'paid_at', 'created_at']
    list_filter = ['status', 'currency', 'created_at']
    search_fields = ['id', 'order__order_number', 'user__email', 'stripe_payment_intent_id']
    readonly_fields = [
        'id', 'order', 'user', 'amount', 'currency', 'payment_method',
        'method_type', 'stripe_payment_intent_id', 'stripe_charge_id',
        'gateway_response', 'refunded_amount', 'created_at', 'updated_at', 'paid_at'
    ]
    
    fieldsets = (
        (None, {
            'fields': ('id', 'order', 'user')
        }),
        ('Amount', {
            'fields': ('amount', 'currency', 'refunded_amount')
        }),
        ('Payment Method', {
            'fields': ('payment_method', 'method_type')
        }),
        ('Status', {
            'fields': ('status', 'failure_reason')
        }),
        ('Stripe', {
            'fields': ('stripe_payment_intent_id', 'stripe_charge_id', 'gateway_response')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'paid_at')
        }),
    )
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


class RefundInline(admin.TabularInline):
    model = Refund
    extra = 0
    readonly_fields = ['id', 'amount', 'reason', 'status', 'stripe_refund_id', 'created_by', 'created_at', 'processed_at']
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Refund)
class RefundAdmin(admin.ModelAdmin):
    list_display = ['id', 'payment', 'amount', 'reason', 'status', 'created_by', 'created_at']
    list_filter = ['status', 'reason', 'created_at']
    search_fields = ['id', 'payment__stripe_payment_intent_id', 'stripe_refund_id']
    readonly_fields = [
        'id', 'payment', 'amount', 'stripe_refund_id',
        'created_by', 'created_at', 'processed_at'
    ]
    
    fieldsets = (
        (None, {
            'fields': ('id', 'payment', 'amount')
        }),
        ('Details', {
            'fields': ('reason', 'notes', 'status')
        }),
        ('Stripe', {
            'fields': ('stripe_refund_id',)
        }),
        ('Admin', {
            'fields': ('created_by', 'created_at', 'processed_at')
        }),
    )
    
    def has_add_permission(self, request):
        return False


@admin.register(PaymentTransaction)
class PaymentTransactionAdmin(admin.ModelAdmin):
    list_display = ['reference', 'gateway', 'event_type', 'fee_amount', 'created_at']
    list_filter = ['gateway', 'event_type', 'created_at']
    search_fields = ['reference', 'payload']
    readonly_fields = ['id', 'gateway', 'payment', 'order', 'reference', 'payload', 'fee_amount', 'created_at']


@admin.register(PaymentLink)
class PaymentLinkAdmin(admin.ModelAdmin):
    list_display = ['code', 'order', 'gateway', 'amount', 'currency', 'is_active', 'expires_at', 'created_at']
    list_filter = ['is_active', 'gateway', 'currency']
    search_fields = ['code', 'order__order_number']
    readonly_fields = ['id', 'code', 'created_at']


@admin.register(BNPLProvider)
class BNPLProviderAdmin(admin.ModelAdmin):
    list_display = ['code', 'name', 'is_active', 'created_at']
    list_filter = ['is_active']
    search_fields = ['code', 'name']


@admin.register(BNPLAgreement)
class BNPLAgreementAdmin(admin.ModelAdmin):
    list_display = ['provider', 'user', 'approved', 'approved_at', 'created_at']
    list_filter = ['approved', 'provider']
    search_fields = ['provider__code', 'user__email']


@admin.register(RecurringCharge)
class RecurringChargeAdmin(admin.ModelAdmin):
    list_display = ['subscription', 'amount', 'currency', 'status', 'attempt_at', 'processed_at']
    list_filter = ['status', 'currency']
    search_fields = ['subscription__stripe_subscription_id']