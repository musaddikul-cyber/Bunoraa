"""
Pre-orders admin configuration - Comprehensive admin panel for custom pre-order management
"""
from decimal import Decimal
from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from django.db.models import Count, Sum
from django.http import HttpResponseRedirect
from django.contrib import messages

from .models import (
    PreOrderCategory, PreOrderOption, PreOrderOptionChoice,
    PreOrder, PreOrderItem, PreOrderOptionValue, PreOrderDesign,
    PreOrderReference, PreOrderStatusHistory, PreOrderPayment,
    PreOrderMessage, PreOrderRevision, PreOrderQuote, PreOrderTemplate
)


# Inlines
class PreOrderOptionInline(admin.TabularInline):
    model = PreOrderOption
    extra = 0
    fields = ['name', 'option_type', 'is_required', 'price_modifier', 'order', 'is_active']
    ordering = ['order']


class PreOrderOptionChoiceInline(admin.TabularInline):
    model = PreOrderOptionChoice
    extra = 0
    fields = ['value', 'display_name', 'price_modifier', 'color_code', 'order', 'is_active']
    ordering = ['order']


class PreOrderItemInline(admin.TabularInline):
    model = PreOrderItem
    extra = 0
    readonly_fields = ['total_price', 'created_at']
    fields = ['name', 'quantity', 'unit_price', 'total_price', 'status', 'notes']


class PreOrderOptionValueInline(admin.TabularInline):
    model = PreOrderOptionValue
    extra = 0
    readonly_fields = ['option', 'price_modifier_applied', 'created_at']
    fields = ['option', 'text_value', 'number_value', 'choice_value', 'boolean_value', 'price_modifier_applied']
    can_delete = False


class PreOrderDesignInline(admin.TabularInline):
    model = PreOrderDesign
    extra = 0
    readonly_fields = ['file_size', 'version', 'uploaded_by', 'approved_by', 'approved_at', 'created_at']
    fields = ['file', 'original_filename', 'design_type', 'is_approved', 'is_current', 'version', 'notes']


class PreOrderReferenceInline(admin.TabularInline):
    model = PreOrderReference
    extra = 0
    readonly_fields = ['created_at']
    fields = ['file', 'original_filename', 'description', 'created_at']


class PreOrderStatusHistoryInline(admin.TabularInline):
    model = PreOrderStatusHistory
    extra = 0
    readonly_fields = ['from_status', 'to_status', 'changed_by', 'notes', 'is_system', 'notification_sent', 'created_at']
    can_delete = False
    ordering = ['-created_at']
    max_num = 0


class PreOrderPaymentInline(admin.TabularInline):
    model = PreOrderPayment
    extra = 0
    readonly_fields = ['transaction_id', 'status', 'paid_at', 'created_at']
    fields = ['payment_type', 'amount', 'payment_method', 'status', 'transaction_id', 'notes', 'paid_at']


class PreOrderMessageInline(admin.StackedInline):
    model = PreOrderMessage
    extra = 0
    readonly_fields = ['sender', 'is_from_customer', 'is_read', 'read_at', 'created_at']
    fields = ['subject', 'message', 'sender', 'is_from_customer', 'is_from_system', 'attachment', 'is_read', 'created_at']
    ordering = ['-created_at']


class PreOrderRevisionInline(admin.TabularInline):
    model = PreOrderRevision
    extra = 0
    readonly_fields = ['revision_number', 'requested_by', 'created_at', 'completed_at']
    fields = ['revision_number', 'description', 'status', 'additional_cost', 'admin_response', 'requested_by', 'created_at']


class PreOrderQuoteInline(admin.StackedInline):
    model = PreOrderQuote
    extra = 0
    readonly_fields = ['quote_number', 'created_by', 'created_at', 'sent_at', 'responded_at']
    fields = [
        'quote_number', 'status', 
        ('base_price', 'customization_cost', 'rush_fee'),
        ('discount', 'shipping', 'tax'),
        'total', ('valid_from', 'valid_until'),
        'estimated_production_days', 'estimated_delivery_date',
        'terms', 'notes', 'customer_response_notes',
        'created_by', 'created_at', 'sent_at'
    ]


# Admin Classes
@admin.register(PreOrderCategory)
class PreOrderCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'base_price', 'deposit_percentage', 'min_production_days', 
                    'max_production_days', 'preorder_count', 'is_active', 'order']
    list_filter = ['is_active', 'requires_design', 'requires_approval', 'allow_rush_order']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    ordering = ['order', 'name']
    inlines = [PreOrderOptionInline]
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'slug', 'description', 'icon', 'image')
        }),
        ('Pricing', {
            'fields': ('base_price', 'deposit_percentage')
        }),
        ('Production Time', {
            'fields': ('min_production_days', 'max_production_days')
        }),
        ('Settings', {
            'fields': ('requires_design', 'requires_approval', 'allow_rush_order', 'rush_order_fee_percentage')
        }),
        ('Quantity Limits', {
            'fields': ('min_quantity', 'max_quantity')
        }),
        ('Display', {
            'fields': ('is_active', 'order')
        }),
    )
    
    def preorder_count(self, obj):
        return obj.preorders.count()
    preorder_count.short_description = 'Pre-orders'


@admin.register(PreOrderOption)
class PreOrderOptionAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'option_type', 'is_required', 'price_modifier', 'is_active', 'order']
    list_filter = ['category', 'option_type', 'is_required', 'is_active']
    search_fields = ['name', 'description', 'category__name']
    ordering = ['category', 'order', 'name']
    inlines = [PreOrderOptionChoiceInline]
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('category', 'name', 'description', 'option_type')
        }),
        ('Validation', {
            'fields': ('is_required', 'min_length', 'max_length')
        }),
        ('Pricing', {
            'fields': ('price_modifier',)
        }),
        ('Display', {
            'fields': ('placeholder', 'help_text', 'order', 'is_active')
        }),
    )


@admin.register(PreOrder)
class PreOrderAdmin(admin.ModelAdmin):
    list_display = [
        'preorder_number', 'customer_info', 'category', 'title_short',
        'status_badge', 'priority_badge', 'total_amount_display',
        'payment_progress', 'assigned_to', 'created_at'
    ]
    list_filter = [
        'status', 'priority', 'category', 'is_rush_order',
        'is_gift', 'shipping_method', 'created_at', 'assigned_to'
    ]
    search_fields = [
        'preorder_number', 'email', 'full_name', 'phone',
        'title', 'description', 'user__email'
    ]
    readonly_fields = [
        'id', 'preorder_number', 'created_at', 'updated_at',
        'submitted_at', 'quoted_at', 'approved_at', 'production_started_at',
        'completed_at', 'shipped_at', 'delivered_at', 'cancelled_at',
        'amount_remaining', 'deposit_is_paid_display', 'is_fully_paid_display'
    ]
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    raw_id_fields = ['user', 'assigned_to', 'base_product', 'coupon']
    
    inlines = [
        PreOrderItemInline, PreOrderOptionValueInline, PreOrderDesignInline,
        PreOrderReferenceInline, PreOrderQuoteInline, PreOrderPaymentInline,
        PreOrderRevisionInline, PreOrderMessageInline, PreOrderStatusHistoryInline
    ]
    
    fieldsets = (
        ('Pre-Order Info', {
            'fields': ('id', 'preorder_number', 'category', 'base_product')
        }),
        ('Customer', {
            'fields': ('user', 'full_name', 'email', 'phone')
        }),
        ('Order Details', {
            'fields': ('title', 'description', 'quantity', 'special_instructions')
        }),
        ('Gift Options', {
            'fields': ('is_gift', 'gift_wrap', 'gift_message'),
            'classes': ('collapse',)
        }),
        ('Status & Priority', {
            'fields': ('status', 'priority', 'is_rush_order', 'assigned_to')
        }),
        ('Pricing', {
            'fields': (
                'estimated_price', 'final_price', 'rush_order_fee',
                'discount_amount', 'tax_amount', 'shipping_cost', 'total_amount'
            )
        }),
        ('Payments', {
            'fields': (
                'deposit_required', 'deposit_paid', 'deposit_is_paid_display',
                'amount_paid', 'amount_remaining', 'is_fully_paid_display',
                'currency', 'coupon'
            )
        }),
        ('Shipping Address', {
            'fields': (
                'shipping_first_name', 'shipping_last_name',
                'shipping_address_line_1', 'shipping_address_line_2',
                'shipping_city', 'shipping_state', 'shipping_postal_code', 'shipping_country'
            ),
            'classes': ('collapse',)
        }),
        ('Shipping', {
            'fields': ('shipping_method', 'tracking_number', 'tracking_url')
        }),
        ('Dates', {
            'fields': (
                'requested_delivery_date', 'estimated_completion_date',
                'actual_completion_date', 'production_start_date'
            )
        }),
        ('Quote', {
            'fields': ('quote_valid_until', 'quote_notes'),
            'classes': ('collapse',)
        }),
        ('Revisions', {
            'fields': ('revision_count', 'max_revisions'),
            'classes': ('collapse',)
        }),
        ('Notes', {
            'fields': ('customer_notes', 'admin_notes', 'production_notes'),
            'classes': ('collapse',)
        }),
        ('Tracking', {
            'fields': ('source', 'utm_source', 'utm_medium', 'utm_campaign'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': (
                'created_at', 'updated_at', 'submitted_at', 'quoted_at',
                'approved_at', 'production_started_at', 'completed_at',
                'shipped_at', 'delivered_at', 'cancelled_at'
            ),
            'classes': ('collapse',)
        }),
    )
    
    actions = [
        'mark_under_review', 'approve_for_production', 'start_production',
        'mark_completed', 'mark_ready_to_ship', 'export_preorders'
    ]
    
    def customer_info(self, obj):
        return format_html(
            '<strong>{}</strong><br><small>{}</small>',
            obj.full_name,
            obj.email
        )
    customer_info.short_description = 'Customer'
    
    def title_short(self, obj):
        title = obj.title[:40]
        if len(obj.title) > 40:
            title += '...'
        return title
    title_short.short_description = 'Title'
    
    def status_badge(self, obj):
        colors = {
            'draft': 'gray',
            'submitted': 'blue',
            'under_review': 'purple',
            'quoted': 'indigo',
            'quote_accepted': 'green',
            'quote_rejected': 'red',
            'deposit_pending': 'yellow',
            'deposit_paid': 'green',
            'in_production': 'cyan',
            'quality_check': 'teal',
            'awaiting_approval': 'orange',
            'revision_requested': 'amber',
            'final_payment_pending': 'yellow',
            'completed': 'emerald',
            'ready_to_ship': 'lime',
            'shipped': 'sky',
            'delivered': 'green',
            'cancelled': 'red',
            'refunded': 'pink',
            'on_hold': 'gray',
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; '
            'border-radius: 3px; font-size: 11px; text-transform: uppercase;">{}</span>',
            self.get_status_color(obj.status),
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def get_status_color(self, status):
        colors = {
            'draft': '#6b7280',
            'submitted': '#3b82f6',
            'under_review': '#8b5cf6',
            'quoted': '#6366f1',
            'quote_accepted': '#10b981',
            'quote_rejected': '#ef4444',
            'deposit_pending': '#f59e0b',
            'deposit_paid': '#22c55e',
            'in_production': '#06b6d4',
            'quality_check': '#14b8a6',
            'awaiting_approval': '#f97316',
            'revision_requested': '#fbbf24',
            'final_payment_pending': '#eab308',
            'completed': '#10b981',
            'ready_to_ship': '#84cc16',
            'shipped': '#0ea5e9',
            'delivered': '#22c55e',
            'cancelled': '#ef4444',
            'refunded': '#ec4899',
            'on_hold': '#9ca3af',
        }
        return colors.get(status, '#6b7280')
    
    def priority_badge(self, obj):
        colors = {
            'low': '#9ca3af',
            'normal': '#3b82f6',
            'high': '#f97316',
            'urgent': '#ef4444',
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 6px; '
            'border-radius: 3px; font-size: 10px; text-transform: uppercase;">{}</span>',
            colors.get(obj.priority, '#9ca3af'),
            obj.get_priority_display()
        )
    priority_badge.short_description = 'Priority'
    
    def total_amount_display(self, obj):
        if obj.total_amount:
            return format_html(
                '<strong>{} {}</strong>',
                obj.currency,
                obj.total_amount
            )
        return '-'
    total_amount_display.short_description = 'Total'
    
    def payment_progress(self, obj):
        if not obj.total_amount or obj.total_amount == 0:
            return '-'
        percentage = (obj.amount_paid / obj.total_amount) * 100
        color = '#22c55e' if percentage >= 100 else '#f59e0b' if percentage > 0 else '#ef4444'
        return format_html(
            '<div style="width: 100px; background: #e5e7eb; border-radius: 4px;">'
            '<div style="width: {}%; background: {}; height: 8px; border-radius: 4px;"></div>'
            '</div><small>{}%</small>',
            min(percentage, 100), color, int(percentage)
        )
    payment_progress.short_description = 'Paid'
    
    def deposit_is_paid_display(self, obj):
        return obj.deposit_is_paid
    deposit_is_paid_display.short_description = 'Deposit Paid'
    deposit_is_paid_display.boolean = True
    
    def is_fully_paid_display(self, obj):
        return obj.is_fully_paid
    is_fully_paid_display.short_description = 'Fully Paid'
    is_fully_paid_display.boolean = True
    
    def mark_under_review(self, request, queryset):
        updated = queryset.filter(status=PreOrder.STATUS_SUBMITTED).update(
            status=PreOrder.STATUS_UNDER_REVIEW
        )
        self.message_user(request, f'{updated} pre-orders marked as under review.')
    mark_under_review.short_description = 'Mark as Under Review'
    
    def approve_for_production(self, request, queryset):
        updated = 0
        for preorder in queryset.filter(status__in=[PreOrder.STATUS_DEPOSIT_PAID, PreOrder.STATUS_QUOTE_ACCEPTED]):
            if preorder.deposit_is_paid:
                preorder.status = PreOrder.STATUS_IN_PRODUCTION
                preorder.production_started_at = timezone.now()
                preorder.production_start_date = timezone.now().date()
                preorder.save()
                updated += 1
        self.message_user(request, f'{updated} pre-orders approved for production.')
    approve_for_production.short_description = 'Approve for Production'
    
    def start_production(self, request, queryset):
        updated = queryset.filter(
            status=PreOrder.STATUS_DEPOSIT_PAID
        ).update(
            status=PreOrder.STATUS_IN_PRODUCTION,
            production_started_at=timezone.now(),
            production_start_date=timezone.now().date()
        )
        self.message_user(request, f'{updated} pre-orders moved to production.')
    start_production.short_description = 'Start Production'
    
    def mark_completed(self, request, queryset):
        updated = queryset.filter(
            status=PreOrder.STATUS_IN_PRODUCTION
        ).update(
            status=PreOrder.STATUS_COMPLETED,
            completed_at=timezone.now(),
            actual_completion_date=timezone.now().date()
        )
        self.message_user(request, f'{updated} pre-orders marked as completed.')
    mark_completed.short_description = 'Mark as Completed'
    
    def mark_ready_to_ship(self, request, queryset):
        updated = 0
        for preorder in queryset.filter(status=PreOrder.STATUS_COMPLETED):
            if preorder.is_fully_paid:
                preorder.status = PreOrder.STATUS_READY_TO_SHIP
                preorder.save()
                updated += 1
        self.message_user(request, f'{updated} pre-orders marked as ready to ship.')
    mark_ready_to_ship.short_description = 'Mark Ready to Ship'
    
    def export_preorders(self, request, queryset):
        # TODO: Implement CSV export
        self.message_user(request, 'Export functionality coming soon.', messages.INFO)
    export_preorders.short_description = 'Export Selected'
    
    def save_model(self, request, obj, form, change):
        # Track status changes
        if change:
            old_obj = PreOrder.objects.get(pk=obj.pk)
            if old_obj.status != obj.status:
                PreOrderStatusHistory.objects.create(
                    preorder=obj,
                    from_status=old_obj.status,
                    to_status=obj.status,
                    changed_by=request.user,
                    notes=f'Status changed via admin by {request.user.email}'
                )
        super().save_model(request, obj, form, change)


@admin.register(PreOrderPayment)
class PreOrderPaymentAdmin(admin.ModelAdmin):
    list_display = [
        'preorder', 'payment_type', 'amount_display', 'payment_method',
        'status_badge', 'transaction_id', 'paid_at', 'created_at'
    ]
    list_filter = ['payment_type', 'status', 'payment_method', 'created_at']
    search_fields = ['preorder__preorder_number', 'transaction_id']
    readonly_fields = ['created_at', 'updated_at']
    raw_id_fields = ['preorder', 'recorded_by']
    date_hierarchy = 'created_at'
    
    def amount_display(self, obj):
        return format_html('<strong>{} {}</strong>', obj.currency, obj.amount)
    amount_display.short_description = 'Amount'
    
    def status_badge(self, obj):
        colors = {
            'pending': '#f59e0b',
            'processing': '#3b82f6',
            'completed': '#22c55e',
            'failed': '#ef4444',
            'cancelled': '#6b7280',
            'refunded': '#ec4899',
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 6px; '
            'border-radius: 3px; font-size: 10px;">{}</span>',
            colors.get(obj.status, '#6b7280'),
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'


@admin.register(PreOrderQuote)
class PreOrderQuoteAdmin(admin.ModelAdmin):
    list_display = [
        'quote_number', 'preorder', 'total', 'status_badge',
        'valid_until', 'is_expired_display', 'created_at'
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['quote_number', 'preorder__preorder_number']
    readonly_fields = ['quote_number', 'created_at', 'updated_at', 'sent_at', 'responded_at']
    raw_id_fields = ['preorder', 'created_by']
    date_hierarchy = 'created_at'
    
    def status_badge(self, obj):
        colors = {
            'pending': '#f59e0b',
            'sent': '#3b82f6',
            'viewed': '#8b5cf6',
            'accepted': '#22c55e',
            'rejected': '#ef4444',
            'expired': '#6b7280',
            'superseded': '#9ca3af',
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 6px; '
            'border-radius: 3px; font-size: 10px;">{}</span>',
            colors.get(obj.status, '#6b7280'),
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def is_expired_display(self, obj):
        return obj.is_expired
    is_expired_display.short_description = 'Expired'
    is_expired_display.boolean = True


@admin.register(PreOrderMessage)
class PreOrderMessageAdmin(admin.ModelAdmin):
    list_display = ['preorder', 'subject', 'sender_info', 'is_read', 'created_at']
    list_filter = ['is_from_customer', 'is_from_system', 'is_read', 'created_at']
    search_fields = ['preorder__preorder_number', 'subject', 'message']
    readonly_fields = ['created_at', 'read_at']
    raw_id_fields = ['preorder', 'sender']
    
    def sender_info(self, obj):
        if obj.is_from_system:
            return format_html('<em>System</em>')
        elif obj.is_from_customer:
            return format_html('<span style="color: #3b82f6;">Customer</span>')
        else:
            return format_html('<span style="color: #10b981;">Admin</span>')
    sender_info.short_description = 'From'


@admin.register(PreOrderRevision)
class PreOrderRevisionAdmin(admin.ModelAdmin):
    list_display = ['preorder', 'revision_number', 'status_badge', 'additional_cost', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['preorder__preorder_number', 'description']
    readonly_fields = ['created_at', 'updated_at', 'completed_at']
    raw_id_fields = ['preorder', 'requested_by', 'responded_by']
    
    def status_badge(self, obj):
        colors = {
            'pending': '#f59e0b',
            'in_progress': '#3b82f6',
            'completed': '#22c55e',
            'rejected': '#ef4444',
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 6px; '
            'border-radius: 3px; font-size: 10px;">{}</span>',
            colors.get(obj.status, '#6b7280'),
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'


@admin.register(PreOrderTemplate)
class PreOrderTemplateAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'base_price', 'estimated_days', 'use_count', 'is_active', 'is_featured', 'order']
    list_filter = ['category', 'is_active', 'is_featured']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    ordering = ['order', 'name']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'slug', 'description', 'category', 'image')
        }),
        ('Defaults', {
            'fields': ('default_quantity', 'base_price', 'estimated_days', 'default_options')
        }),
        ('Display', {
            'fields': ('is_active', 'is_featured', 'order', 'use_count')
        }),
    )


@admin.register(PreOrderDesign)
class PreOrderDesignAdmin(admin.ModelAdmin):
    list_display = ['preorder', 'original_filename', 'design_type', 'version', 'is_current', 'is_approved', 'created_at']
    list_filter = ['design_type', 'is_approved', 'is_current', 'created_at']
    search_fields = ['preorder__preorder_number', 'original_filename']
    readonly_fields = ['file_size', 'created_at', 'updated_at', 'approved_at']
    raw_id_fields = ['preorder', 'uploaded_by', 'approved_by']


@admin.register(PreOrderStatusHistory)
class PreOrderStatusHistoryAdmin(admin.ModelAdmin):
    list_display = ['preorder', 'from_status', 'to_status', 'changed_by', 'is_system', 'notification_sent', 'created_at']
    list_filter = ['to_status', 'is_system', 'notification_sent', 'created_at']
    search_fields = ['preorder__preorder_number', 'notes']
    readonly_fields = ['created_at']
    raw_id_fields = ['preorder', 'changed_by']
    date_hierarchy = 'created_at'
