"""
Orders admin configuration
"""
from django.contrib import admin
from django.utils.html import format_html
from .models import Order, OrderItem, OrderStatusHistory


class OrderItemInline(admin.TabularInline):
    model = OrderItem
    extra = 0
    readonly_fields = ['product', 'variant', 'product_name', 'variant_name', 'unit_price', 'quantity', 'line_total']
    can_delete = False
    
    def line_total(self, obj):
        return obj.line_total
    line_total.short_description = 'Line Total'


class OrderStatusHistoryInline(admin.TabularInline):
    model = OrderStatusHistory
    extra = 0
    readonly_fields = ['old_status', 'new_status', 'changed_by', 'notes', 'created_at']
    can_delete = False
    ordering = ['-created_at']


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = [
        'order_number', 'user_email', 'status', 'status_badge', 'item_count',
        'total', 'payment_status_badge', 'created_at'
    ]
    list_editable = ['status']
    list_filter = ['status', 'payment_status', 'payment_method', 'shipping_method', 'created_at']
    search_fields = ['order_number', 'email', 'user__email', 'phone', 'shipping_first_name', 'shipping_last_name']
    raw_id_fields = ['user']
    readonly_fields = [
        'id', 'order_number', 'subtotal', 'discount', 'tax', 'total',
        'stripe_payment_intent_id', 'created_at', 'updated_at',
        'confirmed_at', 'shipped_at', 'delivered_at', 'cancelled_at'
    ]
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    
    inlines = [OrderItemInline, OrderStatusHistoryInline]

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.filter(is_deleted=False)
    
    fieldsets = (
        ('Order Info', {
            'fields': ('id', 'order_number', 'user', 'email', 'phone', 'status')
        }),
        ('Shipping Address', {
            'fields': (
                'shipping_first_name', 'shipping_last_name',
                'shipping_address_line_1', 'shipping_address_line_2',
                'shipping_city', 'shipping_state', 'shipping_postal_code', 'shipping_country'
            )
        }),
        ('Billing Address', {
            'fields': (
                'billing_first_name', 'billing_last_name',
                'billing_address_line_1', 'billing_address_line_2',
                'billing_city', 'billing_state', 'billing_postal_code', 'billing_country'
            ),
            'classes': ('collapse',)
        }),
        ('Shipping', {
            'fields': ('shipping_method', 'shipping_cost', 'tracking_number', 'tracking_url')
        }),
        ('Payment', {
            'fields': ('payment_method', 'payment_status', 'stripe_payment_intent_id')
        }),
        ('Amounts', {
            'fields': ('subtotal', 'discount', 'coupon', 'coupon_code', 'tax', 'total')
        }),
        ('Notes', {
            'fields': ('customer_notes', 'admin_notes')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'confirmed_at', 'shipped_at', 'delivered_at', 'cancelled_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['mark_confirmed', 'mark_processing', 'mark_shipped', 'mark_delivered', 'mark_cancelled', 'mark_refunded', 'soft_delete_selected']

    def save_model(self, request, obj, form, change):
        """Override save to record status changes in history with the admin user."""
        if change:
            try:
                old = Order.objects.get(pk=obj.pk)
                if old.status != obj.status:
                    OrderStatusHistory.objects.create(
                        order=obj,
                        old_status=old.status or '',
                        new_status=obj.status,
                        changed_by=request.user,
                        notes=f"Status changed via admin from {old.status} to {obj.status}"
                    )
            except Order.DoesNotExist:
                pass
        super().save_model(request, obj, form, change)

    def payment_status_badge(self, obj):
        colors = {
            'pending': '#f59e0b',
            'succeeded': '#10b981',
            'failed': '#ef4444',
            'refunded': '#6b7280',
        }
        color = colors.get(obj.payment_status, '#6b7280')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; '
            'border-radius: 4px; font-size: 11px;">{}</span>',
            color, obj.get_payment_status_display()
        )
    payment_status_badge.short_description = 'Payment Status'

    def mark_cancelled(self, request, queryset):
        """Admin action to cancel selected orders (uses OrderService to handle refunds) and ensure stock restore."""
        from .services import OrderService
        count = 0
        for order in queryset.exclude(status=Order.STATUS_CANCELLED):
            success, message = OrderService.cancel_order(order, reason='Cancelled by admin', cancelled_by=request.user)
            if success:
                # Ensure stock restore if not handled (signal handles it normally)
                for item in order.items.all():
                    if item.variant:
                        item.variant.stock_quantity += item.quantity
                        item.variant.save(update_fields=['stock_quantity'])
                    elif item.product:
                        item.product.stock_quantity += item.quantity
                        item.product.save(update_fields=['stock_quantity'])
                count += 1
        self.message_user(request, f'{count} orders cancelled.')
    mark_cancelled.short_description = 'Cancel selected orders'

    def mark_refunded(self, request, queryset):
        """Mark selected paid orders as refunded (best-effort, does not call Stripe)."""
        updated = queryset.filter(payment_status__in=['succeeded', 'paid']).update(status=Order.STATUS_REFUNDED)
        self.message_user(request, f'{updated} orders marked as refunded.')
    mark_refunded.short_description = 'Mark selected orders as refunded'

    def soft_delete_selected(self, request, queryset):
        """Soft-delete selected orders."""
        count = 0
        for order in queryset:
            if not order.is_deleted:
                order.soft_delete()
                count += 1
        self.message_user(request, f'{count} orders soft-deleted.')
    soft_delete_selected.short_description = 'Soft-delete selected orders'
    
    def user_email(self, obj):
        return obj.user.email if obj.user else obj.email
    user_email.short_description = 'Customer'
    
    def status_badge(self, obj):
        colors = {
            'pending': '#f59e0b',
            'confirmed': '#3b82f6',
            'processing': '#8b5cf6',
            'shipped': '#06b6d4',
            'delivered': '#10b981',
            'cancelled': '#ef4444',
            'refunded': '#6b7280',
        }
        color = colors.get(obj.status, '#6b7280')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 2px 8px; '
            'border-radius: 4px; font-size: 11px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def mark_confirmed(self, request, queryset):
        """Iterate and mark eligible orders as confirmed, recording history."""
        count = 0
        for order in queryset.filter(status=Order.STATUS_PENDING):
            old = order.status
            order.status = Order.STATUS_CONFIRMED
            order.save()
            OrderStatusHistory.objects.create(
                order=order,
                old_status=old,
                new_status=order.status,
                changed_by=request.user,
                notes='Marked as confirmed via admin action'
            )
            count += 1
        self.message_user(request, f'{count} orders marked as confirmed.')
    mark_confirmed.short_description = 'Mark selected orders as confirmed'

    def mark_processing(self, request, queryset):
        """Iterate and mark eligible orders as processing, recording history."""
        count = 0
        for order in queryset.filter(status=Order.STATUS_CONFIRMED):
            old = order.status
            order.status = Order.STATUS_PROCESSING
            order.save()
            OrderStatusHistory.objects.create(
                order=order,
                old_status=old,
                new_status=order.status,
                changed_by=request.user,
                notes='Marked as processing via admin action'
            )
            count += 1
        self.message_user(request, f'{count} orders marked as processing.')
    mark_processing.short_description = 'Mark selected orders as processing'

    def mark_shipped(self, request, queryset):
        """Mark as shipped using OrderService to set shipped_at and history."""
        from .services import OrderService
        count = 0
        for order in queryset.filter(status__in=[Order.STATUS_CONFIRMED, Order.STATUS_PROCESSING]):
            OrderService.mark_shipped(order, shipped_by=request.user)
            OrderStatusHistory.objects.create(
                order=order,
                old_status=Order.STATUS_CONFIRMED if order.status == Order.STATUS_SHIPPED else order.status,
                new_status=Order.STATUS_SHIPPED,
                changed_by=request.user,
                notes='Marked as shipped via admin action'
            )
            count += 1
        self.message_user(request, f'{count} orders marked as shipped.')
    mark_shipped.short_description = 'Mark selected orders as shipped'

    def mark_delivered(self, request, queryset):
        """Mark shipped orders as delivered and record history."""
        from django.utils import timezone
        count = 0
        for order in queryset.filter(status=Order.STATUS_SHIPPED):
            old = order.status
            order.status = Order.STATUS_DELIVERED
            order.delivered_at = timezone.now()
            order.save()
            OrderStatusHistory.objects.create(
                order=order,
                old_status=old,
                new_status=order.status,
                changed_by=request.user,
                notes='Marked as delivered via admin action'
            )
            count += 1
        self.message_user(request, f'{count} orders marked as delivered.')
    mark_delivered.short_description = 'Mark selected orders as delivered'


@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ['order', 'product_name', 'variant_name', 'unit_price', 'quantity', 'line_total']
    list_filter = ['order__status', 'created_at']
    search_fields = ['order__order_number', 'product_name', 'product_sku']
    readonly_fields = ['order', 'product', 'variant', 'line_total']
    
    def line_total(self, obj):
        return obj.line_total
    line_total.short_description = 'Line Total'


@admin.register(OrderStatusHistory)
class OrderStatusHistoryAdmin(admin.ModelAdmin):
    list_display = ['order', 'old_status', 'new_status', 'changed_by', 'created_at']
    list_filter = ['new_status', 'created_at']
    search_fields = ['order__order_number', 'notes']
    readonly_fields = ['order', 'old_status', 'new_status', 'changed_by', 'notes', 'created_at']
