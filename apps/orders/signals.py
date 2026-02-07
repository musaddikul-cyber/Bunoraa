"""
Orders signals
"""
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone

from .models import Order, OrderStatusHistory
from apps.payments.models import Payment
from .services import OrderService


@receiver(pre_save, sender=Order)
def track_status_change(sender, instance, **kwargs):
    """Track status changes before saving."""
    if instance.pk:
        try:
            old_instance = Order.objects.get(pk=instance.pk)
            instance._old_status = old_instance.status
        except Order.DoesNotExist:
            instance._old_status = None
    else:
        instance._old_status = None


@receiver(post_save, sender=Order)
def create_status_history(sender, instance, created, **kwargs):
    """Create status history entry on status change."""
    old_status = getattr(instance, '_old_status', None)
    
    if created:
        # New order
        OrderStatusHistory.objects.create(
            order=instance,
            old_status='',
            new_status=instance.status,
            notes='Order created'
        )
    elif old_status and old_status != instance.status:
        # Status changed
        OrderStatusHistory.objects.create(
            order=instance,
            old_status=old_status,
            new_status=instance.status
        )
        
        # Update timestamp fields based on status
        if instance.status == Order.STATUS_CONFIRMED and not instance.confirmed_at:
            Order.objects.filter(pk=instance.pk).update(confirmed_at=timezone.now())
        elif instance.status == Order.STATUS_SHIPPED and not instance.shipped_at:
            Order.objects.filter(pk=instance.pk).update(shipped_at=timezone.now())
        elif instance.status == Order.STATUS_DELIVERED and not instance.delivered_at:
            Order.objects.filter(pk=instance.pk).update(delivered_at=timezone.now())
        elif instance.status == Order.STATUS_CANCELLED and not instance.cancelled_at:
            Order.objects.filter(pk=instance.pk).update(cancelled_at=timezone.now())


@receiver(post_save, sender=Order)
def update_coupon_usage(sender, instance, created, **kwargs):
    """Update coupon usage count when order is created."""
    if created and instance.coupon:
        instance.coupon.times_used += 1
        instance.coupon.save(update_fields=['times_used'])


@receiver(post_save, sender=Order)
def restore_stock_on_cancel(sender, instance, **kwargs):
    """Restore stock when order is cancelled."""
    old_status = getattr(instance, '_old_status', None)
    
    if old_status and old_status != Order.STATUS_CANCELLED and instance.status == Order.STATUS_CANCELLED:
        for item in instance.items.all():
            if item.variant:
                item.variant.stock_quantity += item.quantity
                item.variant.save(update_fields=['stock_quantity'])
            elif item.product:
                item.product.stock_quantity += item.quantity
                item.product.save(update_fields=['stock_quantity'])


@receiver(post_save, sender=Payment)
def update_order_on_payment(sender, instance, created, **kwargs):
    """Update order status and payment_status when a Payment transitions to succeeded/failed."""
    # Only act on updates where payment has an associated order
    payment = instance
    order = getattr(payment, 'order', None)
    if not order:
        return

    # Succeeded
    if payment.status == payment.STATUS_SUCCEEDED and order.payment_status != Order.PAYMENT_SUCCEEDED:
        # Mark payment_status
        order.payment_status = Order.PAYMENT_SUCCEEDED
        # If order is pending, move to confirmed
        if order.status == Order.STATUS_PENDING:
            OrderService.update_order_status(order, Order.STATUS_CONFIRMED, changed_by=None, notes='Payment received')
        else:
            order.save(update_fields=['payment_status'])

    # Failed
    if payment.status == payment.STATUS_FAILED and order.payment_status != Order.PAYMENT_FAILED:
        order.payment_status = Order.PAYMENT_FAILED
        order.save(update_fields=['payment_status'])
