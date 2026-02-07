"""
Pre-orders signals - Signal handlers for pre-order events
"""
import logging
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone

from .models import (
    PreOrder, PreOrderPayment, PreOrderMessage, 
    PreOrderRevision, PreOrderQuote, PreOrderStatusHistory
)
from .services import PreOrderService

logger = logging.getLogger(__name__)


@receiver(pre_save, sender=PreOrder)
def track_status_changes(sender, instance, **kwargs):
    """Track status changes before save."""
    if instance.pk:
        try:
            old_instance = PreOrder.objects.get(pk=instance.pk)
            instance._old_status = old_instance.status
        except PreOrder.DoesNotExist:
            instance._old_status = None
    else:
        instance._old_status = None


@receiver(post_save, sender=PreOrder)
def handle_preorder_status_change(sender, instance, created, **kwargs):
    """Handle post-save actions for pre-order status changes."""
    if created:
        return
    
    old_status = getattr(instance, '_old_status', None)
    if old_status and old_status != instance.status:
        # Log status change
        logger.info(
            f"Pre-order {instance.preorder_number} status changed: "
            f"{old_status} -> {instance.status}"
        )
        
        # Check if we need to create status history (if not already created)
        if not PreOrderStatusHistory.objects.filter(
            preorder=instance,
            to_status=instance.status,
            created_at__gte=timezone.now() - timezone.timedelta(seconds=5)
        ).exists():
            PreOrderStatusHistory.objects.create(
                preorder=instance,
                from_status=old_status,
                to_status=instance.status,
                is_system=True,
                notes='Status changed automatically'
            )


@receiver(post_save, sender=PreOrderPayment)
def handle_payment_completed(sender, instance, created, **kwargs):
    """Handle payment completion events."""
    if not created:
        return
    
    if instance.status == PreOrderPayment.STATUS_COMPLETED:
        preorder = instance.preorder
        
        # Check if deposit is now paid
        if (instance.payment_type == PreOrderPayment.PAYMENT_DEPOSIT and 
            preorder.status == PreOrder.STATUS_DEPOSIT_PENDING and
            preorder.deposit_is_paid):
            
            PreOrderService.update_status(
                preorder,
                PreOrder.STATUS_DEPOSIT_PAID,
                notes=f'Deposit payment completed: {instance.amount}'
            )
            logger.info(f"Pre-order {preorder.preorder_number} deposit paid")
        
        # Check if fully paid
        if preorder.is_fully_paid and preorder.status == PreOrder.STATUS_FINAL_PAYMENT_PENDING:
            PreOrderService.update_status(
                preorder,
                PreOrder.STATUS_READY_TO_SHIP,
                notes='Final payment completed, ready to ship'
            )
            logger.info(f"Pre-order {preorder.preorder_number} fully paid and ready to ship")


@receiver(post_save, sender=PreOrderMessage)
def handle_new_message(sender, instance, created, **kwargs):
    """Handle new message notifications."""
    if not created:
        return
    
    # Log the message
    logger.info(
        f"New message on pre-order {instance.preorder.preorder_number} "
        f"from {'customer' if instance.is_from_customer else 'admin'}"
    )


@receiver(post_save, sender=PreOrderRevision)
def handle_revision_request(sender, instance, created, **kwargs):
    """Handle revision request events."""
    if not created:
        return
    
    preorder = instance.preorder
    
    # Update preorder revision count if not already updated
    if preorder.revision_count < instance.revision_number:
        preorder.revision_count = instance.revision_number
        preorder.save(update_fields=['revision_count'])
    
    logger.info(
        f"Revision #{instance.revision_number} requested for "
        f"pre-order {preorder.preorder_number}"
    )


@receiver(post_save, sender=PreOrderQuote)
def handle_quote_created(sender, instance, created, **kwargs):
    """Handle quote creation events."""
    if not created:
        return
    
    logger.info(
        f"Quote {instance.quote_number} created for "
        f"pre-order {instance.preorder.preorder_number}"
    )


@receiver(pre_save, sender=PreOrderQuote)
def handle_quote_expiry(sender, instance, **kwargs):
    """Check and update quote expiry status."""
    if instance.pk and instance.status in [PreOrderQuote.STATUS_PENDING, PreOrderQuote.STATUS_SENT]:
        if instance.is_expired and instance.status != PreOrderQuote.STATUS_EXPIRED:
            instance.status = PreOrderQuote.STATUS_EXPIRED
            logger.info(f"Quote {instance.quote_number} marked as expired")
