"""
Signal handlers for Payments app.
Handles payment processing events and notifications.
"""
import logging

from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone

logger = logging.getLogger("bunoraa.payments")


def _broadcast_admin_payment_event(event_type: str, entity_type: str, entity_id: str, payload: dict):
    try:
        from asgiref.sync import async_to_sync
        from channels.layers import get_channel_layer

        channel_layer = get_channel_layer()
        if not channel_layer:
            return

        async_to_sync(channel_layer.group_send)(
            "admin_updates",
            {
                "type": "admin_update",
                "event_type": event_type,
                "module": "payments",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "timestamp": timezone.now().isoformat(),
                "payload": payload,
            },
        )
    except Exception:
        logger.exception("Failed to emit admin payment websocket event")


@receiver(pre_save, sender="payments.Payment")
def on_payment_pre_save(sender, instance, **kwargs):
    """Track status changes before save."""
    if instance.pk:
        try:
            old = sender.objects.get(pk=instance.pk)
            instance._status_changed = old.status != instance.status
            instance._old_status = old.status
        except sender.DoesNotExist:
            instance._status_changed = False
    else:
        instance._status_changed = True
        instance._old_status = None


@receiver(post_save, sender="payments.Payment")
def on_payment_saved(sender, instance, created, **kwargs):
    """Handle payment status updates."""
    if created:
        logger.info(f"Payment created: {instance.transaction_id}")

    # Check if status changed
    if getattr(instance, "_status_changed", False):
        logger.info(
            f"Payment status changed: {instance.transaction_id} "
            f"{getattr(instance, '_old_status', '')} -> {instance.status}"
        )

        _broadcast_admin_payment_event(
            event_type="payment_update",
            entity_type="payment",
            entity_id=str(instance.id),
            payload={
                "payment_id": str(instance.id),
                "old_status": getattr(instance, "_old_status", ""),
                "status": instance.status,
                "amount": str(instance.amount),
                "currency": instance.currency,
                "order_id": str(instance.order_id) if instance.order_id else "",
            },
        )

        # Send notification on payment completion
        if instance.status == "completed":
            try:
                from apps.notifications.tasks import send_notification

                send_notification.delay(
                    user_id=instance.order.user_id if instance.order else None,
                    notification_type="payment_success",
                    context={
                        "payment_id": instance.id,
                        "amount": str(instance.amount),
                        "currency": instance.currency,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to send payment notification: {e}")

        # Handle failed payment
        elif instance.status == "failed":
            try:
                from apps.notifications.tasks import send_notification

                send_notification.delay(
                    user_id=instance.order.user_id if instance.order else None,
                    notification_type="payment_failed",
                    context={
                        "payment_id": instance.id,
                        "reason": instance.failure_reason,
                    },
                )
            except Exception:
                pass


@receiver(post_save, sender="payments.Refund")
def on_refund_saved(sender, instance, created, **kwargs):
    """Handle refund creation."""
    if created:
        logger.info(f"Refund requested: {instance.amount} for payment {instance.payment_id}")

        _broadcast_admin_payment_event(
            event_type="refund_created",
            entity_type="refund",
            entity_id=str(instance.id),
            payload={
                "refund_id": str(instance.id),
                "payment_id": str(instance.payment_id),
                "amount": str(instance.amount),
                "status": instance.status,
            },
        )

        # Notify customer
        try:
            from apps.notifications.tasks import send_notification

            payment = instance.payment
            if payment and payment.order:
                send_notification.delay(
                    user_id=payment.order.user_id,
                    notification_type="refund_initiated",
                    context={
                        "refund_amount": str(instance.amount),
                        "order_number": payment.order.order_number,
                    },
                )
        except Exception:
            pass
