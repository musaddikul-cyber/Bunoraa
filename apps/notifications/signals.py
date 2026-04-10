"""
Notifications signals
"""
import logging
import os

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models.signals import post_migrate, post_save
from django.dispatch import receiver
from django.utils import timezone


logger = logging.getLogger("bunoraa.notifications")


def _module_for_notification(notification_type: str | None, reference_type: str | None = None) -> str:
    ntype = (notification_type or "").lower()
    rtype = (reference_type or "").lower()
    if ntype.startswith("order_") or rtype == "order":
        return "orders"
    if ntype.startswith("payment_") or "refund" in ntype or rtype == "payment":
        return "payments"
    if ntype.startswith("review_") or rtype == "review":
        return "reviews"
    if "stock" in ntype or "price" in ntype:
        return "catalog"
    if "promo" in ntype or "coupon" in ntype:
        return "promotions"
    if "subscription" in ntype:
        return "subscriptions"
    if "chat" in ntype or rtype in {"conversation", "message"}:
        return "support"
    return "notifications"


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_notification_preferences(sender, instance, created, **kwargs):
    """Create notification preferences when user is created."""
    if kwargs.get("raw", False) or os.environ.get("BUNORAA_IMPORTING_FIXTURES") == "1":
        return
    if created:
        from .models import NotificationPreference

        NotificationPreference.objects.get_or_create(user=instance)


@receiver(post_save, sender="accounts.UserPreferences")
def sync_preferences_from_account(sender, instance, **kwargs):
    """Sync notification preferences from user account preferences."""
    if kwargs.get("raw", False) or os.environ.get("BUNORAA_IMPORTING_FIXTURES") == "1":
        return
    from .services import NotificationService

    try:
        NotificationService.sync_from_user_preferences(instance)
    except Exception:
        pass


@receiver(post_migrate)
def ensure_notification_preferences(sender, **kwargs):
    """Backfill notification preferences for existing users after migrations."""
    if getattr(sender, "name", "") != "apps.notifications":
        return
    try:
        from .models import NotificationPreference
        from apps.accounts.models import UserPreferences
        from .services import NotificationService

        user_model = get_user_model()
        existing_user_ids = set(NotificationPreference.objects.values_list("user_id", flat=True))
        missing_ids = list(
            user_model.objects.exclude(id__in=existing_user_ids).values_list("id", flat=True)
        )

        if missing_ids:
            NotificationPreference.objects.bulk_create(
                [NotificationPreference(user_id=user_id) for user_id in missing_ids],
                batch_size=500,
            )

        for user_pref in UserPreferences.objects.select_related("user").all():
            try:
                NotificationService.sync_from_user_preferences(user_pref)
            except Exception:
                continue
    except Exception:
        pass


@receiver(post_save, sender="notifications.Notification")
def broadcast_notification(sender, instance, created, **kwargs):
    """Broadcast new notifications to user and admin websocket groups."""
    if kwargs.get("raw", False) or os.environ.get("BUNORAA_IMPORTING_FIXTURES") == "1":
        return
    if not created:
        return
    try:
        from asgiref.sync import async_to_sync
        from channels.layers import get_channel_layer

        from .api.serializers import NotificationSerializer

        channel_layer = get_channel_layer()
        if not channel_layer:
            return

        payload = NotificationSerializer(instance).data
        async_to_sync(channel_layer.group_send)(
            f"user_{instance.user_id}",
            {
                "type": "notification_message",
                "notification": payload,
            },
        )

        async_to_sync(channel_layer.group_send)(
            "admin_updates",
            {
                "type": "admin_update",
                "event_type": "notification",
                "module": _module_for_notification(instance.type, instance.reference_type),
                "entity_type": instance.reference_type or "notification",
                "entity_id": instance.reference_id or str(instance.id),
                "timestamp": timezone.now().isoformat(),
                "payload": {
                    "notification_id": str(instance.id),
                    "notification_type": instance.type,
                    "title": instance.title,
                    "message": instance.message,
                    "url": instance.url or "",
                    "user_id": str(instance.user_id),
                },
            },
        )
    except Exception:
        logger.exception("Failed to broadcast notification websocket event")
