"""
Notifications signals
"""
from django.db.models.signals import post_save, post_migrate
from django.dispatch import receiver
from django.conf import settings
from django.contrib.auth import get_user_model


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_notification_preferences(sender, instance, created, **kwargs):
    """Create notification preferences when user is created."""
    if created:
        from .models import NotificationPreference
        NotificationPreference.objects.get_or_create(user=instance)


@receiver(post_save, sender='accounts.UserPreferences')
def sync_preferences_from_account(sender, instance, **kwargs):
    """Sync notification preferences from user account preferences."""
    from .services import NotificationService
    try:
        NotificationService.sync_from_user_preferences(instance)
    except Exception:
        pass


@receiver(post_migrate)
def ensure_notification_preferences(sender, **kwargs):
    """Backfill notification preferences for existing users after migrations."""
    if getattr(sender, 'name', '') != 'apps.notifications':
        return
    try:
        from .models import NotificationPreference
        from apps.accounts.behavior_models import UserPreferences
        from .services import NotificationService

        User = get_user_model()
        existing_user_ids = set(NotificationPreference.objects.values_list('user_id', flat=True))
        missing_ids = list(
            User.objects.exclude(id__in=existing_user_ids).values_list('id', flat=True)
        )

        if missing_ids:
            NotificationPreference.objects.bulk_create(
                [NotificationPreference(user_id=user_id) for user_id in missing_ids],
                batch_size=500,
            )

        for user_pref in UserPreferences.objects.select_related('user').all():
            try:
                NotificationService.sync_from_user_preferences(user_pref)
            except Exception:
                continue
    except Exception:
        pass


@receiver(post_save, sender='notifications.Notification')
def broadcast_notification(sender, instance, created, **kwargs):
    """Broadcast new notifications to the user's WebSocket group."""
    if not created:
        return
    try:
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        from .api.serializers import NotificationSerializer
        
        channel_layer = get_channel_layer()
        if not channel_layer:
            return
        
        payload = NotificationSerializer(instance).data
        async_to_sync(channel_layer.group_send)(
            f'user_{instance.user_id}',
            {
                'type': 'notification_message',
                'notification': payload,
            }
        )
    except Exception:
        pass
