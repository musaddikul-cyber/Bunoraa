"""
Notifications signals
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_notification_preferences(sender, instance, created, **kwargs):
    """Create notification preferences when user is created."""
    if created:
        from .models import NotificationPreference
        NotificationPreference.objects.get_or_create(user=instance)
