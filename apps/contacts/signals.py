"""
Contacts Signals
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.cache import cache

from .models import ContactSettings, StoreLocation


@receiver(post_save, sender=ContactSettings)
def settings_saved(sender, instance, **kwargs):
    """Clear settings cache when saved."""
    cache.delete('contact_settings')


@receiver(post_save, sender=StoreLocation)
def location_saved(sender, instance, **kwargs):
    """Clear location cache when saved."""
    cache.delete('store_locations')
    cache.delete('main_store_location')
    cache.delete('pickup_locations')
    cache.delete('returns_locations')
