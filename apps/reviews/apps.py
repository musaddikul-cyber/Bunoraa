"""
Reviews app configuration
"""
from django.apps import AppConfig


class ReviewsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.reviews'
    verbose_name = 'Reviews'
    
    def ready(self):
        # Canonical review counters are maintained by apps.catalog.signals.
        # Keep this app lightweight to avoid duplicate signal side effects.
        return
