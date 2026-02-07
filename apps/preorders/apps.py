"""
Pre-orders app configuration
"""
from django.apps import AppConfig


class PreordersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.preorders'
    verbose_name = 'Pre-orders'

    def ready(self):
        try:
            import apps.preorders.signals  # noqa
        except ImportError:
            pass
