"""
Internationalization App Configuration

Unified i18n app combining language, currency, timezone, and localization features.
"""
from django.apps import AppConfig


class I18nConfig(AppConfig):
    """Configuration for the i18n app."""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.i18n'
    verbose_name = 'Internationalization'
    
    def ready(self):
        """Import signals and connect handlers when app is ready."""
        try:
            from . import signals  # noqa
            # Connect user creation signal for auto-creating locale preferences
            signals.connect_user_signal()
        except ImportError:
            pass
