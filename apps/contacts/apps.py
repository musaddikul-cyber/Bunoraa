"""
Contacts App Configuration
"""
from django.apps import AppConfig


class ContactsConfig(AppConfig):
    """Contacts app configuration."""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.contacts'
    verbose_name = 'Contacts'
    
    def ready(self):
        """Import signals when app is ready."""
        try:
            import importlib
            importlib.import_module('apps.contacts.signals')
        except ImportError:
            pass
