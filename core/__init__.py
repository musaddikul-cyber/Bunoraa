# Bunoraa Core Module
default_app_config = 'core.apps.CoreConfig'

# Import Celery app for Django
from .celery import app as celery_app

__all__ = ('celery_app',)
