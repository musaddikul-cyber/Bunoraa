"""Helpers for resolving the ML Redis backend consistently."""

from django.conf import settings


def get_ml_redis_url(default: str = 'redis://localhost:6379/1') -> str:
    """Return the dedicated ML Redis URL with sane fallbacks."""
    return (
        getattr(settings, 'ML_REDIS_URL', '').strip()
        or getattr(settings, 'CELERY_REDIS_URL', '').strip()
        or getattr(settings, 'REDIS_URL', '').strip()
        or default
    )
