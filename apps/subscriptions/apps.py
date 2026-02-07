# apps/subscriptions/apps.py
from django.apps import AppConfig


class SubscriptionsConfig(AppConfig):
    name = "apps.subscriptions"
    verbose_name = "Subscriptions"

    def ready(self):
        try:
            from . import signals  # noqa: F401
        except Exception:
            pass
