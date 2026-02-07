from django.apps import AppConfig


class RecommendationsConfig(AppConfig):
    name = "apps.recommendations"
    verbose_name = "Recommendations"

    def ready(self):
        # import signal handlers
        try:
            from . import signals  # noqa: F401
        except Exception:
            pass
