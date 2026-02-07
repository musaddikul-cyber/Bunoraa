from django.apps import AppConfig


class CatalogConfig(AppConfig):
    name = "apps.catalog"
    verbose_name = "Catalog"

    def ready(self):
        # import signals to ensure m2m handler registration
        from . import signals  # noqa: F401
