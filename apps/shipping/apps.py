from django.apps import AppConfig


class ShippingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.shipping'
    verbose_name = 'Shipping'

    def ready(self):
        import apps.shipping.signals  # noqa
