from django.core.management.base import BaseCommand
from django.apps import apps
from django.conf import settings
from django.core.files.storage import default_storage

TARGET_FIELDS = [
    'apps.pages.SiteSettings:logo',
    'apps.pages.SiteSettings:logo_dark',
    'apps.pages.SiteSettings:favicon',
    'apps.promotions.Banner:image',
    'apps.promotions.Banner:image_mobile',
    'apps.payments.PaymentGateway:icon',
    'apps.accounts.User:avatar',
    'apps.shipping.ShippingCarrier:logo',
    'apps.categories.Category:image',
]

class Command(BaseCommand):
    help = 'Check media storage configuration and fields for common upload targets.'

    def handle(self, *args, **options):
        self.stdout.write(f'DEFAULT_FILE_STORAGE: {getattr(settings, "DEFAULT_FILE_STORAGE", None)}')
        self.stdout.write(f'default_storage class: {default_storage.__class__}\n')

        for spec in TARGET_FIELDS:
            try:
                app_model, field_name = spec.split(':')
                app_label, model_name = app_model.split('.')[-2:]
                model = apps.get_model(app_label, model_name)
                field = model._meta.get_field(field_name)
                storage = getattr(field, 'storage', None)
                self.stdout.write(f'{model.__name__}.{field_name} -> field.storage: {storage.__class__}');
            except Exception as e:
                self.stdout.write(f'Failed to inspect {spec}: {e}')
