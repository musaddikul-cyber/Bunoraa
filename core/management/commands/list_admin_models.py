from django.core.management.base import BaseCommand
from django.contrib import admin

class Command(BaseCommand):
    help = 'List registered admin models and their app labels'

    def handle(self, *args, **options):
        registry = admin.site._registry
        apps = {}
        for model, model_admin in registry.items():
            app_label = model._meta.app_label
            models_for_app = apps.setdefault(app_label, [])
            models_for_app.append((model.__name__, type(model_admin).__name__))

        for app_label in sorted(apps.keys()):
            self.stdout.write(f'App: {app_label}')
            for model_name, admin_name in sorted(apps[app_label]):
                self.stdout.write(f'  - {model_name} (admin: {admin_name})')
            self.stdout.write('')
