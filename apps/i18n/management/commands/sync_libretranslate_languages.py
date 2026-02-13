"""
Sync languages from a LibreTranslate instance into the Language model.
"""
from django.core.management.base import BaseCommand

from apps.i18n.models import Language, I18nSettings
from apps.i18n.services import LanguageService, MachineTranslationService


RTL_LANGUAGES = {
    'ar', 'fa', 'he', 'ur', 'ps', 'dv', 'ku', 'yi',
}


class Command(BaseCommand):
    help = "Sync languages from LibreTranslate into the database"

    def handle(self, *args, **options):
        languages = MachineTranslationService.list_languages()
        if not languages:
            self.stdout.write(self.style.WARNING("No languages returned from LibreTranslate"))
            return

        settings = I18nSettings.get_settings()
        default_code = settings.default_language.code if settings.default_language else 'en'

        created = 0
        updated = 0

        for entry in languages:
            code = entry.get('code') or entry.get('language') or ''
            name = entry.get('name') or code
            code = LanguageService.normalize_code(code)
            if not code:
                continue

            is_rtl = code.split('-')[0] in RTL_LANGUAGES

            obj, was_created = Language.objects.update_or_create(
                code=code,
                defaults={
                    'name': name,
                    'native_name': name,
                    'locale_code': code,
                    'is_active': True,
                    'is_rtl': is_rtl,
                    'is_default': code == default_code,
                }
            )
            if was_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(self.style.SUCCESS(
            f"Synced languages. Created: {created}, Updated: {updated}"
        ))
