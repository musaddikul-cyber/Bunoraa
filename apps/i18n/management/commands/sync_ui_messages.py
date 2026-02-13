"""
Sync UI message JSON files into TranslationKey rows and optionally auto-translate.
"""
from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings

from apps.i18n.models import TranslationNamespace, TranslationKey, I18nSettings
from apps.i18n.tasks import auto_translate_key


def _flatten_messages(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_messages(value, full_key))
        else:
            flat[full_key] = str(value)
    return flat


class Command(BaseCommand):
    help = "Sync UI message JSON files into TranslationKey and enqueue translations"

    def add_arguments(self, parser):
        parser.add_argument(
            "--source-language",
            default=None,
            help="Source language code (default from I18nSettings)",
        )
        parser.add_argument(
            "--messages-dir",
            default=None,
            help="Base messages directory (default: frontend/messages/<lang>)",
        )
        parser.add_argument(
            "--no-translate",
            action="store_true",
            help="Only sync keys, do not auto-translate",
        )

    def handle(self, *args, **options):
        settings_obj = I18nSettings.get_settings()
        source_language = options["source_language"] or settings_obj.source_language or "en"

        base_dir = options["messages_dir"]
        if base_dir:
            messages_dir = Path(base_dir)
        else:
            messages_dir = Path(settings.BASE_DIR) / "frontend" / "messages" / source_language

        if not messages_dir.exists():
            self.stdout.write(self.style.ERROR(f"Messages directory not found: {messages_dir}"))
            return

        synced = 0
        for path in messages_dir.glob("*.json"):
            namespace = path.stem
            namespace_obj, _ = TranslationNamespace.objects.get_or_create(
                name=namespace,
                defaults={'description': f"UI namespace: {namespace}"}
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            flat = _flatten_messages(payload)

            for key, value in flat.items():
                tkey, _ = TranslationKey.objects.update_or_create(
                    namespace=namespace_obj,
                    key=key,
                    defaults={
                        'source_text': value,
                    }
                )
                synced += 1
                if not options["no_translate"]:
                    auto_translate_key.delay(namespace, key, value, source_language)

        self.stdout.write(self.style.SUCCESS(f"Synced {synced} UI message keys"))
