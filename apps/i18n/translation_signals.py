"""
Signals to enqueue automatic translation for registered content models and keys.
"""
from __future__ import annotations

import logging
from django.apps import apps as django_apps
from django.db.models.signals import post_save

from .translation_registry import get_translation_targets
from .tasks import auto_translate_content, auto_translate_key, auto_translate_notification_template
from .models import I18nSettings, TranslationKey

logger = logging.getLogger(__name__)


def _should_auto_translate() -> bool:
    try:
        settings = I18nSettings.get_settings()
        return settings.enable_machine_translation and settings.auto_translate_new_content
    except Exception:
        return False


def _content_handler_factory(content_type: str, fields: list[str]):
    def _handler(sender, instance, created, **kwargs):
        if not _should_auto_translate():
            return
        try:
            settings = I18nSettings.get_settings()
            source_language = settings.source_language or 'en'
        except Exception:
            source_language = 'en'

        for field in fields:
            value = getattr(instance, field, None)
            if not value:
                continue
            try:
                auto_translate_content.delay(
                    content_type=content_type,
                    content_id=str(instance.pk),
                    field_name=field,
                    source_text=str(value),
                    source_language=source_language,
                )
            except Exception as exc:
                logger.error(f"Failed to enqueue translation for {content_type}.{field}: {exc}")
    return _handler


def _translation_key_handler(sender, instance: TranslationKey, created, **kwargs):
    if not _should_auto_translate():
        return
    if not instance or not instance.source_text:
        return
    try:
        settings = I18nSettings.get_settings()
        source_language = settings.source_language or 'en'
    except Exception:
        source_language = 'en'

    namespace = instance.namespace.name if instance.namespace else 'default'
    try:
        auto_translate_key.delay(
            namespace=namespace,
            key=instance.key,
            source_text=instance.source_text,
            source_language=source_language,
        )
    except Exception as exc:
        logger.error(f"Failed to enqueue translation key {namespace}:{instance.key}: {exc}")


def register_translation_signals():
    for target in get_translation_targets():
        try:
            model = django_apps.get_model(target.model_label)
        except Exception:
            model = None
        if model is None:
            continue
        post_save.connect(
            _content_handler_factory(target.content_type, target.fields),
            sender=model,
            weak=False,
            dispatch_uid=f"i18n_autotranslate_{target.model_label}",
        )

    post_save.connect(
        _translation_key_handler,
        sender=TranslationKey,
        weak=False,
        dispatch_uid="i18n_translate_translation_key",
    )
