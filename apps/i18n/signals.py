"""
Internationalization Signals

Signal handlers for cache invalidation and data synchronization.
"""
import logging
from django.db.models.signals import post_save, post_delete, pre_save
from django.dispatch import receiver
from django.core.cache import cache
from django.conf import settings

from .models import (
    Language, Currency, ExchangeRate, ExchangeRateHistory,
    Timezone, Country, Division, District, Upazila,
    Translation, TranslationKey, ContentTranslation,
    UserLocalePreference, I18nSettings
)

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Keys
# =============================================================================

CACHE_KEYS = {
    'languages': [
        'i18n_active_languages',
        'i18n_default_language',
    ],
    'currencies': [
        'i18n_active_currencies',
        'i18n_default_currency',
    ],
    'timezones': [
        'i18n_all_timezones',
        'i18n_common_timezones',
    ],
    'countries': [
        'i18n_all_countries',
        'i18n_shipping_countries',
    ],
    'settings': [
        'i18n_settings',
    ],
}


def clear_cache_keys(category: str):
    """Clear all cache keys for a category."""
    keys = CACHE_KEYS.get(category, [])
    for key in keys:
        cache.delete(key)
    logger.debug(f"Cleared {len(keys)} cache keys for category: {category}")


# =============================================================================
# Language Signals
# =============================================================================

@receiver(post_save, sender=Language)
def language_post_save(sender, instance, created, **kwargs):
    """Handle language save."""
    # Clear language caches
    clear_cache_keys('languages')
    cache.delete(f'i18n_language_{instance.code}')
    
    # Ensure only one default
    if instance.is_default:
        Language.objects.filter(is_default=True).exclude(pk=instance.pk).update(is_default=False)
    
    logger.info(f"Language {'created' if created else 'updated'}: {instance.code}")


@receiver(post_delete, sender=Language)
def language_post_delete(sender, instance, **kwargs):
    """Handle language deletion."""
    clear_cache_keys('languages')
    cache.delete(f'i18n_language_{instance.code}')
    logger.info(f"Language deleted: {instance.code}")


# =============================================================================
# Currency Signals
# =============================================================================

@receiver(post_save, sender=Currency)
def currency_post_save(sender, instance, created, **kwargs):
    """Handle currency save."""
    # Clear currency caches
    clear_cache_keys('currencies')
    cache.delete(f'i18n_currency_{instance.code}')
    
    # Ensure only one default
    if instance.is_default:
        Currency.objects.filter(is_default=True).exclude(pk=instance.pk).update(is_default=False)
    
    logger.info(f"Currency {'created' if created else 'updated'}: {instance.code}")


@receiver(post_delete, sender=Currency)
def currency_post_delete(sender, instance, **kwargs):
    """Handle currency deletion."""
    clear_cache_keys('currencies')
    cache.delete(f'i18n_currency_{instance.code}')
    logger.info(f"Currency deleted: {instance.code}")


# =============================================================================
# Exchange Rate Signals
# =============================================================================

@receiver(post_save, sender=ExchangeRate)
def exchange_rate_post_save(sender, instance, created, **kwargs):
    """Handle exchange rate save."""
    # Clear rate caches
    cache.delete(f'i18n_rate_{instance.from_currency.code}_{instance.to_currency.code}')
    cache.delete(f'i18n_rate_{instance.to_currency.code}_{instance.from_currency.code}')
    
    # Update currency's exchange_rate field if this is the active rate
    if instance.is_active and instance.from_currency.is_default:
        try:
            instance.to_currency.exchange_rate = instance.rate
            instance.to_currency.save(update_fields=['exchange_rate', 'last_rate_update'])
        except Exception as e:
            logger.error(f"Error updating currency exchange rate: {e}")
    
    logger.info(f"Exchange rate {'created' if created else 'updated'}: "
                f"{instance.from_currency.code}/{instance.to_currency.code}")


@receiver(post_delete, sender=ExchangeRate)
def exchange_rate_post_delete(sender, instance, **kwargs):
    """Handle exchange rate deletion."""
    cache.delete(f'i18n_rate_{instance.from_currency.code}_{instance.to_currency.code}')
    cache.delete(f'i18n_rate_{instance.to_currency.code}_{instance.from_currency.code}')
    logger.info(f"Exchange rate deleted: {instance.from_currency.code}/{instance.to_currency.code}")


# =============================================================================
# Timezone Signals
# =============================================================================

@receiver(post_save, sender=Timezone)
def timezone_post_save(sender, instance, created, **kwargs):
    """Handle timezone save."""
    clear_cache_keys('timezones')
    logger.info(f"Timezone {'created' if created else 'updated'}: {instance.name}")


@receiver(post_delete, sender=Timezone)
def timezone_post_delete(sender, instance, **kwargs):
    """Handle timezone deletion."""
    clear_cache_keys('timezones')
    logger.info(f"Timezone deleted: {instance.name}")


# =============================================================================
# Country Signals
# =============================================================================

@receiver(post_save, sender=Country)
def country_post_save(sender, instance, created, **kwargs):
    """Handle country save."""
    clear_cache_keys('countries')
    logger.info(f"Country {'created' if created else 'updated'}: {instance.code}")


@receiver(post_delete, sender=Country)
def country_post_delete(sender, instance, **kwargs):
    """Handle country deletion."""
    clear_cache_keys('countries')
    logger.info(f"Country deleted: {instance.code}")


# =============================================================================
# Division/District/Upazila Signals
# =============================================================================

@receiver(post_save, sender=Division)
def division_post_save(sender, instance, created, **kwargs):
    """Handle division save."""
    cache.delete(f'i18n_divisions_{instance.country.code}')
    logger.debug(f"Division {'created' if created else 'updated'}: {instance.name}")


@receiver(post_delete, sender=Division)
def division_post_delete(sender, instance, **kwargs):
    """Handle division deletion."""
    cache.delete(f'i18n_divisions_{instance.country.code}')
    logger.debug(f"Division deleted: {instance.name}")


@receiver(post_save, sender=District)
def district_post_save(sender, instance, created, **kwargs):
    """Handle district save."""
    # Districts are typically loaded via their division, so clear division cache
    cache.delete(f'i18n_divisions_{instance.division.country.code}')
    logger.debug(f"District {'created' if created else 'updated'}: {instance.name}")


@receiver(post_save, sender=Upazila)
def upazila_post_save(sender, instance, created, **kwargs):
    """Handle upazila save."""
    logger.debug(f"Upazila {'created' if created else 'updated'}: {instance.name}")


# =============================================================================
# Translation Signals
# =============================================================================

@receiver(post_save, sender=Translation)
def translation_post_save(sender, instance, created, **kwargs):
    """Handle translation save."""
    # Clear specific translation cache
    namespace = instance.key.namespace.name if instance.key.namespace else 'default'
    cache.delete(f'i18n_trans_{namespace}_{instance.key.key}_{instance.language.code}')
    
    # Update language translation progress
    if instance.status == 'approved':
        try:
            language = instance.language
            total = TranslationKey.objects.count()
            translated = Translation.objects.filter(
                language=language,
                status='approved'
            ).count()
            language.total_strings = total
            language.translated_strings = translated
            language.save(update_fields=['total_strings', 'translated_strings'])
        except Exception as e:
            logger.error(f"Error updating translation progress: {e}")
    
    logger.debug(f"Translation {'created' if created else 'updated'}: {instance.key.key} ({instance.language.code})")


@receiver(post_delete, sender=Translation)
def translation_post_delete(sender, instance, **kwargs):
    """Handle translation deletion."""
    namespace = instance.key.namespace.name if instance.key.namespace else 'default'
    cache.delete(f'i18n_trans_{namespace}_{instance.key.key}_{instance.language.code}')
    logger.debug(f"Translation deleted: {instance.key.key} ({instance.language.code})")


@receiver(post_save, sender=ContentTranslation)
def content_translation_post_save(sender, instance, created, **kwargs):
    """Handle content translation save."""
    cache.delete(f'i18n_content_{instance.content_type}_{instance.content_id}'
                 f'_{instance.field_name}_{instance.language.code}')
    logger.debug(f"Content translation {'created' if created else 'updated'}: "
                 f"{instance.content_type}/{instance.content_id}/{instance.field_name}")


@receiver(post_delete, sender=ContentTranslation)
def content_translation_post_delete(sender, instance, **kwargs):
    """Handle content translation deletion."""
    cache.delete(f'i18n_content_{instance.content_type}_{instance.content_id}'
                 f'_{instance.field_name}_{instance.language.code}')
    logger.debug(f"Content translation deleted: {instance.content_type}/{instance.content_id}/{instance.field_name}")


# =============================================================================
# User Preference Signals
# =============================================================================

@receiver(post_save, sender=UserLocalePreference)
def user_preference_post_save(sender, instance, created, **kwargs):
    """Handle user preference save."""
    # Could update session here if we had request context
    logger.debug(f"User locale preference {'created' if created else 'updated'} for user: {instance.user_id}")


# =============================================================================
# Settings Signals
# =============================================================================

@receiver(post_save, sender=I18nSettings)
def settings_post_save(sender, instance, created, **kwargs):
    """Handle settings save."""
    clear_cache_keys('settings')
    logger.info("I18n settings updated")


# =============================================================================
# User Creation Signal (Auto-create preference)
# =============================================================================

def user_created_handler(sender, instance, created, **kwargs):
    """Create locale preference when user is created."""
    if created:
        try:
            UserLocalePreference.objects.get_or_create(user=instance)
            logger.debug(f"Created locale preference for new user: {instance.id}")
        except Exception as e:
            logger.error(f"Error creating locale preference for user: {e}")


def connect_user_signal():
    """Connect to User model's post_save signal."""
    from django.contrib.auth import get_user_model
    User = get_user_model()
    post_save.connect(user_created_handler, sender=User)


# Connect user signal when app is ready (called from apps.py)
