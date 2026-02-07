"""
Internationalization Celery Tasks

Background tasks for exchange rate updates, cleanup, and maintenance.
"""
import logging
from celery import shared_task
from django.utils import timezone
from django.core.cache import cache
from datetime import timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# Exchange Rate Tasks
# =============================================================================

@shared_task(name='i18n.update_exchange_rates')
def update_exchange_rates():
    """
    Update exchange rates from configured API provider.
    Scheduled to run based on I18nSettings.exchange_rate_update_frequency.
    """
    from .services import ExchangeRateUpdateService
    from .models import I18nSettings
    
    try:
        settings = I18nSettings.get_settings()
        
        if not settings.auto_update_exchange_rates:
            logger.info("Exchange rate auto-update is disabled")
            return {'status': 'skipped', 'reason': 'auto_update_disabled'}
        
        # Check if update is needed based on frequency
        if settings.last_exchange_rate_update:
            frequency_hours = {
                'hourly': 1,
                '6_hours': 6,
                '12_hours': 12,
                'daily': 24,
                'weekly': 168,
            }
            hours = frequency_hours.get(settings.exchange_rate_update_frequency, 24)
            next_update = settings.last_exchange_rate_update + timedelta(hours=hours)
            
            if timezone.now() < next_update:
                logger.info("Exchange rate update not yet due")
                return {'status': 'skipped', 'reason': 'not_due'}
        
        count = ExchangeRateUpdateService.update_rates()
        
        logger.info(f"Updated {count} exchange rates")
        return {
            'status': 'success',
            'rates_updated': count,
            'provider': settings.exchange_rate_provider
        }
        
    except Exception as e:
        logger.error(f"Error updating exchange rates: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='i18n.update_exchange_rates_from_provider')
def update_exchange_rates_from_provider(provider: str = None, api_key: str = None):
    """
    Update exchange rates from a specific provider.
    Can be called manually with override parameters.
    """
    from .services import ExchangeRateUpdateService
    from .models import I18nSettings
    
    try:
        settings = I18nSettings.get_settings()
        provider = provider or settings.exchange_rate_provider
        api_key = api_key or settings.exchange_rate_api_key
        
        count = 0
        
        if provider == 'openexchange':
            count = ExchangeRateUpdateService._update_from_openexchange(api_key)
        elif provider == 'fixer':
            count = ExchangeRateUpdateService._update_from_fixer(api_key)
        elif provider == 'ecb':
            count = ExchangeRateUpdateService._update_from_ecb()
        elif provider == 'exchangerate_api':
            count = ExchangeRateUpdateService._update_from_exchangerate_api(api_key)
        else:
            return {'status': 'error', 'error': f'Unknown provider: {provider}'}
        
        if count > 0:
            settings.last_exchange_rate_update = timezone.now()
            settings.save(update_fields=['last_exchange_rate_update'])
        
        logger.info(f"Updated {count} exchange rates from {provider}")
        return {
            'status': 'success',
            'rates_updated': count,
            'provider': provider
        }
        
    except Exception as e:
        logger.error(f"Error updating exchange rates from {provider}: {e}")
        return {'status': 'error', 'error': str(e)}


# =============================================================================
# Exchange Rate History Tasks
# =============================================================================

@shared_task(name='i18n.record_daily_exchange_rates')
def record_daily_exchange_rates():
    """
    Record daily exchange rate snapshots for historical tracking.
    Should run once per day.
    """
    from .models import Currency, ExchangeRate, ExchangeRateHistory
    
    try:
        today = timezone.now().date()
        count = 0
        
        # Get all active rates
        active_rates = ExchangeRate.objects.filter(is_active=True).select_related(
            'from_currency', 'to_currency'
        )
        
        for rate in active_rates:
            # Update or create history record
            history, created = ExchangeRateHistory.objects.update_or_create(
                from_currency=rate.from_currency,
                to_currency=rate.to_currency,
                date=today,
                defaults={
                    'rate': rate.rate,
                    'close_rate': rate.rate,
                    'source': rate.source,
                }
            )
            
            # If this is the first record of the day, set open rate
            if created:
                history.open_rate = rate.rate
                history.high_rate = rate.rate
                history.low_rate = rate.rate
                history.save(update_fields=['open_rate', 'high_rate', 'low_rate'])
            else:
                # Update high/low
                update_fields = []
                if history.high_rate is None or rate.rate > history.high_rate:
                    history.high_rate = rate.rate
                    update_fields.append('high_rate')
                if history.low_rate is None or rate.rate < history.low_rate:
                    history.low_rate = rate.rate
                    update_fields.append('low_rate')
                if update_fields:
                    history.save(update_fields=update_fields)
            
            count += 1
        
        logger.info(f"Recorded {count} daily exchange rate snapshots")
        return {'status': 'success', 'records_created': count}
        
    except Exception as e:
        logger.error(f"Error recording daily exchange rates: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='i18n.cleanup_old_exchange_rate_history')
def cleanup_old_exchange_rate_history(days_to_keep: int = 365):
    """
    Clean up old exchange rate history records.
    Keeps only the last N days of history.
    """
    from .models import ExchangeRateHistory
    
    try:
        cutoff_date = timezone.now().date() - timedelta(days=days_to_keep)
        
        deleted_count, _ = ExchangeRateHistory.objects.filter(
            date__lt=cutoff_date
        ).delete()
        
        logger.info(f"Deleted {deleted_count} old exchange rate history records")
        return {'status': 'success', 'deleted_count': deleted_count}
        
    except Exception as e:
        logger.error(f"Error cleaning up exchange rate history: {e}")
        return {'status': 'error', 'error': str(e)}


# =============================================================================
# Cache Management Tasks
# =============================================================================

@shared_task(name='i18n.clear_all_caches')
def clear_all_caches():
    """Clear all i18n related caches."""
    try:
        from .signals import CACHE_KEYS
        
        count = 0
        
        # Clear category caches
        for category, keys in CACHE_KEYS.items():
            for key in keys:
                cache.delete(key)
                count += 1
        
        # Clear pattern-based caches using wildcard (if supported by cache backend)
        cache_patterns = [
            'i18n_language_*',
            'i18n_currency_*',
            'i18n_rate_*',
            'i18n_trans_*',
            'i18n_content_*',
            'i18n_divisions_*',
        ]
        
        # Note: Pattern deletion requires cache backend support (Redis)
        # For other backends, we'd need to track keys differently
        
        logger.info(f"Cleared {count}+ i18n caches")
        return {'status': 'success', 'cleared_count': count}
        
    except Exception as e:
        logger.error(f"Error clearing i18n caches: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='i18n.warm_caches')
def warm_caches():
    """Warm up commonly used caches."""
    try:
        from .services import (
            LanguageService, CurrencyService, 
            TimezoneService, GeoService
        )
        
        # Warm language cache
        LanguageService.get_active_languages()
        LanguageService.get_default_language()
        
        # Warm currency cache
        CurrencyService.get_active_currencies()
        CurrencyService.get_default_currency()
        
        # Warm timezone cache
        TimezoneService.get_common_timezones()
        
        # Warm country cache
        GeoService.get_all_countries()
        GeoService.get_shipping_countries()
        
        logger.info("Warmed i18n caches")
        return {'status': 'success'}
        
    except Exception as e:
        logger.error(f"Error warming i18n caches: {e}")
        return {'status': 'error', 'error': str(e)}


# =============================================================================
# Translation Tasks
# =============================================================================

@shared_task(name='i18n.update_translation_progress')
def update_translation_progress():
    """Update translation progress for all languages."""
    from .models import Language, Translation, TranslationKey
    
    try:
        total_keys = TranslationKey.objects.count()
        updated_count = 0
        
        for language in Language.objects.filter(is_active=True):
            translated = Translation.objects.filter(
                language=language,
                status='approved'
            ).count()
            
            language.total_strings = total_keys
            language.translated_strings = translated
            language.save(update_fields=['total_strings', 'translated_strings'])
            updated_count += 1
        
        logger.info(f"Updated translation progress for {updated_count} languages")
        return {'status': 'success', 'languages_updated': updated_count}
        
    except Exception as e:
        logger.error(f"Error updating translation progress: {e}")
        return {'status': 'error', 'error': str(e)}


@shared_task(name='i18n.auto_translate_content')
def auto_translate_content(content_type: str, content_id: str, field_name: str, 
                           source_text: str, source_language: str = 'en'):
    """
    Auto-translate content using machine translation.
    Called when new content is created that needs translation.
    """
    from .models import Language, ContentTranslation, I18nSettings
    
    try:
        settings = I18nSettings.get_settings()
        
        if not settings.auto_translate_content:
            return {'status': 'skipped', 'reason': 'auto_translate_disabled'}
        
        provider = settings.machine_translation_provider
        api_key = settings.machine_translation_api_key
        
        if not provider or provider == 'none':
            return {'status': 'skipped', 'reason': 'no_provider_configured'}
        
        # Get target languages
        target_languages = Language.objects.filter(
            is_active=True
        ).exclude(code=source_language)
        
        translated_count = 0
        
        for target_lang in target_languages:
            try:
                translated_text = _machine_translate(
                    source_text, source_language, 
                    target_lang.code, provider, api_key
                )
                
                if translated_text:
                    ContentTranslation.objects.update_or_create(
                        content_type=content_type,
                        content_id=content_id,
                        field_name=field_name,
                        language=target_lang,
                        defaults={
                            'source_text': source_text,
                            'translated_text': translated_text,
                            'is_machine_translated': True,
                            'is_approved': False,  # Machine translations need review
                        }
                    )
                    translated_count += 1
                    
            except Exception as e:
                logger.error(f"Error translating to {target_lang.code}: {e}")
                continue
        
        logger.info(f"Auto-translated content to {translated_count} languages")
        return {
            'status': 'success',
            'languages_translated': translated_count,
            'provider': provider
        }
        
    except Exception as e:
        logger.error(f"Error in auto_translate_content: {e}")
        return {'status': 'error', 'error': str(e)}


def _machine_translate(text: str, source_lang: str, target_lang: str, 
                       provider: str, api_key: str) -> str:
    """Perform machine translation using configured provider."""
    import requests
    
    if provider == 'google':
        # Google Cloud Translation API
        url = 'https://translation.googleapis.com/language/translate/v2'
        params = {
            'key': api_key,
            'q': text,
            'source': source_lang,
            'target': target_lang,
            'format': 'text'
        }
        response = requests.post(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['data']['translations'][0]['translatedText']
    
    elif provider == 'deepl':
        # DeepL API
        url = 'https://api-free.deepl.com/v2/translate'
        data = {
            'auth_key': api_key,
            'text': text,
            'source_lang': source_lang.upper(),
            'target_lang': target_lang.upper()
        }
        response = requests.post(url, data=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['translations'][0]['text']
    
    elif provider == 'azure':
        # Azure Translator
        url = 'https://api.cognitive.microsofttranslator.com/translate'
        params = {
            'api-version': '3.0',
            'from': source_lang,
            'to': target_lang
        }
        headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Content-Type': 'application/json'
        }
        body = [{'text': text}]
        response = requests.post(url, params=params, headers=headers, 
                                json=body, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result[0]['translations'][0]['text']
    
    return None


# =============================================================================
# Periodic Task Registration
# =============================================================================

@shared_task(name='i18n.register_periodic_tasks')
def register_periodic_tasks():
    """
    Register periodic tasks with Celery Beat.
    Call this after deployment to set up scheduled tasks.
    """
    from django_celery_beat.models import PeriodicTask, IntervalSchedule, CrontabSchedule
    import json
    
    try:
        # Exchange rate update - every 6 hours
        schedule_6h, _ = IntervalSchedule.objects.get_or_create(
            every=6,
            period=IntervalSchedule.HOURS
        )
        
        PeriodicTask.objects.update_or_create(
            name='Update Exchange Rates',
            defaults={
                'task': 'i18n.update_exchange_rates',
                'interval': schedule_6h,
                'enabled': True,
            }
        )
        
        # Daily rate snapshots - every day at midnight
        schedule_midnight, _ = CrontabSchedule.objects.get_or_create(
            minute='0',
            hour='0',
            day_of_week='*',
            day_of_month='*',
            month_of_year='*'
        )
        
        PeriodicTask.objects.update_or_create(
            name='Record Daily Exchange Rates',
            defaults={
                'task': 'i18n.record_daily_exchange_rates',
                'crontab': schedule_midnight,
                'enabled': True,
            }
        )
        
        # Monthly cleanup - first day of month
        schedule_monthly, _ = CrontabSchedule.objects.get_or_create(
            minute='0',
            hour='2',
            day_of_week='*',
            day_of_month='1',
            month_of_year='*'
        )
        
        PeriodicTask.objects.update_or_create(
            name='Cleanup Old Exchange Rate History',
            defaults={
                'task': 'i18n.cleanup_old_exchange_rate_history',
                'crontab': schedule_monthly,
                'kwargs': json.dumps({'days_to_keep': 365}),
                'enabled': True,
            }
        )
        
        # Translation progress update - every day at 3 AM
        schedule_3am, _ = CrontabSchedule.objects.get_or_create(
            minute='0',
            hour='3',
            day_of_week='*',
            day_of_month='*',
            month_of_year='*'
        )
        
        PeriodicTask.objects.update_or_create(
            name='Update Translation Progress',
            defaults={
                'task': 'i18n.update_translation_progress',
                'crontab': schedule_3am,
                'enabled': True,
            }
        )
        
        logger.info("Registered i18n periodic tasks")
        return {'status': 'success'}
        
    except Exception as e:
        logger.error(f"Error registering periodic tasks: {e}")
        return {'status': 'error', 'error': str(e)}
