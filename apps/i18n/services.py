"""
Internationalization Services

Comprehensive services for multi-language, multi-currency, and localization operations.
"""
import logging
from decimal import Decimal, ROUND_HALF_UP, ROUND_UP
from typing import Optional, List, Dict, Any
from django.db import models, transaction
from django.utils import timezone
from django.core.cache import cache
from django.conf import settings as django_settings

from .models import (
    Language, Currency, ExchangeRate, ExchangeRateHistory,
    Timezone, Country, Division, District, Upazila,
    Translation, TranslationKey, ContentTranslation,
    UserLocalePreference, I18nSettings
)

logger = logging.getLogger(__name__)


# =============================================================================
# Language Service
# =============================================================================

class LanguageService:
    """Service for language operations."""
    
    CACHE_TIMEOUT = 3600  # 1 hour
    
    @staticmethod
    def get_active_languages() -> List[Language]:
        """Get all active languages."""
        cache_key = 'i18n_active_languages'
        languages = cache.get(cache_key)
        
        if languages is None:
            languages = list(Language.objects.filter(is_active=True).order_by('sort_order', 'name'))
            cache.set(cache_key, languages, LanguageService.CACHE_TIMEOUT)
        
        return languages
    
    @staticmethod
    def get_default_language() -> Optional[Language]:
        """Get default language."""
        cache_key = 'i18n_default_language'
        language = cache.get(cache_key)
        
        if language is None:
            language = Language.objects.filter(is_default=True).first()
            if not language:
                language = Language.objects.filter(is_active=True).first()
            if language:
                cache.set(cache_key, language, LanguageService.CACHE_TIMEOUT)
        
        return language
    
    @staticmethod
    def get_language_by_code(code: str) -> Optional[Language]:
        """Get language by code."""
        cache_key = f'i18n_language_{code}'
        language = cache.get(cache_key)
        
        if language is None:
            language = Language.objects.filter(code=code, is_active=True).first()
            if language:
                cache.set(cache_key, language, LanguageService.CACHE_TIMEOUT)
        
        return language
    
    @staticmethod
    def detect_language(request) -> Optional[Language]:
        """Detect language from request."""
        # Priority 1: Query parameter
        if request.GET.get('lang'):
            lang = LanguageService.get_language_by_code(request.GET['lang'])
            if lang:
                return lang
        
        # Priority 2: Session
        if hasattr(request, 'session'):
            lang_code = request.session.get('language')
            if lang_code:
                lang = LanguageService.get_language_by_code(lang_code)
                if lang:
                    return lang
        
        # Priority 3: Cookie
        if hasattr(request, 'COOKIES'):
            lang_code = request.COOKIES.get('language')
            if lang_code:
                lang = LanguageService.get_language_by_code(lang_code)
                if lang:
                    return lang
        
        # Priority 4: Accept-Language header
        accept_lang = request.META.get('HTTP_ACCEPT_LANGUAGE', '')
        if accept_lang:
            for lang_item in accept_lang.split(','):
                lang_code = lang_item.split(';')[0].split('-')[0].strip()
                lang = LanguageService.get_language_by_code(lang_code)
                if lang:
                    return lang
        
        return LanguageService.get_default_language()
    
    @staticmethod
    def get_user_language(user=None, request=None) -> Optional[Language]:
        """Get language for user or request."""
        # Check user preference
        if user and user.is_authenticated:
            try:
                pref = UserLocalePreference.objects.filter(user=user).select_related('language').first()
                if pref and pref.language and not pref.auto_detect_language:
                    return pref.language
            except Exception:
                pass
        
        # Detect from request
        if request:
            return LanguageService.detect_language(request)
        
        return LanguageService.get_default_language()
    
    @staticmethod
    def set_user_language(user, language_code: str) -> Optional[UserLocalePreference]:
        """Set user's preferred language."""
        language = LanguageService.get_language_by_code(language_code)
        if not language:
            return None
        
        pref, _ = UserLocalePreference.objects.get_or_create(user=user)
        pref.language = language
        pref.auto_detect_language = False
        pref.save()
        
        return pref


# =============================================================================
# Currency Service
# =============================================================================

class CurrencyService:
    """Service for currency operations."""
    
    CACHE_TIMEOUT = 3600  # 1 hour
    
    @staticmethod
    def get_active_currencies() -> List[Currency]:
        """Get all active currencies."""
        cache_key = 'i18n_active_currencies'
        currencies = cache.get(cache_key)
        
        if currencies is None:
            currencies = list(Currency.objects.filter(is_active=True).order_by('sort_order', 'code'))
            cache.set(cache_key, currencies, CurrencyService.CACHE_TIMEOUT)
        
        return currencies
    
    @staticmethod
    def get_default_currency() -> Optional[Currency]:
        """Get default currency."""
        cache_key = 'i18n_default_currency_v2'
        currency = cache.get(cache_key)
        
        if currency is None:
            currency = None
            try:
                settings = I18nSettings.get_settings()
                if settings.default_currency and settings.default_currency.is_active:
                    currency = settings.default_currency
            except Exception:
                currency = None

            if not currency:
                currency = Currency.objects.filter(code='BDT', is_active=True).first()

            if not currency:
                currency = Currency.objects.filter(is_default=True).first()
            if not currency:
                currency = Currency.objects.filter(is_active=True).first()
            if currency:
                cache.set(cache_key, currency, CurrencyService.CACHE_TIMEOUT)
        
        return currency
    
    @staticmethod
    def get_currency_by_code(code: str) -> Optional[Currency]:
        """Get currency by code."""
        cache_key = f'i18n_currency_{code.upper()}'
        currency = cache.get(cache_key)
        
        if currency is None:
            currency = Currency.objects.filter(code=code.upper(), is_active=True).first()
            if currency:
                cache.set(cache_key, currency, CurrencyService.CACHE_TIMEOUT)
        
        return currency
    
    @staticmethod
    def detect_currency(request) -> Optional[Currency]:
        """Detect currency from request."""
        # Priority 1: Query parameter
        if request.GET.get('currency'):
            curr = CurrencyService.get_currency_by_code(request.GET['currency'])
            if curr:
                return curr
        
        # Priority 2: Custom header
        header_currency = request.META.get('HTTP_X_USER_CURRENCY')
        if header_currency:
            curr = CurrencyService.get_currency_by_code(header_currency)
            if curr:
                return curr
        
        # Priority 3: Session
        if hasattr(request, 'session'):
            curr_code = request.session.get('currency_code')
            if curr_code:
                curr = CurrencyService.get_currency_by_code(curr_code)
                if curr:
                    return curr
        
        # Priority 4: Cookie
        if hasattr(request, 'COOKIES'):
            curr_code = request.COOKIES.get('currency')
            if curr_code:
                curr = CurrencyService.get_currency_by_code(curr_code)
                if curr:
                    return curr

        # Priority 5: Geo-location (only if allowed by settings)
        try:
            settings = I18nSettings.get_settings()
            if not settings.auto_detect_currency:
                return None
        except Exception:
            pass

        currency = CurrencyService._detect_from_geo(request)
        if currency:
            return currency

        return None
    
    @staticmethod
    def _detect_from_geo(request) -> Optional[Currency]:
        """Detect currency from geo-location."""
        # Country to currency mapping
        COUNTRY_CURRENCIES = {
            'BD': 'BDT', 'IN': 'INR', 'US': 'USD', 'GB': 'GBP',
            'CA': 'CAD', 'AU': 'AUD', 'DE': 'EUR', 'FR': 'EUR',
            'IT': 'EUR', 'ES': 'EUR', 'NL': 'EUR', 'JP': 'JPY',
            'CN': 'CNY', 'KR': 'KRW', 'SG': 'SGD', 'AE': 'AED',
            'SA': 'SAR', 'MY': 'MYR', 'PK': 'PKR', 'NP': 'NPR',
        }
        
        # Try Cloudflare header
        country_code = request.META.get('HTTP_CF_IPCOUNTRY')
        
        # Try AWS CloudFront
        if not country_code:
            country_code = request.META.get('HTTP_CLOUDFRONT_VIEWER_COUNTRY')
        
        # Try generic GeoIP header
        if not country_code:
            country_code = request.META.get('HTTP_X_COUNTRY_CODE')
        
        if country_code and country_code in COUNTRY_CURRENCIES:
            return CurrencyService.get_currency_by_code(COUNTRY_CURRENCIES[country_code])
        
        return None
    
    @staticmethod
    def get_user_currency(user=None, request=None) -> Optional[Currency]:
        """Get currency for user or request."""
        import logging
        logger = logging.getLogger('bunoraa.i18n')
        
        # Check force default setting
        if getattr(django_settings, 'FORCE_DEFAULT_CURRENCY', False):
            logger.debug("[Currency] FORCE_DEFAULT_CURRENCY is set, returning default")
            return CurrencyService.get_default_currency()
        
        # Check user preference
        if user and user.is_authenticated:
            try:
                pref = UserLocalePreference.objects.filter(user=user).select_related('currency').first()
                logger.debug(f"[Currency] User {user.id} preference: currency={pref.currency if pref else None}, auto_detect={pref.auto_detect_currency if pref else None}")
                if pref and pref.currency and not pref.auto_detect_currency:
                    logger.debug(f"[Currency] Returning user preference: {pref.currency.code}")
                    return pref.currency
            except Exception as e:
                logger.warning(f"[Currency] Error getting user preference: {e}")
        
        # Detect from request
        if request:
            detected = CurrencyService.detect_currency(request)
            logger.debug(f"[Currency] Detected from request: {detected.code if detected else None}")
            if detected:
                return detected

        return CurrencyService.get_default_currency()
    
    @staticmethod
    def set_user_currency(user, currency_code: str) -> Optional[UserLocalePreference]:
        """Set user's preferred currency."""
        currency = CurrencyService.get_currency_by_code(currency_code)
        if not currency:
            return None
        
        pref, _ = UserLocalePreference.objects.get_or_create(user=user)
        pref.currency = currency
        pref.auto_detect_currency = False
        pref.save()
        
        return pref


# =============================================================================
# Exchange Rate Service
# =============================================================================

class ExchangeRateService:
    """Service for exchange rate operations."""
    
    CACHE_TIMEOUT = 300  # 5 minutes
    
    @staticmethod
    def get_exchange_rate(from_currency: Currency, to_currency: Currency) -> Optional[Decimal]:
        """Get current exchange rate between two currencies."""
        if from_currency.id == to_currency.id:
            return Decimal('1')
        
        cache_key = f'i18n_rate_{from_currency.code}_{to_currency.code}'
        rate = cache.get(cache_key)
        
        if rate is None:
            rate = ExchangeRateService._fetch_rate(from_currency, to_currency)
            if rate:
                cache.set(cache_key, rate, ExchangeRateService.CACHE_TIMEOUT)
        
        return rate
    
    @staticmethod
    def _fetch_rate(from_currency: Currency, to_currency: Currency) -> Optional[Decimal]:
        """Fetch rate from database."""
        now = timezone.now()
        
        # Try direct rate
        rate_obj = ExchangeRate.objects.filter(
            from_currency=from_currency,
            to_currency=to_currency,
            is_active=True,
            valid_from__lte=now
        ).filter(
            models.Q(valid_until__isnull=True) | models.Q(valid_until__gte=now)
        ).order_by('-valid_from').first()
        
        if rate_obj:
            return rate_obj.rate
        
        # Try inverse rate
        inverse = ExchangeRate.objects.filter(
            from_currency=to_currency,
            to_currency=from_currency,
            is_active=True,
            valid_from__lte=now
        ).filter(
            models.Q(valid_until__isnull=True) | models.Q(valid_until__gte=now)
        ).order_by('-valid_from').first()
        
        if inverse:
            return Decimal('1') / inverse.rate
        
        # Try via base currency
        return ExchangeRateService._get_rate_via_base(from_currency, to_currency)
    
    @staticmethod
    def _get_rate_via_base(from_currency: Currency, to_currency: Currency) -> Optional[Decimal]:
        """Calculate rate via base currency (EUR or USD)."""
        for base_code in ['USD', 'EUR']:
            base = Currency.objects.filter(code=base_code, is_active=True).first()
            if not base or base.id in (from_currency.id, to_currency.id):
                continue
            
            rate = ExchangeRateService._calculate_via_base(from_currency, to_currency, base)
            if rate:
                return rate
        
        return None
    
    @staticmethod
    def _calculate_via_base(from_currency: Currency, to_currency: Currency, base: Currency) -> Optional[Decimal]:
        """Calculate rate via specific base currency."""
        now = timezone.now()
        
        # Get from_currency -> base rate
        from_to_base = ExchangeRateService._get_single_rate(from_currency, base, now)
        if from_to_base is None:
            return None
        
        # Get base -> to_currency rate
        base_to_target = ExchangeRateService._get_single_rate(base, to_currency, now)
        if base_to_target is None:
            return None
        
        return from_to_base * base_to_target
    
    @staticmethod
    def _get_single_rate(from_curr: Currency, to_curr: Currency, now) -> Optional[Decimal]:
        """Get single rate (direct or inverse)."""
        # Direct
        rate_obj = ExchangeRate.objects.filter(
            from_currency=from_curr,
            to_currency=to_curr,
            is_active=True,
            valid_from__lte=now
        ).filter(
            models.Q(valid_until__isnull=True) | models.Q(valid_until__gte=now)
        ).order_by('-valid_from').first()
        
        if rate_obj:
            return rate_obj.rate
        
        # Inverse
        inverse = ExchangeRate.objects.filter(
            from_currency=to_curr,
            to_currency=from_curr,
            is_active=True,
            valid_from__lte=now
        ).filter(
            models.Q(valid_until__isnull=True) | models.Q(valid_until__gte=now)
        ).order_by('-valid_from').first()
        
        if inverse:
            return Decimal('1') / inverse.rate
        
        return None
    
    @staticmethod
    @transaction.atomic
    def set_exchange_rate(
        from_currency: Currency,
        to_currency: Currency,
        rate: Decimal,
        source: str = 'manual',
        bid_rate: Optional[Decimal] = None,
        ask_rate: Optional[Decimal] = None
    ) -> ExchangeRate:
        """Set exchange rate."""
        now = timezone.now()
        
        # Deactivate old rates
        ExchangeRate.objects.filter(
            from_currency=from_currency,
            to_currency=to_currency,
            is_active=True
        ).update(is_active=False, valid_until=now)
        
        # Create new rate
        exchange_rate = ExchangeRate.objects.create(
            from_currency=from_currency,
            to_currency=to_currency,
            rate=Decimal(str(rate)),
            bid_rate=Decimal(str(bid_rate)) if bid_rate else None,
            ask_rate=Decimal(str(ask_rate)) if ask_rate else None,
            source=source,
            valid_from=now
        )
        
        # Record history
        ExchangeRateHistory.objects.update_or_create(
            from_currency=from_currency,
            to_currency=to_currency,
            date=now.date(),
            defaults={
                'rate': Decimal(str(rate)),
                'source': source
            }
        )
        
        # Clear cache
        cache.delete(f'i18n_rate_{from_currency.code}_{to_currency.code}')
        cache.delete(f'i18n_rate_{to_currency.code}_{from_currency.code}')
        
        return exchange_rate
    
    @staticmethod
    def get_rate_history(
        from_currency: Currency,
        to_currency: Currency,
        days: int = 30
    ) -> List[ExchangeRateHistory]:
        """Get historical exchange rates."""
        cutoff = (timezone.now() - timezone.timedelta(days=days)).date()
        return list(ExchangeRateHistory.objects.filter(
            from_currency=from_currency,
            to_currency=to_currency,
            date__gte=cutoff
        ).order_by('date'))


# =============================================================================
# Currency Conversion Service
# =============================================================================

class CurrencyConversionService:
    """Service for converting amounts between currencies."""
    
    @staticmethod
    def convert(
        amount: Decimal,
        from_currency: Currency,
        to_currency: Currency,
        round_result: bool = True
    ) -> Decimal:
        """Convert an amount from one currency to another."""
        if from_currency.id == to_currency.id:
            return Decimal(str(amount))
        
        rate = ExchangeRateService.get_exchange_rate(from_currency, to_currency)
        
        if rate is None:
            raise ValueError(f"No exchange rate available for {from_currency.code} to {to_currency.code}")
        
        converted = Decimal(str(amount)) * rate
        
        if round_result:
            converted = CurrencyConversionService._apply_rounding(converted, to_currency)
        
        return converted
    
    @staticmethod
    def convert_by_code(
        amount: Decimal,
        from_code: str,
        to_code: str,
        round_result: bool = True
    ) -> Decimal:
        """Convert amount using currency codes."""
        from_currency = CurrencyService.get_currency_by_code(from_code)
        to_currency = CurrencyService.get_currency_by_code(to_code)
        
        if not from_currency:
            raise ValueError(f"Currency not found: {from_code}")
        if not to_currency:
            raise ValueError(f"Currency not found: {to_code}")
        
        return CurrencyConversionService.convert(amount, from_currency, to_currency, round_result)
    
    @staticmethod
    def _apply_rounding(amount: Decimal, currency: Currency) -> Decimal:
        """Apply rounding based on settings."""
        try:
            settings = I18nSettings.get_settings()
            method = settings.rounding_method
        except Exception:
            method = 'nearest_cent'
        
        if method == 'none':
            return amount.quantize(Decimal(10) ** -currency.decimal_places, rounding=ROUND_HALF_UP)
        elif method == 'nearest_cent':
            return amount.quantize(Decimal(10) ** -currency.decimal_places, rounding=ROUND_HALF_UP)
        elif method == 'nearest_99':
            integer = int(amount)
            return Decimal(f'{integer}.99')
        elif method == 'nearest_95':
            integer = int(amount)
            return Decimal(f'{integer}.95')
        elif method == 'nearest_integer':
            return amount.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        elif method == 'round_up':
            return amount.quantize(Decimal(10) ** -currency.decimal_places, rounding=ROUND_UP)
        
        return amount.quantize(Decimal(10) ** -currency.decimal_places, rounding=ROUND_HALF_UP)
    
    @staticmethod
    def format_price(
        amount: Decimal,
        currency: Currency,
        include_symbol: bool = True,
        use_native: bool = False
    ) -> str:
        """Format a price in a specific currency."""
        if include_symbol:
            return currency.format_amount(amount, use_native_symbol=use_native)
        return currency._format_number(amount)


# =============================================================================
# Exchange Rate Update Service
# =============================================================================

class ExchangeRateUpdateService:
    """Service for updating exchange rates from external APIs."""
    
    # Provider order for fallback
    PROVIDERS = ['exchangerate_api', 'openexchange', 'exchangeratesapi', 'fixer', 'ecb']
    
    @staticmethod
    def update_rates() -> int:
        """Update exchange rates from configured provider."""
        try:
            settings = I18nSettings.get_settings()
        except Exception:
            return 0
        
        if not settings.auto_update_exchange_rates:
            return 0
        
        provider = settings.exchange_rate_provider
        api_key = settings.exchange_rate_api_key
        
        count = 0
        
        if provider == 'openexchange':
            count = ExchangeRateUpdateService._update_from_openexchange(api_key)
        elif provider == 'fixer':
            count = ExchangeRateUpdateService._update_from_fixer(api_key)
        elif provider == 'ecb':
            count = ExchangeRateUpdateService._update_from_ecb()
        elif provider == 'exchangerate_api':
            count = ExchangeRateUpdateService._update_from_exchangerate_api(api_key)
        elif provider == 'exchangeratesapi':
            count = ExchangeRateUpdateService._update_from_exchangeratesapi(api_key)
        
        if count > 0:
            settings.last_exchange_rate_update = timezone.now()
            settings.save(update_fields=['last_exchange_rate_update'])
        
        return count
    
    @staticmethod
    def update_rates_with_fallback(api_keys: dict = None) -> tuple[int, str]:
        """
        Update rates trying providers in order until one succeeds.
        
        Args:
            api_keys: Dict mapping provider to API key, e.g. 
                     {'exchangerate_api': 'xxx', 'openexchange': 'yyy', 'fixer': 'zzz'}
        
        Returns:
            Tuple of (count, provider_used)
        """
        if api_keys is None:
            api_keys = {
                'exchangerate_api': getattr(django_settings, 'EXCHANGERATE_API_KEY', ''),
                'openexchange': getattr(django_settings, 'OPENEXCHANGE_RATES_API_KEY', ''),
                'exchangeratesapi': getattr(django_settings, 'EXCHANGERATESAPI_KEY', ''),
                'fixer': getattr(django_settings, 'FIXER_API_KEY', ''),
            }
        
        for provider in ExchangeRateUpdateService.PROVIDERS:
            api_key = api_keys.get(provider, '')
            
            # ECB doesn't need API key
            if provider == 'ecb':
                count = ExchangeRateUpdateService._update_from_ecb()
                if count > 0:
                    return count, 'ecb'
                continue
            
            if not api_key:
                continue
            
            try:
                if provider == 'exchangerate_api':
                    count = ExchangeRateUpdateService._update_from_exchangerate_api(api_key)
                elif provider == 'openexchange':
                    count = ExchangeRateUpdateService._update_from_openexchange(api_key)
                elif provider == 'exchangeratesapi':
                    count = ExchangeRateUpdateService._update_from_exchangeratesapi(api_key)
                elif provider == 'fixer':
                    count = ExchangeRateUpdateService._update_from_fixer(api_key)
                
                if count > 0:
                    return count, provider
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
                continue
        
        return 0, None
    
    @staticmethod
    def _update_from_openexchange(api_key: str) -> int:
        """Update from Open Exchange Rates (openexchangerates.org)."""
        import requests
        
        if not api_key:
            return 0
        
        try:
            response = requests.get(
                f'https://openexchangerates.org/api/latest.json?app_id={api_key}',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            rates = data.get('rates', {})
            base = Currency.objects.filter(code='USD', is_active=True).first()
            
            if not base:
                return 0
            
            count = 0
            for code, rate in rates.items():
                target = Currency.objects.filter(code=code, is_active=True).first()
                if target and target.id != base.id:
                    ExchangeRateService.set_exchange_rate(base, target, rate, 'openexchange')
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error updating from OpenExchange: {e}")
            return 0
    
    @staticmethod
    def _update_from_exchangeratesapi(api_key: str) -> int:
        """Update from ExchangeRatesAPI.io."""
        import requests
        
        if not api_key:
            return 0
        
        try:
            # exchangeratesapi.io uses EUR as base on free tier
            response = requests.get(
                f'http://api.exchangeratesapi.io/v1/latest?access_key={api_key}',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success', True):
                error = data.get('error', {})
                logger.error(f"ExchangeRatesAPI error: {error.get('info', 'Unknown')}")
                return 0
            
            rates = data.get('rates', {})
            base_code = data.get('base', 'EUR')
            base = Currency.objects.filter(code=base_code, is_active=True).first()
            
            if not base:
                return 0
            
            count = 0
            for code, rate in rates.items():
                target = Currency.objects.filter(code=code, is_active=True).first()
                if target and target.id != base.id:
                    ExchangeRateService.set_exchange_rate(base, target, rate, 'exchangeratesapi')
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error updating from ExchangeRatesAPI: {e}")
            return 0
    
    @staticmethod
    def _update_from_fixer(api_key: str) -> int:
        """Update from Fixer.io."""
        import requests
        
        if not api_key:
            return 0
        
        try:
            response = requests.get(
                f'http://data.fixer.io/api/latest?access_key={api_key}',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                error = data.get('error', {})
                logger.error(f"Fixer error: {error.get('info', 'Unknown')}")
                return 0
            
            rates = data.get('rates', {})
            base_code = data.get('base', 'EUR')
            base = Currency.objects.filter(code=base_code, is_active=True).first()
            
            if not base:
                return 0
            
            count = 0
            for code, rate in rates.items():
                target = Currency.objects.filter(code=code, is_active=True).first()
                if target and target.id != base.id:
                    ExchangeRateService.set_exchange_rate(base, target, rate, 'fixer')
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error updating from Fixer: {e}")
            return 0
    
    @staticmethod
    def _update_from_ecb() -> int:
        """Update from European Central Bank (free, no API key needed)."""
        import requests
        import xml.etree.ElementTree as ET
        
        try:
            response = requests.get(
                'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml',
                timeout=30
            )
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            ns = {
                'gesmes': 'http://www.gesmes.org/xml/2002-08-01',
                'eurofxref': 'http://www.ecb.int/vocabulary/2002-08-01/eurofxref'
            }
            
            base = Currency.objects.filter(code='EUR', is_active=True).first()
            if not base:
                return 0
            
            count = 0
            for cube in root.findall('.//eurofxref:Cube[@currency]', ns):
                code = cube.get('currency')
                rate = Decimal(cube.get('rate'))
                
                target = Currency.objects.filter(code=code, is_active=True).first()
                if target:
                    ExchangeRateService.set_exchange_rate(base, target, rate, 'ecb')
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error updating from ECB: {e}")
            return 0
    
    @staticmethod
    def _update_from_exchangerate_api(api_key: str) -> int:
        """Update from ExchangeRate-API (exchangerate-api.com)."""
        import requests
        
        if not api_key:
            return 0
        
        try:
            response = requests.get(
                f'https://v6.exchangerate-api.com/v6/{api_key}/latest/USD',
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('result') != 'success':
                logger.error(f"ExchangeRate-API error: {data.get('error-type', 'Unknown')}")
                return 0
            
            rates = data.get('conversion_rates', {})
            base = Currency.objects.filter(code='USD', is_active=True).first()
            
            if not base:
                return 0
            
            count = 0
            for code, rate in rates.items():
                target = Currency.objects.filter(code=code, is_active=True).first()
                if target and target.id != base.id:
                    ExchangeRateService.set_exchange_rate(base, target, rate, 'exchangerate_api')
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error updating from ExchangeRate-API: {e}")
            return 0


# =============================================================================
# Timezone Service
# =============================================================================

class TimezoneService:
    """Service for timezone operations."""
    
    CACHE_TIMEOUT = 3600
    
    @staticmethod
    def get_all_timezones() -> List[Timezone]:
        """Get all active timezones."""
        cache_key = 'i18n_all_timezones'
        timezones = cache.get(cache_key)
        
        if timezones is None:
            timezones = list(Timezone.objects.filter(is_active=True))
            cache.set(cache_key, timezones, TimezoneService.CACHE_TIMEOUT)
        
        return timezones
    
    @staticmethod
    def get_common_timezones() -> List[Timezone]:
        """Get common timezones."""
        cache_key = 'i18n_common_timezones'
        timezones = cache.get(cache_key)
        
        if timezones is None:
            timezones = list(Timezone.objects.filter(is_active=True, is_common=True))
            cache.set(cache_key, timezones, TimezoneService.CACHE_TIMEOUT)
        
        return timezones
    
    @staticmethod
    def get_timezone_by_name(name: str) -> Optional[Timezone]:
        """Get timezone by IANA name."""
        return Timezone.objects.filter(name=name, is_active=True).first()
    
    @staticmethod
    def detect_timezone(request) -> Optional[Timezone]:
        """Detect timezone from request."""
        # Check session
        if hasattr(request, 'session'):
            tz_name = request.session.get('timezone')
            if tz_name:
                tz = TimezoneService.get_timezone_by_name(tz_name)
                if tz:
                    return tz
        
        # Check cookie (usually set by JS)
        if hasattr(request, 'COOKIES'):
            tz_name = request.COOKIES.get('timezone')
            if tz_name:
                tz = TimezoneService.get_timezone_by_name(tz_name)
                if tz:
                    return tz
        
        return None
    
    @staticmethod
    def get_user_timezone(user=None, request=None) -> Optional[Timezone]:
        """Get timezone for user or request."""
        if user and user.is_authenticated:
            try:
                pref = UserLocalePreference.objects.filter(user=user).select_related('timezone').first()
                if pref and pref.timezone and not pref.auto_detect_timezone:
                    return pref.timezone
            except Exception:
                pass
        
        if request:
            return TimezoneService.detect_timezone(request)
        
        try:
            settings = I18nSettings.get_settings()
            return settings.default_timezone
        except Exception:
            return None


# =============================================================================
# Geographic Service
# =============================================================================

class GeoService:
    """Service for geographic operations."""
    
    CACHE_TIMEOUT = 3600
    
    @staticmethod
    def get_all_countries() -> List[Country]:
        """Get all active countries."""
        cache_key = 'i18n_all_countries'
        countries = cache.get(cache_key)
        
        if countries is None:
            countries = list(Country.objects.filter(is_active=True))
            cache.set(cache_key, countries, GeoService.CACHE_TIMEOUT)
        
        return countries
    
    @staticmethod
    def get_shipping_countries() -> List[Country]:
        """Get countries where shipping is available."""
        cache_key = 'i18n_shipping_countries'
        countries = cache.get(cache_key)
        
        if countries is None:
            countries = list(Country.objects.filter(is_active=True, is_shipping_available=True))
            cache.set(cache_key, countries, GeoService.CACHE_TIMEOUT)
        
        return countries
    
    @staticmethod
    def get_country_by_code(code: str) -> Optional[Country]:
        """Get country by ISO code."""
        return Country.objects.filter(code=code.upper(), is_active=True).first()
    
    @staticmethod
    def get_divisions(country_code: str = 'BD') -> List[Division]:
        """Get divisions for a country."""
        cache_key = f'i18n_divisions_{country_code}'
        divisions = cache.get(cache_key)
        
        if divisions is None:
            divisions = list(Division.objects.filter(
                country__code=country_code,
                is_active=True
            ).order_by('sort_order', 'name'))
            cache.set(cache_key, divisions, GeoService.CACHE_TIMEOUT)
        
        return divisions
    
    @staticmethod
    def get_districts(division_id: str) -> List[District]:
        """Get districts for a division."""
        return list(District.objects.filter(
            division_id=division_id,
            is_active=True
        ).order_by('sort_order', 'name'))
    
    @staticmethod
    def get_upazilas(district_id: str) -> List[Upazila]:
        """Get upazilas for a district."""
        return list(Upazila.objects.filter(
            district_id=district_id,
            is_active=True
        ).order_by('sort_order', 'name'))
    
    @staticmethod
    def detect_country(request) -> Optional[Country]:
        """Detect country from request."""
        # Try Cloudflare header
        country_code = request.META.get('HTTP_CF_IPCOUNTRY')
        
        # Try AWS CloudFront
        if not country_code:
            country_code = request.META.get('HTTP_CLOUDFRONT_VIEWER_COUNTRY')
        
        # Try generic GeoIP header
        if not country_code:
            country_code = request.META.get('HTTP_X_COUNTRY_CODE')
        
        if country_code:
            return GeoService.get_country_by_code(country_code)
        
        return None


# =============================================================================
# Translation Service
# =============================================================================

class TranslationService:
    """Service for translation operations."""
    
    CACHE_TIMEOUT = 3600
    
    @staticmethod
    def get_translation(key: str, language_code: str, namespace: str = None) -> Optional[str]:
        """Get translation for a key."""
        cache_key = f'i18n_trans_{namespace or "default"}_{key}_{language_code}'
        translation = cache.get(cache_key)
        
        if translation is not None:
            return translation
        
        # Build query
        query = Translation.objects.filter(
            key__key=key,
            language__code=language_code,
            status='approved'
        )
        
        if namespace:
            query = query.filter(key__namespace__name=namespace)
        
        trans_obj = query.select_related('key').first()
        
        if trans_obj:
            cache.set(cache_key, trans_obj.translated_text, TranslationService.CACHE_TIMEOUT)
            return trans_obj.translated_text
        
        return None
    
    @staticmethod
    def get_content_translation(
        content_type: str,
        content_id: str,
        field_name: str,
        language_code: str
    ) -> Optional[str]:
        """Get translation for dynamic content."""
        cache_key = f'i18n_content_{content_type}_{content_id}_{field_name}_{language_code}'
        translation = cache.get(cache_key)
        
        if translation is not None:
            return translation
        
        trans_obj = ContentTranslation.objects.filter(
            content_type=content_type,
            content_id=content_id,
            field_name=field_name,
            language__code=language_code,
            is_approved=True
        ).first()
        
        if trans_obj:
            cache.set(cache_key, trans_obj.translated_text, TranslationService.CACHE_TIMEOUT)
            return trans_obj.translated_text
        
        return None
    
    @staticmethod
    def set_content_translation(
        content_type: str,
        content_id: str,
        field_name: str,
        language_code: str,
        text: str,
        user=None,
        is_machine: bool = False
    ) -> ContentTranslation:
        """Set translation for dynamic content."""
        language = LanguageService.get_language_by_code(language_code)
        if not language:
            raise ValueError(f"Language not found: {language_code}")
        
        translation, created = ContentTranslation.objects.update_or_create(
            content_type=content_type,
            content_id=content_id,
            field_name=field_name,
            language=language,
            defaults={
                'translated_text': text,
                'translated_by': user,
                'is_machine_translated': is_machine,
                'is_approved': not is_machine,
            }
        )
        
        # Clear cache
        cache_key = f'i18n_content_{content_type}_{content_id}_{field_name}_{language_code}'
        cache.delete(cache_key)
        
        return translation


# =============================================================================
# User Preference Service
# =============================================================================

class UserPreferenceService:
    """Service for user locale preferences."""

    @staticmethod
    def _clear_missing_refs(pref: UserLocalePreference) -> List[str]:
        """Clear invalid FK references on the preference to avoid FK constraint errors."""
        cleared = []
        if pref.language_id and not Language.objects.filter(id=pref.language_id).exists():
            pref.language = None
            cleared.append('language')
        if pref.currency_id and not Currency.objects.filter(id=pref.currency_id).exists():
            pref.currency = None
            cleared.append('currency')
        if pref.timezone_id and not Timezone.objects.filter(id=pref.timezone_id).exists():
            pref.timezone = None
            cleared.append('timezone')
        if pref.country_id and not Country.objects.filter(id=pref.country_id).exists():
            pref.country = None
            cleared.append('country')
        return cleared
    
    @staticmethod
    def get_or_create_preference(user) -> UserLocalePreference:
        """Get or create locale preference for user."""
        pref, _ = UserLocalePreference.objects.get_or_create(user=user)
        cleared = UserPreferenceService._clear_missing_refs(pref)
        if cleared:
            pref.save(update_fields=cleared)
        return pref
    
    @staticmethod
    def update_preference(user, **kwargs) -> UserLocalePreference:
        """Update user's locale preference."""
        import logging
        logger = logging.getLogger('bunoraa.i18n')
        
        pref = UserPreferenceService.get_or_create_preference(user)
        UserPreferenceService._clear_missing_refs(pref)
        logger.debug(f"[Preference] Updating for user {user.id} with: {kwargs}")
        
        if 'language_code' in kwargs:
            pref.language = LanguageService.get_language_by_code(kwargs['language_code'])
        
        if 'currency_code' in kwargs:
            currency = CurrencyService.get_currency_by_code(kwargs['currency_code'])
            if currency:
                logger.debug(f"[Preference] Setting currency to: {currency.code}")
                pref.currency = currency
            else:
                logger.warning(f"[Preference] Currency code not found: {kwargs['currency_code']}")
        
        if 'timezone_name' in kwargs:
            pref.timezone = TimezoneService.get_timezone_by_name(kwargs['timezone_name'])
        
        if 'country_code' in kwargs:
            pref.country = GeoService.get_country_by_code(kwargs['country_code'])
        
        for field in ['date_format', 'time_format', 'first_day_of_week', 
                      'measurement_system', 'temperature_unit',
                      'auto_detect_language', 'auto_detect_currency', 'auto_detect_timezone']:
            if field in kwargs:
                setattr(pref, field, kwargs[field])
                if field == 'auto_detect_currency':
                    logger.debug(f"[Preference] Setting auto_detect_currency to: {kwargs[field]}")
        
        pref.save()
        logger.debug(f"[Preference] Saved: currency={pref.currency.code if pref.currency else None}, auto_detect={pref.auto_detect_currency}")
        return pref
    
    @staticmethod
    def get_user_locale(user=None, request=None) -> Dict[str, Any]:
        """Get complete locale info for user/request."""
        return {
            'language': LanguageService.get_user_language(user, request),
            'currency': CurrencyService.get_user_currency(user, request),
            'timezone': TimezoneService.get_user_timezone(user, request),
            'country': GeoService.detect_country(request) if request else None,
        }


# =============================================================================
# Bangladesh Commerce Service
# =============================================================================

class BangladeshCommerceService:
    """
    Comprehensive service for Bangladesh-specific commerce operations.
    Handles BDT as default currency, local payment methods, and shipping zones.
    """
    
    # Bangladesh Configuration
    DEFAULT_CURRENCY_CODE = 'BDT'
    CURRENCY_SYMBOL = '৳'
    CURRENCY_SYMBOL_NATIVE = '৳'
    DECIMAL_PLACES = 0  # BDT typically shown without decimals
    DEFAULT_TIMEZONE = 'Asia/Dhaka'
    COUNTRY_CODE = 'BD'
    
    # Shipping Configuration
    DIVISIONS = [
        'Dhaka', 'Chittagong', 'Rajshahi', 'Khulna',
        'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh'
    ]
    
    SHIPPING_ZONES = {
        'dhaka_city': {
            'name': 'Dhaka City',
            'shipping_cost': Decimal('60'),
            'free_shipping_threshold': Decimal('2000'),
            'delivery_days_min': 1,
            'delivery_days_max': 2,
            'areas': ['Dhaka North', 'Dhaka South', 'Gulshan', 'Banani', 'Dhanmondi', 'Mirpur', 'Uttara']
        },
        'dhaka_suburbs': {
            'name': 'Dhaka Suburbs',
            'shipping_cost': Decimal('80'),
            'free_shipping_threshold': Decimal('2000'),
            'delivery_days_min': 2,
            'delivery_days_max': 3,
            'areas': ['Gazipur', 'Narayanganj', 'Savar', 'Keraniganj', 'Tongi']
        },
        'chittagong': {
            'name': 'Chittagong Division',
            'shipping_cost': Decimal('100'),
            'free_shipping_threshold': Decimal('3000'),
            'delivery_days_min': 2,
            'delivery_days_max': 4,
            'areas': ['Chittagong City', "Cox's Bazar", 'Comilla', 'Feni', 'Brahmanbaria']
        },
        'major_cities': {
            'name': 'Major Cities',
            'shipping_cost': Decimal('100'),
            'free_shipping_threshold': Decimal('3000'),
            'delivery_days_min': 2,
            'delivery_days_max': 4,
            'areas': ['Sylhet', 'Rajshahi', 'Khulna', 'Rangpur', 'Barisal', 'Mymensingh']
        },
        'district': {
            'name': 'District Areas',
            'shipping_cost': Decimal('120'),
            'free_shipping_threshold': Decimal('3500'),
            'delivery_days_min': 3,
            'delivery_days_max': 5,
            'areas': []  # All other district areas
        },
        'remote': {
            'name': 'Remote Areas',
            'shipping_cost': Decimal('150'),
            'free_shipping_threshold': Decimal('5000'),
            'delivery_days_min': 5,
            'delivery_days_max': 7,
            'areas': ['Char areas', 'Hill tracts', 'Sundarbans periphery']
        }
    }
    
    # Payment Methods
    PAYMENT_METHODS = {
        'bkash': {
            'name': 'bKash',
            'type': 'mobile_wallet',
            'fee_percentage': Decimal('1.5'),
            'fee_fixed': Decimal('0'),
            'min_amount': Decimal('10'),
            'max_amount': Decimal('50000'),
            'is_instant': True,
            'icon': 'bkash.svg'
        },
        'nagad': {
            'name': 'Nagad',
            'type': 'mobile_wallet',
            'fee_percentage': Decimal('1.5'),
            'fee_fixed': Decimal('0'),
            'min_amount': Decimal('10'),
            'max_amount': Decimal('50000'),
            'is_instant': True,
            'icon': 'nagad.svg'
        },
        'rocket': {
            'name': 'Rocket',
            'type': 'mobile_wallet',
            'fee_percentage': Decimal('1.8'),
            'fee_fixed': Decimal('0'),
            'min_amount': Decimal('10'),
            'max_amount': Decimal('25000'),
            'is_instant': True,
            'icon': 'rocket.svg'
        },
        'upay': {
            'name': 'Upay',
            'type': 'mobile_wallet',
            'fee_percentage': Decimal('1.5'),
            'fee_fixed': Decimal('0'),
            'min_amount': Decimal('10'),
            'max_amount': Decimal('25000'),
            'is_instant': True,
            'icon': 'upay.svg'
        },
        'cod': {
            'name': 'Cash on Delivery',
            'type': 'cod',
            'fee_percentage': Decimal('0'),
            'fee_fixed': Decimal('20'),
            'min_amount': Decimal('0'),
            'max_amount': Decimal('50000'),
            'is_instant': False,
            'icon': 'cod.svg'
        },
        'card': {
            'name': 'Credit/Debit Card',
            'type': 'card',
            'fee_percentage': Decimal('2.5'),
            'fee_fixed': Decimal('0'),
            'min_amount': Decimal('100'),
            'max_amount': None,  # No limit
            'is_instant': True,
            'icon': 'card.svg'
        },
        'bank_transfer': {
            'name': 'Bank Transfer',
            'type': 'bank',
            'fee_percentage': Decimal('0'),
            'fee_fixed': Decimal('0'),
            'min_amount': Decimal('500'),
            'max_amount': None,
            'is_instant': False,
            'icon': 'bank.svg'
        }
    }
    
    # VAT Rate (Bangladesh standard)
    VAT_RATE = Decimal('0.10')  # 10% VAT
    
    CACHE_TIMEOUT = 3600  # 1 hour
    
    @classmethod
    def get_bdt_currency(cls) -> Optional[Currency]:
        """Get BDT currency object."""
        cache_key = 'bd_currency_bdt'
        currency = cache.get(cache_key)
        
        if currency is None:
            currency = Currency.objects.filter(code=cls.DEFAULT_CURRENCY_CODE, is_active=True).first()
            if currency:
                cache.set(cache_key, currency, cls.CACHE_TIMEOUT)
        
        return currency
    
    @classmethod
    def format_bdt(cls, amount, include_symbol: bool = True) -> str:
        """Format amount in BDT with Bangladeshi number formatting."""
        try:
            amount = Decimal(str(amount))
            # Round to nearest integer for BDT
            rounded = amount.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            
            # Format with Bangladeshi lakhs/crores system
            formatted = cls._format_bangladeshi_number(rounded)
            
            if include_symbol:
                return f'{cls.CURRENCY_SYMBOL}{formatted}'
            return formatted
        except Exception:
            return f'{cls.CURRENCY_SYMBOL}0'
    
    @classmethod
    def _format_bangladeshi_number(cls, number: Decimal) -> str:
        """
        Format number using Bangladeshi numbering system.
        Pattern: X,XX,XX,XXX (lakhs and crores)
        """
        try:
            num_str = str(int(abs(number)))
            length = len(num_str)
            
            if length <= 3:
                result = num_str
            elif length <= 5:
                result = num_str[:-3] + ',' + num_str[-3:]
            elif length <= 7:
                result = num_str[:-5] + ',' + num_str[-5:-3] + ',' + num_str[-3:]
            else:
                # Handle crores
                prefix = num_str[:-7]
                rest = num_str[-7:-5] + ',' + num_str[-5:-3] + ',' + num_str[-3:]
                # Format prefix with commas every 2 digits
                formatted_prefix = ''
                for i, char in enumerate(reversed(prefix)):
                    if i > 0 and i % 2 == 0:
                        formatted_prefix = ',' + formatted_prefix
                    formatted_prefix = char + formatted_prefix
                result = formatted_prefix + ',' + rest
            
            return '-' + result if number < 0 else result
        except Exception:
            return str(number)
    
    @classmethod
    def convert_to_bdt(cls, amount: Decimal, from_currency_code: str) -> Decimal:
        """Convert any amount to BDT."""
        if from_currency_code.upper() == cls.DEFAULT_CURRENCY_CODE:
            return Decimal(str(amount))
        
        try:
            return CurrencyConversionService.convert_by_code(
                amount, from_currency_code, cls.DEFAULT_CURRENCY_CODE
            )
        except ValueError:
            # Fallback: use approximate rates
            fallback_rates = {
                'USD': Decimal('110'),
                'EUR': Decimal('120'),
                'GBP': Decimal('140'),
                'INR': Decimal('1.32'),
                'AED': Decimal('30'),
                'SAR': Decimal('29'),
                'MYR': Decimal('24'),
                'SGD': Decimal('82'),
            }
            rate = fallback_rates.get(from_currency_code.upper(), Decimal('100'))
            return Decimal(str(amount)) * rate
    
    @classmethod
    def convert_from_bdt(cls, amount: Decimal, to_currency_code: str) -> Decimal:
        """Convert BDT amount to another currency."""
        if to_currency_code.upper() == cls.DEFAULT_CURRENCY_CODE:
            return Decimal(str(amount))
        
        try:
            return CurrencyConversionService.convert_by_code(
                amount, cls.DEFAULT_CURRENCY_CODE, to_currency_code
            )
        except ValueError:
            return Decimal(str(amount))  # Return original if conversion fails
    
    @classmethod
    def get_shipping_zone(cls, division: str = None, district: str = None, area: str = None) -> Dict[str, Any]:
        """Get shipping zone based on location."""
        if not division:
            division = 'Dhaka'
        
        division_lower = division.lower()
        area_lower = (area or '').lower()
        
        # Check Dhaka zones
        if division_lower == 'dhaka':
            if any(a.lower() in area_lower for a in cls.SHIPPING_ZONES['dhaka_city']['areas']):
                return cls.SHIPPING_ZONES['dhaka_city']
            if any(a.lower() in area_lower for a in cls.SHIPPING_ZONES['dhaka_suburbs']['areas']):
                return cls.SHIPPING_ZONES['dhaka_suburbs']
            # Default Dhaka to city
            return cls.SHIPPING_ZONES['dhaka_city']
        
        # Check Chittagong
        if division_lower == 'chittagong':
            return cls.SHIPPING_ZONES['chittagong']
        
        # Check major cities
        if division_lower in ['sylhet', 'rajshahi', 'khulna', 'rangpur', 'barisal', 'mymensingh']:
            return cls.SHIPPING_ZONES['major_cities']
        
        # Default to district
        return cls.SHIPPING_ZONES['district']
    
    @classmethod
    def calculate_shipping(
        cls,
        subtotal: Decimal,
        division: str = None,
        district: str = None,
        weight_kg: Decimal = None
    ) -> Dict[str, Any]:
        """Calculate shipping cost for Bangladesh."""
        zone = cls.get_shipping_zone(division, district)
        
        base_cost = zone['shipping_cost']
        
        # Weight-based additional cost (৳10 per kg after first 500g)
        weight_cost = Decimal('0')
        if weight_kg and weight_kg > Decimal('0.5'):
            extra_kg = weight_kg - Decimal('0.5')
            weight_cost = extra_kg * Decimal('10')
        
        total_cost = base_cost + weight_cost
        
        # Check free shipping threshold
        is_free = subtotal >= zone['free_shipping_threshold']
        final_cost = Decimal('0') if is_free else total_cost
        
        return {
            'zone': zone['name'],
            'base_cost': base_cost,
            'weight_cost': weight_cost,
            'total_cost': total_cost,
            'final_cost': final_cost,
            'is_free': is_free,
            'free_threshold': zone['free_shipping_threshold'],
            'remaining_for_free': max(Decimal('0'), zone['free_shipping_threshold'] - subtotal),
            'delivery_days_min': zone['delivery_days_min'],
            'delivery_days_max': zone['delivery_days_max'],
            'formatted_cost': cls.format_bdt(final_cost) if not is_free else 'FREE',
            'formatted_threshold': cls.format_bdt(zone['free_shipping_threshold'])
        }
    
    @classmethod
    def calculate_vat(cls, amount: Decimal) -> Dict[str, Any]:
        """Calculate VAT for Bangladesh."""
        vat_amount = (Decimal(str(amount)) * cls.VAT_RATE).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        total = Decimal(str(amount)) + vat_amount
        
        return {
            'subtotal': amount,
            'vat_rate': cls.VAT_RATE,
            'vat_percentage': int(cls.VAT_RATE * 100),
            'vat_amount': vat_amount,
            'total': total,
            'formatted_vat': cls.format_bdt(vat_amount),
            'formatted_total': cls.format_bdt(total)
        }
    
    @classmethod
    def get_payment_methods(cls, amount: Decimal = None) -> List[Dict[str, Any]]:
        """Get available payment methods, optionally filtered by amount."""
        methods = []
        
        for code, method in cls.PAYMENT_METHODS.items():
            # Check amount limits
            if amount:
                if method['min_amount'] and amount < method['min_amount']:
                    continue
                if method['max_amount'] and amount > method['max_amount']:
                    continue
            
            # Calculate fee
            fee = Decimal('0')
            if amount:
                fee = (amount * method['fee_percentage'] / 100) + method['fee_fixed']
                fee = fee.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            
            methods.append({
                'code': code,
                'name': method['name'],
                'type': method['type'],
                'fee': fee,
                'formatted_fee': cls.format_bdt(fee) if fee > 0 else 'Free',
                'is_instant': method['is_instant'],
                'icon': method['icon'],
                'min_amount': method['min_amount'],
                'max_amount': method['max_amount']
            })
        
        return methods
    
    @classmethod
    def calculate_payment_fee(cls, amount: Decimal, method_code: str) -> Decimal:
        """Calculate payment processing fee."""
        method = cls.PAYMENT_METHODS.get(method_code)
        if not method:
            return Decimal('0')
        
        fee = (Decimal(str(amount)) * method['fee_percentage'] / 100) + method['fee_fixed']
        return fee.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    
    @classmethod
    def get_order_summary(
        cls,
        subtotal: Decimal,
        division: str = 'Dhaka',
        payment_method: str = 'cod',
        include_vat: bool = True
    ) -> Dict[str, Any]:
        """Get complete order summary with all costs."""
        # Calculate shipping
        shipping = cls.calculate_shipping(subtotal, division)
        
        # Calculate VAT if applicable
        vat_info = cls.calculate_vat(subtotal) if include_vat else None
        
        # Calculate payment fee
        payment_fee = cls.calculate_payment_fee(subtotal, payment_method)
        
        # Calculate total
        total = subtotal + shipping['final_cost'] + payment_fee
        if include_vat:
            total += vat_info['vat_amount']
        
        return {
            'subtotal': subtotal,
            'formatted_subtotal': cls.format_bdt(subtotal),
            'shipping': shipping,
            'vat': vat_info,
            'payment_fee': payment_fee,
            'formatted_payment_fee': cls.format_bdt(payment_fee) if payment_fee > 0 else 'Free',
            'total': total,
            'formatted_total': cls.format_bdt(total),
            'currency': cls.DEFAULT_CURRENCY_CODE,
            'currency_symbol': cls.CURRENCY_SYMBOL
        }
    
    @classmethod
    def get_divisions(cls) -> List[Dict[str, str]]:
        """Get list of Bangladesh divisions."""
        return [
            {'code': d.lower(), 'name': d, 'name_bn': cls._get_division_bn(d)}
            for d in cls.DIVISIONS
        ]
    
    @classmethod
    def _get_division_bn(cls, division: str) -> str:
        """Get Bengali name for division."""
        names = {
            'Dhaka': 'ঢাকা',
            'Chittagong': 'চট্টগ্রাম',
            'Rajshahi': 'রাজশাহী',
            'Khulna': 'খুলনা',
            'Barisal': 'বরিশাল',
            'Sylhet': 'সিলেট',
            'Rangpur': 'রংপুর',
            'Mymensingh': 'ময়মনসিংহ'
        }
        return names.get(division, division)

