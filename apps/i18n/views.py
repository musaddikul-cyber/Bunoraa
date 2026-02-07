"""
Internationalization Views

Views for language, currency, timezone switching and geographic data.
"""
from django.shortcuts import redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods, require_GET, require_POST
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.decorators import login_required
from django.utils.translation import activate
from django.conf import settings
from decimal import Decimal
import json

from .services import (
    LanguageService, CurrencyService, TimezoneService, 
    GeoService, CurrencyConversionService, UserPreferenceService
)
from .models import UserLocalePreference


def get_next_url(request):
    """Get the next URL from request, with fallback to HTTP_REFERER or home."""
    next_url = request.GET.get('next') or request.POST.get('next')
    if not next_url:
        next_url = request.META.get('HTTP_REFERER')
    if not next_url:
        next_url = '/'
    return next_url


# =============================================================================
# Language Views
# =============================================================================

@require_http_methods(['GET', 'POST'])
@csrf_protect
def set_language(request, code):
    """Set user's language preference."""
    language = LanguageService.get_language_by_code(code)
    
    if language:
        # Set in session
        request.session['language'] = code
        request.session['django_language'] = code
        
        # Activate for this request
        activate(code)
        
        # Update user preference if authenticated
        if request.user.is_authenticated:
            LanguageService.set_user_language(request.user, code)
        
        # Set cookie
        response = redirect(get_next_url(request))
        response.set_cookie(
            'language', code,
            max_age=365 * 24 * 60 * 60,  # 1 year
            httponly=False,
            samesite='Lax'
        )
        return response
    
    return redirect(get_next_url(request))


@require_GET
def detect_language(request):
    """Detect and return suggested language."""
    language = LanguageService.detect_language(request)
    
    if language:
        return JsonResponse({
            'detected': True,
            'code': language.code,
            'name': language.name,
            'native_name': language.native_name,
            'is_rtl': language.is_rtl
        })
    
    return JsonResponse({'detected': False})


# =============================================================================
# Currency Views
# =============================================================================

@require_http_methods(['GET', 'POST'])
@csrf_protect
def set_currency(request, code):
    """Set user's currency preference."""
    currency = CurrencyService.get_currency_by_code(code)
    
    if currency:
        # Set in session
        request.session['currency_code'] = code
        
        # Update user preference if authenticated
        if request.user.is_authenticated:
            CurrencyService.set_user_currency(request.user, code)
        
        # Set cookie
        response = redirect(get_next_url(request))
        response.set_cookie(
            'currency', code,
            max_age=365 * 24 * 60 * 60,
            httponly=False,
            samesite='Lax'
        )
        return response
    
    return redirect(get_next_url(request))


@require_GET
def detect_currency(request):
    """Detect and return suggested currency."""
    currency = CurrencyService.detect_currency(request)
    
    if currency:
        return JsonResponse({
            'detected': True,
            'code': currency.code,
            'name': currency.name,
            'symbol': currency.symbol
        })
    
    return JsonResponse({'detected': False})


# =============================================================================
# Timezone Views
# =============================================================================

@require_http_methods(['GET', 'POST'])
@csrf_protect
def set_timezone(request, name):
    """Set user's timezone preference."""
    timezone_obj = TimezoneService.get_timezone_by_name(name)
    
    if timezone_obj:
        # Set in session
        request.session['timezone'] = name
        
        # Update user preference if authenticated
        if request.user.is_authenticated:
            pref = UserPreferenceService.get_or_create_preference(request.user)
            pref.timezone = timezone_obj
            pref.auto_detect_timezone = False
            pref.save()
        
        # Set cookie
        response = redirect(get_next_url(request))
        response.set_cookie(
            'timezone', name,
            max_age=365 * 24 * 60 * 60,
            httponly=False,
            samesite='Lax'
        )
        return response
    
    return redirect(get_next_url(request))


@require_GET
def detect_timezone(request):
    """Detect and return suggested timezone."""
    timezone_obj = TimezoneService.detect_timezone(request)
    
    if timezone_obj:
        return JsonResponse({
            'detected': True,
            'name': timezone_obj.name,
            'display_name': timezone_obj.display_name,
            'offset': timezone_obj.formatted_offset
        })
    
    return JsonResponse({'detected': False})


# =============================================================================
# Country Views
# =============================================================================

@require_http_methods(['GET', 'POST'])
@csrf_protect
def set_country(request, code):
    """Set user's country preference."""
    country = GeoService.get_country_by_code(code)
    
    if country:
        # Set in session
        request.session['country_code'] = code
        
        # Update user preference if authenticated
        if request.user.is_authenticated:
            pref = UserPreferenceService.get_or_create_preference(request.user)
            pref.country = country
            pref.save()
        
        # Set cookie
        response = redirect(get_next_url(request))
        response.set_cookie(
            'country', code,
            max_age=365 * 24 * 60 * 60,
            httponly=False,
            samesite='Lax'
        )
        return response
    
    return redirect(get_next_url(request))


@require_GET
def detect_country(request):
    """Detect country from IP."""
    country = GeoService.detect_country(request)
    
    if country:
        return JsonResponse({
            'detected': True,
            'code': country.code,
            'name': country.name,
            'flag': country.flag_emoji,
            'currency': country.default_currency.code if country.default_currency else None
        })
    
    return JsonResponse({'detected': False})


# =============================================================================
# Geographic Data Views (for address forms)
# =============================================================================

@require_GET
def get_divisions(request, country_code):
    """Get divisions for a country."""
    divisions = GeoService.get_divisions(country_code)
    
    return JsonResponse({
        'divisions': [
            {
                'id': str(d.id),
                'name': d.name,
                'native_name': d.native_name,
                'code': d.code
            }
            for d in divisions
        ]
    })


@require_GET
def get_districts(request, division_id):
    """Get districts for a division."""
    districts = GeoService.get_districts(division_id)
    
    return JsonResponse({
        'districts': [
            {
                'id': str(d.id),
                'name': d.name,
                'native_name': d.native_name,
                'code': d.code
            }
            for d in districts
        ]
    })


@require_GET
def get_upazilas(request, district_id):
    """Get upazilas for a district."""
    upazilas = GeoService.get_upazilas(district_id)
    
    return JsonResponse({
        'upazilas': [
            {
                'id': str(u.id),
                'name': u.name,
                'native_name': u.native_name,
                'code': u.code,
                'post_codes': u.post_codes
            }
            for u in upazilas
        ]
    })


# =============================================================================
# Currency Conversion Views
# =============================================================================

@require_GET
def convert_currency(request):
    """Convert amount between currencies."""
    try:
        amount = Decimal(request.GET.get('amount', '0'))
        from_code = request.GET.get('from', '').upper()
        to_code = request.GET.get('to', '').upper()
        
        if not from_code or not to_code:
            return JsonResponse({'error': 'Missing from or to currency'}, status=400)
        
        from_currency = CurrencyService.get_currency_by_code(from_code)
        to_currency = CurrencyService.get_currency_by_code(to_code)
        
        if not from_currency or not to_currency:
            return JsonResponse({'error': 'Invalid currency code'}, status=400)
        
        converted = CurrencyConversionService.convert(amount, from_currency, to_currency)
        formatted = CurrencyConversionService.format_price(converted, to_currency)
        
        return JsonResponse({
            'original': {
                'amount': str(amount),
                'currency': from_code,
                'formatted': from_currency.format_amount(amount)
            },
            'converted': {
                'amount': str(converted),
                'currency': to_code,
                'formatted': formatted
            }
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@require_GET
def get_exchange_rates(request):
    """Get exchange rates for currencies."""
    from .models import ExchangeRate, Currency
    
    base_code = request.GET.get('base', '')
    
    if base_code:
        base_currency = CurrencyService.get_currency_by_code(base_code)
        if not base_currency:
            return JsonResponse({'error': 'Invalid base currency'}, status=400)
    else:
        base_currency = CurrencyService.get_default_currency()
    
    rates = {}
    for currency in CurrencyService.get_active_currencies():
        if currency.id != base_currency.id:
            from .services import ExchangeRateService
            rate = ExchangeRateService.get_exchange_rate(base_currency, currency)
            if rate:
                rates[currency.code] = str(rate)
    
    return JsonResponse({
        'base': base_currency.code,
        'rates': rates,
        'timestamp': timezone.now().isoformat() if 'timezone' in dir() else None
    })


# =============================================================================
# User Preferences Views
# =============================================================================

@login_required
@require_GET
def user_preferences(request):
    """Get user's locale preferences."""
    pref = UserPreferenceService.get_or_create_preference(request.user)
    
    return JsonResponse({
        'language': {
            'code': pref.language.code if pref.language else None,
            'name': pref.language.name if pref.language else None,
            'auto_detect': pref.auto_detect_language
        },
        'currency': {
            'code': pref.currency.code if pref.currency else None,
            'name': pref.currency.name if pref.currency else None,
            'auto_detect': pref.auto_detect_currency
        },
        'timezone': {
            'name': pref.timezone.name if pref.timezone else None,
            'display_name': pref.timezone.display_name if pref.timezone else None,
            'auto_detect': pref.auto_detect_timezone
        },
        'country': {
            'code': pref.country.code if pref.country else None,
            'name': pref.country.name if pref.country else None
        },
        'formatting': {
            'date_format': pref.date_format,
            'time_format': pref.time_format,
            'first_day_of_week': pref.first_day_of_week,
            'measurement_system': pref.measurement_system,
            'temperature_unit': pref.temperature_unit
        }
    })


@login_required
@require_POST
@csrf_protect
def update_preferences(request):
    """Update user's locale preferences."""
    try:
        data = json.loads(request.body) if request.body else request.POST.dict()
        
        pref = UserPreferenceService.update_preference(request.user, **data)
        
        # Update session with new preferences
        if pref.language:
            request.session['language'] = pref.language.code
            request.session['django_language'] = pref.language.code
        if pref.currency:
            request.session['currency_code'] = pref.currency.code
        if pref.timezone:
            request.session['timezone'] = pref.timezone.name
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# =============================================================================
# Translation Views
# =============================================================================

@require_GET
def get_translations(request, namespace):
    """Get all translations for a namespace."""
    from .models import Translation
    
    language_code = request.GET.get('lang')
    if not language_code:
        language = LanguageService.get_user_language(
            request.user if request.user.is_authenticated else None,
            request
        )
        language_code = language.code if language else 'en'
    
    translations = Translation.objects.filter(
        key__namespace__name=namespace,
        language__code=language_code,
        status='approved'
    ).select_related('key').values_list('key__key', 'translated_text')
    
    return JsonResponse({
        'namespace': namespace,
        'language': language_code,
        'translations': dict(translations)
    })


# =============================================================================
# Data Export Views (for JS)
# =============================================================================

@require_GET
def get_locale_data(request):
    """Get all locale data for JavaScript."""
    language = LanguageService.get_user_language(
        request.user if request.user.is_authenticated else None,
        request
    )
    currency = CurrencyService.get_user_currency(
        request.user if request.user.is_authenticated else None,
        request
    )
    
    return JsonResponse({
        'language': {
            'code': language.code if language else 'en',
            'name': language.name if language else 'English',
            'is_rtl': language.is_rtl if language else False
        },
        'currency': {
            'code': currency.code if currency else 'BDT',
            'symbol': currency.symbol if currency else '৳',
            'decimal_places': currency.decimal_places if currency else 2,
            'symbol_position': currency.symbol_position if currency else 'before',
            'thousands_separator': currency.thousands_separator if currency else ',',
            'decimal_separator': currency.decimal_separator if currency else '.'
        },
        'available_languages': [
            {'code': lang.code, 'name': lang.name, 'native_name': lang.native_name}
            for lang in LanguageService.get_active_languages()
        ],
        'available_currencies': [
            {'code': curr.code, 'name': curr.name, 'symbol': curr.symbol}
            for curr in CurrencyService.get_active_currencies()
        ]
    })


# Import timezone at the top if needed
from django.utils import timezone
