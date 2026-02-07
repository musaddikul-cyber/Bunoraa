"""
Internationalization Template Tags

Template tags and filters for i18n operations in templates.
"""
from django import template
from django.utils.safestring import mark_safe
from decimal import Decimal

from ..services import (
    LanguageService, CurrencyService, CurrencyConversionService,
    TranslationService
)

register = template.Library()


# =============================================================================
# Currency Tags
# =============================================================================

@register.filter(name='format_currency')
def format_currency(value, currency=None):
    """
    Format a value as currency.
    
    Usage:
        {{ product.price|format_currency }}
        {{ product.price|format_currency:request.currency }}
    """
    if value is None:
        return ''
    
    try:
        if currency is None:
            currency = CurrencyService.get_default_currency()
        
        if currency:
            return currency.format_amount(Decimal(str(value)))
        return str(value)
    except Exception:
        return str(value)


@register.filter(name='convert_currency')
def convert_currency(value, target_currency):
    """
    Convert a value to target currency.
    
    Usage:
        {{ product.price|convert_currency:request.currency }}
    """
    if value is None:
        return ''
    
    try:
        from_currency = CurrencyService.get_default_currency()
        if not from_currency or not target_currency:
            return value
        
        if from_currency.id == target_currency.id:
            return value
        
        return CurrencyConversionService.convert(
            Decimal(str(value)), from_currency, target_currency
        )
    except Exception:
        return value


@register.simple_tag(takes_context=True)
def price(context, value, currency=None, show_both=False):
    """
    Display price with proper formatting.
    
    Usage:
        {% price product.price %}
        {% price product.price request.currency %}
        {% price product.price show_both=True %}
    """
    if value is None:
        return ''
    
    try:
        request = context.get('request')
        
        # Get currency from request if not provided
        if currency is None and request and hasattr(request, 'currency'):
            currency = request.currency
        
        if currency is None:
            currency = CurrencyService.get_default_currency()
        
        if not currency:
            return str(value)
        
        # Convert and format
        default = CurrencyService.get_default_currency()
        
        if default and default.id != currency.id:
            converted = CurrencyConversionService.convert(
                Decimal(str(value)), default, currency
            )
            formatted = currency.format_amount(converted)
            
            if show_both:
                original = default.format_amount(Decimal(str(value)))
                return mark_safe(f'{formatted} <small class="text-muted">({original})</small>')
            
            return formatted
        
        return currency.format_amount(Decimal(str(value)))
        
    except Exception:
        return str(value)


@register.simple_tag
def currency_symbol(currency=None):
    """
    Get currency symbol.
    
    Usage:
        {% currency_symbol %}
        {% currency_symbol request.currency %}
    """
    if currency is None:
        currency = CurrencyService.get_default_currency()
    
    if currency:
        return currency.native_symbol or currency.symbol
    return ''


@register.simple_tag(takes_context=True)
def format_price(context, amount, from_currency=None, show_original=False):
    """
    Format an amount in the user's current currency with conversion.
    
    Usage:
        {% format_price product.price %}
        {% format_price product.price product.currency %}
        {% format_price 12.5 "USD" %}
        {% format_price product.price product.currency show_original=True %}
    """
    if amount is None:
        return ''
    
    try:
        request = context.get('request')
        user = None
        if request and hasattr(request, 'user') and request.user.is_authenticated:
            user = request.user
        
        # Get from_currency code
        from_code = None
        if from_currency is not None:
            if hasattr(from_currency, 'code'):
                from_code = from_currency.code
            else:
                from_code = str(from_currency).upper()
        else:
            # Try to get from context product
            product = context.get('product') or context.get('object')
            if product and hasattr(product, 'currency'):
                cur = product.currency
                if hasattr(cur, 'code'):
                    from_code = cur.code
                elif cur:
                    from_code = str(cur).upper()
        
        # Get target currency
        target = CurrencyService.get_user_currency(user=user, request=request)
        if not target:
            target = CurrencyService.get_default_currency()
        
        if not target:
            return str(amount)
        
        # Convert if needed
        if from_code and from_code != target.code:
            try:
                converted = CurrencyConversionService.convert_by_code(
                    Decimal(str(amount)), from_code, target.code
                )
                formatted = target.format_amount(converted)
                
                if show_original:
                    from_curr = CurrencyService.get_currency_by_code(from_code)
                    if from_curr:
                        original = from_curr.format_amount(Decimal(str(amount)))
                        return mark_safe(f'{formatted} <small class="text-muted">({original})</small>')
                
                return formatted
            except Exception:
                # Conversion failed, format in target currency
                return target.format_amount(Decimal(str(amount)))
        
        # No conversion needed
        return target.format_amount(Decimal(str(amount)))
        
    except Exception:
        # Fallback
        try:
            return f"{Decimal(str(amount)):.2f}"
        except Exception:
            return str(amount)


# =============================================================================
# Language Tags
# =============================================================================

@register.simple_tag
def active_languages():
    """
    Get list of active languages.
    
    Usage:
        {% active_languages as languages %}
        {% for lang in languages %}...{% endfor %}
    """
    return LanguageService.get_active_languages()


@register.simple_tag(takes_context=True)
def current_language(context):
    """
    Get current language.
    
    Usage:
        {% current_language as lang %}
        {{ lang.name }}
    """
    request = context.get('request')
    if request and hasattr(request, 'language'):
        return request.language
    
    user = None
    if request and hasattr(request, 'user') and request.user.is_authenticated:
        user = request.user
    
    return LanguageService.get_user_language(user, request)


@register.simple_tag
def rtl_class(language):
    """
    Get RTL class for language.
    
    Usage:
        <html class="{% rtl_class current_lang %}">
    """
    if language and language.is_rtl:
        return 'rtl'
    return 'ltr'


@register.filter(name='is_rtl')
def is_rtl(language):
    """
    Check if language is RTL.
    
    Usage:
        {% if current_lang|is_rtl %}...{% endif %}
    """
    return language and language.is_rtl


# =============================================================================
# Translation Tags
# =============================================================================

@register.simple_tag(takes_context=True)
def trans_key(context, key, namespace=None, default=''):
    """
    Get translation for a key.
    
    Usage:
        {% trans_key "welcome_message" %}
        {% trans_key "add_to_cart" namespace="shop" %}
    """
    request = context.get('request')
    language = None
    
    if request and hasattr(request, 'language'):
        language = request.language
    
    if not language:
        language = LanguageService.get_default_language()
    
    if not language:
        return default or key
    
    translation = TranslationService.get_translation(key, language.code, namespace)
    return translation or default or key


@register.simple_tag(takes_context=True)
def translate_content(context, content_type, content_id, field_name, default=''):
    """
    Get translation for dynamic content.
    
    Usage:
        {% translate_content "product" product.id "name" default=product.name %}
    """
    request = context.get('request')
    language = None
    
    if request and hasattr(request, 'language'):
        language = request.language
    
    if not language:
        language = LanguageService.get_default_language()
    
    if not language:
        return default
    
    translation = TranslationService.get_content_translation(
        content_type, str(content_id), field_name, language.code
    )
    return translation or default


# =============================================================================
# Currency Switcher Tag
# =============================================================================

@register.inclusion_tag('i18n/currency_switcher.html', takes_context=True)
def currency_switcher(context):
    """
    Render currency switcher dropdown.
    
    Usage:
        {% currency_switcher %}
    """
    request = context.get('request')
    current = None
    
    if request and hasattr(request, 'currency'):
        current = request.currency
    
    if not current:
        current = CurrencyService.get_default_currency()
    
    return {
        'currencies': CurrencyService.get_active_currencies(),
        'current': current,
        'request': request,
    }


# =============================================================================
# Language Switcher Tag
# =============================================================================

@register.inclusion_tag('i18n/language_switcher.html', takes_context=True)
def language_switcher(context):
    """
    Render language switcher dropdown.
    
    Usage:
        {% language_switcher %}
    """
    request = context.get('request')
    current = None
    
    if request and hasattr(request, 'language'):
        current = request.language
    
    if not current:
        current = LanguageService.get_default_language()
    
    return {
        'languages': LanguageService.get_active_languages(),
        'current': current,
        'request': request,
    }


# =============================================================================
# Locale Data Tag (for JavaScript)
# =============================================================================

@register.simple_tag(takes_context=True)
def locale_json(context):
    """
    Output locale data as JSON for JavaScript.
    
    Usage:
        <script>const LOCALE = {% locale_json %};</script>
    """
    import json
    
    request = context.get('request')
    
    language = None
    currency = None
    
    if request:
        if hasattr(request, 'language'):
            language = request.language
        if hasattr(request, 'currency'):
            currency = request.currency
    
    if not language:
        language = LanguageService.get_default_language()
    if not currency:
        currency = CurrencyService.get_default_currency()
    
    data = {
        'language': {
            'code': language.code if language else 'en',
            'name': language.name if language else 'English',
            'is_rtl': language.is_rtl if language else False,
        },
        'currency': {
            'code': currency.code if currency else 'BDT',
            'symbol': currency.symbol if currency else '৳',
            'native_symbol': getattr(currency, 'native_symbol', None) or (currency.symbol if currency else '৳'),
            'decimal_places': currency.decimal_places if currency else 2,
            'symbol_position': currency.symbol_position if currency else 'before',
            'thousand_separator': getattr(currency, 'thousand_separator', ',') if currency else ',',
            'decimal_separator': getattr(currency, 'decimal_separator', '.') if currency else '.',
            'number_system': getattr(currency, 'number_system', 'western') if currency else 'western',
        }
    }
    
    return mark_safe(json.dumps(data))
