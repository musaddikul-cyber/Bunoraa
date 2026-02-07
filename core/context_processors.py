"""
Context processors for templates
"""
from django.conf import settings
from django.core.cache import cache
from decimal import Decimal
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def site_settings(request):
    """Add site settings to template context. Optimized for memory-constrained environments."""
    # Cache the site settings for performance
    cache_key = 'site_settings_context'
    cached_settings = cache.get(cache_key)
    
    if cached_settings is None:
        try:
            from apps.pages.models import SiteSettings, SocialLink
            site = SiteSettings.get_settings()
            
            # Get tax rate from site settings
            tax_rate = float(site.tax_rate) if site.tax_rate else 0

            # Build social links list - optimized for memory
            social_links = []
            try:
                qs = site.social_links.filter(is_active=True).order_by('order')
                social_links = [
                    {'name': s.name, 'url': s.url, 'icon': s.get_icon_url()}
                    for s in qs
                ]
            except Exception:
                # Fallback to global SocialLink objects for compatibility
                try:
                    qs = SocialLink.objects.filter(is_active=True).order_by('order')
                    social_links = [
                        {'name': s.name, 'url': s.url, 'icon': s.get_icon_url()}
                        for s in qs
                    ]
                except Exception:
                    social_links = []

            cached_settings = {
                'SITE_NAME': site.site_name or 'Bunoraa',
                'SITE_TAGLINE': site.site_tagline or 'Premium Quality Products',
                'SITE_DESCRIPTION': site.site_description or '',
                'SITE_LOGO': site.logo.url if site.logo else None,
                'SITE_LOGO_DARK': site.logo_dark.url if site.logo_dark else None,
                'SITE_FAVICON': site.favicon.url if site.favicon else None,
                'CONTACT_EMAIL': site.contact_email or '',
                'CONTACT_PHONE': site.contact_phone or '',
                'CONTACT_ADDRESS': site.contact_address or '',
                'FACEBOOK_URL': site.facebook_url or '',
                'INSTAGRAM_URL': site.instagram_url or '',
                'TWITTER_URL': site.twitter_url or '',
                'LINKEDIN_URL': site.linkedin_url or '',
                'YOUTUBE_URL': site.youtube_url or '',
                'TIKTOK_URL': site.tiktok_url or '',
                'COPYRIGHT_TEXT': site.copyright_text or '',
                'FOOTER_TEXT': site.footer_text or '',
                'GOOGLE_ANALYTICS_ID': site.google_analytics_id or '',
                'FACEBOOK_PIXEL_ID': site.facebook_pixel_id or '',
                'CUSTOM_HEAD_SCRIPTS': site.custom_head_scripts or '',
                'CUSTOM_BODY_SCRIPTS': site.custom_body_scripts or '',
                'SOCIAL_LINKS': social_links,
                'TAX_RATE': tax_rate,
            }
            # Cache for 5 minutes
            cache.set(cache_key, cached_settings, 300)
        except Exception:
            # Fallback if database is not available
            cached_settings = {
                'SITE_NAME': 'Bunoraa',
                'SITE_TAGLINE': 'Premium Quality Products',
                'SITE_DESCRIPTION': '',
                'SITE_LOGO': None,
                'SITE_LOGO_DARK': None,
                'SITE_FAVICON': None,
                'CONTACT_EMAIL': '',
                'CONTACT_PHONE': '',
                'CONTACT_ADDRESS': '',
                'FACEBOOK_URL': '',
                'INSTAGRAM_URL': '',
                'TWITTER_URL': '',
                'LINKEDIN_URL': '',
                'YOUTUBE_URL': '',
                'TIKTOK_URL': '',
                'COPYRIGHT_TEXT': '',
                'FOOTER_TEXT': '',
                'GOOGLE_ANALYTICS_ID': '',
                'FACEBOOK_PIXEL_ID': '',
                'CUSTOM_HEAD_SCRIPTS': '',
                'CUSTOM_BODY_SCRIPTS': '',
                'SOCIAL_LINKS': [],
                'TAX_RATE': 0,
            }
    
    # Determine per-request currency (do not cache - user/session based)
    # Also fetch shipping and payment settings
    free_shipping_threshold = 2000  # Default
    default_shipping_cost = 60  # Default
    cod_fee = 0  # Default
    
    try:
        from apps.i18n.services import CurrencyService, CurrencyConversionService
        currency = CurrencyService.get_user_currency(
            user=request.user if request.user.is_authenticated else None,
            request=request
        )
        currency_code = currency.code if currency else 'BDT'
        currency_symbol = currency.symbol if currency else '৳'
        currency_native_symbol = getattr(currency, 'native_symbol', currency_symbol) if currency else '৳'
        currency_decimal_places = currency.decimal_places if currency else 2
        currency_thousand_separator = getattr(currency, 'thousand_separator', ',') if currency else ','
        currency_decimal_separator = getattr(currency, 'decimal_separator', '.') if currency else '.'
        currency_symbol_position = currency.symbol_position if currency else 'before'
        currency_number_system = getattr(currency, 'number_system', 'western') if currency else 'western'
        currency_locale = 'bn-BD' if currency and getattr(currency, 'code', '') == 'BDT' else 'en-US'
        
        # Get free shipping threshold from ShippingSettings (robust source)
        try:
            from apps.shipping.models import ShippingSettings, ShippingRate
            shipping_settings = ShippingSettings.get_settings()
            if shipping_settings.enable_free_shipping and shipping_settings.free_shipping_threshold:
                free_shipping_threshold = float(shipping_settings.free_shipping_threshold)
            
            # Get default shipping cost from first active rate or settings
            default_rate = ShippingRate.objects.filter(
                zone__is_active=True,
                method__is_active=True
            ).order_by('base_rate').first()
            if default_rate:
                default_shipping_cost = float(default_rate.base_rate)
        except Exception:
            pass
        
        # Get COD fee from PaymentGateway
        try:
            from apps.payments.models import PaymentGateway
            cod_gateway = PaymentGateway.objects.filter(code='cod', is_active=True).first()
            if cod_gateway and cod_gateway.fee_amount:
                cod_fee = float(cod_gateway.fee_amount)
        except Exception:
            pass

        # Convert monetary settings to user's currency if needed
        free_shipping_threshold_converted = free_shipping_threshold
        default_shipping_cost_converted = default_shipping_cost
        cod_fee_converted = cod_fee
        if currency_code and currency_code != 'BDT':
            try:
                free_shipping_threshold_converted = float(CurrencyConversionService.convert_by_code(
                    Decimal(str(free_shipping_threshold)), 'BDT', currency_code, round_result=True
                ))
            except Exception:
                free_shipping_threshold_converted = free_shipping_threshold
            try:
                default_shipping_cost_converted = float(CurrencyConversionService.convert_by_code(
                    Decimal(str(default_shipping_cost)), 'BDT', currency_code, round_result=True
                ))
            except Exception:
                default_shipping_cost_converted = default_shipping_cost
            try:
                cod_fee_converted = float(CurrencyConversionService.convert_by_code(
                    Decimal(str(cod_fee)), 'BDT', currency_code, round_result=True
                ))
            except Exception:
                cod_fee_converted = cod_fee
            
    except Exception:
        currency_code = 'BDT'
        currency_symbol = '৳'
        currency_native_symbol = '৳'
        currency_decimal_places = 2
        currency_thousand_separator = ','
        currency_decimal_separator = '.'
        currency_symbol_position = 'before'
        currency_number_system = 'western'
        currency_locale = 'en-US'
        free_shipping_threshold_converted = free_shipping_threshold
        default_shipping_cost_converted = default_shipping_cost
        cod_fee_converted = cod_fee

    # Build a canonical URL (strip common tracking params, keep search/page where relevant)
    def build_canonical(req):
        full = req.build_absolute_uri()
        parsed = urlparse(full)
        qs = parse_qs(parsed.query)
        allowed = {'q', 'page', 'category', 'brand', 'sort'}
        filtered = {k: v for k, v in qs.items() if k in allowed}
        # remove page=1
        if 'page' in filtered and filtered['page'] == ['1']:
            filtered.pop('page')
        new_q = urlencode([(k, item) for k, vals in filtered.items() for item in vals], doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', new_q, ''))

    # Default robots meta; mark non-content pages as noindex
    def compute_meta_robots(req):
        path = req.path or ''
        noindex_paths = ['/search', '/account', '/cart', '/checkout', '/wishlist', '/preorders/wizard']
        for p in noindex_paths:
            if path.startswith(p):
                return 'noindex, follow'
        return 'index, follow'

    ua = (request.META.get('HTTP_USER_AGENT') or '').lower()
    is_crawler = False
    for bot in ['googlebot', 'bingbot', 'yandex', 'baiduspider', 'duckduckbot']:
        if bot in ua:
            is_crawler = True
            break

    return {
        **cached_settings,
        'IS_DEBUG': settings.DEBUG,
        'STRIPE_PUBLIC_KEY': getattr(settings, 'STRIPE_PUBLIC_KEY', ''),
        'currency_code': currency_code,
        'currency_symbol': currency_symbol,
        'currency_native_symbol': currency_native_symbol,
        'currency_locale': currency_locale,
        'currency_decimal_places': currency_decimal_places,
        'currency_thousand_separator': currency_thousand_separator,
        'currency_decimal_separator': currency_decimal_separator,
        'currency_symbol_position': currency_symbol_position,
        'currency_number_system': currency_number_system,
        'free_shipping_threshold': free_shipping_threshold,
        'free_shipping_threshold_converted': free_shipping_threshold_converted,
        'default_shipping_cost': default_shipping_cost,
        'default_shipping_cost_converted': default_shipping_cost_converted,
        'cod_fee': cod_fee,
        'cod_fee_converted': cod_fee_converted,
        'tax_rate': cached_settings.get('TAX_RATE', 0),
        'canonical_url': build_canonical(request),
        'meta_robots': compute_meta_robots(request),
        'IS_CRAWLER': is_crawler,
        'SOCIAL_AUTH_GOOGLE_OAUTH2_KEY': getattr(settings, 'SOCIAL_AUTH_GOOGLE_OAUTH2_KEY', ''),
        'SOCIAL_AUTH_GOOGLE_ENABLED': bool(getattr(settings, 'SOCIAL_AUTH_GOOGLE_OAUTH2_KEY', '')),
        'SOCIAL_AUTH_GOOGLE_REDIRECT_URI': getattr(settings, 'SOCIAL_AUTH_GOOGLE_OAUTH2_REDIRECT_URI', ''),
    }
