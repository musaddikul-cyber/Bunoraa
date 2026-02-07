"""
Internationalization Middleware

Middleware for automatic locale detection and setting.
"""
import logging
from django.utils import timezone as django_timezone
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import activate

logger = logging.getLogger(__name__)


class LocaleMiddleware(MiddlewareMixin):
    """
    Middleware that sets language, currency, and timezone based on user preferences.
    
    Order of preference:
    1. Session value
    2. Cookie value
    3. User preference (if authenticated)
    4. Request detection (Accept-Language, GeoIP)
    5. Default from settings
    """
    
    def process_request(self, request):
        """Process incoming request to set locale."""
        from .services import (
            LanguageService, CurrencyService, TimezoneService
        )
        
        # Get user if authenticated
        user = request.user if hasattr(request, 'user') and request.user.is_authenticated else None
        
        # Set Language
        language = self._get_language(request, user)
        if language:
            request.LANGUAGE_CODE = language.code
            request.language = language
            activate(language.code)
            request.session['language'] = language.code
        
        # Set Currency
        currency = self._get_currency(request, user)
        if currency:
            request.currency = currency
            request.session['currency_code'] = currency.code
        
        # Set Timezone
        tz = self._get_timezone(request, user)
        if tz:
            request.timezone = tz
            try:
                import pytz
                django_timezone.activate(pytz.timezone(tz.name))
            except Exception:
                pass
    
    def process_response(self, request, response):
        """Process response to set cookies."""
        # Set language cookie if changed
        if hasattr(request, 'language') and request.language:
            current_cookie = request.COOKIES.get('language')
            if current_cookie != request.language.code:
                response.set_cookie(
                    'language', request.language.code,
                    max_age=365 * 24 * 60 * 60,
                    httponly=False,
                    samesite='Lax'
                )
        
        # Set currency cookie if changed
        if hasattr(request, 'currency') and request.currency:
            current_cookie = request.COOKIES.get('currency')
            if current_cookie != request.currency.code:
                response.set_cookie(
                    'currency', request.currency.code,
                    max_age=365 * 24 * 60 * 60,
                    httponly=False,
                    samesite='Lax'
                )
        
        return response
    
    def _get_language(self, request, user):
        """Get language for request."""
        from .services import LanguageService
        
        # 1. Session
        session_lang = request.session.get('language')
        if session_lang:
            lang = LanguageService.get_language_by_code(session_lang)
            if lang:
                return lang
        
        # 2. Cookie
        cookie_lang = request.COOKIES.get('language')
        if cookie_lang:
            lang = LanguageService.get_language_by_code(cookie_lang)
            if lang:
                return lang
        
        # 3. User preference
        if user:
            lang = LanguageService.get_user_language(user)
            if lang:
                return lang
        
        # 4. Detect from request
        return LanguageService.detect_language(request)
    
    def _get_currency(self, request, user):
        """Get currency for request."""
        from .services import CurrencyService
        
        # 1. Session
        session_curr = request.session.get('currency_code')
        if session_curr:
            curr = CurrencyService.get_currency_by_code(session_curr)
            if curr:
                return curr
        
        # 2. Cookie
        cookie_curr = request.COOKIES.get('currency')
        if cookie_curr:
            curr = CurrencyService.get_currency_by_code(cookie_curr)
            if curr:
                return curr
        
        # 3. User preference or detect
        return CurrencyService.get_user_currency(user, request)
    
    def _get_timezone(self, request, user):
        """Get timezone for request."""
        from .services import TimezoneService
        
        # 1. Session
        session_tz = request.session.get('timezone')
        if session_tz:
            tz = TimezoneService.get_timezone_by_name(session_tz)
            if tz:
                return tz
        
        # 2. Cookie
        cookie_tz = request.COOKIES.get('timezone')
        if cookie_tz:
            tz = TimezoneService.get_timezone_by_name(cookie_tz)
            if tz:
                return tz
        
        # 3. User preference or detect
        return TimezoneService.get_user_timezone(user, request)


class CurrencyMiddleware(MiddlewareMixin):
    """
    Lightweight middleware for currency detection only.
    Use this instead of LocaleMiddleware if you only need currency.
    """
    
    def process_request(self, request):
        """Set currency on request."""
        from .services import CurrencyService
        
        user = request.user if hasattr(request, 'user') and request.user.is_authenticated else None
        currency = CurrencyService.get_user_currency(user, request)
        
        if currency:
            request.currency = currency


class TimezoneMiddleware(MiddlewareMixin):
    """
    Lightweight middleware for timezone activation only.
    Use this instead of LocaleMiddleware if you only need timezone.
    """
    
    def process_request(self, request):
        """Activate timezone for request."""
        from .services import TimezoneService
        
        user = request.user if hasattr(request, 'user') and request.user.is_authenticated else None
        tz = TimezoneService.get_user_timezone(user, request)
        
        if tz:
            request.timezone = tz
            try:
                import pytz
                django_timezone.activate(pytz.timezone(tz.name))
            except Exception:
                pass
    
    def process_response(self, request, response):
        """Deactivate timezone."""
        django_timezone.deactivate()
        return response
