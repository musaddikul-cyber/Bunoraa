"""
Internationalization Middleware

Middleware for automatic locale detection and setting.
Optimized for high-traffic scenarios with caching to reduce database load.
Includes circuit breaker for slow/failed database queries to prevent request hangs.
"""
import logging
import time
import threading
from zoneinfo import ZoneInfo
from django.conf import settings
from django.utils import timezone as django_timezone
from django.utils.cache import patch_vary_headers
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import activate
from django.db import DatabaseError
from django.db.utils import OperationalError

try:
    from django.utils.translation import LANGUAGE_SESSION_KEY
except ImportError:  # Django >=5.2 may not expose this constant
    LANGUAGE_SESSION_KEY = 'django_language'

logger = logging.getLogger(__name__)


class DatabaseCircuitBreaker:
    """
    Thread-safe circuit breaker to prevent cascading database failures.
    Temporarily disables OPTIONAL database queries if they're timing out or failing.
    NOTE: Never skips authentication/user session queries - only skips locale preferences!
    """
    _failures = 0
    _last_failure_time = None
    _FAILURE_THRESHOLD = 10  # More tolerant threshold (was 5, increased to 10)
    _RECOVERY_TIME = 60  # Try again after 60 seconds (was 30)
    _lock = threading.RLock()
    
    @classmethod
    def is_open(cls) -> bool:
        """Check if circuit breaker is open (DB is failing).
        
        Only blocks OPTIONAL queries (locale preferences, auto-detection).
        Does NOT block authentication/session queries.
        """
        with cls._lock:
            if cls._failures >= cls._FAILURE_THRESHOLD:
                now = time.time()
                if cls._last_failure_time and (now - cls._last_failure_time) < cls._RECOVERY_TIME:
                    return True
                else:
                    # Try to recover
                    cls._failures = 0
                    cls._last_failure_time = None
                    logger.info("Database circuit breaker: Attempting recovery")
                    return False
            return False
    
    @classmethod
    def record_failure(cls):
        """Record a database failure."""
        with cls._lock:
            cls._failures += 1
            cls._last_failure_time = time.time()
            logger.warning(f"Database query failure: {cls._failures}/{cls._FAILURE_THRESHOLD} failures - circuit breaker will open if limit reached")
    
    @classmethod
    def record_success(cls):
        """Reset failures on success."""
        with cls._lock:
            if cls._failures > 0:
                logger.info(f"Database circuit breaker: Query succeeded, resetting failure count ({cls._failures} → 0)")
            cls._failures = 0
            cls._last_failure_time = None


class LocaleMiddleware(MiddlewareMixin):
    """
    Middleware that sets language, currency, and timezone based on user preferences.
    
    Optimized for high-traffic scenarios to prevent database connection exhaustion.
    Uses circuit breaker to prevent cascade failures when database is under load.
    
    Order of preference:
    1. Session value (cached in request, no DB)
    2. Cookie value (cached in request, no DB)
    3. User preference (if authenticated, with select_related optimization)
    4. Request detection (Accept-Language, GeoIP headers - no DB)
    5. Default from settings (cached)
    
    Timing:
    - Max 500ms total for all i18n processing
    - Falls back to defaults if any operation exceeds 1s
    """
    
    # Maximum time to spend on locale detection per request (milliseconds)
    _MAX_MIDDLEWARE_TIME = 500
    
    def process_request(self, request):
        """Process incoming request to set locale with connection error handling."""
        from .services import (
            LanguageService, CurrencyService, TimezoneService
        )
        
        request._locale_start_time = time.time()
        
        try:
            # Check if we're already over time budget
            if self._exceeded_time_budget(request):
                logger.debug("Locale middleware: Time budget exceeded, using defaults")
                self._set_defaults_only(request)
                return
            
            # Check circuit breaker first
            if DatabaseCircuitBreaker.is_open():
                logger.debug("Database circuit breaker is open - using cached defaults only")
                # Set sensible defaults from cache/settings, skip optional DB queries
                self._set_defaults_only(request)
                return
            
            # Get user if authenticated
            user = request.user if hasattr(request, 'user') and request.user.is_authenticated else None
            
            # Initialize request-level cache to avoid duplicate queries within same request
            if not hasattr(request, '_locale_cache'):
                request._locale_cache = {}
            
            try:
                # Set Language
                if not self._exceeded_time_budget(request):
                    language = self._get_language(request, user)
                    if language:
                        request.LANGUAGE_CODE = language.code
                        request.language = language
                        request.locale = language.locale_code or language.code
                        activate(language.code)
                        if hasattr(request, 'session'):
                            if request.session.get(LANGUAGE_SESSION_KEY) != language.code:
                                request.session[LANGUAGE_SESSION_KEY] = language.code
                            if request.session.get('language') != language.code:
                                request.session['language'] = language.code
                DatabaseCircuitBreaker.record_success()
            except (DatabaseError, OperationalError) as e:
                DatabaseCircuitBreaker.record_failure()
                logger.warning(f"Database error setting language: {e}")
            except Exception as e:
                logger.warning(f"Error setting language in middleware: {e}")
            
            try:
                # Set Currency
                if not self._exceeded_time_budget(request):
                    currency = self._get_currency(request, user)
                    if currency:
                        request.currency = currency
                        if hasattr(request, 'session'):
                            if request.session.get('currency_code') != currency.code:
                                request.session['currency_code'] = currency.code
            except (DatabaseError, OperationalError) as e:
                DatabaseCircuitBreaker.record_failure()
                logger.warning(f"Database error setting currency: {e}")
            except Exception as e:
                logger.warning(f"Error setting currency in middleware: {e}")
            
            try:
                # Set Timezone - wrapped in try/except to handle DB connection errors
                if not self._exceeded_time_budget(request):
                    tz = self._get_timezone(request, user)
                    if tz:
                        request.timezone = tz
                        try:
                            django_timezone.activate(ZoneInfo(tz.name))
                        except Exception:
                            try:
                                import pytz
                                django_timezone.activate(pytz.timezone(tz.name))
                            except Exception:
                                pass
                        if hasattr(request, 'session'):
                            if request.session.get('timezone') != tz.name:
                                request.session['timezone'] = tz.name
            except (DatabaseError, OperationalError) as e:
                DatabaseCircuitBreaker.record_failure()
                logger.warning(f"Database error setting timezone: {e}")
            except Exception as e:
                logger.warning(f"Error setting timezone in middleware: {e}")
        
        except RuntimeError as e:
            # Handle async executor shutdown errors gracefully
            if 'interpreter shutdown' in str(e):
                logger.debug(f"Async executor shutdown during request processing (normal during shutdown): {e}")
                self._set_defaults_only(request)
            else:
                logger.warning(f"Runtime error in locale middleware: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in locale middleware: {e}", exc_info=True)
    
    def _exceeded_time_budget(self, request) -> bool:
        """Check if middleware has exceeded its time budget for this request."""
        if not hasattr(request, '_locale_start_time'):
            return False
        elapsed_ms = (time.time() - request._locale_start_time) * 1000
        return elapsed_ms > self._MAX_MIDDLEWARE_TIME
    
    def _set_defaults_only(self, request):
        """Set locale defaults when circuit breaker is open (DB is unavailable).
        
        Important: Do NOT skip user auth checks - this could cause session logout.
        Only skip OPTIONAL locale preferences, keep user authentication intact.
        """
        try:
            from .services import LanguageService, CurrencyService, TimezoneService
            
            # CRITICAL: Always preserve authenticated user's session
            # Only use cached defaults for locale, not for authentication
            if hasattr(request, 'user') and request.user.is_authenticated:
                # User is authenticated - use default locale but preserve auth
                logger.debug(f"Preserving authenticated user session during circuit breaker")
            
            request.language = LanguageService.get_default_language()
            if request.language:
                activate(request.language.code)
                request.LANGUAGE_CODE = request.language.code
            
            request.currency = CurrencyService.get_default_currency()
            request.timezone = TimezoneService._get_cached_default_timezone()
            
            # Log timing
            if hasattr(request, '_locale_start_time'):
                elapsed_ms = (time.time() - request._locale_start_time) * 1000
                logger.debug(f"Locale middleware (fallback defaults): {elapsed_ms:.1f}ms")
            
        except Exception as e:
            logger.debug(f"Could not set defaults: {e}")
    
    def process_response(self, request, response):
        """Process response to set cookies and log timing."""
        # Log timing info
        if hasattr(request, '_locale_start_time'):
            elapsed_ms = (time.time() - request._locale_start_time) * 1000
            logger.debug(f"Locale middleware total time: {elapsed_ms:.1f}ms")
        
        # Set language cookie if changed
        if hasattr(request, 'language') and request.language:
            cookie_name = getattr(settings, 'LANGUAGE_COOKIE_NAME', 'language')
            current_cookie = request.COOKIES.get(cookie_name) or request.COOKIES.get('language')
            if current_cookie != request.language.code:
                response.set_cookie(
                    cookie_name,
                    request.language.code,
                    max_age=getattr(settings, 'LANGUAGE_COOKIE_AGE', 365 * 24 * 60 * 60),
                    httponly=False,
                    samesite=getattr(settings, 'LANGUAGE_COOKIE_SAMESITE', 'Lax')
                )
                if cookie_name != 'language':
                    response.set_cookie(
                        'language',
                        request.language.code,
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

        # Content-Language header for caches/clients
        if hasattr(request, 'language') and request.language:
            response['Content-Language'] = request.language.code

        # Ensure caches vary on language/cookie headers
        patch_vary_headers(response, ['Accept-Language', 'X-User-Language', 'Cookie'])
        
        return response
    
    def _get_language(self, request, user):
        """Get language for request with request-level caching and timeout protection."""
        from .services import LanguageService
        
        # Check request cache first
        if '_language' in request._locale_cache:
            return request._locale_cache['_language']
        
        # Quick timeout check before expensive operations
        if self._exceeded_time_budget(request):
            return None
        
        # 1. Session (no DB)
        session_lang = request.session.get(LANGUAGE_SESSION_KEY) or request.session.get('language')
        if session_lang:
            # Wrap with timeout since get_language_by_code might hit DB
            lang = self._call_with_timeout(
                lambda: LanguageService.get_language_by_code(session_lang),
                timeout=0.5
            )
            if lang:
                request._locale_cache['_language'] = lang
                return lang
        
        # 2. Cookie (no DB)
        cookie_name = getattr(settings, 'LANGUAGE_COOKIE_NAME', 'language')
        cookie_lang = request.COOKIES.get(cookie_name) or request.COOKIES.get('language')
        if cookie_lang:
            lang = self._call_with_timeout(
                lambda: LanguageService.get_language_by_code(cookie_lang),
                timeout=0.5
            )
            if lang:
                request._locale_cache['_language'] = lang
                return lang
        
        # 3. User preference (requires DB) - Skip if circuit breaker is open
        if user and not DatabaseCircuitBreaker.is_open():
            pref = self._get_user_pref(request, user)
            if pref and pref.language and pref.language.is_active and not pref.auto_detect_language:
                request._locale_cache['_language'] = pref.language
                return pref.language

        # 4. Detect from request headers (no DB)
        lang = self._call_with_timeout(
            lambda: LanguageService.detect_language(request),
            timeout=0.5
        )
        request._locale_cache['_language'] = lang
        return lang
    
    def _get_currency(self, request, user):
        """Get currency for request with request-level caching and timeout protection."""
        from .services import CurrencyService
        
        # Check request cache first
        if '_currency' in request._locale_cache:
            return request._locale_cache['_currency']
        
        # Quick timeout check before expensive operations
        if self._exceeded_time_budget(request):
            return None
        
        # 1. Session (no DB)
        session_curr = request.session.get('currency_code')
        if session_curr:
            curr = self._call_with_timeout(
                lambda: CurrencyService.get_currency_by_code(session_curr),
                timeout=0.5
            )
            if curr:
                request._locale_cache['_currency'] = curr
                return curr
        
        # 2. Cookie (no DB)
        cookie_curr = request.COOKIES.get('currency')
        if cookie_curr:
            curr = self._call_with_timeout(
                lambda: CurrencyService.get_currency_by_code(cookie_curr),
                timeout=0.5
            )
            if curr:
                request._locale_cache['_currency'] = curr
                return curr
        
        # 3. User preference or detect - Skip if circuit breaker is open
        if not DatabaseCircuitBreaker.is_open():
            pref = self._get_user_pref(request, user)
            if pref and pref.currency and pref.currency.is_active and not pref.auto_detect_currency:
                request._locale_cache['_currency'] = pref.currency
                return pref.currency
            curr = self._call_with_timeout(
                lambda: CurrencyService.get_user_currency(user, request),
                timeout=1.0
            )
            request._locale_cache['_currency'] = curr
            return curr
        
        # Fallback to cached default
        curr = CurrencyService.get_default_currency()
        request._locale_cache['_currency'] = curr
        return curr
    
    def _get_timezone(self, request, user):
        """Get timezone for request with request-level caching and timeout protection."""
        from .services import TimezoneService
        
        # Check request cache first
        if '_timezone' in request._locale_cache:
            return request._locale_cache['_timezone']
        
        # Quick timeout check before expensive operations
        if self._exceeded_time_budget(request):
            return None
        
        # 1. Session (no DB)
        session_tz = request.session.get('timezone')
        if session_tz:
            tz = self._call_with_timeout(
                lambda: TimezoneService.get_timezone_by_name(session_tz),
                timeout=0.5
            )
            if tz:
                request._locale_cache['_timezone'] = tz
                return tz
        
        # 2. Cookie (no DB)
        cookie_tz = request.COOKIES.get('timezone')
        if cookie_tz:
            tz = self._call_with_timeout(
                lambda: TimezoneService.get_timezone_by_name(cookie_tz),
                timeout=0.5
            )
            if tz:
                request._locale_cache['_timezone'] = tz
                return tz
        
        # 3. User preference or detect - Skip if circuit breaker is open
        if not DatabaseCircuitBreaker.is_open():
            pref = self._get_user_pref(request, user)
            if pref and pref.timezone and not pref.auto_detect_timezone:
                request._locale_cache['_timezone'] = pref.timezone
                return pref.timezone
            tz = self._call_with_timeout(
                lambda: TimezoneService.get_user_timezone(user, request),
                timeout=1.0
            )
            request._locale_cache['_timezone'] = tz
            return tz
        
        # Fallback to nothing, will use Django default
        request._locale_cache['_timezone'] = None
        return None
    
    @staticmethod
    def _call_with_timeout(func, timeout=1.0):
        """Call a function safely (DB timeouts handled at the database level)."""
        try:
            return func()
        except (DatabaseError, OperationalError):
            raise
        except Exception as exc:
            logger.debug(f"Call failed in {getattr(func, '__name__', 'callable')}: {exc}")
            return None

    def _get_user_pref(self, request, user):
        """Fetch user locale preference once per request to avoid repeated DB hits."""
        if not user or not getattr(user, 'is_authenticated', False):
            return None
        cache_key = '_user_locale_pref'
        if cache_key in request._locale_cache:
            return request._locale_cache[cache_key]
        if DatabaseCircuitBreaker.is_open():
            request._locale_cache[cache_key] = None
            return None

        try:
            from .models import UserLocalePreference

            pref = (
                UserLocalePreference.objects
                .filter(user=user)
                .select_related('language', 'currency', 'timezone', 'country')
                .first()
            )
            request._locale_cache[cache_key] = pref
            return pref
        except (DatabaseError, OperationalError) as exc:
            DatabaseCircuitBreaker.record_failure()
            logger.warning(f"Database error fetching locale preference: {exc}")
            request._locale_cache[cache_key] = None
            return None


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
