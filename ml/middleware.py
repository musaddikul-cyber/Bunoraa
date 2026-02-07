"""
ML Data Collection Middleware

Silent data collection middleware for tracking user behavior.
"""

import logging
import uuid
from datetime import datetime, timedelta

try:
    from django.conf import settings
    from django.utils import timezone
    from django.utils.deprecation import MiddlewareMixin
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    class MiddlewareMixin:
        pass

logger = logging.getLogger("bunoraa.ml.middleware")


class MLTrackingMiddleware(MiddlewareMixin):
    """
    Middleware for silent ML data collection.
    
    Tracks:
    - Page views with timing
    - Session data
    - Device/browser info
    - Location (from IP)
    - Referrer information
    """
    
    def __init__(self, get_response=None):
        self.get_response = get_response
        self._is_production = self._check_production()
        super().__init__(get_response)
    
    def _check_production(self) -> bool:
        """Check if running in production."""
        environment = getattr(settings, 'ENVIRONMENT', 'development')
        return environment == 'production'
    
    def _should_track(self, request) -> bool:
        """Determine if request should be tracked."""
        # Skip static files and media
        path = request.path.lower()
        if any(path.startswith(p) for p in ['/static/', '/media/', '/__debug__/']):
            return False
        
        # Skip health checks
        if path.startswith('/health'):
            return False
        
        # Skip API tracking endpoints (avoid recursion)
        if '/api/ml/track' in path or '/api/v1/track' in path:
            return False
        
        # Skip admin for non-admin tracking
        if path.startswith('/admin/') and not getattr(settings, 'ML_TRACK_ADMIN', False):
            return False
        
        # Skip bots
        user_agent = request.META.get('HTTP_USER_AGENT', '').lower()
        bot_patterns = ['bot', 'crawler', 'spider', 'scraper', 'curl', 'wget']
        if any(p in user_agent for p in bot_patterns):
            return False
        
        return True
    
    def _ensure_anonymous_id(self, request, response):
        """Ensure anonymous tracking ID exists."""
        if '_bunoraa_aid' not in request.COOKIES:
            aid = str(uuid.uuid4())
            max_age = 365 * 24 * 60 * 60  # 1 year
            response.set_cookie(
                '_bunoraa_aid',
                aid,
                max_age=max_age,
                httponly=True,
                secure=not settings.DEBUG,
                samesite='Lax',
            )
        return response
    
    def _get_page_type(self, request) -> str:
        """Determine page type from URL."""
        path = request.path.lower()
        
        if path == '/' or path == '':
            return 'home'
        elif '/product/' in path or '/products/' in path:
            return 'product'
        elif '/category/' in path or '/categories/' in path:
            return 'category'
        elif '/cart' in path:
            return 'cart'
        elif '/checkout' in path:
            return 'checkout'
        elif '/wishlist' in path:
            return 'wishlist'
        elif '/search' in path:
            return 'search'
        elif '/account' in path or '/profile' in path:
            return 'account'
        elif '/order' in path:
            return 'order'
        else:
            return 'other'
    
    def process_request(self, request):
        """Process incoming request."""
        if not self._should_track(request):
            request._ml_track = False
            return None
        
        request._ml_track = True
        request._ml_start_time = timezone.now() if DJANGO_AVAILABLE else datetime.now()
        
        # Ensure session exists
        if not request.session.session_key:
            request.session.create()
        
        return None
    
    def process_response(self, request, response):
        """Process outgoing response and track data."""
        # Ensure anonymous ID
        response = self._ensure_anonymous_id(request, response)
        
        # Skip if not tracking
        if not getattr(request, '_ml_track', False):
            return response
        
        # Skip non-200 responses
        if response.status_code != 200:
            return response
        
        # Skip non-HTML responses
        content_type = response.get('Content-Type', '')
        if 'text/html' not in content_type:
            return response
        
        try:
            from ml.data_collection.collector import DataCollector
            
            collector = DataCollector()
            page_type = self._get_page_type(request)
            
            # Calculate response time
            start_time = getattr(request, '_ml_start_time', None)
            response_time = 0
            if start_time:
                now = timezone.now() if DJANGO_AVAILABLE else datetime.now()
                response_time = (now - start_time).total_seconds() * 1000  # ms
            
            # Collect page view
            collector.collect_user_interaction(
                request=request,
                event_type='page_view',
                page_type=page_type,
                metadata={
                    'response_time_ms': response_time,
                    'status_code': response.status_code,
                }
            )
            
        except Exception as e:
            logger.error(f"ML tracking error: {e}")
        
        return response


class MLProductTrackingMiddleware(MiddlewareMixin):
    """
    Middleware for tracking product views.
    
    Captures product view events with source attribution.
    """
    
    def process_view(self, request, view_func, view_args, view_kwargs):
        """Track product views."""
        # Check if this is a product detail view
        view_name = getattr(view_func, '__name__', '')
        
        if 'product' not in view_name.lower() and 'detail' not in view_name.lower():
            return None
        
        # Check for product ID in kwargs
        product_id = view_kwargs.get('pk') or view_kwargs.get('product_id') or view_kwargs.get('slug')
        
        if not product_id:
            return None
        
        try:
            from ml.data_collection.collector import DataCollector
            from apps.catalog.models import Product
            
            # Get product
            if isinstance(product_id, int) or product_id.isdigit():
                product = Product.objects.get(id=int(product_id))
            else:
                product = Product.objects.get(slug=product_id)
            
            # Determine source
            referrer = request.META.get('HTTP_REFERER', '')
            source_type = 'direct'
            search_query = ''
            
            if 'search' in referrer:
                source_type = 'search'
                search_query = request.GET.get('q', '')
            elif 'category' in referrer:
                source_type = 'category'
            elif 'recommendation' in referrer or 'rec=' in request.GET.urlencode():
                source_type = 'recommendation'
            
            # Collect product interaction
            collector = DataCollector()
            collector.collect_product_interaction(
                request=request,
                product=product,
                interaction_type='view',
                source_info={
                    'source_page': referrer,
                    'source_type': source_type,
                    'search_query': search_query,
                    'position': int(request.GET.get('pos', 0)),
                }
            )
            
        except Exception as e:
            logger.debug(f"Product tracking error: {e}")
        
        return None
