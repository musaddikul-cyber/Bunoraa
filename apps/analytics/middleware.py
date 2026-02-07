"""
Analytics middleware for tracking page views
"""
from django.conf import settings


class AnalyticsMiddleware:
    """Middleware to track page views."""
    
    EXCLUDED_PATHS = [
        '/admin/',
        '/api/',
        '/static/',
        '/media/',
        '/__debug__/',
    ]
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Track page view for successful GET requests
        if (
            request.method == 'GET' 
            and response.status_code == 200
            and not self._is_excluded(request.path)
            and not request.is_ajax() if hasattr(request, 'is_ajax') else 'XMLHttpRequest' not in request.headers.get('X-Requested-With', '')
        ):
            self._track_page_view(request)
        
        return response
    
    def _is_excluded(self, path):
        """Check if path should be excluded from tracking."""
        for excluded in self.EXCLUDED_PATHS:
            if path.startswith(excluded):
                return True
        return False
    
    def _track_page_view(self, request):
        """Track the page view asynchronously."""
        try:
            from .services import AnalyticsService
            AnalyticsService.track_page_view(request)
        except Exception:
            # Fail silently - analytics should not break the site
            pass
