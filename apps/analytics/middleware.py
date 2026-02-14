"""
Analytics middleware for tracking page views
"""

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

        is_ajax = request.headers.get('X-Requested-With', '') == 'XMLHttpRequest'

        # Track page view for successful non-AJAX GET requests.
        if (
            request.method == 'GET'
            and response.status_code == 200
            and not self._is_excluded(request.path)
            and not is_ajax
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
