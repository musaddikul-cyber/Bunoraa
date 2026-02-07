"""
API Response middleware for consistent response format
"""
import json
from django.http import JsonResponse


class APIResponseMiddleware:
    """
    Middleware to ensure consistent API response format.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Only process API responses
        if not request.path.startswith('/api/'):
            return response
        
        # Skip if already in correct format or not JSON
        if not hasattr(response, 'content'):
            return response
        
        content_type = response.get('Content-Type', '')
        if 'application/json' not in content_type:
            return response
        
        try:
            data = json.loads(response.content.decode('utf-8'))
            
            # Skip if already in correct format
            if isinstance(data, dict) and 'success' in data:
                return response
            
            # Wrap in standard format
            wrapped_data = {
                'success': 200 <= response.status_code < 300,
                'message': 'Request processed successfully.' if response.status_code < 400 else 'Request failed.',
                'data': data,
                'meta': None
            }
            
            response.content = json.dumps(wrapped_data).encode('utf-8')
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        
        return response
