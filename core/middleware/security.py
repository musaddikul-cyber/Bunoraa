"""
Security Middleware for Bunoraa.
Implements rate limiting, security headers, and request validation.
"""
import time
import hashlib
import json
from typing import Optional, Callable
from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.utils.deprecation import MiddlewareMixin


class SecurityMiddleware(MiddlewareMixin):
    """
    Enhanced security middleware with additional protections.
    """
    
    def __init__(self, get_response: Callable = None):
        self.get_response = get_response
        super().__init__(get_response)
    
    def process_request(self, request: HttpRequest) -> Optional[HttpResponse]:
        """Process incoming requests for security checks."""
        
        # Check for suspicious headers
        if self._has_suspicious_headers(request):
            return HttpResponse('Forbidden', status=403)
        
        # Validate content type for POST requests
        if request.method == 'POST':
            content_type = request.META.get('CONTENT_TYPE', '')
            if not self._is_valid_content_type(content_type):
                pass  # Allow but log
        
        return None
    
    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        """Add security headers to response."""
        
        # Permissions Policy (formerly Feature-Policy)
        response['Permissions-Policy'] = (
            'accelerometer=(), autoplay=(self), camera=(), '
            'cross-origin-isolated=(), display-capture=(), encrypted-media=(self), '
            'fullscreen=(self), geolocation=(), gyroscope=(), keyboard-map=(), '
            'magnetometer=(), microphone=(), midi=(), payment=(self), '
            'picture-in-picture=(self), publickey-credentials-get=(), '
            'screen-wake-lock=(), sync-xhr=(self), usb=(), xr-spatial-tracking=()'
        )
        
        # Referrer Policy (stricter default)
        if 'Referrer-Policy' not in response:
            response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Cross-Origin headers for API responses
        if request.path.startswith('/api/'):
            response['Cross-Origin-Resource-Policy'] = 'cross-origin'
        
        # Cache headers for sensitive endpoints
        if self._is_sensitive_endpoint(request):
            response['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
        
        return response
    
    def _has_suspicious_headers(self, request: HttpRequest) -> bool:
        """Check for suspicious header patterns."""
        # Check for potential attacks in headers
        suspicious_patterns = [
            '${', '#{', '<%', '%>', '<script', 'javascript:',
            'data:text/html', 'vbscript:', 'onload=', 'onerror='
        ]
        
        for header, value in request.META.items():
            if header.startswith('HTTP_') and isinstance(value, str):
                value_lower = value.lower()
                for pattern in suspicious_patterns:
                    if pattern.lower() in value_lower:
                        return True
        
        return False
    
    def _is_valid_content_type(self, content_type: str) -> bool:
        """Validate content type for form submissions."""
        valid_types = [
            'application/x-www-form-urlencoded',
            'multipart/form-data',
            'application/json',
            'text/plain'
        ]
        
        for valid in valid_types:
            if valid in content_type:
                return True
        return False
    
    def _is_sensitive_endpoint(self, request: HttpRequest) -> bool:
        """Check if endpoint handles sensitive data."""
        sensitive_paths = [
            '/accounts/', '/checkout/', '/orders/',
            '/api/auth/', '/api/users/', '/admin/'
        ]
        
        for path in sensitive_paths:
            if request.path.startswith(path):
                return True
        return False


class RateLimitMiddleware:
    """
    Rate limiting middleware using sliding window algorithm.
    """
    
    def __init__(self, get_response: Callable):
        self.get_response = get_response
        
        # Rate limit configurations
        self.limits = {
            'default': {'requests': 100, 'window': 60},  # 100 req/min
            'api': {'requests': 60, 'window': 60},       # 60 req/min
            'auth': {'requests': 5, 'window': 60},        # 5 req/min
            'search': {'requests': 30, 'window': 60},     # 30 req/min
        }
        
        # Paths and their rate limit categories
        self.path_categories = {
            '/api/': 'api',
            '/api/auth/': 'auth',
            '/api/v1/auth/': 'auth',
            '/accounts/login/': 'auth',
            '/accounts/register/': 'auth',
            '/api/search/': 'search',
            '/search/': 'search',
        }
        
        # Exempt paths
        self.exempt_paths = [
            '/static/',
            '/media/',
            '/health/',
            '/admin/',
        ]
        
        # Trusted IPs (e.g., monitoring services)
        self.trusted_ips = getattr(settings, 'RATE_LIMIT_TRUSTED_IPS', [])
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Check if rate limiting is enabled
        if not getattr(settings, 'RATE_LIMIT_ENABLED', True):
            return self.get_response(request)
        
        # Skip exempt paths
        for path in self.exempt_paths:
            if request.path.startswith(path):
                return self.get_response(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check if trusted IP
        client_ip = self._get_client_ip(request)
        if client_ip in self.trusted_ips:
            return self.get_response(request)
        
        # Get rate limit category
        category = self._get_category(request.path)
        limits = self.limits.get(category, self.limits['default'])
        
        # Check rate limit
        if self._is_rate_limited(client_id, category, limits):
            return self._rate_limit_response(limits)
        
        # Process request
        response = self.get_response(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_id, category, limits)
        
        return response
    
    def _get_client_id(self, request: HttpRequest) -> str:
        """Generate unique client identifier."""
        # Use user ID if authenticated
        if request.user.is_authenticated:
            return f'user_{request.user.id}'
        
        # Fall back to IP + User Agent hash
        ip = self._get_client_ip(request)
        ua = request.META.get('HTTP_USER_AGENT', '')
        combined = f'{ip}:{ua}'
        return f'anon_{hashlib.md5(combined.encode()).hexdigest()[:16]}'
    
    def _get_client_ip(self, request: HttpRequest) -> str:
        """Get client IP address, handling proxies."""
        # Check for Cloudflare header first
        cf_ip = request.META.get('HTTP_CF_CONNECTING_IP')
        if cf_ip:
            return cf_ip
        
        # Check for X-Forwarded-For
        xff = request.META.get('HTTP_X_FORWARDED_FOR')
        if xff:
            return xff.split(',')[0].strip()
        
        # Check for X-Real-IP
        real_ip = request.META.get('HTTP_X_REAL_IP')
        if real_ip:
            return real_ip
        
        return request.META.get('REMOTE_ADDR', '')
    
    def _get_category(self, path: str) -> str:
        """Get rate limit category for path."""
        for prefix, category in sorted(self.path_categories.items(), 
                                       key=lambda x: -len(x[0])):
            if path.startswith(prefix):
                return category
        return 'default'
    
    def _is_rate_limited(self, client_id: str, category: str, limits: dict) -> bool:
        """Check if client has exceeded rate limit."""
        cache_key = f'rate_limit:{category}:{client_id}'
        current_time = time.time()
        window = limits['window']
        max_requests = limits['requests']
        
        # Get current window data
        data = cache.get(cache_key, {'requests': [], 'count': 0})
        
        # Clean old requests outside window
        cutoff = current_time - window
        data['requests'] = [t for t in data['requests'] if t > cutoff]
        
        # Check if limit exceeded
        if len(data['requests']) >= max_requests:
            return True
        
        # Add current request
        data['requests'].append(current_time)
        data['count'] = len(data['requests'])
        
        # Store with expiry
        cache.set(cache_key, data, timeout=window * 2)
        
        return False
    
    def _rate_limit_response(self, limits: dict) -> HttpResponse:
        """Generate rate limit exceeded response."""
        response = JsonResponse({
            'error': 'Rate limit exceeded',
            'message': f'Too many requests. Please try again in {limits["window"]} seconds.',
            'retry_after': limits['window']
        }, status=429)
        
        response['Retry-After'] = str(limits['window'])
        return response
    
    def _add_rate_limit_headers(self, response: HttpResponse, client_id: str, 
                                 category: str, limits: dict) -> None:
        """Add rate limit headers to response."""
        cache_key = f'rate_limit:{category}:{client_id}'
        data = cache.get(cache_key, {'requests': [], 'count': 0})
        
        remaining = max(0, limits['requests'] - len(data['requests']))
        
        response['X-RateLimit-Limit'] = str(limits['requests'])
        response['X-RateLimit-Remaining'] = str(remaining)
        response['X-RateLimit-Window'] = str(limits['window'])
        response['X-RateLimit-Category'] = category


class CORSMiddleware:
    """
    Cross-Origin Resource Sharing middleware.
    More flexible than django-cors-headers for specific needs.
    """
    
    def __init__(self, get_response: Callable):
        self.get_response = get_response
        
        self.allowed_origins = getattr(settings, 'CORS_ALLOWED_ORIGINS', [
            'https://bunoraa.com',
            'https://www.bunoraa.com',
        ])
        
        self.allowed_methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS']
        self.allowed_headers = [
            'Accept', 'Accept-Language', 'Content-Type', 'Content-Language',
            'Authorization', 'X-CSRFToken', 'X-Requested-With'
        ]
        self.expose_headers = [
            'X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Window'
        ]
        self.max_age = 86400  # 24 hours
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Handle preflight requests
        if request.method == 'OPTIONS':
            response = HttpResponse()
            response = self._add_cors_headers(request, response)
            return response
        
        response = self.get_response(request)
        return self._add_cors_headers(request, response)
    
    def _add_cors_headers(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        """Add CORS headers to response."""
        origin = request.META.get('HTTP_ORIGIN', '')
        
        # Check if origin is allowed
        if origin in self.allowed_origins or '*' in self.allowed_origins:
            response['Access-Control-Allow-Origin'] = origin
            response['Access-Control-Allow-Credentials'] = 'true'
        
        # Preflight response headers
        if request.method == 'OPTIONS':
            response['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            response['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            response['Access-Control-Max-Age'] = str(self.max_age)
        
        # Expose headers
        response['Access-Control-Expose-Headers'] = ', '.join(self.expose_headers)
        
        return response


class RequestLoggingMiddleware:
    """
    Request logging middleware for debugging and monitoring.
    """
    
    def __init__(self, get_response: Callable):
        self.get_response = get_response
        self.logger = self._get_logger()
    
    def _get_logger(self):
        import logging
        return logging.getLogger('bunoraa.requests')
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Skip static files
        if request.path.startswith('/static/') or request.path.startswith('/media/'):
            return self.get_response(request)
        
        start_time = time.time()
        
        response = self.get_response(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Log request
        self._log_request(request, response, duration)
        
        return response
    
    def _log_request(self, request: HttpRequest, response: HttpResponse, duration: float) -> None:
        """Log request details."""
        log_data = {
            'method': request.method,
            'path': request.path,
            'status': response.status_code,
            'duration_ms': round(duration * 1000, 2),
            'ip': self._get_ip(request),
            'user': str(request.user) if request.user.is_authenticated else 'anonymous',
        }
        
        # Log at appropriate level based on status
        if response.status_code >= 500:
            self.logger.error('Request error', extra=log_data)
        elif response.status_code >= 400:
            self.logger.warning('Request warning', extra=log_data)
        else:
            self.logger.info('Request completed', extra=log_data)
    
    def _get_ip(self, request: HttpRequest) -> str:
        """Get client IP."""
        return (
            request.META.get('HTTP_CF_CONNECTING_IP') or
            request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')[0].strip() or
            request.META.get('HTTP_X_REAL_IP') or
            request.META.get('REMOTE_ADDR', '')
        )
