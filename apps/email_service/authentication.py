"""
Email Service Authentication
=============================

API key authentication for the email service.
"""

from rest_framework import authentication
from rest_framework.exceptions import AuthenticationFailed
from django.utils.translation import gettext_lazy as _
from django.core.cache import cache
from django.utils import timezone

from .models import APIKey


class APIKeyAuthentication(authentication.BaseAuthentication):
    """
    API key authentication for email service endpoints.
    
    Clients authenticate by passing the API key in the Authorization header:
        Authorization: Bearer BN.xxxxxxxxxxxxx
    
    Or via X-API-Key header:
        X-API-Key: BN.xxxxxxxxxxxxx
    """
    
    keyword = 'Bearer'
    
    def authenticate(self, request):
        """
        Authenticate the request and return a tuple of (user, api_key).
        """
        api_key = self._get_api_key(request)
        
        if not api_key:
            return None
        
        # Verify the API key
        key_obj = APIKey.verify_key(api_key)
        
        if not key_obj:
            raise AuthenticationFailed(_('Invalid or expired API key'))
        
        # Check IP restriction
        if not self._check_ip_allowed(request, key_obj):
            raise AuthenticationFailed(_('Request from unauthorized IP address'))
        
        # Check rate limit
        if not self._check_rate_limit(key_obj):
            raise AuthenticationFailed(_('Rate limit exceeded'))
        
        return (key_obj.user, key_obj)
    
    def _get_api_key(self, request):
        """Extract API key from request headers."""
        # Try Authorization header first
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        
        if auth_header.startswith(self.keyword + ' '):
            return auth_header[len(self.keyword) + 1:].strip()
        
        # Try X-API-Key header
        api_key_header = request.META.get('HTTP_X_API_KEY', '')
        if api_key_header:
            return api_key_header.strip()
        
        # Try query parameter (not recommended but supported)
        return request.query_params.get('api_key', '')
    
    def _check_ip_allowed(self, request, api_key):
        """Check if request IP is allowed for this API key."""
        if not api_key.allowed_ips:
            return True
        
        client_ip = self._get_client_ip(request)
        return client_ip in api_key.allowed_ips
    
    def _get_client_ip(self, request):
        """Get client IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', '')
    
    def _check_rate_limit(self, api_key):
        """
        Check if API key is within rate limits.
        Uses sliding window rate limiting.
        """
        now = timezone.now()
        key_id = str(api_key.id)
        
        # Check per-minute limit
        minute_key = f"email_api_rate:{key_id}:min:{now.strftime('%Y%m%d%H%M')}"
        minute_count = cache.get(minute_key, 0)
        if minute_count >= api_key.rate_limit_per_minute:
            return False
        
        # Check per-hour limit
        hour_key = f"email_api_rate:{key_id}:hour:{now.strftime('%Y%m%d%H')}"
        hour_count = cache.get(hour_key, 0)
        if hour_count >= api_key.rate_limit_per_hour:
            return False
        
        # Check per-day limit
        day_key = f"email_api_rate:{key_id}:day:{now.strftime('%Y%m%d')}"
        day_count = cache.get(day_key, 0)
        if day_count >= api_key.rate_limit_per_day:
            return False
        
        # Increment counters
        cache.set(minute_key, minute_count + 1, 120)
        cache.set(hour_key, hour_count + 1, 7200)
        cache.set(day_key, day_count + 1, 90000)
        
        return True
    
    def authenticate_header(self, request):
        """
        Return a string to be used as the value of the WWW-Authenticate
        header in a 401 Unauthenticated response.
        """
        return self.keyword
