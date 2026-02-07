from django.utils.deprecation import MiddlewareMixin
from django.conf import settings

BOT_AGENTS = ['googlebot', 'bingbot', 'yandex', 'baiduspider', 'duckduckbot']

class CacheControlHTMLMiddleware(MiddlewareMixin):
    """Set Cache-Control headers for anonymous HTML responses to enable edge caching.

    Rules:
    - Only GET requests
    - Only anonymous users
    - Skip admin, account, cart, checkout, api
    - For bots, set longer s-maxage
    """
    def process_response(self, request, response):
        try:
            if request.method != 'GET':
                return response
            if request.user.is_authenticated:
                return response
            path = request.path
            # Skip dynamic/private paths
            skip_prefixes = ['/admin', '/account', '/cart', '/checkout', '/api/', '/ws/']
            for p in skip_prefixes:
                if path.startswith(p):
                    return response
            content_type = response.get('Content-Type', '')
            if 'text/html' not in content_type:
                return response
            ua = request.META.get('HTTP_USER_AGENT', '').lower()
            is_bot = any(b in ua for b in BOT_AGENTS)
            if is_bot:
                # Long cache for bots
                response['Cache-Control'] = 'public, max-age=0, s-maxage=86400, stale-while-revalidate=3600'
                response['X-Edge-Cache'] = 'bot-cache'
            else:
                # Short cache for anonymous users
                response['Cache-Control'] = 'public, max-age=0, s-maxage=60, stale-while-revalidate=300'
                response['X-Edge-Cache'] = 'anonymous-cache'
        except Exception:
            pass
        return response