import os
from django.conf import settings
from django.http import HttpResponse
from django.utils.encoding import iri_to_uri
from django.utils.cache import patch_response_headers
from urllib.parse import urlparse

BOT_AGENTS = ['googlebot', 'bingbot', 'yandex', 'baiduspider', 'duckduckbot']


def _sanitize_path(path: str) -> str:
    # map path to filesystem-friendly name
    if path == '/' or path == '':
        return 'index.html'
    # remove leading/trailing slashes
    p = path.strip('/').replace('/', '_')
    if not p:
        p = 'index'
    return f"{p}.html"


class BotPreRenderMiddleware:
    """Serve pre-rendered cached HTML for known crawlers to reduce TTFB.

    Behavior:
    - Only for GET requests
    - Only serve when user is not authenticated
    - Only serve for configured paths (PRERENDER_PATHS)
    - Adds header X-PreRendered: 1
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.cache_dir = os.path.join(settings.BASE_DIR, getattr(settings, 'PRERENDER_CACHE_DIR', 'prerender_cache'))
        self.paths = getattr(settings, 'PRERENDER_PATHS', ['/'])

    def __call__(self, request):
        # Quick checks to avoid overhead
        # Be defensive: AuthenticationMiddleware may not have run yet so guard request.user
        if request.method != 'GET' or (getattr(request, 'user', None) and request.user.is_authenticated):
            return self.get_response(request)
        ua = request.META.get('HTTP_USER_AGENT', '').lower()
        is_bot = any(bot in ua for bot in BOT_AGENTS)
        path = request.path
        if not is_bot:
            return self.get_response(request)

        # Only prerender configured paths and root + product/category detail patterns
        allowed = False
        for p in self.paths:
            if p == '/' and path == '/':
                allowed = True
            elif p.endswith('*') and path.startswith(p[:-1]):
                allowed = True
            elif path == p:
                allowed = True
        if not allowed:
            # allow products and categories by pattern for bots
            if path.startswith('/products/') or path.startswith('/categories/'):
                allowed = True
        if not allowed:
            return self.get_response(request)

        filename = _sanitize_path(path)
        full_path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(full_path):
            # No prerendered content; fallback
            return self.get_response(request)
        with open(full_path, 'rb') as f:
            content = f.read()
        resp = HttpResponse(content, content_type='text/html; charset=utf-8')
        resp['X-PreRendered'] = '1'
        # Cache for a short period so repeated crawls are fast
        patch_response_headers(resp, cache_timeout=300)
        return resp