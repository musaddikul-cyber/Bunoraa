from django.http import HttpResponsePermanentRedirect
from django.conf import settings
from django.urls import resolve, Resolver404
from urllib.parse import urlencode


class EnsureTrailingSlashMiddleware:
    """Redirect requests missing a trailing slash to their slash form when the
    slash-appended path resolves to a valid URL pattern.

    Rules:
    - Only for GET and HEAD requests
    - Skip if path already ends with '/'
    - Skip assets (has a file extension) and API/static/media/ws prefixes
    - If resolving `path/` succeeds and `path` does not resolve, redirect to `path/` preserving query string

    The middleware is conservative to avoid surprising redirects for API endpoints.
    """

    SKIP_PREFIXES = ('/api/', '/static/', '/media/', '/ws/')

    def __init__(self, get_response):
        self.get_response = get_response

    def _should_check(self, request):
        path = request.path or ''
        if request.method not in ('GET', 'HEAD'):
            return False
        if not path or path.endswith('/'):
            return False
        # skip common static/api prefixes
        for p in self.SKIP_PREFIXES:
            if path.startswith(p):
                return False
        # skip obvious static assets with an extension
        if '.' in path.split('/')[-1]:
            return False
        return True

    def __call__(self, request):
        try:
            if not self._should_check(request):
                return self.get_response(request)

            path = request.path
            # If `path` already resolves, don't interfere
            try:
                resolve(path)
                return self.get_response(request)
            except Resolver404:
                pass

            # Try resolving with trailing slash
            try:
                resolve(path + '/')
                # Build redirect URL preserving query string
                qs = request.META.get('QUERY_STRING', '')
                scheme = 'https' if not settings.DEBUG else request.scheme
                host = request.get_host()
                new_path = path + '/'
                if qs:
                    new_url = f"{scheme}://{host}{new_path}?{qs}"
                else:
                    new_url = f"{scheme}://{host}{new_path}"
                return HttpResponsePermanentRedirect(new_url)
            except Resolver404:
                # No matching slash-appended view; allow normal 404/processing
                return self.get_response(request)
        except Exception:
            # Be defensive: do not break the request if something unexpected occurs
            return self.get_response(request)
