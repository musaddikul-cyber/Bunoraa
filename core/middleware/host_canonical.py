from django.shortcuts import redirect
from django.conf import settings

class HostCanonicalMiddleware:
    """Redirect www host to apex domain (non-www) to enforce a single canonical host."""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        host = request.get_host().lower()
        if host.startswith('www.'):
            new_host = host[4:]
            scheme = 'https' if not settings.DEBUG else request.scheme
            new_url = f"{scheme}://{new_host}{request.get_full_path()}"
            return redirect(new_url, permanent=True)
        return self.get_response(request)
