from django.http import HttpResponse

class HealthCheckMiddleware:
    """Short-circuit middleware for lightweight health checks.

    Returns 200 OK for `/healthz` or `/status` without touching sessions, auth, or DB.
    Add this as the very first middleware to avoid expensive setup on pings.
    """
    HEALTH_PATHS = ['/healthz', '/status', '/_health']

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path in self.HEALTH_PATHS:
            return HttpResponse('OK', content_type='text/plain')
        return self.get_response(request)
