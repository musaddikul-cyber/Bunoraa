from django.http import HttpResponse


class ApiPreflightMiddleware:
    """Short-circuit API OPTIONS requests to avoid slow session/DB work."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS" and request.path.startswith("/api/"):
            return HttpResponse(status=200)
        return self.get_response(request)
