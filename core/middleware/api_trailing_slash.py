class ApiTrailingSlashMiddleware:
    """Normalize API paths to include a trailing slash without redirecting.

    This avoids 301 loops from APPEND_SLASH for API clients that omit the slash.
    """

    API_PREFIXES = ("/api/", "/oauth/")

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            path = request.path_info or ""
            if (
                path
                and not path.endswith("/")
                and path.startswith(self.API_PREFIXES)
                and "." not in path.split("/")[-1]
            ):
                new_path = f"{path}/"
                request.path_info = new_path
                request.path = new_path
                request.META["PATH_INFO"] = new_path
        except Exception:
            # Fail-safe: never break the request pipeline.
            pass
        return self.get_response(request)
