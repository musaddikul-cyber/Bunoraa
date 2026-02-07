"""Middleware to set SEO-related response headers (X-Robots-Tag) for certain paths."""
from django.utils.deprecation import MiddlewareMixin

class SEOHeadersMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        path = request.path or ''
        noindex_paths = ['/search', '/account', '/cart', '/checkout', '/wishlist', '/preorders/wizard']
        for p in noindex_paths:
            if path.startswith(p):
                # Prevent indexing of these pages
                response.setdefault('X-Robots-Tag', 'noindex, follow')
                break
        return response
