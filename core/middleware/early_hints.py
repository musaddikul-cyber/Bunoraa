from django.utils.deprecation import MiddlewareMixin
from django.conf import settings

class EarlyHintsMiddleware(MiddlewareMixin):
    """Attach Link headers for preload to speed up resource fetch.

    Sends Link headers only when explicit preload assets are configured.
    """

    @staticmethod
    def _get_configured_asset(name: str) -> str:
        value = getattr(settings, name, "")
        if isinstance(value, str):
            return value.strip()
        return ""

    def process_response(self, request, response):
        try:
            if 'text/html' in response.get('Content-Type', '') and request.method == 'GET':
                links = []
                main_css = self._get_configured_asset('MAIN_CSS')
                if main_css:
                    links.append(f'<{main_css}>; rel=preload; as=style')
                # Preconnect to CDN/Asset host if configured
                asset_host = getattr(settings, 'ASSET_HOST', None)
                if asset_host:
                    links.append(f'<{asset_host}>; rel=preconnect; crossorigin')
                main_js = self._get_configured_asset('MAIN_JS')
                if main_js:
                    links.append(f'<{main_js}>; rel=preload; as=script')
                if not links:
                    return response
                existing = response.get('Link')
                link_header = ', '.join(links)
                if existing:
                    response['Link'] = existing + ', ' + link_header
                else:
                    response['Link'] = link_header
        except Exception:
            pass
        return response
