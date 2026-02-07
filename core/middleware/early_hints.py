from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
from django.contrib.staticfiles.storage import staticfiles_storage

class EarlyHintsMiddleware(MiddlewareMixin):
    """Attach Link headers for preload to speed up resource fetch.

    Sends Link headers for the main CSS and main JS files. Gives browsers a head-start.
    """
    def process_response(self, request, response):
        try:
            if 'text/html' in response.get('Content-Type', '') and request.method == 'GET':
                links = []
                # Resolve hashed static file URLs when possible (fall back to sensible defaults)
                try:
                    default_css = staticfiles_storage.url('css/styles.css')
                except Exception:
                    default_css = '/static/css/styles.css'
                main_css = getattr(settings, 'MAIN_CSS', default_css)
                links.append(f'<{main_css}>; rel=preload; as=style')
                # Preconnect to CDN/Asset host if configured
                asset_host = getattr(settings, 'ASSET_HOST', None)
                if asset_host:
                    links.append(f'<{asset_host}>; rel=preconnect; crossorigin')
                # Resolve JS bundle URL
                try:
                    default_js = staticfiles_storage.url('js/app.bundle.js')
                except Exception:
                    default_js = '/static/js/app.bundle.js'
                main_js = getattr(settings, 'MAIN_JS', default_js)
                links.append(f'<{main_js}>; rel=preload; as=script')
                existing = response.get('Link')
                link_header = ', '.join(links)
                if existing:
                    response['Link'] = existing + ', ' + link_header
                else:
                    response['Link'] = link_header
        except Exception:
            pass
        return response