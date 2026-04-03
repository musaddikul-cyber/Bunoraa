"""
ASGI config for Bunoraa project.
Supports HTTP, WebSocket, and background tasks.
"""
import os
import signal
from pathlib import Path

# Load .env file before Django initializes
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

# Set default settings module BEFORE importing Django
# Prefer explicit DJANGO_SETTINGS_MODULE; otherwise infer from ENVIRONMENT.
settings_module = os.environ.get('DJANGO_SETTINGS_MODULE', '').strip()
if not settings_module or settings_module == 'core.settings':
    environment = os.environ.get('ENVIRONMENT', '').lower()
    if environment == 'production':
        settings_module = 'core.settings.production'
    elif environment == 's3':
        settings_module = 'core.settings.s3'
    else:
        settings_module = 'core.settings.local'
    os.environ['DJANGO_SETTINGS_MODULE'] = settings_module

from django.core.asgi import get_asgi_application

_SHUTTING_DOWN = False


def _mark_shutting_down(*_args):
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True


for _sig_name in ("SIGINT", "SIGTERM"):
    _sig = getattr(signal, _sig_name, None)
    if _sig is None:
        continue
    try:
        signal.signal(_sig, _mark_shutting_down)
    except (ValueError, RuntimeError):
        # Signal handling only works in the main thread; ignore otherwise.
        pass

# Initialize Django ASGI application early to ensure apps are loaded
django_asgi_app = get_asgi_application()

def _print_media_storage_settings() -> None:
    try:
        from django.conf import settings

        media_url = getattr(settings, "MEDIA_URL", None)
        storage = getattr(settings, "DEFAULT_FILE_STORAGE", None)
        if media_url or storage:
            print(f"[media-storage] MEDIA_URL={media_url} STORAGE={storage}")
    except Exception:
        pass


_print_media_storage_settings()


async def _shutdown_aware_http_app(scope, receive, send):
    if _SHUTTING_DOWN:
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"Service shutting down",
                "more_body": False,
            }
        )
        return
    return await django_asgi_app(scope, receive, send)


async def _shutdown_aware_ws_app(scope, receive, send):
    if _SHUTTING_DOWN:
        await send(
            {
                "type": "websocket.close",
                "code": 1013,  # Try again later
                "reason": "Service shutting down",
            }
        )
        return
    return await AllowedHostsOriginValidator(
        JWTOrSessionAuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        )
    )(scope, receive, send)

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from core.routing import websocket_urlpatterns
from core.websocket.auth import JWTOrSessionAuthMiddlewareStack


application = ProtocolTypeRouter({
    'http': _shutdown_aware_http_app,
    'websocket': _shutdown_aware_ws_app,
})
