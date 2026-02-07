"""
ASGI config for Bunoraa project.
Supports HTTP, WebSocket, and background tasks.
"""
import os
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
    elif environment in ('development', 's3'):
        settings_module = 'core.settings.s3'
    else:
        settings_module = 'core.settings.local'
    os.environ['DJANGO_SETTINGS_MODULE'] = settings_module

from django.core.asgi import get_asgi_application

# Initialize Django ASGI application early to ensure apps are loaded
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator
from core.routing import websocket_urlpatterns


application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        )
    ),
})
