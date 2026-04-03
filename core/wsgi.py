"""
WSGI config for Bunoraa project.
"""
import os
import time
from pathlib import Path

# Load .env file before Django initializes
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

from django.core.wsgi import get_wsgi_application

_start_ts = time.time()

# Choose settings module based on environment
environment = os.environ.get('ENVIRONMENT', 'development').lower()
if environment == 'production':
    settings_module = 'core.settings.production'
elif environment == 's3':
    settings_module = 'core.settings.s3'
else:
    # Local/development defaults to local settings.
    settings_module = 'core.settings.local'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)

application = get_wsgi_application()

# Print effective media/storage configuration for startup diagnostics.
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

# Log startup time for observability (useful to detect cold-starts)
try:
    _startup_time = time.time() - _start_ts
    print(f"WSGI application ready in {_startup_time:.2f}s (using {settings_module})")
except Exception:
    pass
