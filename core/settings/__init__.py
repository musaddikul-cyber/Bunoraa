"""
Settings module for Bunoraa.

Prefer explicit DJANGO_SETTINGS_MODULE (e.g., core.settings.local).
If this package module is loaded directly, fall back to an environment-based
selection to avoid missing required settings.
"""

import os

_environment = os.environ.get('ENVIRONMENT', '').lower()
_settings_module = os.environ.get('DJANGO_SETTINGS_MODULE', '').strip()

# If Django is pointed at the package (core.settings), resolve to a concrete module.
if _settings_module in ('core.settings', 'core.settings.__init__', ''):
    if _environment == 'production':
        from .production import *  # noqa: F401,F403
    elif _environment in ('development', 's3'):
        from .s3 import *  # noqa: F401,F403
    else:
        from .local import *  # noqa: F401,F403
