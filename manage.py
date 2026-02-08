#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


def main():
    """Run administrative tasks."""
    # Prefer explicit DJANGO_SETTINGS_MODULE; otherwise infer from ENVIRONMENT.
    settings_module = os.environ.get("DJANGO_SETTINGS_MODULE", "").strip()
    if not settings_module or settings_module == "core.settings":
        environment = os.environ.get("ENVIRONMENT", "").lower()
        if environment == "production":
            settings_module = "core.settings.production"
        elif environment in ("development", "s3"):
            settings_module = "core.settings.s3"
        else:
            settings_module = "core.settings.local"
        os.environ["DJANGO_SETTINGS_MODULE"] = settings_module
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
