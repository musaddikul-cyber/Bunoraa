import logging
import os

from django.apps import AppConfig


logger = logging.getLogger("bunoraa.env_registry")
_autosync_done = False


class EnvRegistryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.env_registry"
    verbose_name = "Env Registry"

    def ready(self):
        try:
            from . import signals  # noqa: F401
            from django.conf import settings
            from django.core.signals import request_started
            from django.db import connection
            from django.db.models.signals import post_migrate

            def auto_sync(*args, **kwargs):
                global _autosync_done
                if _autosync_done:
                    return
                if not getattr(settings, "ENV_REGISTRY_AUTOSEED", True):
                    return
                try:
                    table_names = connection.introspection.table_names()
                except Exception:
                    return
                if "env_registry_envvariable" not in table_names:
                    return
                if "env_registry_historicalenvvariable" not in table_names:
                    return

                from .schema_loader import sync_from_schema

                schema_path = getattr(settings, "ENV_REGISTRY_SCHEMA_PATH", "")
                sync_from_schema(schema_path or None, env=settings.ENVIRONMENT, force=False, prune=False)

                if getattr(settings, "ENV_REGISTRY_AUTOSYNC_RUNTIME", True):
                    from .runtime import apply_runtime_overrides

                    apply_runtime_overrides(settings.ENVIRONMENT)

                _autosync_done = True

            post_migrate.connect(auto_sync, sender=self, dispatch_uid="env_registry_auto_sync")
            request_started.connect(auto_sync, dispatch_uid="env_registry_auto_sync_request")
        except Exception as exc:
            if os.environ.get("ENV_REGISTRY_DEBUG"):
                logger.exception("Env registry auto-seed failed: %s", exc)
