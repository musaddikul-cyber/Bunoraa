import logging
import os
import threading
import time

from django.apps import AppConfig


logger = logging.getLogger("bunoraa.env_registry")
_autosync_done = False
_autosync_lock = threading.Lock()
_last_autosync_attempt = 0.0
_AUTOSYNC_RETRY_COOLDOWN_SECONDS = 15.0


class EnvRegistryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.env_registry"
    verbose_name = "Env Registry"

    def ready(self):
        try:
            from . import signals  # noqa: F401
            from django.conf import settings
            from django.core.signals import request_started
            from django.db import OperationalError, connection
            from django.db.models.signals import post_migrate

            def auto_sync(*args, **kwargs):
                global _autosync_done, _last_autosync_attempt
                if _autosync_done:
                    return
                if not getattr(settings, "ENV_REGISTRY_AUTOSEED", True):
                    return
                if not _autosync_lock.acquire(blocking=False):
                    return
                try:
                    now = time.monotonic()
                    if (
                        _last_autosync_attempt
                        and now - _last_autosync_attempt < _AUTOSYNC_RETRY_COOLDOWN_SECONDS
                    ):
                        return
                    _last_autosync_attempt = now

                    try:
                        table_names = connection.introspection.table_names()
                    except Exception:
                        return

                    if "env_registry_envvariable" not in table_names:
                        return
                    if "env_registry_historicalenvvariable" not in table_names:
                        return

                    from .services import sync_registry_from_schema

                    schema_path = getattr(settings, "ENV_REGISTRY_SCHEMA_PATH", "")
                    sync_registry_from_schema(
                        schema_path or None,
                        env=settings.ENVIRONMENT,
                        force=False,
                        prune=False,
                    )

                    if getattr(settings, "ENV_REGISTRY_AUTOSYNC_RUNTIME", True):
                        from .services import apply_runtime_overrides

                        apply_runtime_overrides(settings.ENVIRONMENT)

                    _autosync_done = True
                except OperationalError as exc:
                    if "database is locked" in str(exc).lower():
                        logger.warning("Env registry auto-sync skipped because SQLite database is locked.")
                        return
                    raise
                except Exception as exc:
                    if os.environ.get("ENV_REGISTRY_DEBUG"):
                        logger.exception("Env registry auto-seed failed: %s", exc)
                    else:
                        logger.warning("Env registry auto-seed skipped: %s", exc)
                finally:
                    _autosync_lock.release()

            post_migrate.connect(auto_sync, sender=self, dispatch_uid="env_registry_auto_sync")
            sync_on_request = bool(getattr(settings, "ENV_REGISTRY_SYNC_ON_REQUEST", True))
            db_engine = str(settings.DATABASES.get("default", {}).get("ENGINE", ""))
            if sync_on_request and db_engine.endswith("sqlite3"):
                logger.info("Disabled env registry request-start auto-sync for SQLite to avoid lock contention.")
                sync_on_request = False
            if sync_on_request:
                request_started.connect(auto_sync, dispatch_uid="env_registry_auto_sync_request")
        except Exception as exc:
            if os.environ.get("ENV_REGISTRY_DEBUG"):
                logger.exception("Env registry auto-seed failed: %s", exc)
