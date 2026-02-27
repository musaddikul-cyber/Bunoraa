"""
Database router for environment-driven read/write access control.
"""

from django.conf import settings
from django.db.utils import DatabaseError


class EnvDatabaseAccessRouter:
    """Guard ORM writes based on DB_ALLOW_WRITE."""

    def db_for_write(self, model, **hints):
        if not getattr(settings, 'DB_ALLOW_WRITE', True):
            raise DatabaseError("Database write access is disabled (DB_ALLOW_WRITE=False).")
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if not getattr(settings, 'DB_ALLOW_WRITE', True):
            return False
        return None
