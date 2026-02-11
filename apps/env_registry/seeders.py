from __future__ import annotations

from django.conf import settings

from core.seed.base import SeedContext, SeedResult, SeedSpec
from core.seed.registry import register_seed


class EnvRegistrySeedSpec(SeedSpec):
    name = "env_registry.sync"
    app_label = "env_registry"
    kind = "prod"
    description = "Sync env registry from schema"

    def apply(self, ctx: SeedContext) -> SeedResult:
        if not getattr(settings, "ENV_REGISTRY_AUTOSEED", True):
            return SeedResult(skipped=1)

        try:
            from apps.env_registry.schema_loader import sync_from_schema
        except Exception:
            return SeedResult(errors=1)

        schema_path = getattr(settings, "ENV_REGISTRY_SCHEMA_PATH", "")
        env_name = getattr(settings, "ENVIRONMENT", None)
        if ctx.dry_run:
            return SeedResult(skipped=1)

        stats = sync_from_schema(schema_path or None, env=env_name, force=False, prune=False)
        result = SeedResult()
        result.updated = stats.get("variables_changed", 0)
        return result


register_seed(EnvRegistrySeedSpec())
