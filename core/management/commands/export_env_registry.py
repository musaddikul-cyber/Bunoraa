from django.conf import settings
from django.core.management.base import BaseCommand

from apps.env_registry.models import EnvValue
from apps.env_registry import env_manager


class Command(BaseCommand):
    help = "Export env registry values to .env and Render YAML targets via schema"

    def add_arguments(self, parser):
        parser.add_argument("--schema", help="Path to env schema YAML")
        parser.add_argument("--env", help="Environment name (development|production)")
        parser.add_argument("--targets", help="Comma-separated target names")
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
        parser.add_argument(
            "--include-secrets",
            action="store_true",
            help="Include decrypted secrets when exporting dotenv targets",
        )

    def handle(self, *args, **options):
        schema_path = options.get("schema") or getattr(settings, "ENV_REGISTRY_SCHEMA_PATH", None)
        env = (options.get("env") or settings.ENVIRONMENT).lower()
        targets_raw = options.get("targets") or ""
        targets = [t.strip() for t in targets_raw.split(",") if t.strip()]

        overrides = {}
        values = (
            EnvValue.objects.select_related("variable")
            .filter(environment=env, variable__is_active=True)
        )
        for env_value in values:
            variable = env_value.variable
            if variable.is_secret and not options.get("include_secrets"):
                continue
            value = env_value.get_value()
            if value is not None:
                overrides[variable.key] = value

        env_manager.sync_from_schema(
            schema_path=schema_path,
            env=env,
            targets=targets,
            overrides=overrides,
            prune=False,
            dry_run=options.get("dry_run"),
        )

        self.stdout.write(self.style.SUCCESS("Export complete."))
