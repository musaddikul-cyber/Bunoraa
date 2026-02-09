from django.conf import settings
from django.core.management.base import BaseCommand

from apps.env_registry.schema_loader import sync_from_schema


class Command(BaseCommand):
    help = "Sync env registry entries from config/env.schema.yml"

    def add_arguments(self, parser):
        parser.add_argument("--schema", help="Path to env schema YAML")
        parser.add_argument("--env", help="Environment name (development|production)")
        parser.add_argument("--force", action="store_true", help="Overwrite existing values from schema")
        parser.add_argument("--prune", action="store_true", help="Mark variables missing from schema as inactive")

    def handle(self, *args, **options):
        schema_path = options.get("schema")
        env = (options.get("env") or settings.ENVIRONMENT).lower()
        result = sync_from_schema(schema_path, env=env, force=options.get("force"), prune=options.get("prune"))
        self.stdout.write(
            self.style.SUCCESS(
                f"Env registry synced. Variables processed: {result['variables_processed']}, changed: {result['variables_changed']}"
            )
        )
