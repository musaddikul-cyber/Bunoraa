from __future__ import annotations

import os

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed product tags (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument("--file", "-f", help="Path to JSON file with tag items")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Allow pruning in production (confirm destructive prune).",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning of missing tags.")

    def handle(self, *args, **options):
        file_path = options.get("file")
        if file_path:
            os.environ["SEED_TAGS_PATH"] = file_path

        runner = SeedRunner(
            dry_run=options.get("dry_run", False),
            prune=not options.get("no_prune", False),
            confirm_prune=options.get("overwrite", False),
            logger=self.stdout.write,
        )
        result = runner.run(only=["catalog.tags"], kind="prod")
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded tags. Created: {result.created}, Updated: {result.updated}, Pruned: {result.pruned}, Errors: {result.errors}"
            )
        )
