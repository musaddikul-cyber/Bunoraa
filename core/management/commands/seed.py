"""
Unified production seed command.
"""
import os

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed production configuration data (taxonomy, settings, reference tables)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--only",
            type=str,
            help="Comma-separated list of seed spec names to run",
        )
        parser.add_argument(
            "--exclude",
            type=str,
            help="Comma-separated list of seed spec names to exclude",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would change without writing to the database",
        )
        parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Disable pruning for this run",
        )
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning in production",
        )

    def handle(self, *args, **options):
        only = options.get("only")
        exclude = options.get("exclude")
        dry_run = options.get("dry_run", False)
        no_prune = options.get("no_prune", False)
        confirm_prune = options.get("confirm_prune", False)
        if os.environ.get("SEED_CONFIRM_PRUNE") in {"1", "true", "True", "yes", "YES"}:
            confirm_prune = True

        runner = SeedRunner(
            dry_run=dry_run,
            prune=not no_prune,
            confirm_prune=confirm_prune,
            logger=self.stdout.write,
        )

        result = runner.run(
            only=only.split(",") if only else None,
            exclude=exclude.split(",") if exclude else None,
            kind="prod",
        )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Seed completed."))
        self.stdout.write(f"  Created: {result.created}")
        self.stdout.write(f"  Updated: {result.updated}")
        self.stdout.write(f"  Pruned:  {result.pruned}")
        self.stdout.write(f"  Skipped: {result.skipped}")
        self.stdout.write(f"  Errors:  {result.errors}")
