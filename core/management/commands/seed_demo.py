"""
Unified demo/sample seed command.
"""
import random

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed demo/sample data only (non-production)."

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
            help="Confirm pruning (if enabled) in production",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed for deterministic demo data",
        )

    def handle(self, *args, **options):
        only = options.get("only")
        exclude = options.get("exclude")
        dry_run = options.get("dry_run", False)
        no_prune = options.get("no_prune", False)
        confirm_prune = options.get("confirm_prune", False)
        seed = options.get("seed")

        if seed is not None:
            random.seed(seed)

        runner = SeedRunner(
            dry_run=dry_run,
            prune=not no_prune,
            confirm_prune=confirm_prune,
            logger=self.stdout.write,
        )

        result = runner.run(
            only=only.split(",") if only else None,
            exclude=exclude.split(",") if exclude else None,
            kind="demo",
        )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Demo seed completed."))
        self.stdout.write(f"  Created: {result.created}")
        self.stdout.write(f"  Updated: {result.updated}")
        self.stdout.write(f"  Pruned:  {result.pruned}")
        self.stdout.write(f"  Skipped: {result.skipped}")
        self.stdout.write(f"  Errors:  {result.errors}")
