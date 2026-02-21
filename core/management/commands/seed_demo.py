"""
Unified demo/sample seed command.
"""
from django.core.management.base import BaseCommand

from core.management.seed_utils import run_seed_command, split_csv_option


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
        only = split_csv_option(options.get("only"))
        exclude = split_csv_option(options.get("exclude"))
        dry_run = bool(options.get("dry_run", False))
        no_prune = bool(options.get("no_prune", False))
        confirm_prune = bool(options.get("confirm_prune", False))
        seed = options.get("seed")

        run_seed_command(
            self,
            kind="demo",
            success_label="Demo seed completed.",
            only=only,
            exclude=exclude,
            dry_run=dry_run,
            no_prune=no_prune,
            confirm_prune=confirm_prune,
            seed=seed,
        )
