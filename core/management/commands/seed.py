"""
Unified production seed command.
"""
import os

from django.core.management.base import BaseCommand

from core.management.seed_utils import (
    resolve_confirm_prune,
    run_seed_command,
    split_csv_option,
)


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
        only = split_csv_option(options.get("only"))
        exclude = split_csv_option(options.get("exclude"))
        dry_run = bool(options.get("dry_run", False))
        no_prune = bool(options.get("no_prune", False))
        confirm_prune = resolve_confirm_prune(options)
        if os.environ.get("SEED_CONFIRM_PRUNE") in {"1", "true", "True", "yes", "YES"}:
            confirm_prune = True

        run_seed_command(
            self,
            kind="prod",
            success_label="Seed completed.",
            only=only,
            exclude=exclude,
            dry_run=dry_run,
            no_prune=no_prune,
            confirm_prune=confirm_prune,
        )
