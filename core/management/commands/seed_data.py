"""
Production-safe seeding wrapper.
Usage: python manage.py seed_data
"""

from django.core.management.base import BaseCommand
from django.core.management import call_command


class Command(BaseCommand):
    help = "Seeds production-safe configuration data (taxonomy, settings, reference tables)."

    def add_arguments(self, parser):
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
        self.stdout.write("Starting production seed...\n")
        args = []
        if options.get("dry_run"):
            args.append("--dry-run")
        if options.get("no_prune"):
            args.append("--no-prune")
        if options.get("confirm_prune"):
            args.append("--confirm-prune")

        call_command("seed", *args)
        self.stdout.write(self.style.SUCCESS("\nProduction seeding completed successfully!"))



