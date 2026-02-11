from __future__ import annotations

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed Bangladesh location data (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Allow pruning in production (confirm destructive prune).",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning for this run.")

    def handle(self, *args, **options):
        runner = SeedRunner(
            dry_run=options.get("dry_run", False),
            prune=not options.get("no_prune", False),
            confirm_prune=options.get("clear", False),
            logger=self.stdout.write,
        )
        result = runner.run(
            only=[
                "i18n.languages",
                "i18n.currencies",
                "i18n.timezones",
                "i18n.countries",
                "i18n.divisions",
                "i18n.districts",
            ],
            kind="prod",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded Bangladesh locations. Created: {result.created}, Updated: {result.updated}, Pruned: {result.pruned}, Errors: {result.errors}"
            )
        )
