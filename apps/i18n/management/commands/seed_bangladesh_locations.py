from __future__ import annotations

from django.core.management.base import BaseCommand

from core.management.seed_utils import resolve_confirm_prune, run_seed_command


class Command(BaseCommand):
    help = "Seed Bangladesh location data (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Deprecated alias for --confirm-prune.",
        )
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning in production.",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning for this run.")

    def handle(self, *args, **options):
        run_seed_command(
            self,
            kind="prod",
            success_label="Seeded Bangladesh locations.",
            only=[
                "i18n.languages",
                "i18n.currencies",
                "i18n.timezones",
                "i18n.countries",
                "i18n.divisions",
                "i18n.districts",
            ],
            dry_run=bool(options.get("dry_run", False)),
            no_prune=bool(options.get("no_prune", False)),
            confirm_prune=resolve_confirm_prune(options, "clear"),
        )
