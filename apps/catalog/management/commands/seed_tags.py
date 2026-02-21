from __future__ import annotations

from django.core.management.base import BaseCommand

from core.management.seed_utils import resolve_confirm_prune, run_seed_command


class Command(BaseCommand):
    help = "Seed product tags (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument("--file", "-f", help="Path to JSON file with tag items")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Deprecated alias for --confirm-prune.",
        )
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning in production.",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning of missing tags.")

    def handle(self, *args, **options):
        file_path = options.get("file")
        run_seed_command(
            self,
            kind="prod",
            success_label="Seeded tags.",
            only=["catalog.tags"],
            dry_run=bool(options.get("dry_run", False)),
            no_prune=bool(options.get("no_prune", False)),
            confirm_prune=resolve_confirm_prune(options, "overwrite"),
            env_overrides={"SEED_TAGS_PATH": file_path} if file_path else None,
        )
