from __future__ import annotations

from django.core.management.base import BaseCommand

from core.management.seed_utils import resolve_confirm_prune, run_seed_command


class Command(BaseCommand):
    help = "Seed category taxonomy (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Deprecated alias for --confirm-prune.",
        )
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning in production.",
        )
        parser.add_argument(
            "--assign-facets",
            action="store_true",
            help="Also seed facets and category facet assignments.",
        )
        parser.add_argument(
            "--file",
            help="Path to taxonomy JSON file (overrides default taxonomy).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would change without writing to the database.",
        )
        parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Disable pruning of categories not present in seed data.",
        )

    def handle(self, *args, **options):
        taxonomy_file = options.get("file")
        env_overrides = {"SEED_TAXONOMY_PATH": taxonomy_file} if taxonomy_file else None

        only = ["catalog.categories"]
        if options.get("assign_facets"):
            only.extend(["catalog.facets", "catalog.category_facets"])

        run_seed_command(
            self,
            kind="prod",
            success_label="Seeded categories.",
            only=only,
            dry_run=bool(options.get("dry_run", False)),
            no_prune=bool(options.get("no_prune", False)),
            confirm_prune=resolve_confirm_prune(options, "force"),
            env_overrides=env_overrides,
        )
