from __future__ import annotations

import os

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed category taxonomy (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Allow pruning in production (confirm destructive prune).",
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
        if taxonomy_file:
            os.environ["SEED_TAXONOMY_PATH"] = taxonomy_file

        only = ["catalog.categories"]
        if options.get("assign_facets"):
            only.extend(["catalog.facets", "catalog.category_facets"])

        runner = SeedRunner(
            dry_run=options.get("dry_run", False),
            prune=not options.get("no_prune", False),
            confirm_prune=options.get("force", False),
            logger=self.stdout.write,
        )
        result = runner.run(only=only, kind="prod")
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded categories. Created: {result.created}, Updated: {result.updated}, Pruned: {result.pruned}, Errors: {result.errors}"
            )
        )
