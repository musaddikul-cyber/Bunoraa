from __future__ import annotations

from django.core.management.base import BaseCommand

from core.management.seed_utils import resolve_confirm_prune, run_seed_command


SECTION_TO_SPEC = {
    "categories": "preorders.categories",
    "options": "preorders.options",
    "option_choices": "preorders.option_choices",
    "templates": "preorders.templates",
}


class Command(BaseCommand):
    help = "Sync preorder taxonomy seed data into DB (idempotent + optional prune)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file",
            help="Path to preorder taxonomy JSON file (defaults to apps/preorders/data/taxonomy.json).",
        )
        parser.add_argument(
            "--only",
            help="Comma-separated sections: categories,options,option_choices,templates",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would change without writing to the database.",
        )
        parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Disable pruning/deactivation for missing records.",
        )
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

    def handle(self, *args, **options):
        taxonomy_file = options.get("file")
        section_list = self._parse_sections(options.get("only"))
        selected_specs = [SECTION_TO_SPEC[section] for section in section_list]

        run_seed_command(
            self,
            kind="prod",
            success_label="Preorder taxonomy synced.",
            only=selected_specs,
            dry_run=bool(options.get("dry_run", False)),
            no_prune=bool(options.get("no_prune", False)),
            confirm_prune=resolve_confirm_prune(options, "force"),
            env_overrides={"SEED_PREORDER_TAXONOMY_PATH": taxonomy_file} if taxonomy_file else None,
        )

    def _parse_sections(self, raw: str | None) -> list[str]:
        if not raw:
            return list(SECTION_TO_SPEC.keys())

        requested = [part.strip() for part in raw.split(",") if part.strip()]
        invalid = [part for part in requested if part not in SECTION_TO_SPEC]
        if invalid:
            raise CommandError(
                f"Invalid --only section(s): {', '.join(invalid)}. "
                f"Allowed: {', '.join(SECTION_TO_SPEC.keys())}."
            )
        return requested
