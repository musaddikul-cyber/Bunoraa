from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from core.management.seed_utils import run_seed_command


SECTION_TO_SPECS: dict[str, list[str]] = {
    "languages": ["i18n.languages"],
    "currencies": ["i18n.currencies", "i18n.exchange_rates"],
    "timezones": ["i18n.timezones"],
    "countries": ["i18n.countries"],
    "divisions": ["i18n.divisions"],
    "districts": ["i18n.districts"],
    "settings": ["i18n.settings"],
}


class Command(BaseCommand):
    help = (
        "Load i18n data through the unified seed system "
        "(replaces legacy direct model bootstrapping)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true", help="Load all i18n seed sections.")
        parser.add_argument("--languages", action="store_true", help="Load languages only.")
        parser.add_argument("--currencies", action="store_true", help="Load currencies and exchange rates.")
        parser.add_argument("--timezones", action="store_true", help="Load timezones only.")
        parser.add_argument("--countries", action="store_true", help="Load countries only.")
        parser.add_argument("--divisions", action="store_true", help="Load divisions only.")
        parser.add_argument("--districts", action="store_true", help="Load districts only.")
        parser.add_argument(
            "--i18n-settings",
            action="store_true",
            dest="i18n_settings",
            help="Load i18n settings only.",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument(
            "--prune",
            action="store_true",
            help="Enable pruning/deactivation for records not in seed data.",
        )
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning in production.",
        )

    def handle(self, *args, **options):
        requested_sections = [
            section
            for section in SECTION_TO_SPECS.keys()
            if options.get(section if section != "settings" else "i18n_settings")
        ]

        if options.get("all"):
            sections = list(SECTION_TO_SPECS.keys())
        elif requested_sections:
            sections = requested_sections
        else:
            # Backward-compatible default behavior from legacy command.
            sections = ["languages", "currencies", "timezones", "countries"]

        if not sections:
            raise CommandError("No i18n sections selected.")

        only: list[str] = []
        seen: set[str] = set()
        for section in sections:
            for spec_name in SECTION_TO_SPECS[section]:
                if spec_name in seen:
                    continue
                only.append(spec_name)
                seen.add(spec_name)

        run_seed_command(
            self,
            kind="prod",
            success_label=f"Loaded i18n sections: {', '.join(sections)}.",
            only=only,
            dry_run=bool(options.get("dry_run", False)),
            no_prune=not bool(options.get("prune", False)),
            confirm_prune=bool(options.get("confirm_prune", False)),
        )
