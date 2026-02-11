from __future__ import annotations

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed currencies and exchange rates (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument("codes", nargs="*", help="(deprecated) currency codes")
        parser.add_argument("--base", type=str, default="BDT", help="(deprecated) base currency code")
        parser.add_argument("--fetch", action="store_true", help="(deprecated) fetch live rates")
        parser.add_argument("--provider", type=str, default="", help="(deprecated) rate provider")
        parser.add_argument("--update", action="store_true", help="(deprecated) update existing rates")
        parser.add_argument("--list", action="store_true", help="(deprecated) list available currencies")
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning for this run.")
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning (if enabled) in production",
        )

    def handle(self, *args, **options):
        if options.get("codes") or options.get("fetch") or options.get("update") or options.get("list"):
            self.stdout.write(
                self.style.WARNING(
                    "Legacy seed_currencies flags are ignored; using unified seed system."
                )
            )

        runner = SeedRunner(
            dry_run=options.get("dry_run", False),
            prune=not options.get("no_prune", False),
            confirm_prune=options.get("confirm_prune", False),
            logger=self.stdout.write,
        )
        result = runner.run(
            only=["i18n.currencies", "i18n.exchange_rates"],
            kind="prod",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded currencies. Created: {result.created}, Updated: {result.updated}, Pruned: {result.pruned}, Errors: {result.errors}"
            )
        )
