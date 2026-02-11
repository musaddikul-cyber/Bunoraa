from __future__ import annotations

import os

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed ShippingSettings singleton (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--threshold",
            type=float,
            default=3000,
            help="Free shipping threshold amount (default: 3000)",
        )
        parser.add_argument(
            "--handling-days",
            type=int,
            default=1,
            help="Order handling days (default: 1)",
        )
        parser.add_argument(
            "--enable-free-shipping",
            action="store_true",
            default=True,
            help="Enable free shipping above threshold",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning for this run.")
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning (if enabled) in production",
        )

    def handle(self, *args, **options):
        os.environ["SEED_SHIPPING_FREE_THRESHOLD"] = str(options.get("threshold"))
        os.environ["SEED_SHIPPING_HANDLING_DAYS"] = str(options.get("handling_days"))
        os.environ["SEED_SHIPPING_ENABLE_FREE_SHIPPING"] = "1" if options.get("enable_free_shipping") else "0"

        runner = SeedRunner(
            dry_run=options.get("dry_run", False),
            prune=not options.get("no_prune", False),
            confirm_prune=options.get("confirm_prune", False),
            logger=self.stdout.write,
        )
        result = runner.run(only=["shipping.settings"], kind="prod")
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded shipping settings. Created: {result.created}, Updated: {result.updated}, Pruned: {result.pruned}, Errors: {result.errors}"
            )
        )
