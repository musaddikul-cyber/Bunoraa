from __future__ import annotations

from django.core.management.base import BaseCommand

from core.management.seed_utils import run_seed_command


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
            dest="enable_free_shipping",
            default=True,
            help="Enable free shipping above threshold",
        )
        parser.add_argument(
            "--disable-free-shipping",
            action="store_false",
            dest="enable_free_shipping",
            help="Disable free shipping",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning for this run.")
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning (if enabled) in production",
        )

    def handle(self, *args, **options):
        run_seed_command(
            self,
            kind="prod",
            success_label="Seeded shipping settings.",
            only=["shipping.settings"],
            dry_run=bool(options.get("dry_run", False)),
            no_prune=bool(options.get("no_prune", False)),
            confirm_prune=bool(options.get("confirm_prune", False)),
            env_overrides={
                "SEED_SHIPPING_FREE_THRESHOLD": str(options.get("threshold")),
                "SEED_SHIPPING_HANDLING_DAYS": str(options.get("handling_days")),
                "SEED_SHIPPING_ENABLE_FREE_SHIPPING": "1"
                if options.get("enable_free_shipping")
                else "0",
            },
        )
