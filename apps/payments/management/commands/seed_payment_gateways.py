from __future__ import annotations

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed default payment gateways (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Allow pruning in production (confirm destructive prune).",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning of missing gateways.")

    def handle(self, *args, **options):
        runner = SeedRunner(
            dry_run=options.get("dry_run", False),
            prune=not options.get("no_prune", False),
            confirm_prune=options.get("clear", False),
            logger=self.stdout.write,
        )
        result = runner.run(
            only=["payments.payment_gateways", "payments.bnpl_providers"],
            kind="prod",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded payment gateways. Created: {result.created}, Updated: {result.updated}, Pruned: {result.pruned}, Errors: {result.errors}"
            )
        )
