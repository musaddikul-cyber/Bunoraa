from __future__ import annotations

import os

from django.core.management.base import BaseCommand

from core.seed.runner import SeedRunner


class Command(BaseCommand):
    help = "Seed email service configuration (wrapper around unified seed system)."

    def add_arguments(self, parser):
        parser.add_argument("--user", type=str, help="Email or ID of user to associate with seeded records")
        parser.add_argument("--skip-api-key", action="store_true", help="(deprecated) API keys are not seeded")
        parser.add_argument("--skip-domain", action="store_true", help="Skip sender domain and identity seeding")
        parser.add_argument("--skip-templates", action="store_true", help="Skip email template seeding")
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning for this run.")
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning (if enabled) in production",
        )

    def handle(self, *args, **options):
        user = options.get("user")
        if user:
            os.environ["SEED_DEFAULT_USER_EMAIL"] = user

        if options.get("skip_api_key"):
            self.stdout.write(self.style.WARNING("API key seeding is disabled in the unified seed system."))

        only = [
            "email_service.sender_domains",
            "email_service.sender_identities",
            "email_service.email_templates",
            "email_service.unsubscribe_groups",
        ]

        if options.get("skip_domain"):
            only = [name for name in only if not name.startswith("email_service.sender_")]
        if options.get("skip_templates"):
            only = [name for name in only if name != "email_service.email_templates"]

        runner = SeedRunner(
            dry_run=options.get("dry_run", False),
            prune=not options.get("no_prune", False),
            confirm_prune=options.get("confirm_prune", False),
            logger=self.stdout.write,
        )
        result = runner.run(only=only, kind="prod")
        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded email service. Created: {result.created}, Updated: {result.updated}, Pruned: {result.pruned}, Errors: {result.errors}"
            )
        )
