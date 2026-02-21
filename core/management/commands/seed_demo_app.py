from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from core.management.seed_utils import run_seed_command


TARGET_SPECS: dict[str, list[str]] = {
    "artisans": ["artisans.demo"],
    "catalog": ["catalog.demo"],
    "chat": ["chat.demo"],
    "contacts": ["contacts.demo"],
    "notifications": ["notifications.demo"],
    "referral": ["referral.demo"],
}


class Command(BaseCommand):
    help = "Seed demo data for one or more apps (consolidates legacy seed_*_data commands)."

    def add_arguments(self, parser):
        parser.add_argument(
            "targets",
            nargs="*",
            choices=sorted(TARGET_SPECS.keys()),
            help="One or more demo targets to seed",
        )
        parser.add_argument(
            "--list-targets",
            action="store_true",
            help="List available targets and exit",
        )
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--seed", type=int, help="Random seed for deterministic demo data")
        parser.add_argument("--no-prune", action="store_true", help="Disable pruning for this run.")
        parser.add_argument(
            "--confirm-prune",
            action="store_true",
            help="Confirm pruning (if enabled) in production",
        )

    def handle(self, *args, **options):
        if options.get("list_targets"):
            self.stdout.write("Available targets:")
            for target in sorted(TARGET_SPECS.keys()):
                self.stdout.write(f"  - {target}")
            return

        targets = options.get("targets") or []
        if not targets:
            raise CommandError("Provide at least one target or use --list-targets.")

        only: list[str] = []
        for target in targets:
            only.extend(TARGET_SPECS[target])

        run_seed_command(
            self,
            kind="demo",
            success_label=f"Demo seed completed for targets: {', '.join(targets)}.",
            only=only,
            dry_run=bool(options.get("dry_run", False)),
            no_prune=bool(options.get("no_prune", False)),
            confirm_prune=bool(options.get("confirm_prune", False)),
            seed=options.get("seed"),
        )
