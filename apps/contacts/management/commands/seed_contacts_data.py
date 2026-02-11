from __future__ import annotations

from django.core.management.base import BaseCommand
from django.core.management import call_command


class Command(BaseCommand):
    help = "Seed demo contacts data (wrapper around seed_demo)."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
        parser.add_argument("--seed", type=int, help="Random seed for deterministic demo data")

    def handle(self, *args, **options):
        args = ["--only=contacts.demo"]
        if options.get("dry_run"):
            args.append("--dry-run")
        if options.get("seed") is not None:
            args.append(f"--seed={options['seed']}")
        call_command("seed_demo", *args)
