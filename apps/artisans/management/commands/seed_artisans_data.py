from __future__ import annotations

from django.core.management.base import BaseCommand
from django.core.management import call_command


class Command(BaseCommand):
    help = "Seed demo artisans data (wrapper around seed_demo)."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")

    def handle(self, *args, **options):
        args = ["--only=artisans.demo"]
        if options.get("dry_run"):
            args.append("--dry-run")
        call_command("seed_demo", *args)
