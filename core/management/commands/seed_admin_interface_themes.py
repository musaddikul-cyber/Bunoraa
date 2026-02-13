import json
from pathlib import Path

import admin_interface
from admin_interface.models import Theme
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Seed admin_interface theme presets without overwriting existing themes."

    def handle(self, *args, **options):
        fixtures_dir = Path(admin_interface.__file__).resolve().parent / "fixtures"
        if not fixtures_dir.exists():
            self.stdout.write(self.style.WARNING("admin_interface fixtures directory not found."))
            return

        fixture_files = sorted(fixtures_dir.glob("admin_interface_theme_*.json"))
        if not fixture_files:
            self.stdout.write(self.style.WARNING("No admin_interface theme fixtures found."))
            return

        active_exists = Theme.objects.filter(active=True).exists()
        active_set = active_exists
        created = 0
        skipped = 0

        for fixture_path in fixture_files:
            try:
                payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            except Exception:
                self.stdout.write(self.style.WARNING(f"Failed to read {fixture_path.name}."))
                continue

            for entry in payload:
                if entry.get("model") != "admin_interface.theme":
                    continue
                fields = dict(entry.get("fields", {}))
                name = fields.get("name")
                if not name:
                    continue
                if Theme.objects.filter(name=name).exists():
                    skipped += 1
                    continue

                # Preserve any existing active theme; only set active if none exists.
                if active_set:
                    fields["active"] = False
                else:
                    fields["active"] = True
                    active_set = True

                Theme.objects.create(**fields)
                created += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Admin interface themes seeded. Created: {created}, Skipped: {skipped}."
            )
        )
