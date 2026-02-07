"""Management command to seed Tag objects.

Usage:
  python manage.py seed_tags --file=apps/catalog/data/tags.json
  python manage.py seed_tags           # uses built-in defaults
  python manage.py seed_tags --overwrite   # delete existing tags first
  python manage.py seed_tags --dry-run     # show what would be created
"""
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
import json
import os

DEFAULT_TAGS = [
    "New",
    "Popular",
    "Sale",
    "Limited",
    "Handmade",
    "Organic",
    "Trending",
    "Gift",
    "Eco-friendly",
    "Clearance",
    "Bestseller",
    "Exclusive",
    "Featured",
    "Artisan",
    "Traditional",
    "Modern",
    "Vintage",
    "Premium",
    "Budget-friendly",
    "Sustainable",
]


class Command(BaseCommand):
    help = 'Seed product tags from a JSON array or newline-separated text file. If no file is provided inserts a default list.'

    def add_arguments(self, parser):
        parser.add_argument('--file', '-f', help='Path to a JSON array or newline-separated file with tag names')
        parser.add_argument('--overwrite', action='store_true', help='Delete all existing tags before seeding')
        parser.add_argument('--dry-run', action='store_true', help="Don't write to DB, only show actions")

    def handle(self, *args, **options):
        file_path = options.get('file')
        overwrite = options.get('overwrite')
        dry_run = options.get('dry_run')

        names = []
        if file_path:
            if not os.path.exists(file_path):
                raise CommandError(f"File not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as fh:
                content = fh.read().strip()
                # Try JSON parse first
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        names = [str(x).strip() for x in data if str(x).strip()]
                    else:
                        raise CommandError('JSON file must contain an array of strings')
                except json.JSONDecodeError:
                    # Fallback: newline separated list
                    names = [ln.strip() for ln in content.splitlines() if ln.strip()]
        else:
            names = DEFAULT_TAGS

        if not names:
            self.stdout.write(self.style.WARNING('No tags found to seed.'))
            return

        from apps.catalog.models import Tag

        if overwrite:
            existing = Tag.objects.count()
            self.stdout.write(self.style.WARNING(f'--overwrite: will delete {existing} existing tags'))
            if not dry_run:
                with transaction.atomic():
                    Tag.objects.all().delete()

        created = 0
        touched = 0
        for name in names:
            if dry_run:
                # Show what would be done
                self.stdout.write(self.style.NOTICE(f'Would create: {name}'))
                continue
            obj, was_created = Tag.objects.get_or_create(name=name)
            if was_created:
                created += 1
            else:
                touched += 1
        self.stdout.write(self.style.SUCCESS(f'Tags processed: {len(names)}; created: {created}; existed: {touched}'))
