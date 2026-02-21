from django.core.management.base import BaseCommand
from django.conf import settings

from apps.seo.prerender import is_prerender_enabled, prerender_paths


class Command(BaseCommand):
    help = 'Prerender configured pages and save HTML snapshots to PRERENDER_CACHE_DIR'

    def add_arguments(self, parser):
        parser.add_argument('--paths', nargs='*', help='Paths to prerender (overrides settings PRERENDER_PATHS)')

    def handle(self, *args, **options):
        if not is_prerender_enabled():
            self.stdout.write(self.style.WARNING('PRERENDER_ENABLED is false; skipping prerender.'))
            return
        paths = [str(path) for path in (options.get('paths') or getattr(settings, 'PRERENDER_PATHS', ['/']))]

        saved, successes, failures = prerender_paths(paths=paths)
        for _, output in successes:
            self.stdout.write(self.style.SUCCESS(f'Saved {output}'))
        for url, error in failures:
            self.stdout.write(self.style.ERROR(f'Failed {url}: {error}'))
        self.stdout.write(self.style.SUCCESS(f'Prerendered {saved} pages'))
