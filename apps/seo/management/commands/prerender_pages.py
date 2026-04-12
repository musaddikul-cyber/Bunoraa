from django.core.management.base import BaseCommand
from django.conf import settings

from apps.seo.services import is_prerender_enabled, prerender_paths


class Command(BaseCommand):
    help = 'Prerender configured pages and save HTML snapshots to PRERENDER_CACHE_DIR'

    def add_arguments(self, parser):
        parser.add_argument('--paths', nargs='*', help='Paths to prerender (overrides settings PRERENDER_PATHS)')
        parser.add_argument('--timeout', type=int, default=15, help='HTTP timeout (seconds)')
        parser.add_argument('--retries', type=int, default=None, help='Retry attempts after first request')
        parser.add_argument('--force', action='store_true', help='Force refresh even when a fresh snapshot exists')
        parser.add_argument(
            '--user-agent',
            type=str,
            default=None,
            help='Override prerender user agent',
        )

    def handle(self, *args, **options):
        if not is_prerender_enabled():
            self.stdout.write(self.style.WARNING('PRERENDER_ENABLED is false; skipping prerender.'))
            return
        paths = [str(path) for path in (options.get('paths') or getattr(settings, 'PRERENDER_PATHS', ['/']))]
        timeout = options.get('timeout') or 15
        retries = options.get('retries')
        force = bool(options.get('force'))
        user_agent = options.get('user_agent') or None

        saved, successes, failures = prerender_paths(
            paths=paths,
            timeout=timeout,
            retries=retries,
            force=force,
            user_agent=user_agent or getattr(settings, 'PRERENDER_USER_AGENT', None),
        )
        for _, output in successes:
            self.stdout.write(self.style.SUCCESS(f'Saved {output}'))
        for url, error in failures:
            self.stdout.write(self.style.ERROR(f'Failed {url}: {error}'))
        total = len(successes) + len(failures)
        if failures:
            self.stdout.write(self.style.WARNING(f'Prerendered {saved}/{total} pages; {len(failures)} failures'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Prerendered {saved} pages'))

