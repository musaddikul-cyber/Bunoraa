from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import requests
from urllib.parse import urljoin


class Command(BaseCommand):
    help = 'Prerender configured pages and save HTML snapshots to PRERENDER_CACHE_DIR'

    def add_arguments(self, parser):
        parser.add_argument('--paths', nargs='*', help='Paths to prerender (overrides settings PRERENDER_PATHS)')

    def handle(self, *args, **options):
        paths = options.get('paths') or getattr(settings, 'PRERENDER_PATHS', ['/'])
        cache_dir = os.path.join(settings.BASE_DIR, getattr(settings, 'PRERENDER_CACHE_DIR', 'prerender_cache'))
        os.makedirs(cache_dir, exist_ok=True)
        site = getattr(settings, 'SITE_URL', 'https://bunoraa.com')

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; BunoraaPrerender/1.0)'}
        saved = 0
        for p in paths:
            url = urljoin(site, p.lstrip('/'))
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                # sanitize filename
                fname = p.strip('/').replace('/', '_') or 'index'
                fname = f"{fname}.html"
                full_path = os.path.join(cache_dir, fname)
                with open(full_path, 'wb') as f:
                    f.write(resp.content)
                self.stdout.write(self.style.SUCCESS(f'Saved {full_path}'))
                saved += 1
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f'Failed {url}: {exc}'))
        self.stdout.write(self.style.SUCCESS(f'Prerendered {saved} pages'))