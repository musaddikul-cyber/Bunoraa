from django.core.management.base import BaseCommand
from django.conf import settings
import os
import requests
import time
from urllib.parse import urljoin

class Command(BaseCommand):
    help = 'Check prerender cache files and verify they are served to bots with X-PreRendered header and low TTFB'

    def handle(self, *args, **options):
        cache_dir = os.path.join(settings.BASE_DIR, getattr(settings, 'PRERENDER_CACHE_DIR', 'prerender_cache'))
        if not os.path.exists(cache_dir):
            self.stdout.write(self.style.ERROR(f'Prerender cache dir not found: {cache_dir}'))
            return
        files = [f for f in os.listdir(cache_dir) if f.endswith('.html')]
        if not files:
            self.stdout.write(self.style.WARNING('No prerendered HTML files found.'))
            return
        site = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
        passed = 0
        total = 0
        for f in files:
            path = '/' + f.replace('.html', '').replace('_', '/')
            if path == '/index':
                path = '/'
            total += 1
            url = urljoin(site, path.lstrip('/'))
            try:
                start = time.time()
                r = requests.get(url, headers=headers, timeout=10)
                elapsed = (time.time() - start) * 1000
                header = r.headers.get('X-PreRendered')
                ok = (r.status_code == 200 and header == '1' and elapsed < 500)
                status = 'OK' if ok else 'FAIL'
                self.stdout.write(f'{status}: {url} -> {r.status_code} in {int(elapsed)}ms X-PreRendered:{header}')
                if ok:
                    passed += 1
            except Exception as exc:
                self.stdout.write(f'ERR: {url} -> {exc}')
        self.stdout.write(self.style.SUCCESS(f'Checked {total} files, {passed} passed.'))
