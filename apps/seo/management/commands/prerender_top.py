from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import requests
from urllib.parse import urljoin


class Command(BaseCommand):
    help = 'Prerender top landing pages: homepage, top categories, top products and save HTML snapshots'

    def add_arguments(self, parser):
        parser.add_argument('--categories', type=int, default=10, help='Number of top categories to prerender')
        parser.add_argument('--products', type=int, default=20, help='Number of top products to prerender')
        parser.add_argument('--include-static', action='store_true', help='Include static pages like contact, about, faq')

    def handle(self, *args, **options):
        categories_n = options.get('categories')
        products_n = options.get('products')
        include_static = options.get('include_static')

        cache_dir = os.path.join(settings.BASE_DIR, getattr(settings, 'PRERENDER_CACHE_DIR', 'prerender_cache'))
        os.makedirs(cache_dir, exist_ok=True)
        site = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; BunoraaPrerender/1.0)'}

        paths = set(['/'])

        # Top categories by product_count
        try:
            from apps.catalog.models import Category
            cats = Category.objects.filter(is_visible=True, is_deleted=False)
            cats = sorted(list(cats), key=lambda c: c.product_count, reverse=True)[:categories_n]
            for c in cats:
                paths.add(f"/categories/{c.slug}/")
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f'Could not fetch categories: {exc}'))

        # Top products by sales_count then views_count
        try:
            from apps.catalog.models import Product
            prods = Product.objects.filter(is_active=True, is_deleted=False).order_by('-sales_count', '-views_count')[:products_n]
            for p in prods:
                paths.add(f"/products/{p.slug}/")
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f'Could not fetch products: {exc}'))

        # Static pages
        if include_static:
            paths |= {'/about/', '/contact/', '/faq/'}

        saved = 0
        for p in paths:
            url = urljoin(site, p.lstrip('/'))
            try:
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                fname = p.strip('/').replace('/', '_') or 'index'
                fname = f"{fname}.html"
                full = os.path.join(cache_dir, fname)
                with open(full, 'wb') as fh:
                    fh.write(r.content)
                self.stdout.write(self.style.SUCCESS(f'Saved {full}'))
                saved += 1
            except Exception as exc:
                self.stdout.write(self.style.ERROR(f'Failed {url}: {exc}'))
        self.stdout.write(self.style.SUCCESS(f'Prerendered {saved} pages'))
