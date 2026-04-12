from django.core.management.base import BaseCommand
from django.conf import settings

from apps.seo.services import is_prerender_enabled, prerender_paths


class Command(BaseCommand):
    help = 'Prerender top landing pages: homepage, top categories, top products and save HTML snapshots'

    def add_arguments(self, parser):
        parser.add_argument('--categories', type=int, default=10, help='Number of top categories to prerender')
        parser.add_argument('--products', type=int, default=20, help='Number of top products to prerender')
        parser.add_argument('--include-static', action='store_true', help='Include static pages like contact, about, faq')
        parser.add_argument('--timeout', type=int, default=15, help='HTTP timeout (seconds)')
        parser.add_argument('--retries', type=int, default=None, help='Retry attempts after first request')
        parser.add_argument('--force', action='store_true', help='Force refresh even when snapshots are fresh')

    def handle(self, *args, **options):
        if not is_prerender_enabled():
            self.stdout.write(self.style.WARNING('PRERENDER_ENABLED is false; skipping prerender.'))
            return
        categories_n = options.get('categories')
        products_n = options.get('products')
        include_static = options.get('include_static')
        timeout = options.get('timeout') or 15
        retries = options.get('retries')
        force = bool(options.get('force'))

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

        saved, successes, failures = prerender_paths(
            paths=sorted(paths),
            timeout=timeout,
            retries=retries,
            force=force,
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

