from django.core.management.base import BaseCommand
from django.conf import settings

from apps.seo.prerender import is_prerender_enabled, prerender_paths


class Command(BaseCommand):
    help = 'Prerender top landing pages: homepage, top categories, top products and save HTML snapshots'

    def add_arguments(self, parser):
        parser.add_argument('--categories', type=int, default=10, help='Number of top categories to prerender')
        parser.add_argument('--products', type=int, default=20, help='Number of top products to prerender')
        parser.add_argument('--include-static', action='store_true', help='Include static pages like contact, about, faq')

    def handle(self, *args, **options):
        if not is_prerender_enabled():
            self.stdout.write(self.style.WARNING('PRERENDER_ENABLED is false; skipping prerender.'))
            return
        categories_n = options.get('categories')
        products_n = options.get('products')
        include_static = options.get('include_static')

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

        saved, successes, failures = prerender_paths(paths=sorted(paths))
        for _, output in successes:
            self.stdout.write(self.style.SUCCESS(f'Saved {output}'))
        for url, error in failures:
            self.stdout.write(self.style.ERROR(f'Failed {url}: {error}'))
        self.stdout.write(self.style.SUCCESS(f'Prerendered {saved} pages'))
