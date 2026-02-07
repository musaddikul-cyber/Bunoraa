"""
Management command to manage sitemaps - submit, verify, and track submissions
"""
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db.models import Sum
from apps.seo.models import SitemapSubmission, SitemapError
import requests
from urllib.parse import urljoin


class Command(BaseCommand):
    help = 'Manage sitemap submissions and tracking'

    def add_arguments(self, parser):
        parser.add_argument(
            '--action',
            type=str,
            choices=['add', 'submit', 'list', 'verify', 'status'],
            required=True,
            help='Action to perform'
        )
        
        parser.add_argument(
            '--type',
            type=str,
            choices=['static', 'products', 'categories', 'blog'],
            help='Type of sitemap'
        )
        
        parser.add_argument(
            '--url',
            type=str,
            help='URL of the sitemap'
        )
        
        parser.add_argument(
            '--engines',
            type=str,
            nargs='+',
            help='Search engines to submit to (google, bing, yandex)'
        )
        
        parser.add_argument(
            '--domain',
            type=str,
            help='Domain name for sitemap submission'
        )

    def handle(self, *args, **options):
        action = options['action']

        if action == 'add':
            self.add_sitemap(options)
        elif action == 'submit':
            self.submit_sitemap(options)
        elif action == 'list':
            self.list_sitemaps()
        elif action == 'verify':
            self.verify_sitemaps()
        elif action == 'status':
            self.show_status()

    def add_sitemap(self, options):
        """Add a new sitemap submission record"""
        if not options['type'] or not options['url']:
            raise CommandError('--type and --url are required for add action')

        sitemap, created = SitemapSubmission.objects.get_or_create(
            url=options['url'],
            defaults={
                'sitemap_type': options['type'],
                'status': 'pending',
                'search_engines': options['engines'] or []
            }
        )

        if created:
            self.stdout.write(
                self.style.SUCCESS(f'✓ Sitemap added: {sitemap.url}')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'⚠ Sitemap already exists: {sitemap.url}')
            )

    def submit_sitemap(self, options):
        """Submit a sitemap to search engines"""
        if not options['url']:
            raise CommandError('--url is required')

        try:
            sitemap = SitemapSubmission.objects.get(url=options['url'])
        except SitemapSubmission.DoesNotExist:
            raise CommandError(f'Sitemap not found: {options["url"]}')

        engines = options['engines'] or ['google', 'bing']

        for engine in engines:
            success = self._submit_to_engine(engine, options['url'], options.get('domain'))
            if success:
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Submitted to {engine}')
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'✗ Failed to submit to {engine}')
                )

        sitemap.status = 'submitted'
        sitemap.submitted_at = timezone.now()
        if not sitemap.search_engines:
            sitemap.search_engines = engines
        sitemap.save()

    def list_sitemaps(self):
        """List all sitemap submissions"""
        sitemaps = SitemapSubmission.objects.all()

        if not sitemaps.exists():
            self.stdout.write('No sitemaps found')
            return

        self.stdout.write('\nSitemap Submissions:')
        self.stdout.write('=' * 100)

        for sitemap in sitemaps:
            status_color = {
                'pending': self.style.WARNING,
                'submitted': self.style.SUCCESS,
                'indexed': self.style.SUCCESS,
                'error': self.style.ERROR,
            }.get(sitemap.status, self.style.WARNING)

            self.stdout.write(
                f'{sitemap.id:3d} | {sitemap.sitemap_type:12s} | '
                f'{status_color(sitemap.status):12s} | {sitemap.url:50s} | '
                f'Pages: {sitemap.discovered_pages:6d}'
            )

    def verify_sitemaps(self):
        """Verify that sitemaps are accessible"""
        sitemaps = SitemapSubmission.objects.all()

        for sitemap in sitemaps:
            try:
                response = requests.head(sitemap.url, timeout=5)
                if response.status_code == 200:
                    self.stdout.write(
                        self.style.SUCCESS(f'✓ {sitemap.url}')
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR(f'✗ {sitemap.url} (HTTP {response.status_code})')
                    )
                    # Log error
                    SitemapError.objects.create(
                        submission=sitemap,
                        severity='error',
                        error_code=f'HTTP_{response.status_code}',
                        message=f'Sitemap returned HTTP {response.status_code}'
                    )
            except requests.RequestException as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ {sitemap.url} ({str(e)})')
                )
                SitemapError.objects.create(
                    submission=sitemap,
                    severity='critical',
                    error_code='CONNECTION_ERROR',
                    message=str(e)
                )

    def show_status(self):
        """Show sitemap status summary"""
        sitemaps = SitemapSubmission.objects.all()

        if not sitemaps.exists():
            self.stdout.write('No sitemaps found')
            return

        total = sitemaps.count()
        by_status = {
            'pending': sitemaps.filter(status='pending').count(),
            'submitted': sitemaps.filter(status='submitted').count(),
            'indexed': sitemaps.filter(status='indexed').count(),
            'error': sitemaps.filter(status='error').count(),
        }

        total_pages = sitemaps.aggregate(
            total=Sum('discovered_pages')
        )['total'] or 0

        self.stdout.write('\nSitemap Status Summary:')
        self.stdout.write('=' * 50)
        self.stdout.write(f'Total Sitemaps: {total}')
        self.stdout.write(f'  - Pending: {by_status["pending"]}')
        self.stdout.write(f'  - Submitted: {by_status["submitted"]}')
        self.stdout.write(f'  - Indexed: {by_status["indexed"]}')
        self.stdout.write(f'  - Error: {by_status["error"]}')
        self.stdout.write(f'\nTotal Discovered Pages: {total_pages}')

    def _submit_to_engine(self, engine, sitemap_url, domain):
        """Submit sitemap to a specific search engine"""
        try:
            if engine == 'google':
                url = f'https://www.google.com/ping?sitemap={sitemap_url}'
                requests.get(url, timeout=5)
                return True
            elif engine == 'bing':
                url = f'https://www.bing.com/ping?sitemap={sitemap_url}'
                requests.get(url, timeout=5)
                return True
            elif engine == 'yandex':
                url = f'https://www.yandex.ru/ping?sitemap={sitemap_url}'
                requests.get(url, timeout=5)
                return True
            return False
        except requests.RequestException:
            return False
