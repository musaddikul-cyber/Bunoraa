from celery import shared_task
from django.utils import timezone
from .models import Keyword, SitemapSubmission, SitemapError
from .services import snapshot_keyword_serp
import requests


@shared_task(bind=True)
def snapshot_all_keywords_serp(self, num=10):
    """Take a SERP snapshot for all target keywords."""
    keywords = Keyword.objects.filter(is_target=True)
    date = timezone.now().date().isoformat()
    for k in keywords:
        snapshot_keyword_serp(k.term, num=num)
    return {'date': date, 'keywords': keywords.count()}


@shared_task(bind=True)
def fetch_gsc_for_targets(self, start_date=None, end_date=None):
    """Wrapper that calls management command seo_fetch_gsc via call_command (keeps logic centralised)."""
    from django.core.management import call_command
    kwargs = {}
    if start_date:
        kwargs['start_date'] = start_date
    if end_date:
        kwargs['end_date'] = end_date
    call_command('seo_fetch_gsc', **kwargs)
    return {'started': start_date, 'ended': end_date}


@shared_task(bind=True)
def warmup_service(self):
    """Warm up critical endpoints (home, search, top categories) to prevent cold starts on Render."""
    from django.conf import settings
    import requests
    from urllib.parse import urljoin

    site = getattr(settings, 'SITE_URL', 'https://bunoraa.com')
    paths = getattr(settings, 'PRERENDER_PATHS', ['/'])
    headers = {'User-Agent': 'BunoraaWarmup/1.0'}
    results = []
    for p in paths:
        try:
            url = urljoin(site, p)
            r = requests.get(url, headers=headers, timeout=10)
            results.append({'url': url, 'status': r.status_code})
        except Exception as exc:
            results.append({'url': url, 'error': str(exc)})
    return results


@shared_task(bind=True)
def verify_sitemap_accessibility(self):
    """Verify that all submitted sitemaps are accessible"""
    sitemaps = SitemapSubmission.objects.all()
    results = {'verified': 0, 'errors': 0}

    for sitemap in sitemaps:
        try:
            response = requests.head(sitemap.url, timeout=5)
            if response.status_code == 200:
                results['verified'] += 1
            else:
                results['errors'] += 1
                SitemapError.objects.create(
                    submission=sitemap,
                    severity='error',
                    error_code=f'HTTP_{response.status_code}',
                    message=f'Sitemap returned HTTP {response.status_code}'
                )
        except requests.RequestException as e:
            results['errors'] += 1
            SitemapError.objects.create(
                submission=sitemap,
                severity='critical',
                error_code='CONNECTION_ERROR',
                message=str(e)
            )

    return results


@shared_task(bind=True)
def submit_sitemaps_to_engines(self):
    """Submit pending sitemaps to search engines"""
    sitemaps = SitemapSubmission.objects.filter(status='pending')
    results = {'submitted': 0, 'failed': 0}

    for sitemap in sitemaps:
        engines = sitemap.search_engines or ['google', 'bing']
        
        for engine in engines:
            try:
                if engine == 'google':
                    url = f'https://www.google.com/ping?sitemap={sitemap.url}'
                elif engine == 'bing':
                    url = f'https://www.bing.com/ping?sitemap={sitemap.url}'
                elif engine == 'yandex':
                    url = f'https://www.yandex.ru/ping?sitemap={sitemap.url}'
                else:
                    continue

                response = requests.get(url, timeout=5)
                if response.status_code in [200, 204]:
                    results['submitted'] += 1
                else:
                    results['failed'] += 1
            except requests.RequestException as e:
                results['failed'] += 1
                SitemapError.objects.create(
                    submission=sitemap,
                    severity='warning',
                    error_code=f'SUBMIT_FAILED_{engine.upper()}',
                    message=f'Failed to submit to {engine}: {str(e)}'
                )

        # Update submission status
        sitemap.status = 'submitted'
        sitemap.submitted_at = timezone.now()
        sitemap.save()

    return results

@shared_task(bind=True)
def prerender_top_task(self, categories=10, products=20, include_static=True):
    """Call management command to prerender top pages."""
    from django.core.management import call_command
    args = [f'--categories={categories}', f'--products={products}']
    if include_static:
        args.append('--include-static')
    call_command('prerender_top', *args)
    return {'prerendered': True}