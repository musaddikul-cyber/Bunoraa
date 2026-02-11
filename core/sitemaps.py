"""
Sitemap configuration
"""
from django.contrib.sitemaps import Sitemap
from django.contrib.sitemaps.views import x_robots_tag, _get_latest_lastmod
from django.contrib.sites.models import Site
from django.contrib.sites.requests import RequestSite
from django.contrib.sites.shortcuts import get_current_site
from django.core.paginator import EmptyPage, PageNotAnInteger
from django.db.models import Q
from django.http import Http404
from django.template.response import TemplateResponse
from django.urls import reverse, NoReverseMatch
from django.utils import timezone
from django.utils.http import http_date


class StaticViewSitemap(Sitemap):
    """Sitemap for static pages."""
    priority = 0.5
    changefreq = 'weekly'
    
    def items(self):
        return ['pages:home', 'catalog:product-list', 'catalog:category-list']
    
    def location(self, item):
        if isinstance(item, str) and item.startswith('/'):
            return item
        if not isinstance(item, str):
            if hasattr(item, 'get_absolute_url'):
                return item.get_absolute_url()
            return reverse(item)
        candidates = [item]
        base_name = item.split(':')[-1]
        if base_name != item:
            candidates.extend([f'pages:{base_name}', f'home:{base_name}', base_name])
        else:
            candidates.extend([f'pages:{item}', f'home:{item}'])
        last_exc = None
        for name in candidates:
            try:
                return reverse(name)
            except NoReverseMatch as exc:
                last_exc = exc
        raise last_exc


class ProductSitemap(Sitemap):
    """Sitemap for products, including images for image sitemap support."""
    changefreq = 'daily'
    priority = 0.8
    limit = 50000  # Google sitemap limit
    
    def items(self):
        from apps.catalog.models import Product
        now = timezone.now()
        return (
            Product.objects.filter(is_active=True, is_deleted=False)
            .filter(Q(publish_from__isnull=True) | Q(publish_from__lte=now))
            .filter(Q(publish_until__isnull=True) | Q(publish_until__gte=now))
            .prefetch_related('images')
        )
    
    def lastmod(self, obj):
        return obj.updated_at
    
    def location(self, obj):
        return reverse('catalog:product-detail', kwargs={'slug': obj.slug})

    def images(self, obj):
        """Include images in the sitemap for better indexing"""
        imgs = []
        for img in obj.images.all()[:5]:
            if img.image:
                imgs.append({
                    'location': img.image.url,
                    'title': obj.name,
                    'caption': (obj.short_description[:250] if obj.short_description else ''),
                })
        return imgs


class CategorySitemap(Sitemap):
    """Sitemap for categories."""
    changefreq = 'weekly'
    priority = 0.7
    
    def items(self):
        from apps.catalog.models import Category
        return Category.objects.filter(is_visible=True, is_deleted=False)
    
    def lastmod(self, obj):
        return obj.updated_at
    
    def location(self, obj):
        return obj.get_absolute_url()


class BlogSitemap(Sitemap):
    """Sitemap for blog posts."""
    changefreq = 'weekly'
    priority = 0.6
    
    def items(self):
        from apps.pages.models import Page
        now = timezone.now()
        return Page.objects.filter(is_published=True).filter(
            Q(published_at__isnull=True) | Q(published_at__lte=now)
        )
    
    def lastmod(self, obj):
        return obj.updated_at
    
    def location(self, obj):
        return reverse('pages:detail', kwargs={'slug': obj.slug})


@x_robots_tag
def sitemap_view(
    request,
    sitemaps,
    section=None,
    template_name="sitemap.xml",
    content_type="application/xml",
):
    req_protocol = request.scheme
    try:
        req_site = get_current_site(request)
    except Site.DoesNotExist:
        req_site = RequestSite(request)

    if section is not None:
        if section not in sitemaps:
            raise Http404(f"No sitemap available for section: {section!r}")
        maps = [sitemaps[section]]
    else:
        maps = sitemaps.values()
    page = request.GET.get("p", 1)

    lastmod = None
    all_sites_lastmod = True
    urls = []
    for site in maps:
        try:
            if callable(site):
                site = site()
            urls.extend(site.get_urls(page=page, site=req_site, protocol=req_protocol))
            if all_sites_lastmod:
                site_lastmod = getattr(site, "latest_lastmod", None)
                if site_lastmod is not None:
                    lastmod = _get_latest_lastmod(lastmod, site_lastmod)
                else:
                    all_sites_lastmod = False
        except EmptyPage as exc:
            raise Http404(f"Page {page} empty") from exc
        except PageNotAnInteger as exc:
            raise Http404(f"No page '{page}'") from exc

    headers = {"Last-Modified": http_date(lastmod.timestamp())} if all_sites_lastmod and lastmod else None
    return TemplateResponse(
        request,
        template_name,
        {"urlset": urls},
        content_type=content_type,
        headers=headers,
    )
