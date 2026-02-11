"""
Sitemap configuration
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse, urljoin

from django.conf import settings
from django.contrib.sitemaps import Sitemap
from django.contrib.sitemaps.views import x_robots_tag, _get_latest_lastmod
from django.contrib.sites.models import Site
from django.contrib.sites.requests import RequestSite
from django.contrib.sites.shortcuts import get_current_site
from django.core.paginator import EmptyPage, PageNotAnInteger
from django.db.models import Q, Max
from django.http import Http404
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils import timezone
from django.utils.http import http_date


def _get_site_base_url() -> str | None:
    site_url = getattr(settings, "SITE_URL", "").strip()
    if not site_url:
        return None
    parsed = urlparse(site_url)
    if parsed.scheme and parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            base = f"{base}{parsed.path.rstrip('/')}"
        return base
    if parsed.netloc:
        return f"https://{parsed.netloc}"
    if parsed.path:
        return f"https://{parsed.path.strip('/')}"
    return None


def _get_site_domain() -> str | None:
    site_url = getattr(settings, "SITE_URL", "").strip()
    if not site_url:
        return None
    parsed = urlparse(site_url)
    if parsed.netloc:
        return parsed.netloc
    if parsed.path and "://" not in site_url:
        return parsed.path.strip("/")
    return None


def _absolute_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("http://") or url.startswith("https://"):
        return url
    base = _get_site_base_url()
    if not base:
        return url
    return urljoin(f"{base}/", url.lstrip("/"))


def _is_local_request(request) -> bool:
    try:
        host = (request.get_host() or "").split(":")[0].lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "[::1]"} or host.endswith(".local")


def _rewrite_url_for_request(url: str, request) -> str:
    if not url:
        return url
    try:
        if url.startswith("//"):
            url = f"{request.scheme}:{url}"
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return parsed._replace(scheme=request.scheme, netloc=request.get_host()).geturl()
        if url.startswith("/"):
            return f"{request.scheme}://{request.get_host()}{url}"
    except Exception:
        return url
    return url


def _rewrite_urls_for_request(urls: list[dict], request) -> None:
    for url_info in urls:
        loc = url_info.get("location")
        if loc:
            url_info["location"] = _rewrite_url_for_request(loc, request)
        images = url_info.get("images")
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict) and img.get("location"):
                    img["location"] = _rewrite_url_for_request(img["location"], request)


def _normalize_images(images) -> list[dict]:
    normalized: list[dict] = []
    for image in images or []:
        if not image:
            continue
        if isinstance(image, str):
            location = image
            data = {"location": location}
        elif isinstance(image, dict):
            data = dict(image)
        else:
            continue
        location = data.get("location") or data.get("loc") or data.get("url")
        if not location:
            continue
        normalized.append({"location": _absolute_url(str(location))})
    return normalized


class FrontendSitemap(Sitemap):
    protocol = "https"

    def get_domain(self, site=None):
        domain = _get_site_domain()
        if domain:
            return domain
        return super().get_domain(site=site)

    def get_urls(self, page=1, site=None, protocol=None):
        urls = super().get_urls(page=page, site=site, protocol=protocol)
        if hasattr(self, "images"):
            for url_info in urls:
                item = url_info.get("item")
                images = self.images(item) if item is not None else []
                url_info["images"] = _normalize_images(images)
        return urls


STATIC_URLS: dict[str, dict[str, str | float]] = {
    "/": {"priority": 1.0, "changefreq": "daily"},
    "/products/": {"priority": 0.9, "changefreq": "daily"},
    "/categories/": {"priority": 0.8, "changefreq": "weekly"},
    "/collections/": {"priority": 0.7, "changefreq": "weekly"},
    "/bundles/": {"priority": 0.7, "changefreq": "weekly"},
    "/artisans/": {"priority": 0.6, "changefreq": "weekly"},
    "/contact/": {"priority": 0.5, "changefreq": "monthly"},
    "/faq/": {"priority": 0.5, "changefreq": "monthly"},
    "/pages/": {"priority": 0.3, "changefreq": "monthly"},
}


class StaticViewSitemap(FrontendSitemap):
    """Sitemap for static public pages."""

    def items(self):
        return list(STATIC_URLS.keys())

    def location(self, item):
        return item

    def priority(self, item):
        return STATIC_URLS.get(item, {}).get("priority")

    def changefreq(self, item):
        return STATIC_URLS.get(item, {}).get("changefreq")


class ProductSitemap(FrontendSitemap):
    """Sitemap for product detail pages (includes image sitemap entries)."""

    changefreq = "daily"
    priority = 0.9
    limit = 50000  # Google sitemap limit

    def get_queryset(self):
        from apps.catalog.models import Product

        now = timezone.now()
        return (
            Product.objects.filter(is_active=True, is_deleted=False)
            .filter(Q(publish_from__isnull=True) | Q(publish_from__lte=now))
            .filter(Q(publish_until__isnull=True) | Q(publish_until__gte=now))
        )

    def items(self):
        return self.get_queryset().prefetch_related("images")

    def get_latest_lastmod(self):
        return self.get_queryset().aggregate(Max("updated_at")).get("updated_at__max")

    def lastmod(self, obj):
        return obj.updated_at

    def location(self, obj):
        return f"/products/{obj.slug}/"

    def images(self, obj):
        images = []
        for img in obj.images.all()[:5]:
            if img.image:
                images.append({"location": img.image.url})
        return images


class CategorySitemap(FrontendSitemap):
    """Sitemap for category pages."""

    changefreq = "weekly"
    priority = 0.7

    def get_queryset(self):
        from apps.catalog.models import Category

        return Category.objects.filter(is_visible=True, is_deleted=False)

    def items(self):
        return self.get_queryset()

    def get_latest_lastmod(self):
        return self.get_queryset().aggregate(Max("updated_at")).get("updated_at__max")

    def lastmod(self, obj):
        return obj.updated_at

    def location(self, obj):
        return f"/categories/{obj.get_slug_path()}/"

    def images(self, obj):
        if obj.image:
            return [{"location": obj.image.url}]
        return []


class CollectionSitemap(FrontendSitemap):
    """Sitemap for curated collections."""

    changefreq = "weekly"
    priority = 0.6

    def items(self):
        from apps.catalog.models import Collection

        now = timezone.now()
        return (
            Collection.objects.filter(is_visible=True)
            .filter(Q(visible_from__isnull=True) | Q(visible_from__lte=now))
            .filter(Q(visible_until__isnull=True) | Q(visible_until__gte=now))
        )

    def location(self, obj):
        return f"/collections/{obj.slug}/"

    def images(self, obj):
        if obj.image:
            return [{"location": obj.image.url}]
        return []


class BundleSitemap(FrontendSitemap):
    """Sitemap for bundles/kits."""

    changefreq = "weekly"
    priority = 0.6

    def items(self):
        from apps.catalog.models import Bundle

        return Bundle.objects.filter(is_active=True)

    def location(self, obj):
        return f"/bundles/{obj.slug}/"

    def images(self, obj):
        if obj.image:
            return [{"location": obj.image.url}]
        return []


class ArtisanSitemap(FrontendSitemap):
    """Sitemap for artisan profile pages."""

    changefreq = "monthly"
    priority = 0.5

    def get_queryset(self):
        from apps.artisans.models import Artisan

        return Artisan.objects.filter(is_active=True)

    def items(self):
        return self.get_queryset()

    def get_latest_lastmod(self):
        return self.get_queryset().aggregate(Max("updated_at")).get("updated_at__max")

    def lastmod(self, obj):
        return obj.updated_at

    def location(self, obj):
        return f"/artisans/{obj.slug}/"

    def images(self, obj):
        if obj.photo:
            return [{"location": obj.photo.url}]
        return []


RESERVED_PAGE_SLUGS = {
    "account",
    "artisans",
    "bundles",
    "cart",
    "categories",
    "checkout",
    "collections",
    "compare",
    "contact",
    "faq",
    "notifications",
    "orders",
    "pages",
    "preorders",
    "products",
    "search",
    "subscriptions",
    "wishlist",
}

SPECIAL_PAGE_PATHS = {
    "about": "/about/",
}


class PageSitemap(FrontendSitemap):
    """Sitemap for CMS pages (served under /pages/)."""

    changefreq = "monthly"
    priority = 0.4

    def get_queryset(self):
        from apps.pages.models import Page

        now = timezone.now()
        return (
            Page.objects.filter(is_published=True)
            .filter(Q(published_at__isnull=True) | Q(published_at__lte=now))
            .exclude(slug__in=RESERVED_PAGE_SLUGS)
        )

    def items(self):
        return self.get_queryset()

    def get_latest_lastmod(self):
        return self.get_queryset().aggregate(Max("updated_at")).get("updated_at__max")

    def lastmod(self, obj):
        return obj.updated_at

    def location(self, obj):
        if obj.slug in SPECIAL_PAGE_PATHS:
            return SPECIAL_PAGE_PATHS[obj.slug]
        return f"/pages/{obj.slug}/"


@dataclass
class SitemapIndexItem:
    location: str
    last_mod: object | None = None


@x_robots_tag
def sitemap_index_view(
    request,
    sitemaps,
    template_name="sitemap_index.xml",
    content_type="application/xml",
    sitemap_url_name="django.contrib.sitemaps.views.sitemap",
):
    req_protocol = request.scheme
    base_url = _get_site_base_url()
    try:
        req_site = get_current_site(request)
    except Site.DoesNotExist:
        req_site = RequestSite(request)

    local_override = _is_local_request(request)
    if local_override:
        base_url = f"{req_protocol}://{request.get_host()}"

    sites = []
    all_indexes_lastmod = True
    latest_lastmod = None
    for section, site in sitemaps.items():
        if callable(site):
            site = site()
        sitemap_url = reverse(sitemap_url_name, kwargs={"section": section})
        if base_url:
            absolute_url = f"{base_url}{sitemap_url}"
        else:
            protocol = req_protocol if site.protocol is None else site.protocol
            absolute_url = f"{protocol}://{req_site.domain}{sitemap_url}"

        site_lastmod = site.get_latest_lastmod()
        if all_indexes_lastmod:
            if site_lastmod is not None:
                latest_lastmod = _get_latest_lastmod(latest_lastmod, site_lastmod)
            else:
                all_indexes_lastmod = False

        sites.append(SitemapIndexItem(absolute_url, site_lastmod))

        # Add links to all pages of the sitemap.
        for page in range(2, site.paginator.num_pages + 1):
            sites.append(SitemapIndexItem(f"{absolute_url}?p={page}", site_lastmod))

    if local_override:
        for item in sites:
            item.location = _rewrite_url_for_request(item.location, request)

    headers = {"Last-Modified": http_date(latest_lastmod.timestamp())} if all_indexes_lastmod and latest_lastmod else None
    return TemplateResponse(
        request,
        template_name,
        {"sitemaps": sites},
        content_type=content_type,
        headers=headers,
    )


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

    if _is_local_request(request):
        _rewrite_urls_for_request(urls, request)

    headers = {"Last-Modified": http_date(lastmod.timestamp())} if all_sites_lastmod and lastmod else None
    return TemplateResponse(
        request,
        template_name,
        {"urlset": urls},
        content_type=content_type,
        headers=headers,
    )
