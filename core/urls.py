"""
Bunoraa URL Configuration
"""
from urllib.parse import urlencode

from django.contrib import admin
from django.conf import settings as dj_settings
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.views import redirect_to_login
from django.http import HttpResponseRedirect
from django.urls import path, include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import resolve_url
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.i18n import JavaScriptCatalog
from django.views.decorators.http import require_GET
from django.views.decorators.cache import never_cache
from django.views.static import serve
from two_factor.admin import AdminSiteOTPRequired
from two_factor.utils import default_device
from apps.accounts.views import OAuthBeginView
from .sitemaps import sitemap_view, sitemap_index_view
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView
from .two_factor_views import ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY
from .sitemaps import (
    StaticViewSitemap,
    ProductSitemap,
    CategorySitemap,
    CollectionSitemap,
    BundleSitemap,
    ArtisanSitemap,
    PageSitemap,
    PreOrderCategorySitemap,
)
from .views import health_check, health_check_detailed, readiness_check, liveness_check
from core.api.performance import (
    health_check as api_health_check,
    metrics_overview,
    warm_cache,
    clear_cache,
    reset_db_log,
)

class BunoraaAdminSite(AdminSiteOTPRequired):
    """Admin site that redirects login to the OTP-enabled login view."""

    @staticmethod
    def _is_user_otp_verified(user):
        checker = getattr(user, "is_verified", None)
        if callable(checker):
            return bool(checker())
        return False

    def has_permission(self, request):
        if not admin.AdminSite.has_permission(self, request):
            return False

        if self._is_user_otp_verified(request.user):
            return True

        # Allow this session to continue only when the user has no OTP device
        # and has explicitly skipped setup from the admin 2FA setup page.
        return not default_device(request.user) and bool(
            request.session.get(ADMIN_2FA_SETUP_SKIPPED_SESSION_KEY)
        )

    def login(self, request, extra_context=None):
        redirect_to = request.POST.get(REDIRECT_FIELD_NAME, request.GET.get(REDIRECT_FIELD_NAME))

        if not redirect_to or not url_has_allowed_host_and_scheme(
            url=redirect_to, allowed_hosts=[request.get_host()]
        ):
            redirect_to = resolve_url(dj_settings.LOGIN_REDIRECT_URL)

        # Avoid login loops: authenticated staff without any OTP device should
        # be sent to setup/skip flow instead of being sent back to /admin/login/.
        if request.user.is_authenticated and request.user.is_staff and not self._is_user_otp_verified(request.user):
            if not default_device(request.user):
                request.session["next"] = redirect_to
                request.session.modified = True
                setup_url = resolve_url("two_factor:setup")
                query = urlencode({REDIRECT_FIELD_NAME: redirect_to})
                return HttpResponseRedirect(f"{setup_url}?{query}")

        return redirect_to_login(redirect_to, login_url=resolve_url('two_factor:login'))


# Enforce OTP for admin site without changing registrations
admin.site.__class__ = BunoraaAdminSite

sitemaps = {
    'static': StaticViewSitemap,
    'products': ProductSitemap,
    'categories': CategorySitemap,
    'collections': CollectionSitemap,
    'bundles': BundleSitemap,
    'artisans': ArtisanSitemap,
    'pages': PageSitemap,
    'preorders': PreOrderCategorySitemap,
}

@require_GET
@never_cache
def robots_txt(request):
    response = serve(request, "robots.txt", document_root=settings.STATIC_ROOT)
    response["Content-Type"] = "text/plain; charset=utf-8"
    response["X-Robots-Tag"] = "noindex, nofollow"
    return response


urlpatterns = [
    # Internationalization (language switcher)
    path('i18n/', include('django.conf.urls.i18n')),

    # Two-factor URLs (namespaced for OTP redirects)
    path('', include(('core.two_factor_urls', 'two_factor'), namespace='two_factor')),

    # Serve admin JavaScript catalog without OTP gate to avoid jsi18n MIME/login redirects.
    path(
        "admin/jsi18n/",
        JavaScriptCatalog.as_view(packages=["django.contrib.admin"]),
        name="admin-jsi18n-public",
    ),

    # Admin and staff utilities
    path('admin-tools/', include('core.admin_urls')),
    path('admin/', admin.site.urls),

    # API v1
    path('api/v1/', include('core.urls_api')),

    # ML API (direct path for frontend JS library compatibility)
    *(
        [path('api/ml/', include('ml.api.urls'))]
        if settings.ML_ENABLED and getattr(settings, 'ML_API_ENABLED', True)
        else []
    ),

    # Health checks
    path('health', health_check, name='health_check_noslash'),
    path('health/', health_check, name='health_check'),
    path('health/detailed/', health_check_detailed, name='health_check_detailed'),
    path('health/ready/', readiness_check, name='readiness_check'),
    path('health/live/', liveness_check, name='liveness_check'),

    # Performance & Monitoring API
    path('api/admin/performance/health/', api_health_check, name='api_health_check'),
    path('api/admin/performance/metrics/', metrics_overview, name='api_metrics_overview'),
    path('api/admin/performance/cache/warm/', warm_cache, name='api_cache_warm'),
    path('api/admin/performance/cache/clear/', clear_cache, name='api_cache_clear'),
    path('api/admin/performance/db/reset-log/', reset_db_log, name='api_db_reset_log'),

    # SEO helpers
    path(
        'favicon.ico',
        RedirectView.as_view(url=f"{settings.STATIC_URL}images/assets/favicon.ico", permanent=False),
        name='site-favicon',
    ),
    path('robots.txt', robots_txt, name='robots_txt'),

    # Sitemap index + section sitemaps
    path('sitemap.xml', sitemap_index_view, {'sitemaps': sitemaps}, name='sitemap-index'),
    path('sitemap-<section>.xml', sitemap_view, {'sitemaps': sitemaps}, name='django.contrib.sitemaps.views.sitemap'),

    # API Schema views (Spectacular)
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),

    # Catalog routes aligned to actual app URL patterns
    path('catalog/', include('apps.catalog.urls', namespace='catalog')),
    path('products/', include(('apps.catalog.urls', 'catalog-products'), namespace='catalog-products')),
    path('categories/', include(('apps.catalog.urls', 'catalog-categories'), namespace='catalog-categories')),

    # Artisans
    path('artisans/', include('apps.artisans.urls', namespace='artisans')),

    # Shopping features - Cart, Wishlist, Checkout via commerce app
    path('', include(('apps.commerce.urls', 'commerce'), namespace='commerce')),
    path('orders/', include('apps.orders.urls')),
    path('payments/', include('apps.payments.urls')),

    # Pre-orders and Subscriptions
    path('preorders/', include('apps.preorders.urls')),
    path('subscriptions/', include('apps.subscriptions.urls')),

    path('notifications/', include('apps.notifications.urls')),
    path('account/', include('apps.accounts.urls')),
    path('oauth/login/<str:backend>/', OAuthBeginView.as_view(), name='oauth_begin'),
    path('oauth/', include('social_django.urls', namespace='social')),

    # Email Service API
    path('email/', include('apps.email_service.urls', namespace='email_service')),

    # Register pages URLs under compatibility namespaces for reverse lookups.
    path('', include(('apps.pages.urls', 'home'), namespace='home')),
    path('', include(('apps.pages.urls', 'contacts'), namespace='contacts')),
    path('', include(('apps.pages.urls', 'faq'), namespace='faq')),
    path('', include(('apps.pages.urls', 'about'), namespace='about')),
    path('', include(('apps.pages.urls', 'subscribe'), namespace='subscribe')),
    path('', include(('apps.pages.urls', 'detail'), namespace='detail')),

    # Pages catch-all (must come last) and owns the real homepage.
    path('', include('apps.pages.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    
    # Debug toolbar
    try:
        import debug_toolbar
        urlpatterns = [path('__debug__/', include(debug_toolbar.urls))] + urlpatterns
    except ImportError:
        pass

# Admin site customization
admin.site.site_header = 'Bunoraa Administration'
admin.site.site_title = 'Bunoraa Admin'
admin.site.index_title = 'Dashboard'

# Custom error handlers
handler404 = 'core.views.custom_404_view'
handler500 = 'core.views.custom_500_view'

# Custom error handlers
handler404 = 'core.views.custom_404_view'
handler500 = 'core.views.custom_500_view'
