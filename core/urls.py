"""
Bunoraa URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import RedirectView
from django.conf import settings
from django.conf.urls.static import static
from .sitemaps import sitemap_view, sitemap_index_view
from django.views.generic import RedirectView # Added for API docs redirect
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView # Added for API docs
from .sitemaps import (
    StaticViewSitemap,
    ProductSitemap,
    CategorySitemap,
    CollectionSitemap,
    BundleSitemap,
    ArtisanSitemap,
    PageSitemap,
)
from .views import HomeView, health_check, health_check_detailed, readiness_check, liveness_check

sitemaps = {
    'static': StaticViewSitemap,
    'products': ProductSitemap,
    'categories': CategorySitemap,
    'collections': CollectionSitemap,
    'bundles': BundleSitemap,
    'artisans': ArtisanSitemap,
    'pages': PageSitemap,
}

urlpatterns = [
    # Internationalization (language switcher)
    path('i18n/', include('django.conf.urls.i18n')),
    
    # Admin Dashboard (custom)
    path('admin/dashboard/', include('core.admin_urls')),
    
    # Admin
    path('admin', RedirectView.as_view(url='/admin/', permanent=True)),
    path('admin/', admin.site.urls),
    
    # API v1
    path('api/v1/', include('core.urls_api')),
    
    # ML API (direct path for frontend JS library compatibility)
    # Disabled to reduce memory footprint on Render free tier (torch not available)
    # path('api/ml/', include('ml.api.urls')),
    
    # Health checks
    path('health', health_check, name='health_check_noslash'),
    path('health/', health_check, name='health_check'),
    path('health/detailed/', health_check_detailed, name='health_check_detailed'),
    path('health/ready/', readiness_check, name='readiness_check'),
    path('health/live/', liveness_check, name='liveness_check'),
    
    # SEO (Robots.txt) - Served from static directory by web server
    # No need for Django route - nginx/web server serves static/robots.txt directly
    
    # Sitemap index + section sitemaps
    path('sitemap.xml', sitemap_index_view, {'sitemaps': sitemaps}, name='sitemap-index'),
    path('sitemap-<section>.xml', sitemap_view, {'sitemaps': sitemaps}, name='django.contrib.sitemaps.views.sitemap'),

    # API Schema views (Spectacular)
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),

    # Redirect root URL to API Schema page
    path('', RedirectView.as_view(url='/api/schema/swagger-ui/', permanent=True), name='api-docs-redirect'),

    # Catalog app replaces products and categories - unified catalog routes
    path('catalog/', include('apps.catalog.urls', namespace='catalog')),
    # Legacy product/category routes redirect to catalog
    path('products/', include('apps.catalog.urls', namespace='catalog-products')),
    path('categories/', include('apps.catalog.urls', namespace='catalog-categories')),
    
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
    path('oauth/', include('social_django.urls', namespace='social')),
    
    # Email Service API
    path('email/', include('apps.email_service.urls', namespace='email_service')),

    # Register pages URLs under the 'contacts' namespace for backward compatibility
    # so templates using {% url 'contacts:contact' %} continue to work.
    path('', include(('apps.pages.urls', ''), namespace='home')),
    path('', include(('apps.pages.urls', 'contacts'), namespace='contacts')),
    path('', include(('apps.pages.urls', 'faq'), namespace='faq')),
    path('', include(('apps.pages.urls', 'about'), namespace='about')),
    path('', include(('apps.pages.urls', 'subscribe'), namespace='subscribe')),
    path('', include(('apps.pages.urls', 'detail'), namespace='detail')),

    # Pages catch-all (must come last)
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
