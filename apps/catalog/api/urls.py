"""
Catalog API URL Configuration
"""
from django.urls import path, include, re_path
from django.views.generic import RedirectView
from core.api.routers import DefaultRouter

from .views import (
    CategoryViewSet, ProductViewSet, CollectionViewSet, BundleViewSet,
    TagViewSet, SpotlightViewSet, BadgeViewSet,
    SearchAPIView, HomepageDataView, CustomerPhotoUploadView,
    ProductQuestionListView, ProductAnswerCreateView
)

router = DefaultRouter()
router.register(r'categories', CategoryViewSet, basename='api-category')
router.register(r'products', ProductViewSet, basename='api-product')
router.register(r'collections', CollectionViewSet, basename='api-collection')
router.register(r'bundles', BundleViewSet, basename='api-bundle')
router.register(r'tags', TagViewSet, basename='api-tag')
router.register(r'spotlights', SpotlightViewSet, basename='api-spotlight')
router.register(r'badges', BadgeViewSet, basename='api-badge')

urlpatterns = [
    # Guard against relative favicon/static requests being interpreted as nested category slugs.
    path(
        'categories/static/images/assets/favicon.ico/',
        RedirectView.as_view(url='/static/images/assets/favicon.ico', permanent=False),
        name='api-category-favicon-redirect',
    ),
    path(
        'categories/static/images/assets/favicon.ico',
        RedirectView.as_view(url='/static/images/assets/favicon.ico', permanent=False),
        name='api-category-favicon-redirect-no-slash',
    ),
    re_path(
        r'^categories/static/(?P<asset_path>.+)$',
        RedirectView.as_view(url='/static/images/assets/favicon.ico', permanent=False),
        name='api-category-static-asset-redirect',
    ),

    # Router URLs
    path('', include(router.urls)),

    # Nested category path support for detail actions
    path('categories/<path:slug>/children/', CategoryViewSet.as_view({'get': 'children'}), name='api-category-children-by-path'),
    path('categories/<path:slug>/facets/', CategoryViewSet.as_view({'get': 'facets'}), name='api-category-facets-by-path'),
    path('categories/<path:slug>/products/', CategoryViewSet.as_view({'get': 'products'}), name='api-category-products-by-path'),
    path('categories/<path:slug>/', CategoryViewSet.as_view({'get': 'retrieve'}), name='api-category-detail-by-path'),
    
    # Standalone views
    path('search/', SearchAPIView.as_view(), name='api-search'),
    path('homepage/', HomepageDataView.as_view(), name='api-homepage'),
    path('customer-photos/upload/', CustomerPhotoUploadView.as_view(), name='api-customer-photo-upload'),
    
    # Q&A
    path('products/<uuid:product_pk>/questions/', ProductQuestionListView.as_view({'get': 'list', 'post': 'ask_question'}), name='api-product-questions'),
    path('questions/<uuid:question_pk>/answers/', ProductAnswerCreateView.as_view({'post': 'add_answer'}), name='api-question-answers'),
]
