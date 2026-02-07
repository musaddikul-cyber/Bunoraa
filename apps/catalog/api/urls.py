"""
Catalog API URL Configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

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
    # Router URLs
    path('', include(router.urls)),
    
    # Standalone views
    path('search/', SearchAPIView.as_view(), name='api-search'),
    path('homepage/', HomepageDataView.as_view(), name='api-homepage'),
    path('customer-photos/upload/', CustomerPhotoUploadView.as_view(), name='api-customer-photo-upload'),
    
    # Q&A
    path('products/<uuid:product_pk>/questions/', ProductQuestionListView.as_view({'get': 'list', 'post': 'ask_question'}), name='api-product-questions'),
    path('questions/<uuid:question_pk>/answers/', ProductAnswerCreateView.as_view({'post': 'add_answer'}), name='api-question-answers'),
]
