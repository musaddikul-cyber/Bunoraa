"""
Pre-orders API URL configuration
"""
from django.urls import path, include
from core.api.routers import SimpleRouter
from .views import (
    PreOrderCategoryViewSet, PreOrderViewSet, PreOrderTemplateViewSet,
    PreOrderPriceCalculatorAPIView, PreOrderStatisticsAPIView,
    AdminPreOrderViewSet, PreOrderTrackAPIView
)

router = SimpleRouter()
router.register(r'categories', PreOrderCategoryViewSet, basename='preorder-category')
router.register(r'templates', PreOrderTemplateViewSet, basename='preorder-template')
router.register(r'admin', AdminPreOrderViewSet, basename='admin-preorder')
# Keep catch-all preorder routes last so they do not shadow static endpoints.
router.register(r'', PreOrderViewSet, basename='preorder')

urlpatterns = [
    path('calculate-price/', PreOrderPriceCalculatorAPIView.as_view(), name='calculate-price'),
    path('statistics/', PreOrderStatisticsAPIView.as_view(), name='statistics'),
    path('track/', PreOrderTrackAPIView.as_view(), name='track'),
    path('', include(router.urls)),
]
