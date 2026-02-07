"""
Pre-orders API URL configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    PreOrderCategoryViewSet, PreOrderViewSet, PreOrderTemplateViewSet,
    PreOrderPriceCalculatorAPIView, PreOrderStatisticsAPIView,
    AdminPreOrderViewSet, PreOrderTrackAPIView
)

router = DefaultRouter()
router.register(r'categories', PreOrderCategoryViewSet, basename='preorder-category')
router.register(r'orders', PreOrderViewSet, basename='preorder')
router.register(r'templates', PreOrderTemplateViewSet, basename='preorder-template')
router.register(r'admin/orders', AdminPreOrderViewSet, basename='admin-preorder')

urlpatterns = [
    path('', include(router.urls)),
    path('calculate-price/', PreOrderPriceCalculatorAPIView.as_view(), name='calculate-price'),
    path('statistics/', PreOrderStatisticsAPIView.as_view(), name='statistics'),
    path('track/', PreOrderTrackAPIView.as_view(), name='track'),
]
