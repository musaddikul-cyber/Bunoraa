"""
Promotions API URL configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CouponViewSet, BannerViewSet, SaleViewSet


router = DefaultRouter()
router.register(r'coupons', CouponViewSet, basename='coupon')
router.register(r'banners', BannerViewSet, basename='banner')
router.register(r'sales', SaleViewSet, basename='sale')

urlpatterns = [
    path('', include(router.urls)),
]
