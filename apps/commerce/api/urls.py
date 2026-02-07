"""
Commerce API URL Configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register(r'cart', views.CartViewSet, basename='cart')
router.register(r'wishlist', views.WishlistViewSet, basename='wishlist')
router.register(r'checkout', views.CheckoutViewSet, basename='checkout')

urlpatterns = [
    path('', include(router.urls)),
    path('wishlist/shared/<str:token>/', views.SharedWishlistView.as_view(), name='shared-wishlist'),
]
