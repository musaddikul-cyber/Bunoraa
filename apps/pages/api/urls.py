"""
Pages API URL configuration
"""
from django.urls import path, include
from rest_framework.routers import SimpleRouter

from .views import (
    PageViewSet, FAQViewSet, ContactMessageViewSet,
    SiteSettingsViewSet, SubscriberViewSet
)

router = SimpleRouter()
router.register(r'faqs', FAQViewSet, basename='faqs')
router.register(r'contact', ContactMessageViewSet, basename='contact')
router.register(r'settings', SiteSettingsViewSet, basename='settings')
router.register(r'subscribers', SubscriberViewSet, basename='subscribers')

urlpatterns = [
    path('', PageViewSet.as_view({'get': 'list'}), name='pages-list'),
    path('menu/', PageViewSet.as_view({'get': 'menu'}), name='pages-menu'),
    path('footer/', PageViewSet.as_view({'get': 'footer'}), name='pages-footer'),
    path('', include(router.urls)),
    path('<slug:slug>/', PageViewSet.as_view({'get': 'retrieve'}), name='pages-detail'),
]
