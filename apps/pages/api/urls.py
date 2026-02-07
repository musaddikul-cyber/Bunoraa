"""
Pages API URL configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    PageViewSet, FAQViewSet, ContactMessageViewSet,
    SiteSettingsViewSet, SubscriberViewSet
)

router = DefaultRouter()
router.register(r'pages', PageViewSet, basename='pages')
router.register(r'faqs', FAQViewSet, basename='faqs')
router.register(r'contact', ContactMessageViewSet, basename='contact')
router.register(r'settings', SiteSettingsViewSet, basename='settings')
router.register(r'subscribers', SubscriberViewSet, basename='subscribers')

urlpatterns = [
    path('', include(router.urls)),
]
