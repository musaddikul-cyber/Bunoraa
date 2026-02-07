"""
Analytics API URL configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import DashboardViewSet, DailyStatViewSet, TrackingViewSet, PublicAnalyticsViewSet

router = DefaultRouter()
router.register(r'dashboard', DashboardViewSet, basename='dashboard')
router.register(r'daily', DailyStatViewSet, basename='daily-stats')
router.register(r'track', TrackingViewSet, basename='tracking')

# Create a public analytics instance for direct path access
public_analytics = PublicAnalyticsViewSet.as_view({
    'get': 'active_visitors'
}, name='active-visitors')

public_purchases = PublicAnalyticsViewSet.as_view({
    'get': 'recent_purchases'
}, name='recent-purchases')

urlpatterns = [
    path('', include(router.urls)),
    path('active-visitors/', public_analytics, name='active-visitors'),
    path('recent-purchases/', public_purchases, name='recent-purchases'),
]
