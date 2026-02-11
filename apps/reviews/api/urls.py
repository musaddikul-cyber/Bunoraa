"""
Reviews API URL configuration
"""
from django.urls import path, include
from core.api.routers import SimpleRouter
from .views import ReviewViewSet


router = SimpleRouter()
router.register(r'', ReviewViewSet, basename='reviews')

urlpatterns = [
    path('', include(router.urls)),
]
