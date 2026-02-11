"""
Orders API URL configuration
"""
from django.urls import path, include
from rest_framework.routers import SimpleRouter
from .views import OrderViewSet, OrderAdminViewSet


router = SimpleRouter()
router.register(r'', OrderViewSet, basename='order')

admin_router = SimpleRouter()
admin_router.register(r'', OrderAdminViewSet, basename='admin-order')

urlpatterns = [
    path('', include(router.urls)),
]

# Admin URLs should be included separately in urls_api.py
admin_urlpatterns = [
    path('admin/', include(admin_router.urls)),
]
