"""
Orders API URL configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import OrderViewSet, OrderAdminViewSet


router = DefaultRouter()
router.register(r'orders', OrderViewSet, basename='order')

admin_router = DefaultRouter()
admin_router.register(r'orders', OrderAdminViewSet, basename='admin-order')

urlpatterns = [
    path('', OrderViewSet.as_view({'get': 'list'}), name='order-list-root'),
    path('<uuid:pk>/', OrderViewSet.as_view({'get': 'retrieve'}), name='order-detail-root'),
    path('<uuid:pk>/track/', OrderViewSet.as_view({'get': 'track'}), name='order-track-root'),
    path('<uuid:pk>/cancel/', OrderViewSet.as_view({'post': 'cancel'}), name='order-cancel-root'),
    path('', include(router.urls)),
]

# Admin URLs should be included separately in urls_api.py
admin_urlpatterns = [
    path('admin/', include(admin_router.urls)),
]
