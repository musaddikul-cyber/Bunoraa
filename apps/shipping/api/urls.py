"""
Shipping API URLs
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ShippingZoneViewSet, ShippingCarrierViewSet, ShippingMethodViewSet,
    ShippingRateCalculationView, ShipmentViewSet, TrackingView,
    ShippingSettingsView
)

router = DefaultRouter()
router.register(r'zones', ShippingZoneViewSet, basename='shipping-zone')
router.register(r'carriers', ShippingCarrierViewSet, basename='shipping-carrier')
router.register(r'methods', ShippingMethodViewSet, basename='shipping-method')
router.register(r'shipments', ShipmentViewSet, basename='shipment')

urlpatterns = [
    path('', include(router.urls)),
    path('calculate/', ShippingRateCalculationView.as_view(), name='shipping-calculate'),
    path('track/<str:tracking_number>/', TrackingView.as_view(), name='shipping-track'),
    path('settings/', ShippingSettingsView.as_view(), name='shipping-settings'),
]
