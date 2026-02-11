"""
Payments API URL configuration
"""
from django.urls import path, include
from core.api.routers import DefaultRouter

from .views import (
    PaymentViewSet, PaymentMethodViewSet, RefundAdminViewSet, PaymentGatewayViewSet,
    stripe_webhook, sslcommerz_ipn, bkash_webhook, nagad_webhook,
    PaymentLinkViewSet, BNPLViewSet, RecurringChargeAdminViewSet
)

router = DefaultRouter()
# Register specific prefixes before the root payments viewset to avoid
# the root detail route shadowing nested endpoints like /methods/.
router.register(r'methods', PaymentMethodViewSet, basename='payment-methods')
router.register(r'gateways', PaymentGatewayViewSet, basename='payment-gateways')
router.register(r'admin/refunds', RefundAdminViewSet, basename='admin-refunds')
router.register(r'links', PaymentLinkViewSet, basename='payment-links')
router.register(r'bnpl', BNPLViewSet, basename='bnpl')
router.register(r'admin/recurring', RecurringChargeAdminViewSet, basename='admin-recurring')
router.register(r'', PaymentViewSet, basename='payments')

urlpatterns = [
    path('', include(router.urls)),
    path('stripe/webhook/', stripe_webhook, name='stripe-webhook'),
    path('sslcommerz/ipn/', sslcommerz_ipn, name='sslcommerz-ipn'),
    path('bkash/webhook/', bkash_webhook, name='bkash-webhook'),
    path('nagad/webhook/', nagad_webhook, name='nagad-webhook'),
]
