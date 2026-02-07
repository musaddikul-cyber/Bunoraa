"""
Payments URL configuration
"""
from django.urls import path

from . import views

app_name = 'payments'

urlpatterns = [
	path('methods/', views.PaymentMethodsView.as_view(), name='payment_methods'),
	path('ipn/', views.GatewayIPNView.as_view(), name='gateway-ipn'),
]
