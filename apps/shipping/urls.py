"""
Shipping URL Configuration
"""
from django.urls import path, include

app_name = 'shipping'

urlpatterns = [
    path('api/v1/shipping/', include('apps.shipping.api.urls')),
]
