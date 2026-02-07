"""
Contacts URL Configuration
"""
from django.urls import path, include

app_name = 'contacts'

urlpatterns = [
    path('api/', include('apps.contacts.api.urls')),
]
