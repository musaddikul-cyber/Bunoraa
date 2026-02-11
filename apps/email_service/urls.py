"""
Email Service URL Configuration
================================

URL patterns for public tracking/unsubscribe endpoints.
"""

from django.urls import path

from . import views

app_name = 'email_service'

urlpatterns = [
    # Tracking endpoints (public, no auth)
    path('track/open/<str:message_id>/pixel.gif', views.TrackOpenView.as_view(), name='track_open'),
    path('track/click/<str:message_id>/<str:url>/', views.TrackClickView.as_view(), name='track_click'),
    
    # Unsubscribe endpoint (public)
    path('unsubscribe/<str:message_id>/', views.UnsubscribeView.as_view(), name='unsubscribe'),
]
