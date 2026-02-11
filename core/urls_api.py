"""
API URL Configuration - Version 1
All API endpoints under /api/v1/
    """
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenRefreshView,
    TokenVerifyView,
)
from apps.accounts.api.auth_views import MfaTokenObtainPairView

app_name = 'api'

urlpatterns = [
    # Authentication
    path('auth/token/', MfaTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    
    # ML/AI APIs (comprehensive ML services)
    # Disabled: ML module requires torch and other ML packages not installed on free tier
    # path('ml/', include('ml.api.urls')),
    
    # App APIs
    path('accounts/', include('apps.accounts.api.urls')),
    path('analytics/', include('apps.analytics.api.urls')),
    path('catalog/', include('apps.catalog.api.urls')),
    path('chat/', include('apps.chat.urls')),
    path('commerce/', include('apps.commerce.api.urls')),
    path('contacts/', include('apps.contacts.api.urls')),
    path('i18n/', include('apps.i18n.api.urls')),
    path('email/', include('apps.email_service.api_urls')),
    path('notifications/', include('apps.notifications.api.urls')),
    path('orders/', include('apps.orders.api.urls')),
    path('pages/', include('apps.pages.api.urls')),
    path('payments/', include('apps.payments.api.urls')),
    path('preorders/', include('apps.preorders.api.urls')),
    path('promotions/', include('apps.promotions.api.urls')),
    path('reviews/', include('apps.reviews.api.urls')),
    path('recommendations/', include('apps.recommendations.urls')),
    path('referral/', include('apps.referral.api.urls')),
    path('shipping/', include('apps.shipping.api.urls')),
    path('subscriptions/', include('apps.subscriptions.api.urls')),
]
