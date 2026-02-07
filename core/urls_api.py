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
    
    # Catalog API (consolidated categories + products)
    path('catalog/', include('apps.catalog.api.urls')),
    path('recommendations/', include('apps.recommendations.urls')),
    
    # ML/AI APIs (comprehensive ML services)
    # Disabled: ML module requires torch and other ML packages not installed on free tier
    # path('ml/', include('ml.api.urls')),
    
    # Commerce API (cart, wishlist, checkout)
    path('commerce/', include('apps.commerce.api.urls')),
    
    # App APIs
    path('accounts/', include('apps.accounts.api.urls')),
    # path('categories/', include('apps.categories.api.urls')),  # Merged into catalog API
    # path('products/', include('apps.products.api.urls')),  # Deprecated - merged into catalog API
    # Deprecated - replaced by commerce API with unified cart, wishlist, checkout
    path('orders/', include('apps.orders.api.urls')),
    path('payments/', include('apps.payments.api.urls')),
    path('pages/', include('apps.pages.api.urls')),
    path('promotions/', include('apps.promotions.api.urls')),
    path('reviews/', include('apps.reviews.api.urls')),
    path('notifications/', include('apps.notifications.api.urls')),
    path('analytics/', include('apps.analytics.api.urls')),
    path('shipping/', include('apps.shipping.api.urls')),
    # path('wishlist/', include('apps.wishlist.api.urls')),  # Deprecated
    path('i18n/', include('apps.i18n.api.urls')),  # Unified i18n (replaces currencies and localization)
    path('contacts/', include('apps.contacts.api.urls')),
    path('referral/', include('apps.referral.api.urls')), 
    path('preorders/', include('apps.preorders.api.urls')),
    path('subscriptions/', include('apps.subscriptions.api.urls')),
    path('chat/', include('apps.chat.urls')),  # Live chat system
]
