"""
Contacts API URLs
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    ContactCategoryViewSet, ContactInquiryView, UserInquiriesView,
    InquiryDetailView, StoreLocationViewSet, NearbyLocationsView,
    ContactSettingsView, CustomizationRequestView
)


router = DefaultRouter()
router.register(r'categories', ContactCategoryViewSet, basename='contact-category')
router.register(r'locations', StoreLocationViewSet, basename='store-location')

urlpatterns = [
    path('', include(router.urls)),
    path('inquiries/', ContactInquiryView.as_view(), name='contact-inquiry'),
    path('inquiries/mine/', UserInquiriesView.as_view(), name='user-inquiries'),
    path('inquiries/<uuid:inquiry_id>/', InquiryDetailView.as_view(), name='inquiry-detail'),
    path('locations/nearby/', NearbyLocationsView.as_view(), name='nearby-locations'),
    path('settings/', ContactSettingsView.as_view(), name='contact-settings'),
    path('customization-requests/', CustomizationRequestView.as_view(), name='customization-request'),
]
