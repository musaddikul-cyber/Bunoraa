"""
Internationalization API URLs

URL patterns for i18n REST API.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    LanguageViewSet, CurrencyViewSet, ExchangeRateViewSet,
    CurrencyConversionView, TimezoneViewSet, CountryViewSet,
    DivisionViewSet, DistrictViewSet, UpazilaViewSet,
    TranslationNamespaceViewSet, TranslationKeyViewSet,
    TranslationViewSet, ContentTranslationViewSet,
    UserLocalePreferenceView
)

app_name = 'i18n_api'

router = DefaultRouter()
router.register('languages', LanguageViewSet, basename='language')
router.register('currencies', CurrencyViewSet, basename='currency')
router.register('exchange-rates', ExchangeRateViewSet, basename='exchange-rate')
router.register('timezones', TimezoneViewSet, basename='timezone')
router.register('countries', CountryViewSet, basename='country')
router.register('divisions', DivisionViewSet, basename='division')
router.register('districts', DistrictViewSet, basename='district')
router.register('upazilas', UpazilaViewSet, basename='upazila')
router.register('namespaces', TranslationNamespaceViewSet, basename='namespace')
router.register('keys', TranslationKeyViewSet, basename='key')
router.register('translations', TranslationViewSet, basename='translation')
router.register('content-translations', ContentTranslationViewSet, basename='content-translation')

urlpatterns = [
    # Router URLs
    path('', include(router.urls)),
    
    # Custom views
    path('convert/', CurrencyConversionView.as_view(), name='convert'),
    path('preferences/', UserLocalePreferenceView.as_view(), name='preferences'),
]
