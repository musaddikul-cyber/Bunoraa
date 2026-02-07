"""
Internationalization URLs

URL patterns for i18n views.
"""
from django.urls import path, include

from . import views

app_name = 'i18n'

urlpatterns = [
    # Language switching
    path('language/set/<str:code>/', views.set_language, name='set_language'),
    path('language/detect/', views.detect_language, name='detect_language'),
    
    # Currency switching
    path('currency/set/<str:code>/', views.set_currency, name='set_currency'),
    path('currency/detect/', views.detect_currency, name='detect_currency'),
    
    # Timezone
    path('timezone/set/<str:name>/', views.set_timezone, name='set_timezone'),
    path('timezone/detect/', views.detect_timezone, name='detect_timezone'),
    
    # Country/region selection
    path('country/set/<str:code>/', views.set_country, name='set_country'),
    path('country/detect/', views.detect_country, name='detect_country'),
    
    # Geographic data (AJAX endpoints for address forms)
    path('geo/divisions/<str:country_code>/', views.get_divisions, name='get_divisions'),
    path('geo/districts/<str:division_id>/', views.get_districts, name='get_districts'),
    path('geo/upazilas/<str:district_id>/', views.get_upazilas, name='get_upazilas'),
    
    # Currency conversion
    path('convert/', views.convert_currency, name='convert_currency'),
    path('exchange-rates/', views.get_exchange_rates, name='get_exchange_rates'),
    
    # User preferences (requires authentication)
    path('preferences/', views.user_preferences, name='user_preferences'),
    path('preferences/update/', views.update_preferences, name='update_preferences'),
    
    # Translation
    path('translations/<str:namespace>/', views.get_translations, name='get_translations'),
    
    # API routes (if using REST framework)
    path('api/', include('apps.i18n.api.urls')),
]
