"""
URLs for the artisans app.
"""
from django.urls import path
from .views import ArtisanListView, ArtisanDetailView

app_name = 'artisans'

urlpatterns = [
    path('', ArtisanListView.as_view(), name='artisan_list'),
    path('<slug:slug>/', ArtisanDetailView.as_view(), name='artisan_detail'),
]
