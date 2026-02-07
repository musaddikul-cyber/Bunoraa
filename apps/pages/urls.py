"""
Pages URL configuration
"""
from django.urls import path

from . import views

app_name = 'pages'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('contact/', views.ContactView.as_view(), name='contact'),
    path('faq/', views.FAQListView.as_view(), name='faq'),
    path('about/', views.AboutView.as_view(), name='about'),
    path('subscribe/', views.subscribe_newsletter, name='subscribe'),
    path('<slug:slug>/', views.PageDetailView.as_view(), name='detail'),
]
