"""
Orders URL configuration
"""
from django.urls import path
from . import views


app_name = 'orders'

urlpatterns = [
    path('', views.OrderListView.as_view(), name='list'),
    path('track/', views.OrderTrackView.as_view(), name='track'),
    path('<uuid:pk>/', views.OrderDetailView.as_view(), name='detail'),
    path('<str:order_number>/track/', views.OrderTrackView.as_view(), name='track_detail'),
    path('<str:order_number>/invoice/', views.OrderInvoiceView.as_view(), name='invoice'),
]
