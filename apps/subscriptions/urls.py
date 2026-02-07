"""Subscriptions URL Configuration"""
from django.urls import path
from . import views

app_name = 'subscriptions'

urlpatterns = [
    # Landing and plans
    path('', views.SubscriptionLandingView.as_view(), name='landing'),
    path('plans/<uuid:pk>/', views.PlanDetailView.as_view(), name='plan-detail'),
    
    # User subscriptions
    path('my-subscriptions/', views.MySubscriptionsView.as_view(), name='my-subscriptions'),
    path('subscription/<uuid:pk>/', views.SubscriptionDetailView.as_view(), name='subscription-detail'),
    
    # Actions
    path('subscribe/<uuid:plan_id>/', views.SubscribeView.as_view(), name='subscribe'),
    path('subscription/<uuid:pk>/cancel/', views.CancelSubscriptionView.as_view(), name='cancel'),
    path('subscription/<uuid:pk>/reactivate/', views.ReactivateSubscriptionView.as_view(), name='reactivate'),
    path('subscription/<uuid:pk>/change-plan/', views.ChangePlanView.as_view(), name='change-plan'),
    
    # API
    path('api/status/', views.SubscriptionStatusAPIView.as_view(), name='api-status'),
]
