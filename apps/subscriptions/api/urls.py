from django.urls import path, include
from core.api.routers import SimpleRouter
from .views import PlanViewSet, SubscriptionViewSet, StripeWebhookView

router = SimpleRouter()
router.register(r"plans", PlanViewSet, basename="plans")

urlpatterns = [
    path("", include(router.urls)),
    path("", SubscriptionViewSet.as_view({'get': 'list', 'post': 'create'}), name='subscriptions-list'),
    path("<uuid:pk>/", SubscriptionViewSet.as_view({
        'get': 'retrieve',
        'patch': 'update',
        'put': 'update',
        'delete': 'destroy',
    }), name='subscriptions-detail'),
    path("<uuid:pk>/change_plan/", SubscriptionViewSet.as_view({'post': 'change_plan'}), name='subscriptions-change-plan'),
    path("<uuid:pk>/resume/", SubscriptionViewSet.as_view({'post': 'resume'}), name='subscriptions-resume'),
    path("<uuid:pk>/cancel/", SubscriptionViewSet.as_view({'post': 'cancel'}), name='subscriptions-cancel'),
    path("<uuid:pk>/preview_invoice/", SubscriptionViewSet.as_view({'get': 'preview_invoice'}), name='subscriptions-preview-invoice'),
    path("webhook/", StripeWebhookView.as_view(), name="stripe-webhook"),
]
