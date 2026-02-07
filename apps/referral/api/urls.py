from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserReferralCodeView, ReferralRewardViewSet, ApplyReferralCodeView

router = DefaultRouter()
router.register(r'rewards', ReferralRewardViewSet, basename='referral-reward')

urlpatterns = [
    path('', include(router.urls)),
    path('my-code/', UserReferralCodeView.as_view({'get': 'retrieve'}), name='user-referral-code'),
    path('apply-code/', ApplyReferralCodeView.as_view({'post': 'create'}), name='apply-referral-code'),
]
