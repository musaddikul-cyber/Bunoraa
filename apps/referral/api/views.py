from rest_framework import viewsets, mixins, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from django.db import transaction
from django.utils import timezone

from apps.referral.models import ReferralCode, ReferralReward
from .serializers import ReferralCodeSerializer, ReferralRewardSerializer


class UserReferralCodeView(mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    """
    API endpoint for an authenticated user to view their referral code.
    """
    queryset = ReferralCode.objects.all()
    serializer_class = ReferralCodeSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        # Ensure a referral code exists for the authenticated user
        referral_code, created = ReferralCode.objects.get_or_create(user=self.request.user)
        return referral_code

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)


class ReferralRewardViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    """
    API endpoint for an authenticated user to view their referral rewards.
    """
    serializer_class = ReferralRewardSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return ReferralReward.objects.filter(referrer_user=self.request.user)


class ApplyReferralCodeView(mixins.CreateModelMixin, viewsets.GenericViewSet):
    """
    API endpoint for new users to apply a referral code.
    This would typically be called during signup or checkout.
    """
    serializer_class = ReferralCodeSerializer # Reusing for validation of code presence
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        code_string = request.data.get('code')
        if not code_string:
            return Response({'detail': 'Referral code is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            referral_code = ReferralCode.objects.get(code=code_string, is_active=True, expires_at__gte=timezone.now())
        except ReferralCode.DoesNotExist:
            return Response({'detail': 'Invalid or expired referral code.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Here you would typically link this code to the new user's session
        # or the current checkout process. For demonstration, we'll just
        # acknowledge its validity.
        
        # Example: Store code in session for later use during user creation/order
        request.session['referral_code_applied'] = code_string
        
        return Response({'detail': 'Referral code applied successfully.'}, status=status.HTTP_200_OK)
