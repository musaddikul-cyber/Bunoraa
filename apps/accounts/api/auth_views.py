"""
Custom authentication views with MFA support.
"""
from django.utils.decorators import method_decorator
from django_ratelimit.decorators import ratelimit
from rest_framework import status
from rest_framework.response import Response
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.token_blacklist.models import OutstandingToken
from django.utils import timezone
from datetime import datetime

from apps.accounts.services import MfaService, AuthSessionService


class MfaTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Return MFA challenge when enabled for the user."""

    def validate(self, attrs):
        data = super().validate(attrs)
        user = self.user
        vault = MfaService._get_vault(user)
        if vault.mfa_enabled:
            methods = list(MfaService.available_methods(user))
            return {
                'mfa_required': True,
                'mfa_token': MfaService.create_mfa_token(user),
                'methods': methods,
            }
        data['mfa_required'] = False
        return data


@method_decorator(ratelimit(key='ip', rate='10/m', block=True), name='dispatch')
@method_decorator(ratelimit(key='post:email', rate='10/m', block=True), name='dispatch')
class MfaTokenObtainPairView(TokenObtainPairView):
    """Token endpoint that supports MFA step-up."""
    serializer_class = MfaTokenObtainPairSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        if data.get('mfa_required'):
            return Response({
                'success': True,
                'message': 'MFA required.',
                'data': data,
                'meta': None
            }, status=status.HTTP_200_OK)

        access = data.get('access')
        refresh = data.get('refresh')
        if access and refresh:
            try:
                refresh_obj = RefreshToken(refresh)
                access_obj = AccessToken(access)
                expires_at = datetime.fromtimestamp(refresh_obj['exp'], tz=timezone.utc)
                OutstandingToken.objects.get_or_create(
                    jti=str(refresh_obj['jti']),
                    user=serializer.user,
                    token=str(refresh_obj),
                    expires_at=expires_at,
                )
                AuthSessionService.create_session(
                    serializer.user,
                    request,
                    str(access_obj['jti']),
                    str(refresh_obj['jti'])
                )
            except Exception:
                pass

        return Response({
            'success': True,
            'message': 'Login successful.',
            'data': data,
            'meta': None
        }, status=status.HTTP_200_OK)
