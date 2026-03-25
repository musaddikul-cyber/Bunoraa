"""
Custom JWT views to harden token refresh handling.
"""
from __future__ import annotations

from django.contrib.auth import get_user_model
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.serializers import TokenRefreshSerializer
from rest_framework_simplejwt.views import TokenRefreshView


class SafeTokenRefreshSerializer(TokenRefreshSerializer):
    """
    Token refresh serializer that converts missing user records to 401 errors.
    """

    def validate(self, attrs):
        try:
            return super().validate(attrs)
        except get_user_model().DoesNotExist:
            raise AuthenticationFailed(
                {"detail": "User not found", "code": "user_not_found"}
            )


class SafeTokenRefreshView(TokenRefreshView):
    serializer_class = SafeTokenRefreshSerializer

