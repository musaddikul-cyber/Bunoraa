"""
Authentication helpers for Bunoraa API.
"""
from __future__ import annotations

import logging

from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

logger = logging.getLogger("bunoraa")


class OptionalJWTAuthentication(JWTAuthentication):
    """
    JWT auth that ignores invalid/expired tokens and treats the request as anonymous.

    This prevents public endpoints from failing when a stale token is present.
    Protected endpoints will still enforce permissions and return 401 as needed.
    """

    def authenticate(self, request):
        header = self.get_header(request)
        if header is None:
            return None

        raw_token = self.get_raw_token(header)
        if raw_token is None:
            return None

        try:
            validated_token = self.get_validated_token(raw_token)
        except (InvalidToken, TokenError) as exc:
            # Ignore invalid tokens so AllowAny/ReadOnly endpoints can proceed.
            logger.debug("Ignoring invalid JWT token: %s", exc)
            return None

        return self.get_user(validated_token), validated_token
