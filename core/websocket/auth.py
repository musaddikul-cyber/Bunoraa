"""
JWT-aware WebSocket authentication middleware.

Keeps default session auth behavior (AuthMiddlewareStack) and falls back to
JWT token authentication when no authenticated session user is present.
"""
from __future__ import annotations

from urllib.parse import parse_qs

import logging

from channels.db import database_sync_to_async
from channels.auth import AuthMiddleware
from channels.sessions import CookieMiddleware, SessionMiddleware
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.db import DatabaseError
from django.db.utils import OperationalError
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import UntypedToken

User = get_user_model()
logger = logging.getLogger(__name__)


@database_sync_to_async
def _get_user_from_validated_token(validated_token):
    user_id_claim = settings.SIMPLE_JWT.get("USER_ID_CLAIM", "user_id")
    user_id = validated_token.get(user_id_claim)
    if not user_id:
        return AnonymousUser()
    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return AnonymousUser()
    if not getattr(user, "is_active", True):
        return AnonymousUser()
    return user


def _extract_ws_token(scope) -> str | None:
    query_string = (scope.get("query_string") or b"").decode("utf-8")
    query_params = parse_qs(query_string)
    token = (query_params.get("token") or query_params.get("access_token") or [None])[0]
    if token:
        return token

    # Support Authorization header for non-browser websocket clients.
    for key, value in scope.get("headers", []):
        if key == b"authorization":
            try:
                raw = value.decode("utf-8")
            except Exception:
                return None
            if raw.lower().startswith("bearer "):
                return raw.split(" ", 1)[1].strip()
    return None


class JWTAuthMiddleware:
    """
    Add JWT auth fallback for websocket scopes.

    Expected usage: wrap this around AuthMiddlewareStack so session auth still
    works, then promote user from JWT if scope user is anonymous.
    """

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        user = scope.get("user")
        if user is None or not getattr(user, "is_authenticated", False):
            token = _extract_ws_token(scope)
            if token:
                try:
                    validated_token = UntypedToken(token)
                    scope["user"] = await _get_user_from_validated_token(validated_token)
                except TokenError:
                    scope["user"] = AnonymousUser()
                except Exception:
                    scope["user"] = AnonymousUser()
        return await self.inner(scope, receive, send)


class SafeAuthMiddleware(AuthMiddleware):
    """Auth middleware that degrades to AnonymousUser on DB errors."""

    async def resolve_scope(self, scope):
        try:
            await super().resolve_scope(scope)
        except (DatabaseError, OperationalError) as exc:
            scope["user"]._wrapped = AnonymousUser()
            logger.warning("WebSocket auth fallback to anonymous due to DB error: %s", exc)


def SafeAuthMiddlewareStack(inner):
    return CookieMiddleware(SessionMiddleware(SafeAuthMiddleware(inner)))


def JWTAuthMiddlewareStack(inner):
    """Session auth + JWT fallback stack."""
    return SafeAuthMiddlewareStack(JWTAuthMiddleware(inner))
