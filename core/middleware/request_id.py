"""
Request ID middleware for log correlation and tracing.
"""

from __future__ import annotations

import re
import uuid

from core.request_context import reset_request_id, set_request_id


_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _coerce_request_id(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if not _SAFE_REQUEST_ID.match(value):
        return None
    return value


class RequestIdMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        incoming = _coerce_request_id(request.headers.get("X-Request-ID"))
        request_id = incoming or uuid.uuid4().hex
        token = set_request_id(request_id)
        request.request_id = request_id
        try:
            response = self.get_response(request)
        finally:
            reset_request_id(token)

        # Expose request ID to clients for easier support/debugging.
        try:
            response["X-Request-ID"] = request_id
        except Exception:
            pass
        return response

