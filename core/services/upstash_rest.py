"""
Upstash Redis REST helper.

Uses the official Upstash REST SDK when configured.
"""
from __future__ import annotations

import os
import time
from typing import Optional


_client = None
_client_error: Optional[str] = None


def _load_env() -> tuple[str, str]:
    url = os.environ.get("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "").strip()
    return url, token


def get_client():
    """Return a cached Upstash REST client or None if not configured."""
    global _client, _client_error
    if _client is not None:
        return _client
    if _client_error:
        return None

    url, token = _load_env()
    if not url or not token:
        _client_error = "missing-env"
        return None

    try:
        from upstash_redis import Redis  # type: ignore
    except Exception:
        _client_error = "missing-dependency"
        return None

    _client = Redis(url=url, token=token)
    return _client


def health_check() -> dict[str, object]:
    """Check Upstash REST connectivity."""
    url, token = _load_env()
    if not url or not token:
        return {"status": "skipped", "reason": "UPSTASH_REDIS_REST_URL or token not set"}

    client = get_client()
    if not client:
        return {"status": "skipped", "reason": "upstash-redis not installed"}

    try:
        start = time.time()
        if hasattr(client, "ping"):
            client.ping()
        else:
            client.get("__healthcheck__")
        latency = round((time.time() - start) * 1000, 2)
        return {"status": "ok", "latency_ms": latency}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
