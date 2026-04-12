import socket
from functools import lru_cache

from django.conf import settings
from django.http import HttpResponse, HttpResponseNotModified

from apps.seo.services import (
    get_prerender_config,
    get_snapshot_state,
    is_path_allowed_for_prerender,
    is_prerender_enabled,
    load_prerender_snapshot,
    normalize_path,
    trigger_background_prerender,
)

DEFAULT_BOT_TOKENS = (
    "googlebot",
    "bingbot",
    "yandex",
    "baiduspider",
    "duckduckbot",
    "applebot",
    "facebot",
    "facebookexternalhit",
    "linkedinbot",
    "slackbot",
    "twitterbot",
)


def _client_ip(request) -> str:
    forwarded_for = (request.META.get("HTTP_X_FORWARDED_FOR") or "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return (request.META.get("REMOTE_ADDR") or "").strip()


def _user_agent(request) -> str:
    return (request.META.get("HTTP_USER_AGENT") or "").strip()


def _accepts_html(request) -> bool:
    accept = (request.META.get("HTTP_ACCEPT") or "").lower()
    if not accept:
        return True
    return "text/html" in accept or "*/*" in accept


def _has_allowed_query_string(request, allowed_query_keys: tuple[str, ...]) -> bool:
    if not request.GET:
        return True
    if not allowed_query_keys:
        return False
    allowed = set(allowed_query_keys)
    for key in request.GET.keys():
        if key not in allowed:
            return False
    return True


def _bot_tokens() -> tuple[str, ...]:
    configured = tuple(
        str(token).strip().lower()
        for token in getattr(settings, "PRERENDER_BOT_TOKENS", DEFAULT_BOT_TOKENS)
        if str(token).strip()
    )
    return configured or DEFAULT_BOT_TOKENS


def _is_known_bot(user_agent: str) -> bool:
    ua = user_agent.lower()
    return any(token in ua for token in _bot_tokens())


@lru_cache(maxsize=4096)
def _verify_google_dns(remote_ip: str) -> bool:
    if not remote_ip:
        return False
    try:
        host, _aliases, addresses = socket.gethostbyaddr(remote_ip)
    except Exception:
        return False
    host = (host or "").lower().rstrip(".")
    if not host.endswith(("googlebot.com", "google.com", "googleusercontent.com")):
        return False
    if remote_ip in addresses:
        return True
    try:
        _hostname, _aliaslist, forward_addresses = socket.gethostbyname_ex(host)
    except Exception:
        return False
    return remote_ip in forward_addresses


def _must_verify_google_dns(request, user_agent: str) -> bool:
    enabled = bool(getattr(settings, "PRERENDER_VERIFY_GOOGLE_DNS", False))
    return enabled and "googlebot" in user_agent.lower()


def _not_modified(request, etag: str) -> bool:
    if not etag:
        return False
    request_etag = (request.META.get("HTTP_IF_NONE_MATCH") or "").strip()
    if not request_etag:
        return False
    if request_etag == "*":
        return True
    incoming = [token.strip() for token in request_etag.split(",") if token.strip()]
    return etag in incoming


def _set_vary(response, values: tuple[str, ...]) -> None:
    existing = response.get("Vary", "")
    current = {item.strip() for item in existing.split(",") if item.strip()}
    current.update(values)
    response["Vary"] = ", ".join(sorted(current))


def _apply_snapshot_headers(response: HttpResponse, snapshot: dict, state: str) -> None:
    etag = str(snapshot.get("etag") or "").strip()
    last_modified = str(snapshot.get("last_modified") or "").strip()
    fresh_ttl = int(snapshot.get("fresh_ttl_seconds") or 0)
    swr_ttl = int(snapshot.get("stale_while_revalidate_seconds") or 0)
    sie_ttl = int(snapshot.get("stale_if_error_seconds") or 0)
    cache_control = (
        str(snapshot.get("source_cache_control") or "").strip()
        or f"public, max-age=0, s-maxage={max(fresh_ttl, 0)}, stale-while-revalidate={max(swr_ttl, 0)}, stale-if-error={max(sie_ttl, 0)}"
    )
    response["Cache-Control"] = cache_control
    response["X-PreRendered"] = "1"
    response["X-PreRendered-State"] = state
    if etag:
        response["ETag"] = etag
    if last_modified:
        response["Last-Modified"] = last_modified
    _set_vary(response, ("User-Agent", "Accept-Encoding"))


class BotPreRenderMiddleware:
    """Serve file-cached prerender snapshots to known bots.

    Behavior:
    - GET/HEAD only
    - anonymous requests only
    - bot UA match (+ optional Google reverse+forward DNS verification)
    - path allowlist via PRERENDER_PATHS
    - stale-while-revalidate and stale-if-error serving
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not is_prerender_enabled():
            return self.get_response(request)

        if request.method not in ("GET", "HEAD"):
            return self.get_response(request)
        if getattr(request, "user", None) and request.user.is_authenticated:
            return self.get_response(request)
        if not _accepts_html(request):
            return self.get_response(request)

        user_agent = _user_agent(request)
        if not user_agent or not _is_known_bot(user_agent):
            return self.get_response(request)

        if _must_verify_google_dns(request, user_agent):
            if not _verify_google_dns(_client_ip(request)):
                return self.get_response(request)

        config = get_prerender_config()
        if not _has_allowed_query_string(request, config.allowed_query_keys):
            return self.get_response(request)

        normalized_path = normalize_path(
            request.get_full_path(),
            allow_query=True,
            allowed_query_keys=config.allowed_query_keys,
        )
        normalized_path_only = normalize_path(request.path, allow_query=False)
        if not is_path_allowed_for_prerender(normalized_path_only):
            return self.get_response(request)

        snapshot = load_prerender_snapshot(normalized_path)
        if not snapshot:
            if config.on_demand_enabled:
                trigger_background_prerender(normalized_path, force=False)
            return self.get_response(request)

        state = get_snapshot_state(snapshot)
        if state == "expired":
            if config.on_demand_enabled:
                trigger_background_prerender(normalized_path, force=True)
            return self.get_response(request)
        if state in ("stale-while-revalidate", "stale-if-error") and config.on_demand_enabled:
            trigger_background_prerender(normalized_path, force=True)

        if _not_modified(request, str(snapshot.get("etag") or "")):
            not_modified = HttpResponseNotModified()
            _apply_snapshot_headers(not_modified, snapshot, state)
            return not_modified

        content = b"" if request.method == "HEAD" else (snapshot.get("content") or b"")
        response = HttpResponse(
            content,
            content_type=snapshot.get("content_type") or "text/html; charset=utf-8",
            status=200,
        )
        _apply_snapshot_headers(response, snapshot, state)
        return response
