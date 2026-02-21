from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import requests
from django.conf import settings


DEFAULT_PRERENDER_USER_AGENT = "Mozilla/5.0 (compatible; BunoraaPrerender/1.0)"


def is_prerender_enabled() -> bool:
    return bool(getattr(settings, "PRERENDER_ENABLED", False))


def get_cache_dir() -> Path:
    cache_dir = Path(settings.BASE_DIR) / getattr(settings, "PRERENDER_CACHE_DIR", "prerender_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def normalize_path(path: str) -> str:
    normalized = "/" + str(path or "").strip().lstrip("/")
    if not normalized.endswith("/"):
        normalized = f"{normalized}/"
    return normalized


def path_to_filename(path: str) -> str:
    key = path.strip("/").replace("/", "_") or "index"
    return f"{key}.html"


def prerender_paths(
    *,
    paths: list[str],
    timeout: int = 15,
    user_agent: str = DEFAULT_PRERENDER_USER_AGENT,
) -> tuple[int, list[tuple[str, str]], list[tuple[str, str]]]:
    site_url = getattr(settings, "SITE_URL", "https://bunoraa.com")
    cache_dir = get_cache_dir()
    headers = {"User-Agent": user_agent}

    saved = 0
    successes: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []

    for raw_path in paths:
        path = normalize_path(raw_path)
        url = urljoin(site_url, path.lstrip("/"))
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            output_path = cache_dir / path_to_filename(path)
            output_path.write_bytes(response.content)
            successes.append((path, str(output_path)))
            saved += 1
        except Exception as exc:  # noqa: BLE001 - command layer handles/report failures
            failures.append((url, str(exc)))

    return saved, successes, failures
