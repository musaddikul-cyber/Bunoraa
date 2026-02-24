from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from django.conf import settings
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)


CHALLENGE_RE = re.compile(
    r"(captcha|verify (you are|you're) (human|not a robot)|access denied|challenge|unusual traffic|bot detection)",
    re.I,
)
DEFAULT_PROVIDER_ORDER = "searxng,bing_html,duckduckgo,brave_api,google_cse,serpapi"


class SearchProviderError(Exception):
    pass


class SearchBlockedError(SearchProviderError):
    pass


class SearchUnavailableError(SearchProviderError):
    pass


class SearchProvider:
    """
    Multi-provider search with provider health, challenge detection, and cooldown.
    """

    def __init__(self):
        raw_order = getattr(settings, "PRODUCT_AI_SEARCH_PROVIDER_ORDER", DEFAULT_PROVIDER_ORDER)
        provider_order = [part.strip().lower() for part in str(raw_order).split(",") if part.strip()]
        self.provider_order = provider_order or [item.strip() for item in DEFAULT_PROVIDER_ORDER.split(",")]

        self.serpapi_key = getattr(settings, "SERPAPI_KEY", "")
        self.google_cse_key = getattr(settings, "GOOGLE_CSE_API_KEY", "")
        self.google_cx = getattr(settings, "GOOGLE_CSE_CX", "")
        self.brave_api_key = getattr(settings, "BRAVE_SEARCH_API_KEY", "")
        self.provider_timeout_seconds = max(
            2.0,
            float(getattr(settings, "PRODUCT_AI_PROVIDER_TIMEOUT_SECONDS", 8.0) or 8.0),
        )
        self.provider_cooldown_seconds = max(
            15,
            int(getattr(settings, "PRODUCT_AI_PROVIDER_COOLDOWN_SECONDS", 180) or 180),
        )
        self.searxng_timeout_seconds = max(
            2.0,
            float(getattr(settings, "PRODUCT_AI_SEARXNG_TIMEOUT_SECONDS", self.provider_timeout_seconds) or self.provider_timeout_seconds),
        )
        raw_searxng_urls = getattr(settings, "PRODUCT_AI_SEARXNG_BASE_URLS", [])
        if isinstance(raw_searxng_urls, str):
            self.searxng_base_urls = [item.strip().rstrip("/") for item in raw_searxng_urls.split(",") if item.strip()]
        else:
            self.searxng_base_urls = [str(item).strip().rstrip("/") for item in (raw_searxng_urls or []) if str(item).strip()]
        self.user_agent = "BunoraaProductAI/1.0 (+https://bunoraa.com)"
        self._provider_state: dict[str, dict[str, Any]] = defaultdict(dict)
        self._last_search_diagnostics: dict[str, Any] = {}

        retry = Retry(
            total=2,
            read=2,
            connect=2,
            status=2,
            backoff_factor=0.5,
            allowed_methods=frozenset({"GET"}),
            status_forcelist=(408, 429, 500, 502, 503, 504),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_last_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_search_diagnostics or {})

    def search(self, query: str, max_results: int = 8) -> tuple[list[dict[str, Any]], str]:
        diagnostics: dict[str, Any] = {
            "query": query,
            "provider_order": list(self.provider_order),
            "attempts": [],
            "provider": "none",
            "result_count": 0,
            "status": "failed",
        }
        self._last_search_diagnostics = diagnostics

        for provider in self.provider_order:
            provider_name = str(provider or "").strip().lower()
            if not provider_name:
                continue

            now = time.monotonic()
            cooldown_until = float(self._provider_state.get(provider_name, {}).get("cooldown_until", 0.0) or 0.0)
            if now < cooldown_until:
                diagnostics["attempts"].append(
                    {
                        "provider": provider_name,
                        "status": "skipped_cooldown",
                        "cooldown_remaining_seconds": round(cooldown_until - now, 3),
                    }
                )
                continue

            if not self._provider_is_configured(provider_name):
                diagnostics["attempts"].append(
                    {"provider": provider_name, "status": "skipped_unconfigured"}
                )
                continue

            started = time.monotonic()
            try:
                results = self._run_provider(provider_name, query=query, max_results=max_results)
                normalized = self._dedupe_results(results, max_results=max_results)
                elapsed_ms = int((time.monotonic() - started) * 1000)
                diagnostics["attempts"].append(
                    {
                        "provider": provider_name,
                        "status": "ok" if normalized else "empty",
                        "result_count": len(normalized),
                        "elapsed_ms": elapsed_ms,
                    }
                )
                if normalized:
                    diagnostics["provider"] = provider_name
                    diagnostics["result_count"] = len(normalized)
                    diagnostics["status"] = "ok"
                    self._last_search_diagnostics = diagnostics
                    return normalized, provider_name
            except SearchBlockedError as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                self._mark_provider_failed(provider_name, blocked=True)
                diagnostics["attempts"].append(
                    {
                        "provider": provider_name,
                        "status": "blocked",
                        "reason": str(exc),
                        "elapsed_ms": elapsed_ms,
                    }
                )
            except SearchProviderError as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                self._mark_provider_failed(provider_name, blocked=False)
                diagnostics["attempts"].append(
                    {
                        "provider": provider_name,
                        "status": "failed",
                        "reason": str(exc),
                        "elapsed_ms": elapsed_ms,
                    }
                )
            except Exception as exc:  # pragma: no cover - safety net
                elapsed_ms = int((time.monotonic() - started) * 1000)
                self._mark_provider_failed(provider_name, blocked=False)
                diagnostics["attempts"].append(
                    {
                        "provider": provider_name,
                        "status": "failed",
                        "reason": str(exc),
                        "elapsed_ms": elapsed_ms,
                    }
                )
                logger.warning("Search provider failed (%s): %s", provider_name, exc)

        self._last_search_diagnostics = diagnostics
        return [], "none"

    def _provider_is_configured(self, provider: str) -> bool:
        if provider == "serpapi":
            return bool(self.serpapi_key)
        if provider in {"google", "google_cse"}:
            return bool(self.google_cse_key and self.google_cx)
        if provider == "brave_api":
            return bool(self.brave_api_key)
        if provider == "searxng":
            return bool(self.searxng_base_urls)
        if provider in {"bing", "bing_html", "duckduckgo", "ddg"}:
            return True
        return False

    def _run_provider(self, provider: str, *, query: str, max_results: int) -> list[dict[str, Any]]:
        if provider == "serpapi":
            return self._search_serpapi(query, max_results=max_results)
        if provider in {"google", "google_cse"}:
            return self._search_google_cse(query, max_results=max_results)
        if provider == "brave_api":
            return self._search_brave_api(query, max_results=max_results)
        if provider == "searxng":
            return self._search_searxng(query, max_results=max_results)
        if provider in {"bing", "bing_html"}:
            return self._search_bing_html(query, max_results=max_results)
        if provider in {"duckduckgo", "ddg"}:
            return self._search_duckduckgo(query, max_results=max_results)
        raise SearchUnavailableError(f"Unsupported provider '{provider}'")

    def _mark_provider_failed(self, provider: str, *, blocked: bool):
        until = time.monotonic() + self.provider_cooldown_seconds
        self._provider_state[provider] = {
            "cooldown_until": until,
            "blocked": bool(blocked),
        }

    @staticmethod
    def _looks_like_challenge(text: str, *, status_code: int = 200) -> bool:
        if status_code in {202, 403, 429, 503}:
            return True
        sample = str(text or "")[:7000]
        if not sample:
            return False
        return bool(CHALLENGE_RE.search(sample))

    @staticmethod
    def _dedupe_results(results: list[dict[str, Any]], *, max_results: int) -> list[dict[str, Any]]:
        normalized = []
        seen = set()
        for item in results:
            raw_url = item.get("url") or ""
            cleaned = SearchProvider._normalize_result_url(raw_url)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            record = dict(item)
            record["url"] = cleaned
            normalized.append(record)
            if len(normalized) >= max_results:
                break
        return normalized

    @staticmethod
    def _normalize_result_url(url: str) -> str:
        if not url:
            return ""
        candidate = str(url).strip()
        if candidate.startswith("/l/"):
            candidate = f"https://duckduckgo.com{candidate}"
        try:
            parsed = urlparse(candidate)
        except Exception:
            return candidate
        if parsed.path.startswith("/l/") and "duckduckgo.com" in (parsed.netloc or ""):
            params = parse_qs(parsed.query)
            target = params.get("uddg", [None])[0]
            if target:
                return unquote(target)
        return candidate

    def _search_serpapi(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not self.serpapi_key:
            raise SearchUnavailableError("SERPAPI_KEY missing")
        response = self.session.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "engine": "google",
                "api_key": self.serpapi_key,
                "num": max_results,
            },
            headers={"User-Agent": self.user_agent},
            timeout=(4, self.provider_timeout_seconds),
        )
        if response.status_code >= 400:
            raise SearchProviderError(f"serpapi status={response.status_code}")
        payload = response.json()
        results = []
        for item in payload.get("organic_results", [])[:max_results]:
            url = item.get("link") or ""
            if not url:
                continue
            results.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "provider": "serpapi",
                }
            )
        return results

    def _search_google_cse(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not (self.google_cse_key and self.google_cx):
            raise SearchUnavailableError("GOOGLE_CSE_API_KEY/GOOGLE_CSE_CX missing")
        response = self.session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": self.google_cse_key,
                "cx": self.google_cx,
                "q": query,
                "num": max(1, min(10, max_results)),
            },
            headers={"User-Agent": self.user_agent},
            timeout=(4, self.provider_timeout_seconds),
        )
        if response.status_code >= 400:
            raise SearchProviderError(f"google_cse status={response.status_code}")
        payload = response.json()
        results = []
        for item in payload.get("items", [])[:max_results]:
            url = item.get("link") or ""
            if not url:
                continue
            results.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "provider": "google_cse",
                }
            )
        return results

    def _search_brave_api(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not self.brave_api_key:
            raise SearchUnavailableError("BRAVE_SEARCH_API_KEY missing")
        response = self.session.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max(1, min(max_results, 20))},
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key,
            },
            timeout=(4, self.provider_timeout_seconds),
        )
        if response.status_code in {401, 402, 403, 429}:
            raise SearchBlockedError(f"brave_api blocked status={response.status_code}")
        if response.status_code >= 400:
            raise SearchProviderError(f"brave_api status={response.status_code}")
        payload = response.json()
        results = []
        for item in ((payload.get("web") or {}).get("results") or [])[:max_results]:
            url = item.get("url") or ""
            if not url:
                continue
            results.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "provider": "brave_api",
                }
            )
        return results

    def _search_searxng(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not self.searxng_base_urls:
            raise SearchUnavailableError("PRODUCT_AI_SEARXNG_BASE_URLS missing")
        last_error = ""
        for base_url in self.searxng_base_urls:
            endpoint = f"{base_url.rstrip('/')}/search"
            try:
                response = self.session.get(
                    endpoint,
                    params={
                        "q": query,
                        "format": "json",
                        "language": "en",
                        "safesearch": 1,
                    },
                    headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                    timeout=(4, self.searxng_timeout_seconds),
                )
                if response.status_code in {403, 429, 503}:
                    raise SearchBlockedError(f"searxng blocked status={response.status_code}")
                if response.status_code >= 400:
                    raise SearchProviderError(f"searxng status={response.status_code}")
                payload = response.json()
                results = []
                for item in (payload.get("results") or [])[: max_results * 2]:
                    url = item.get("url") or ""
                    if not url:
                        continue
                    results.append(
                        {
                            "url": url,
                            "title": item.get("title", ""),
                            "snippet": item.get("content", ""),
                            "provider": "searxng",
                        }
                    )
                if results:
                    return results[:max_results]
            except SearchBlockedError:
                raise
            except Exception as exc:
                last_error = str(exc)
                continue
        if last_error:
            raise SearchProviderError(f"searxng unavailable: {last_error}")
        return []

    def _search_bing_html(self, query: str, max_results: int) -> list[dict[str, Any]]:
        response = self.session.get(
            "https://www.bing.com/search",
            params={"q": query, "count": max(max_results * 2, 10)},
            headers={"User-Agent": self.user_agent},
            timeout=(4, self.provider_timeout_seconds),
        )
        if self._looks_like_challenge(response.text, status_code=response.status_code):
            raise SearchBlockedError(f"bing_html blocked status={response.status_code}")
        if response.status_code >= 400:
            raise SearchProviderError(f"bing_html status={response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for node in soup.select("li.b_algo")[: max_results * 2]:
            link = node.select_one("h2 a")
            if not link:
                continue
            url = link.get("href") or ""
            if not url:
                continue
            snippet_node = node.select_one(".b_caption p")
            results.append(
                {
                    "url": url,
                    "title": link.get_text(strip=True),
                    "snippet": snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    "provider": "bing_html",
                }
            )
        return results[:max_results]

    def _search_duckduckgo(self, query: str, max_results: int) -> list[dict[str, Any]]:
        encoded = quote_plus(query)
        response = self.session.get(
            f"https://duckduckgo.com/html/?q={encoded}",
            headers={"User-Agent": self.user_agent},
            timeout=(4, self.provider_timeout_seconds),
        )
        if self._looks_like_challenge(response.text, status_code=response.status_code):
            raise SearchBlockedError(f"duckduckgo blocked status={response.status_code}")
        if response.status_code >= 400:
            raise SearchProviderError(f"duckduckgo status={response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for link in soup.select("a.result__a")[: max_results * 2]:
            url = link.get("href") or ""
            if not url:
                continue
            snippet_node = link.find_parent("div", class_="result")
            snippet_text = ""
            if snippet_node:
                node = snippet_node.select_one(".result__snippet")
                snippet_text = node.get_text(" ", strip=True) if node else ""
            results.append(
                {
                    "url": url,
                    "title": link.get_text(strip=True),
                    "snippet": snippet_text,
                    "provider": "duckduckgo",
                }
            )
        return results[:max_results]
