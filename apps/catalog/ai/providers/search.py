from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from django.conf import settings

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = (4, 8)


class SearchProvider:
    """
    Multi-provider search with free-tier fallback order.
    """

    def __init__(self):
        raw_order = getattr(
            settings,
            "PRODUCT_AI_SEARCH_PROVIDER_ORDER",
            "serpapi,google_cse,duckduckgo",
        )
        self.provider_order = [part.strip().lower() for part in raw_order.split(",") if part.strip()]
        self.serpapi_key = getattr(settings, "SERPAPI_KEY", "")
        self.google_cse_key = getattr(settings, "GOOGLE_CSE_API_KEY", "")
        self.google_cx = getattr(settings, "GOOGLE_CSE_CX", "")
        self.user_agent = "BunoraaProductAI/1.0 (+https://bunoraa.com)"

    def search(self, query: str, max_results: int = 8) -> tuple[list[dict[str, Any]], str]:
        for provider in self.provider_order:
            try:
                if provider == "serpapi" and self.serpapi_key:
                    results = self._search_serpapi(query, max_results=max_results)
                elif provider in {"google", "google_cse"} and self.google_cse_key and self.google_cx:
                    results = self._search_google_cse(query, max_results=max_results)
                elif provider in {"duckduckgo", "ddg"}:
                    results = self._search_duckduckgo(query, max_results=max_results)
                else:
                    continue
                if results:
                    return results, provider
            except Exception as exc:
                logger.warning("Search provider failed (%s): %s", provider, exc)
        return [], "none"

    def _search_serpapi(self, query: str, max_results: int) -> list[dict[str, Any]]:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "engine": "google",
                "api_key": self.serpapi_key,
                "num": max_results,
            },
            headers={"User-Agent": self.user_agent},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
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
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": self.google_cse_key,
                "cx": self.google_cx,
                "q": query,
                "num": max(1, min(10, max_results)),
            },
            headers={"User-Agent": self.user_agent},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
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

    def _search_duckduckgo(self, query: str, max_results: int) -> list[dict[str, Any]]:
        encoded = quote_plus(query)
        response = requests.get(
            f"https://duckduckgo.com/html/?q={encoded}",
            headers={"User-Agent": self.user_agent},
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for link in soup.select("a.result__a")[:max_results]:
            url = link.get("href") or ""
            if not url:
                continue
            snippet_node = link.find_parent("div", class_="result").select_one(".result__snippet")
            results.append(
                {
                    "url": url,
                    "title": link.get_text(strip=True),
                    "snippet": snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    "provider": "duckduckgo",
                }
            )
        return results
