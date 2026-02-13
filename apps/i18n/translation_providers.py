"""
Translation provider implementations.

Currently supports LibreTranslate (self-hosted) with optional API key.
"""
from __future__ import annotations

import time
from typing import Iterable, List, Optional
import requests


class TranslationProviderError(Exception):
    """Raised when a translation provider fails."""


class BaseTranslationProvider:
    """Base translation provider interface."""

    def translate(self, texts: Iterable[str], source: str, target: str, fmt: str = "text") -> List[str]:
        raise NotImplementedError

    def languages(self) -> List[dict]:
        raise NotImplementedError


class LibreTranslateProvider(BaseTranslationProvider):
    """LibreTranslate provider (self-hosted)."""

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 20):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise TranslationProviderError("LIBRETRANSLATE_URL is not configured")
        return f"{self.base_url}{path}"

    def translate(self, texts: Iterable[str], source: str, target: str, fmt: str = "text") -> List[str]:
        payload = {
            "q": list(texts),
            "source": source,
            "target": target,
            "format": fmt,
        }
        if self.api_key:
            payload["api_key"] = self.api_key

        for attempt in range(3):
            try:
                resp = self.session.post(
                    self._url("/translate"),
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                translations = data.get("translatedText")
                if isinstance(translations, list):
                    return [str(item) for item in translations]
                if isinstance(translations, str):
                    return [translations]
                # Some deployments return {"translations":[{"translatedText": "..."}]}
                if isinstance(data, dict) and data.get("translations"):
                    return [t.get("translatedText", "") for t in data.get("translations", [])]
                raise TranslationProviderError("Unexpected response from LibreTranslate")
            except Exception as exc:
                if attempt >= 2:
                    raise TranslationProviderError(str(exc)) from exc
                time.sleep(0.5 * (attempt + 1))

        return []

    def languages(self) -> List[dict]:
        for attempt in range(3):
            try:
                resp = self.session.get(self._url("/languages"), timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return data
                raise TranslationProviderError("Unexpected language list response")
            except Exception as exc:
                if attempt >= 2:
                    raise TranslationProviderError(str(exc)) from exc
                time.sleep(0.5 * (attempt + 1))

        return []
