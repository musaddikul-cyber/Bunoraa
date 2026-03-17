from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from django.conf import settings
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = (4, 10)
BLOCKED_SCHEMES = {"file", "ftp", "gopher", "javascript", "data"}
TRUSTED_CERTIFICATION_DOMAINS = {
    "fsc.org",
    "fairtrade.net",
    "bluesign.com",
    "globalrecycled.org",
}
CHALLENGE_RE = re.compile(
    r"(captcha|verify (you are|you're) (human|not a robot)|access denied|challenge|unusual traffic|bot detection)",
    re.I,
)
NOISE_TEXT_RE = re.compile(
    r"\b(cookie|javascript required|enable javascript|privacy preference|consent manager)\b",
    re.I,
)
PRODUCT_SIGNAL_RE = re.compile(
    r"\b(product|price|sku|model|size|material|fabric|cotton|linen|silk|embroid|kurti|dress|shirt|pant|trouser|set|buy|cart|in stock)\b",
    re.I,
)


def _to_domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _is_private_host(host: str) -> bool:
    if not host:
        return True
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True

    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast:
            return True
    return False


def is_safe_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.scheme in BLOCKED_SCHEMES:
        return False
    host = parsed.hostname or ""
    if _is_private_host(host):
        return False
    return True


def _safe_text(value: str, max_chars: int = 4000) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    return value[:max_chars]


def _robots_allows(url: str, user_agent: str) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.read()
        return parser.can_fetch(user_agent, url)
    except Exception:
        return True


@dataclass(slots=True)
class ResearchDocument:
    url: str
    domain: str
    title: str
    snippet: str
    text: str
    trust_score: float
    metadata: dict[str, Any]


class ResearchProvider:
    """
    Fetch and score web sources while enforcing SSRF controls.
    """

    def __init__(self):
        raw_marketplaces = getattr(settings, "PRODUCT_AI_MARKETPLACE_DOMAINS", [])
        if isinstance(raw_marketplaces, str):
            self.marketplace_domains = {d.strip().lower() for d in raw_marketplaces.split(",") if d.strip()}
        else:
            self.marketplace_domains = {str(d).strip().lower() for d in raw_marketplaces if str(d).strip()}
        self.user_agent = (
            str(getattr(settings, "PRODUCT_AI_RESEARCH_USER_AGENT", "") or "").strip()
            or str(getattr(settings, "PRODUCT_AI_USER_AGENT", "") or "").strip()
            or "Mozilla/5.0"
        )
        self.respect_robots = bool(getattr(settings, "PRODUCT_AI_RESEARCH_RESPECT_ROBOTS", False))
        self.allow_snippet_fallback = bool(
            getattr(settings, "PRODUCT_AI_RESEARCH_ALLOW_SNIPPET_FALLBACK", True)
        )
        self.min_extracted_chars = max(
            40,
            int(getattr(settings, "PRODUCT_AI_RESEARCH_MIN_EXTRACTED_CHARS", 80) or 80),
        )

        retry = Retry(
            total=2,
            read=2,
            connect=2,
            status=2,
            backoff_factor=0.6,
            allowed_methods=frozenset({"GET"}),
            status_forcelist=(408, 429, 500, 502, 503, 504),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_documents(self, search_results: list[dict[str, Any]], max_docs: int = 8) -> list[ResearchDocument]:
        documents, _ = self.fetch_documents_with_diagnostics(search_results, max_docs=max_docs)
        return documents

    def fetch_documents_with_diagnostics(
        self,
        search_results: list[dict[str, Any]],
        *,
        max_docs: int = 8,
    ) -> tuple[list[ResearchDocument], dict[str, Any]]:
        seen = set()
        documents: list[ResearchDocument] = []
        rejection_reasons: Counter[str] = Counter()
        attempted = 0

        for result in search_results:
            url = str(result.get("url", "") or "").strip()
            if not url or url in seen:
                rejection_reasons["duplicate_or_empty_url"] += 1
                continue
            seen.add(url)
            attempted += 1

            if not is_safe_public_url(url):
                rejection_reasons["unsafe_url"] += 1
                continue
            if self.respect_robots and not _robots_allows(url, self.user_agent):
                rejection_reasons["robots_disallowed"] += 1
                continue

            doc, reason = self._fetch_one(result)
            if doc:
                documents.append(doc)
            else:
                rejection_reasons[reason or "fetch_failed"] += 1

            if len(documents) >= max_docs:
                break

        diagnostics = {
            "attempted_urls": attempted,
            "accepted_docs": len(documents),
            "rejected_docs": int(sum(rejection_reasons.values())),
            "rejection_reasons": dict(rejection_reasons),
        }
        return documents, diagnostics

    def _fetch_one(self, result: dict[str, Any]) -> tuple[ResearchDocument | None, str]:
        url = str(result.get("url", "") or "").strip()
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        provider = result.get("provider", "")

        try:
            response = self.session.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=True,
            )
        except Exception as exc:
            logger.debug("research fetch failed for %s: %s", url, exc)
            return None, "request_error"

        if response.status_code >= 400:
            fallback_doc = self._build_snippet_fallback_document(
                url=url,
                title=str(title or ""),
                snippet=str(snippet or ""),
                provider=str(provider or ""),
                reason=f"http_{response.status_code}",
            )
            if fallback_doc:
                return fallback_doc, ""
            return None, f"http_{response.status_code}"
        if self._looks_like_challenge(response.text, status_code=response.status_code):
            fallback_doc = self._build_snippet_fallback_document(
                url=url,
                title=str(title or ""),
                snippet=str(snippet or ""),
                provider=str(provider or ""),
                reason="challenge_or_captcha",
            )
            if fallback_doc:
                return fallback_doc, ""
            return None, "challenge_or_captcha"

        final_url = response.url or url
        if not is_safe_public_url(final_url):
            return None, "unsafe_redirect_url"
        domain = _to_domain(final_url)

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return None, "unsupported_content_type"

        soup = BeautifulSoup(response.text, "html.parser")
        if self._looks_like_challenge(soup.get_text(" ", strip=True), status_code=response.status_code):
            return None, "challenge_or_captcha"

        page_title = soup.title.get_text(strip=True) if soup.title else ""
        text = self._extract_main_text(soup)
        fallback_text = _safe_text(" ".join(part for part in [page_title or title, snippet] if part), max_chars=1200)
        if not text and fallback_text:
            text = fallback_text
        if not text:
            return None, "empty_extracted_text"
        signal_text = " ".join(part for part in [page_title or title, snippet, text] if part)
        has_product_signal = bool(PRODUCT_SIGNAL_RE.search(signal_text))
        if NOISE_TEXT_RE.search(text) and len(text) < 220 and not has_product_signal:
            return None, "low_quality_noise_text"

        structured = self._extract_structured_product_data(soup)
        has_structured_signal = self._structured_has_signal(structured)
        if len(text) < self.min_extracted_chars and not has_structured_signal and not has_product_signal:
            return None, "insufficient_content_signal"
        if len(text) < self.min_extracted_chars and has_product_signal:
            text = _safe_text(" ".join(part for part in [text, fallback_text] if part), max_chars=6500)

        trust = self._trust_score(
            domain=domain,
            content=text,
            has_schema=bool(soup.find(attrs={"itemtype": True})),
            structured=structured,
        )
        return (
            ResearchDocument(
                url=final_url,
                domain=domain,
                title=page_title or title,
                snippet=snippet,
                text=text,
                trust_score=trust,
                metadata={
                    "provider": provider,
                    "content_type": content_type,
                    "structured": structured,
                },
            ),
            "",
        )

    def _build_snippet_fallback_document(
        self,
        *,
        url: str,
        title: str,
        snippet: str,
        provider: str,
        reason: str,
    ) -> ResearchDocument | None:
        if not self.allow_snippet_fallback:
            return None
        text = _safe_text(" ".join(part for part in [title, snippet] if part), max_chars=1200)
        if len(text) < 40:
            return None
        if not PRODUCT_SIGNAL_RE.search(text):
            return None
        domain = _to_domain(url)
        trust = self._snippet_fallback_trust_score(domain=domain, text=text)
        return ResearchDocument(
            url=url,
            domain=domain,
            title=title,
            snippet=snippet,
            text=text,
            trust_score=trust,
            metadata={
                "provider": provider,
                "content_type": "snippet/fallback",
                "structured": {},
                "snippet_fallback": True,
                "snippet_fallback_reason": reason,
            },
        )

    def _snippet_fallback_trust_score(self, *, domain: str, text: str) -> float:
        score = 0.28
        if any(domain.endswith(marketplace) for marketplace in self.marketplace_domains):
            score += 0.22
        if re.search(r"\b(\d{2,6}(?:\.\d{1,2})?)\b", text):
            score += 0.08
        if re.search(r"\b(size|material|fabric|cotton|linen|silk|sku|model|in stock)\b", text, re.I):
            score += 0.08
        if PRODUCT_SIGNAL_RE.search(text):
            score += 0.06
        return max(0.0, min(0.74, score))

    @staticmethod
    def _looks_like_challenge(text: str, *, status_code: int = 200) -> bool:
        if status_code in {202, 403, 429, 503}:
            return True
        sample = str(text or "")[:6000]
        if not sample:
            return False
        return bool(CHALLENGE_RE.search(sample))

    @staticmethod
    def _structured_has_signal(structured: dict[str, Any]) -> bool:
        for key in ("names", "price_amounts", "sku_candidates", "category_names", "material_hints"):
            if structured.get(key):
                return True
        return False

    @staticmethod
    def _extract_structured_product_data(soup: BeautifulSoup) -> dict[str, Any]:
        names: list[str] = []
        descriptions: list[str] = []
        sku_candidates: list[str] = []
        price_amounts: list[str] = []
        brand_names: list[str] = []
        category_names: list[str] = []
        material_hints: list[str] = []

        def _add_unique(target: list[str], value: Any, *, max_chars: int = 280):
            cleaned = _safe_text(str(value or ""), max_chars=max_chars)
            if cleaned and cleaned not in target:
                target.append(cleaned)

        def _extract_from_product_node(node: dict[str, Any]):
            _add_unique(names, node.get("name"))
            _add_unique(descriptions, node.get("description"), max_chars=1200)
            _add_unique(sku_candidates, node.get("sku"), max_chars=80)
            _add_unique(category_names, node.get("category"), max_chars=180)
            _add_unique(material_hints, node.get("material"), max_chars=180)

            brand = node.get("brand")
            if isinstance(brand, dict):
                _add_unique(brand_names, brand.get("name"))
            else:
                _add_unique(brand_names, brand)

            offers = node.get("offers")
            offer_nodes = offers if isinstance(offers, list) else [offers]
            for offer in offer_nodes:
                if not isinstance(offer, dict):
                    continue
                for key in ("price", "lowPrice", "highPrice"):
                    value = offer.get(key)
                    if value not in (None, ""):
                        _add_unique(price_amounts, value, max_chars=40)

        def _walk_json_ld(payload: Any):
            if isinstance(payload, list):
                for item in payload:
                    _walk_json_ld(item)
                return
            if not isinstance(payload, dict):
                return
            if "@graph" in payload:
                _walk_json_ld(payload.get("@graph"))
            node_type = payload.get("@type")
            types = node_type if isinstance(node_type, list) else [node_type]
            lowered = {str(item).lower() for item in types if item}
            if any("product" in value for value in lowered):
                _extract_from_product_node(payload)

        for node in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
            raw = node.string or node.get_text() or ""
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            _walk_json_ld(payload)

        meta_extract_map = {
            "og:title": (names, 280),
            "twitter:title": (names, 280),
            "description": (descriptions, 1200),
            "og:description": (descriptions, 1200),
            "product:price:amount": (price_amounts, 40),
            "product:retailer_item_id": (sku_candidates, 80),
            "product:brand": (brand_names, 180),
            "product:category": (category_names, 180),
        }
        for tag in soup.find_all("meta"):
            key = (tag.get("property") or tag.get("name") or "").strip().lower()
            if key not in meta_extract_map:
                continue
            value = tag.get("content")
            target, max_chars = meta_extract_map[key]
            _add_unique(target, value, max_chars=max_chars)

        return {
            "names": names[:5],
            "descriptions": descriptions[:3],
            "sku_candidates": sku_candidates[:6],
            "price_amounts": price_amounts[:8],
            "brand_names": brand_names[:4],
            "category_names": category_names[:6],
            "material_hints": material_hints[:6],
        }

    @staticmethod
    def _extract_main_text(soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer"]):
            tag.decompose()

        candidates = []
        selectors = ["main", "article", ".product", ".product-detail", ".pdp", "#content", ".content"]
        for selector in selectors:
            for node in soup.select(selector):
                text = _safe_text(node.get_text(" ", strip=True), max_chars=6500)
                if len(text) > 120:
                    candidates.append(text)

        if not candidates:
            body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
            body_text = _safe_text(body, max_chars=6500)
            if body_text:
                candidates.append(body_text)

        if not candidates:
            meta_description = ""
            meta_node = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
            if meta_node:
                meta_description = _safe_text(meta_node.get("content") or "", max_chars=1200)
            if meta_description:
                candidates.append(meta_description)

        candidates = [candidate for candidate in candidates if candidate]
        return max(candidates, key=len) if candidates else ""

    def _trust_score(self, domain: str, content: str, has_schema: bool, structured: dict[str, Any]) -> float:
        score = 0.25
        if domain in TRUSTED_CERTIFICATION_DOMAINS:
            score += 0.4
        if any(domain.endswith(marketplace) for marketplace in self.marketplace_domains):
            score += 0.2
        if has_schema:
            score += 0.15
        structured_signals = 0
        for key in ("names", "price_amounts", "sku_candidates", "category_names"):
            if structured.get(key):
                structured_signals += 1
        if structured_signals:
            score += min(0.2, structured_signals * 0.06)
        if len(content) > 800:
            score += 0.1
        if re.search(r"\b(certificate|certification|material|specification|model)\b", content, re.I):
            score += 0.1
        return max(0.0, min(1.0, score))
