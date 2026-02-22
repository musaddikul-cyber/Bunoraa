from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from django.conf import settings

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = (4, 10)
BLOCKED_SCHEMES = {"file", "ftp", "gopher", "javascript", "data"}
TRUSTED_CERTIFICATION_DOMAINS = {
    "fsc.org",
    "fairtrade.net",
    "bluesign.com",
    "globalrecycled.org",
}


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
        self.user_agent = "BunoraaProductAI/1.0 (+https://bunoraa.com)"

    def fetch_documents(self, search_results: list[dict[str, Any]], max_docs: int = 8) -> list[ResearchDocument]:
        seen = set()
        documents: list[ResearchDocument] = []
        for result in search_results:
            url = result.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            if not is_safe_public_url(url):
                continue
            if not _robots_allows(url, self.user_agent):
                continue
            doc = self._fetch_one(result)
            if doc:
                documents.append(doc)
            if len(documents) >= max_docs:
                break
        return documents

    def _fetch_one(self, result: dict[str, Any]) -> ResearchDocument | None:
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        provider = result.get("provider", "")
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.debug("research fetch failed for %s: %s", url, exc)
            return None

        final_url = response.url or url
        if not is_safe_public_url(final_url):
            return None
        domain = _to_domain(final_url)

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        page_title = soup.title.get_text(strip=True) if soup.title else ""
        text = self._extract_main_text(soup)
        if not text:
            return None

        structured = self._extract_structured_product_data(soup)
        trust = self._trust_score(
            domain=domain,
            content=text,
            has_schema=bool(soup.find(attrs={"itemtype": True})),
            structured=structured,
        )
        return ResearchDocument(
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
        )

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
        selectors = ["main", "article", ".product", ".product-detail", ".pdp"]
        for selector in selectors:
            for node in soup.select(selector):
                text = _safe_text(node.get_text(" ", strip=True))
                if len(text) > 120:
                    candidates.append(text)
        if not candidates:
            body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
            candidates = [_safe_text(body)]
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
