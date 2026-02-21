from __future__ import annotations

import ipaddress
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
        domain = _to_domain(url)
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

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        page_title = soup.title.get_text(strip=True) if soup.title else ""
        text = self._extract_main_text(soup)
        if not text:
            return None

        trust = self._trust_score(domain=domain, content=text, has_schema=bool(soup.find(attrs={"itemtype": True})))
        return ResearchDocument(
            url=url,
            domain=domain,
            title=page_title or title,
            snippet=snippet,
            text=text,
            trust_score=trust,
            metadata={
                "provider": provider,
                "content_type": content_type,
            },
        )

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

    def _trust_score(self, domain: str, content: str, has_schema: bool) -> float:
        score = 0.25
        if domain in TRUSTED_CERTIFICATION_DOMAINS:
            score += 0.4
        if any(domain.endswith(marketplace) for marketplace in self.marketplace_domains):
            score += 0.2
        if has_schema:
            score += 0.15
        if len(content) > 800:
            score += 0.1
        if re.search(r"\b(certificate|certification|material|specification|model)\b", content, re.I):
            score += 0.1
        return max(0.0, min(1.0, score))
