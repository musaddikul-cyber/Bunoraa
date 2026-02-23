from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from django.conf import settings

from .research import ResearchDocument, ResearchProvider
from .search import SearchProvider

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'().+\-_]{1,50}")
NON_PRODUCT_RE = re.compile(
    r"\b(requirements?|guidelines?|policy|policies|help(?:\s*center)?|support|how\s+to|tutorial|forum|community|lounge|documentation|seller(?:\s+central)?)\b",
    re.I,
)
PRODUCT_CUE_RE = re.compile(
    r"\b(product|price|sku|model|size|material|fabric|cotton|linen|silk|embroid|kurti|dress|shirt|pant|trouser|set|buy|cart|artisan|handmade|in stock|add to cart)\b",
    re.I,
)
UI_NOISE_RE = re.compile(
    r"\b(open\s+media|in\s+modal|skip\s+to|listings?\s+lounge|help\s*center|seller\s+central|"
    r"image\s+requirements?|cookie|javascript)\b",
    re.I,
)
HELP_PATH_RE = re.compile(
    r"/(help|support|policy|policies|guides?|blog|forum|community|docs?|documentation|faq|kb|knowledge-base)(?:/|$)",
    re.I,
)
COMMERCE_PATH_HINT_RE = re.compile(
    r"/(product|products|shop|item|items|p|collections?)/",
    re.I,
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _clean_text(value: Any, *, max_chars: int = 280) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0]
    return trimmed or text[:max_chars]


def _to_query_terms(text: str, *, limit: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text or ""):
        lowered = token.lower().strip("._-")
        if len(lowered) < 3 or lowered in STOPWORDS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        terms.append(lowered)
        if len(terms) >= limit:
            break
    return terms


def _structured_signal_count(doc: ResearchDocument) -> int:
    metadata = getattr(doc, "metadata", {}) or {}
    structured = metadata.get("structured") if isinstance(metadata, dict) else {}
    if not isinstance(structured, dict):
        return 0
    signal_keys = ("names", "sku_candidates", "price_amounts", "category_names", "brand_names", "material_hints")
    return sum(1 for key in signal_keys if structured.get(key))


@dataclass(slots=True)
class RankedResearchDocument:
    document: ResearchDocument
    score: float
    relevance: float
    product_likelihood: float


class ProductDeepResearchProvider:
    """
    Product-focused web deep-research pipeline:
    - Build multiple focused queries from product evidence.
    - Search across providers (free-first default).
    - Fetch crawlable pages with ResearchProvider safeguards.
    - Rank for product-likelihood + relevance + trust.
    - Return de-duplicated, domain-diverse source set.
    """

    def __init__(
        self,
        *,
        search_provider: SearchProvider | None = None,
        research_provider: ResearchProvider | None = None,
    ):
        self.search_provider = search_provider or SearchProvider()
        self.research_provider = research_provider or ResearchProvider()
        raw_order = (
            getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_SEARCH_PROVIDER_ORDER", "duckduckgo")
            or "duckduckgo"
        )
        provider_order = [item.strip().lower() for item in raw_order.split(",") if item.strip()]
        if not provider_order:
            provider_order = ["duckduckgo"]
        if hasattr(self.search_provider, "provider_order"):
            self.search_provider.provider_order = provider_order

    def run(
        self,
        *,
        query: str,
        candidate_text: str,
        ocr: dict[str, Any],
        vision: dict[str, Any],
        context_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context_hints = context_hints or {}
        query_plan = self._build_query_plan(
            query=query,
            candidate_text=candidate_text,
            ocr=ocr,
            vision=vision,
            context_hints=context_hints,
        )
        max_per_query = max(2, int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_RESULTS_PER_QUERY", 6) or 6))
        max_search_results = max(6, int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_SEARCH_RESULTS", 24) or 24))
        max_docs = max(4, int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_DOCS", 10) or 10))
        max_sources = max(3, int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_SOURCES", 8) or 8))

        aggregated_results: list[dict[str, Any]] = []
        provider_counts: Counter[str] = Counter()
        query_result_counts: dict[str, int] = {}

        for planned_query in query_plan:
            try:
                results, provider = self.search_provider.search(planned_query, max_results=max_per_query)
            except Exception as exc:
                logger.warning("Product deep research search failed for '%s': %s", planned_query, exc)
                continue
            provider_name = (provider or "none").strip() or "none"
            provider_counts[provider_name] += 1
            filtered_results = [item for item in (results or []) if self._result_is_likely_product(item)]
            query_result_counts[planned_query] = len(filtered_results)

            for result in filtered_results:
                item = dict(result)
                item["query"] = planned_query
                aggregated_results.append(item)
            if len(aggregated_results) >= max_search_results:
                break

        deduped_results = self._dedupe_results(aggregated_results, max_results=max_search_results)
        research_docs = self.research_provider.fetch_documents(deduped_results, max_docs=max_docs * 2)

        query_terms = _to_query_terms(" ".join(query_plan), limit=16)
        reference_terms = _to_query_terms(
            " ".join(
                [
                    candidate_text,
                    str(context_hints.get("name") or ""),
                    str(context_hints.get("primary_category_name") or ""),
                    " ".join(str(v) for v in (ocr.get("sku_candidates") or [])[:3]),
                ]
            ),
            limit=16,
        )
        ranked = self._rank_documents(research_docs, query_terms=query_terms, reference_terms=reference_terms)
        selected = self._select_diverse_sources(ranked, max_sources=max_sources)
        selected_docs = [ranked_item.document for ranked_item in selected][:max_docs]

        for ranked_item in selected:
            doc = ranked_item.document
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata["deep_research"] = {
                "score": round(ranked_item.score, 4),
                "relevance": round(ranked_item.relevance, 4),
                "product_likelihood": round(ranked_item.product_likelihood, 4),
            }
            doc.metadata = metadata

        primary_provider = "none"
        if provider_counts:
            primary_provider = provider_counts.most_common(1)[0][0]

        return {
            "documents": selected_docs,
            "search_results": deduped_results,
            "query_plan": query_plan,
            "query_result_counts": query_result_counts,
            "provider_counts": dict(provider_counts),
            "primary_provider": primary_provider,
            "raw_search_results_count": len(aggregated_results),
            "deduped_search_results_count": len(deduped_results),
            "fetched_docs_count": len(research_docs),
            "selected_docs_count": len(selected_docs),
        }

    def _build_query_plan(
        self,
        *,
        query: str,
        candidate_text: str,
        ocr: dict[str, Any],
        vision: dict[str, Any],
        context_hints: dict[str, Any],
    ) -> list[str]:
        max_subqueries = max(1, int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_SUBQUERIES", 4) or 4))
        hint_name = _clean_text(context_hints.get("name"), max_chars=120)
        category_name = _clean_text(context_hints.get("primary_category_name"), max_chars=80)
        sku_candidates = [str(item).strip() for item in (ocr.get("sku_candidates") or []) if str(item).strip()]
        base_terms = _to_query_terms(candidate_text, limit=8)
        vision_terms = [str(token).strip() for token in (vision.get("tokens") or []) if str(token).strip()]

        variants = [query]
        if sku_candidates:
            variants.append(f"\"{sku_candidates[0][:40]}\" product")
        if hint_name:
            variants.append(f"\"{hint_name}\" product details")
        if hint_name and category_name:
            variants.append(f"\"{hint_name}\" \"{category_name}\" buy")
        if category_name and base_terms:
            variants.append(f"{category_name} {' '.join(base_terms[:4])} product")
        if vision_terms:
            variants.append(f"{' '.join(vision_terms[:4])} product details")
        variants = [
            f"{variant} -requirements -guidelines -policy -help -forum"
            for variant in variants
            if variant
        ]

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            cleaned = _clean_text(variant, max_chars=180).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
            if len(deduped) >= max_subqueries:
                break
        return deduped or [query]

    @staticmethod
    def _dedupe_results(results: list[dict[str, Any]], *, max_results: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in results:
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(item)
            if len(deduped) >= max_results:
                break
        return deduped

    @staticmethod
    def _result_is_likely_product(result: dict[str, Any]) -> bool:
        title = str(result.get("title") or "")
        snippet = str(result.get("snippet") or "")
        url = str(result.get("url") or "")
        parsed = urlparse(url)
        path = (parsed.path or "").lower()
        combined = f"{title} {snippet} {path}".strip()
        lowered = combined.lower()

        has_product_cues = bool(PRODUCT_CUE_RE.search(lowered))
        has_non_product_cues = bool(NON_PRODUCT_RE.search(lowered))
        has_help_path = bool(HELP_PATH_RE.search(path))
        has_commerce_path = bool(COMMERCE_PATH_HINT_RE.search(path))

        if UI_NOISE_RE.search(lowered) and not has_product_cues:
            return False
        if has_help_path and not has_commerce_path and not has_product_cues:
            return False
        if has_non_product_cues and not has_product_cues:
            return False
        return True

    def _rank_documents(
        self,
        docs: list[ResearchDocument],
        *,
        query_terms: list[str],
        reference_terms: list[str],
    ) -> list[RankedResearchDocument]:
        min_score = float(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MIN_SCORE", 0.20) or 0.20)
        ranked: list[RankedResearchDocument] = []
        terms = query_terms or reference_terms

        for doc in docs:
            combined_text = " ".join(
                [
                    str(getattr(doc, "title", "") or ""),
                    str(getattr(doc, "snippet", "") or ""),
                    str(getattr(doc, "text", "") or "")[:3200],
                ]
            )
            lowered = combined_text.lower()
            if not lowered.strip():
                continue
            if UI_NOISE_RE.search(lowered) and not PRODUCT_CUE_RE.search(lowered):
                continue

            hits = sum(1 for term in terms if term in lowered)
            relevance = (hits / len(terms)) if terms else 0.0
            relevance = max(0.0, min(1.0, relevance))

            structured_signals = _structured_signal_count(doc)
            product_likelihood = min(0.55, structured_signals * 0.14)
            if PRODUCT_CUE_RE.search(lowered):
                product_likelihood += 0.30
            if re.search(r"\b(size|color|material|fabric|price|sku|model)\b", lowered):
                product_likelihood += 0.12
            has_non_product = bool(NON_PRODUCT_RE.search(lowered))
            if has_non_product and structured_signals == 0:
                product_likelihood -= 0.45
            if has_non_product and structured_signals < 2 and product_likelihood < 0.62:
                continue
            product_likelihood = max(0.0, min(1.0, product_likelihood))

            trust = float(getattr(doc, "trust_score", 0.0) or 0.0)
            score = max(0.0, min(1.0, (0.45 * relevance) + (0.35 * product_likelihood) + (0.20 * trust)))

            if structured_signals >= 2:
                score = max(score, min(1.0, 0.24 + (0.33 * product_likelihood) + (0.20 * trust)))

            if score < min_score:
                continue

            ranked.append(
                RankedResearchDocument(
                    document=doc,
                    score=score,
                    relevance=relevance,
                    product_likelihood=product_likelihood,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _select_diverse_sources(
        self,
        ranked: list[RankedResearchDocument],
        *,
        max_sources: int,
    ) -> list[RankedResearchDocument]:
        max_domain_repeats = max(
            1,
            int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_DOMAIN_REPEATS", 2) or 2),
        )
        selected: list[RankedResearchDocument] = []
        domain_counts: Counter[str] = Counter()
        for ranked_item in ranked:
            doc = ranked_item.document
            if not doc.url:
                continue
            domain = str(getattr(doc, "domain", "") or "")
            if domain and domain_counts[domain] >= max_domain_repeats:
                continue
            selected.append(ranked_item)
            if domain:
                domain_counts[domain] += 1
            if len(selected) >= max_sources:
                break
        return selected
