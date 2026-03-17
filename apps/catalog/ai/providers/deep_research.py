from __future__ import annotations

import logging
import re
import time
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
    r"image\s+requirements?|cookie|javascript|captcha|access denied|verify you are human)\b",
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
QUERY_NOISE_TERMS = {
    "product",
    "products",
    "details",
    "detail",
    "item",
    "items",
    "buy",
    "shop",
    "online",
    "women",
    "woman",
    "men",
    "man",
    "apparel",
    "fashion",
    "set",
    "white",
    "black",
    "brown",
    "beige",
    "pink",
    "blue",
    "green",
    "red",
    "grey",
    "gray",
    "yellow",
    "purple",
}
QUERY_EXCLUSION_SUFFIX = "-requirements -guidelines -policy -help -forum"
IMAGE_NAME_NOISE_RE = re.compile(r"^(?:image\(\d+\)|img[_-]?\d+|photo[_-]?\d+)$", re.I)


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
        if lowered in QUERY_NOISE_TERMS:
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
    - Build focused query variants from image/OCR/context evidence.
    - Search across resilient providers (free-first by default).
    - Fetch and filter pages with SSRF + quality checks.
    - Rank by relevance + product-likelihood + trust.
    - Select domain-diverse sources with diagnostics.
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
            getattr(
                settings,
                "PRODUCT_AI_DEEP_RESEARCH_SEARCH_PROVIDER_ORDER",
                "searxng,bing_html,duckduckgo,brave_api,google_cse,serpapi",
            )
            or "searxng,bing_html,duckduckgo,brave_api,google_cse,serpapi"
        )
        provider_order = [item.strip().lower() for item in raw_order.split(",") if item.strip()]
        if not provider_order:
            provider_order = ["searxng", "bing_html", "duckduckgo", "brave_api", "google_cse", "serpapi"]
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
        started_at = time.monotonic()
        max_latency_seconds = max(
            10,
            int(getattr(settings, "PRODUCT_AI_MAX_RESEARCH_LATENCY_SECONDS", 90) or 90),
        )
        min_source_diversity = 2

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
        query_diagnostics: list[dict[str, Any]] = []
        search_rejection_reasons: Counter[str] = Counter()
        stopped_due_to_latency = False
        search_phase_started = time.monotonic()

        for planned_query in query_plan:
            if (time.monotonic() - started_at) > max_latency_seconds:
                stopped_due_to_latency = True
                break
            try:
                results, provider = self.search_provider.search(planned_query, max_results=max_per_query)
            except Exception as exc:
                logger.warning("Product deep research search failed for '%s': %s", planned_query, exc)
                query_diagnostics.append(
                    {
                        "query": planned_query,
                        "provider": "none",
                        "status": "failed",
                        "search_results_before_filter": 0,
                        "search_results_after_filter": 0,
                        "error": str(exc),
                    }
                )
                continue

            provider_name = (provider or "none").strip() or "none"
            provider_counts[provider_name] += 1
            provider_diag = {}
            if hasattr(self.search_provider, "get_last_diagnostics"):
                provider_diag = self.search_provider.get_last_diagnostics() or {}

            before_filter = len(results or [])
            filtered_results = [item for item in (results or []) if self._result_is_likely_product(item)]
            anchor_terms = _to_query_terms(planned_query, limit=10)
            anchor_filtered_results = self._filter_results_by_anchor_overlap(
                filtered_results,
                anchor_terms=anchor_terms,
            )
            query_result_counts[planned_query] = len(filtered_results)
            after_filter = len(filtered_results)
            if before_filter > after_filter:
                search_rejection_reasons["non_product_or_noise"] += before_filter - after_filter
            if anchor_terms and filtered_results and not anchor_filtered_results:
                search_rejection_reasons["low_anchor_overlap"] += len(filtered_results)
                fallback_results, fallback_provider, fallback_diag = self._search_with_fallback_providers(
                    query=planned_query,
                    max_results=max_per_query,
                    excluded_provider=provider_name.lower(),
                )
                if fallback_results:
                    provider_name = (fallback_provider or "none").strip() or provider_name
                    provider_counts[provider_name] += 1
                    provider_diag = fallback_diag or provider_diag
                    before_filter = len(fallback_results or [])
                    filtered_results = [
                        item for item in (fallback_results or []) if self._result_is_likely_product(item)
                    ]
                    anchor_filtered_results = self._filter_results_by_anchor_overlap(
                        filtered_results,
                        anchor_terms=anchor_terms,
                    )
                    query_result_counts[planned_query] = len(filtered_results)
                    after_filter = len(filtered_results)
                    if before_filter > after_filter:
                        search_rejection_reasons["non_product_or_noise"] += before_filter - after_filter
            if anchor_terms and anchor_filtered_results:
                filtered_results = anchor_filtered_results
                query_result_counts[planned_query] = len(filtered_results)
                after_filter = len(filtered_results)

            for result in filtered_results:
                item = dict(result)
                item["query"] = planned_query
                aggregated_results.append(item)
            query_diagnostics.append(
                {
                    "query": planned_query,
                    "provider": provider_name,
                    "status": provider_diag.get("status", "ok" if provider_name != "none" else "none"),
                    "search_results_before_filter": before_filter,
                    "search_results_after_filter": after_filter,
                    "provider_attempts": provider_diag.get("attempts", []),
                }
            )
            if len(aggregated_results) >= max_search_results:
                break

        deduped_results = self._dedupe_results(aggregated_results, max_results=max_search_results)
        search_phase_duration_ms = int((time.monotonic() - search_phase_started) * 1000)
        fetch_phase_started = time.monotonic()
        try:
            if hasattr(self.research_provider, "fetch_documents_with_diagnostics"):
                research_docs, fetch_diag = self.research_provider.fetch_documents_with_diagnostics(
                    deduped_results,
                    max_docs=max_docs * 2,
                )
            else:
                research_docs = self.research_provider.fetch_documents(deduped_results, max_docs=max_docs * 2)
                fetch_diag = {
                    "attempted_urls": len(deduped_results),
                    "accepted_docs": len(research_docs),
                    "rejected_docs": max(0, len(deduped_results) - len(research_docs)),
                    "rejection_reasons": {},
                }
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("Deep research fetch phase failed: %s", exc)
            research_docs = []
            fetch_diag = {
                "attempted_urls": len(deduped_results),
                "accepted_docs": 0,
                "rejected_docs": len(deduped_results),
                "rejection_reasons": {"fetch_exception": len(deduped_results)},
                "error": str(exc),
            }
        fetch_phase_duration_ms = int((time.monotonic() - fetch_phase_started) * 1000)

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
        rank_phase_started = time.monotonic()
        ranked, rank_rejections = self._rank_documents(
            research_docs,
            query_terms=query_terms,
            reference_terms=reference_terms,
            enforce_anchor_overlap=True,
        )
        rank_fallback_used = False
        if not ranked and research_docs:
            relaxed_ranked, relaxed_rank_rejections = self._rank_documents(
                research_docs,
                query_terms=query_terms,
                reference_terms=reference_terms,
                enforce_anchor_overlap=False,
            )
            if relaxed_ranked:
                ranked = relaxed_ranked
                rank_rejections = relaxed_rank_rejections
                rank_fallback_used = True
        rank_phase_duration_ms = int((time.monotonic() - rank_phase_started) * 1000)
        selection_phase_started = time.monotonic()
        selected, selection_rejections = self._select_diverse_sources(ranked, max_sources=max_sources)
        selection_phase_duration_ms = int((time.monotonic() - selection_phase_started) * 1000)
        selected_docs = [ranked_item.document for ranked_item in selected][:max_docs]
        backfill_count = 0
        if len(selected_docs) < max_sources and research_docs:
            selected_docs, backfill_count = self._backfill_product_docs(
                selected_docs=selected_docs,
                all_docs=research_docs,
                max_docs=min(max_docs, max_sources),
                anchor_terms=reference_terms or query_terms,
            )

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

        unique_domains = sorted({str(getattr(doc, "domain", "") or "") for doc in selected_docs if getattr(doc, "domain", "")})
        diversity_ok = len(unique_domains) >= min_source_diversity or len(selected_docs) < min_source_diversity

        diagnostics = {
            "query_diagnostics": query_diagnostics,
            "provider_counts": dict(provider_counts),
            "search_results_before_dedupe": len(aggregated_results),
            "search_results_after_dedupe": len(deduped_results),
            "search_rejection_reasons": dict(search_rejection_reasons),
            "fetch_attempted": int(fetch_diag.get("attempted_urls", 0) or 0),
            "fetch_success": int(fetch_diag.get("accepted_docs", 0) or 0),
            "fetch_failed": int(fetch_diag.get("rejected_docs", 0) or 0),
            "fetch_rejection_reasons": fetch_diag.get("rejection_reasons", {}) or {},
            "rank_rejection_reasons": dict(rank_rejections),
            "rank_fallback_used": rank_fallback_used,
            "selection_rejection_reasons": dict(selection_rejections),
            "selection_backfill_count": backfill_count,
            "unique_domains": unique_domains,
            "source_diversity_ok": diversity_ok,
            "stopped_due_to_latency": stopped_due_to_latency,
            "timings_ms": {
                "search": search_phase_duration_ms,
                "fetch": fetch_phase_duration_ms,
                "rank": rank_phase_duration_ms,
                "select": selection_phase_duration_ms,
            },
            "duration_ms": int((time.monotonic() - started_at) * 1000),
        }

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
            "diagnostics": diagnostics,
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
        image_name_codes: list[str] = []
        image_names = context_hints.get("image_names")
        if isinstance(image_names, list):
            for raw_name in image_names[:3]:
                raw = str(raw_name or "").strip()
                if not raw:
                    continue
                stem = raw.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0].strip()
                if not stem:
                    continue
                for token in _to_query_terms(stem, limit=4):
                    lowered = token.lower().strip()
                    if lowered in {"image", "img", "photo", "upload", "file", "screenshot"}:
                        continue
                    if IMAGE_NAME_NOISE_RE.match(lowered):
                        continue
                    if len(token) < 4:
                        continue
                    if token not in image_name_codes:
                        image_name_codes.append(token)
                if len(image_name_codes) >= 3:
                    break
        base_terms = _to_query_terms(candidate_text, limit=8)
        vision_terms = [str(token).strip() for token in (vision.get("tokens") or []) if str(token).strip()]

        variants = [query]
        if sku_candidates:
            variants.append(f"\"{sku_candidates[0][:40]}\" product")
        if image_name_codes:
            variants.append(f"\"{image_name_codes[0][:50]}\" product")
            variants.append(f"\"{image_name_codes[0][:50]}\" site:bunoraa.com")
        if hint_name:
            variants.append(f"\"{hint_name}\" product details")
        if hint_name and category_name:
            variants.append(f"\"{hint_name}\" \"{category_name}\" buy")
        if category_name and base_terms:
            variants.append(f"{category_name} {' '.join(base_terms[:4])} product")
        if vision_terms:
            variants.append(f"{' '.join(vision_terms[:4])} product details")
        variants = [
            f"{variant} {QUERY_EXCLUSION_SUFFIX}"
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

    def _search_with_fallback_providers(
        self,
        *,
        query: str,
        max_results: int,
        excluded_provider: str,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        if not hasattr(self.search_provider, "provider_order"):
            return [], "none", {}
        original_order = list(self.search_provider.provider_order or [])
        fallback_order = [
            str(provider).strip().lower()
            for provider in original_order
            if str(provider).strip() and str(provider).strip().lower() != excluded_provider
        ]
        if not fallback_order:
            return [], "none", {}
        self.search_provider.provider_order = fallback_order
        try:
            results, provider = self.search_provider.search(query, max_results=max_results)
            diagnostics = (
                self.search_provider.get_last_diagnostics()
                if hasattr(self.search_provider, "get_last_diagnostics")
                else {}
            )
            return results, provider, diagnostics or {}
        finally:
            self.search_provider.provider_order = original_order

    @staticmethod
    def _filter_results_by_anchor_overlap(
        results: list[dict[str, Any]],
        *,
        anchor_terms: list[str],
    ) -> list[dict[str, Any]]:
        if not anchor_terms:
            return list(results or [])
        filtered: list[dict[str, Any]] = []
        for item in results or []:
            haystack = " ".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("snippet") or ""),
                    str(item.get("url") or ""),
                ]
            ).lower()
            if any(term in haystack for term in anchor_terms):
                filtered.append(item)
        return filtered

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
        enforce_anchor_overlap: bool = True,
    ) -> tuple[list[RankedResearchDocument], Counter[str]]:
        min_score = float(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MIN_SCORE", 0.20) or 0.20)
        ranked: list[RankedResearchDocument] = []
        rejections: Counter[str] = Counter()
        terms = reference_terms or query_terms
        anchor_terms = [term for term in terms if term not in QUERY_NOISE_TERMS]

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
                rejections["empty_text"] += 1
                continue
            if UI_NOISE_RE.search(lowered) and not PRODUCT_CUE_RE.search(lowered):
                rejections["ui_noise"] += 1
                continue

            hits = sum(1 for term in terms if term in lowered)
            anchor_hits = sum(1 for term in anchor_terms if term in lowered)
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
                rejections["non_product_signals"] += 1
                continue
            if enforce_anchor_overlap and anchor_terms and anchor_hits == 0 and structured_signals < 2:
                rejections["no_anchor_overlap"] += 1
                continue
            product_likelihood = max(0.0, min(1.0, product_likelihood))

            trust = float(getattr(doc, "trust_score", 0.0) or 0.0)
            score = max(0.0, min(1.0, (0.45 * relevance) + (0.35 * product_likelihood) + (0.20 * trust)))

            if structured_signals >= 2:
                score = max(score, min(1.0, 0.24 + (0.33 * product_likelihood) + (0.20 * trust)))

            if score < min_score:
                rejections["below_min_score"] += 1
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
        return ranked, rejections

    def _backfill_product_docs(
        self,
        *,
        selected_docs: list[ResearchDocument],
        all_docs: list[ResearchDocument],
        max_docs: int,
        anchor_terms: list[str] | None = None,
    ) -> tuple[list[ResearchDocument], int]:
        if len(selected_docs) >= max_docs:
            return selected_docs, 0
        max_domain_repeats = max(
            1,
            int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_DOMAIN_REPEATS", 2) or 2),
        )
        selected = list(selected_docs)
        selected_urls = {str(getattr(doc, "url", "") or "").strip() for doc in selected}
        required_anchor_terms = [
            str(term).strip().lower()
            for term in (anchor_terms or [])
            if str(term).strip() and str(term).strip().lower() not in QUERY_NOISE_TERMS and len(str(term).strip()) >= 4
        ]
        domain_counts: Counter[str] = Counter(
            str(getattr(doc, "domain", "") or "").strip().lower()
            for doc in selected
            if str(getattr(doc, "domain", "") or "").strip()
        )
        candidates: list[ResearchDocument] = []
        for doc in all_docs:
            url = str(getattr(doc, "url", "") or "").strip()
            if not url or url in selected_urls:
                continue
            parsed = urlparse(url)
            path = (parsed.path or "").lower()
            text = " ".join(
                [
                    str(getattr(doc, "title", "") or ""),
                    str(getattr(doc, "snippet", "") or ""),
                    str(getattr(doc, "text", "") or "")[:2200],
                ]
            ).lower()
            structured_signals = _structured_signal_count(doc)
            has_help_path = bool(HELP_PATH_RE.search(path))
            has_commerce_path = bool(COMMERCE_PATH_HINT_RE.search(path))
            has_non_product_cues = bool(NON_PRODUCT_RE.search(text))
            has_product_cues = bool(PRODUCT_CUE_RE.search(text))
            if required_anchor_terms and structured_signals < 2:
                anchor_haystack = " ".join(
                    [
                        url.lower(),
                        str(getattr(doc, "title", "") or "").lower(),
                        str(getattr(doc, "snippet", "") or "").lower(),
                        text,
                    ]
                )
                if not any(term in anchor_haystack for term in required_anchor_terms):
                    continue
            if has_help_path and not has_commerce_path and structured_signals == 0:
                continue
            if has_non_product_cues and not has_product_cues and structured_signals == 0:
                continue
            if UI_NOISE_RE.search(text) and structured_signals == 0 and not PRODUCT_CUE_RE.search(text):
                continue
            if not PRODUCT_CUE_RE.search(text) and structured_signals == 0:
                continue
            candidates.append(doc)
        candidates.sort(key=lambda item: float(getattr(item, "trust_score", 0.0) or 0.0), reverse=True)

        added = 0
        for doc in candidates:
            if len(selected) >= max_docs:
                break
            domain = str(getattr(doc, "domain", "") or "").strip().lower()
            if domain and domain_counts[domain] >= max_domain_repeats:
                continue
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata["deep_research_backfill"] = True
            doc.metadata = metadata
            selected.append(doc)
            selected_urls.add(str(getattr(doc, "url", "") or "").strip())
            if domain:
                domain_counts[domain] += 1
            added += 1

        return selected, added

    def _select_diverse_sources(
        self,
        ranked: list[RankedResearchDocument],
        *,
        max_sources: int,
    ) -> tuple[list[RankedResearchDocument], Counter[str]]:
        max_domain_repeats = max(
            1,
            int(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_MAX_DOMAIN_REPEATS", 2) or 2),
        )
        selected: list[RankedResearchDocument] = []
        rejections: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        for ranked_item in ranked:
            doc = ranked_item.document
            if not doc.url:
                rejections["missing_url"] += 1
                continue
            domain = str(getattr(doc, "domain", "") or "")
            if domain and domain_counts[domain] >= max_domain_repeats:
                rejections["domain_repeat_limit"] += 1
                continue
            selected.append(ranked_item)
            if domain:
                domain_counts[domain] += 1
            if len(selected) >= max_sources:
                break
        return selected, rejections
