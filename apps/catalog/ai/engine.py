from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone

from .providers.extractors import build_field_candidates, get_internal_similar_products
from .providers.deep_research import ProductDeepResearchProvider
from .providers.nn_vision import NNVisionInferenceError, NNVisionLoadError, NNVisionProvider
from .providers.ocr import OCRProvider
from .providers.personalization import PersonalizationProvider
from .providers.pricing import PricingProvider
from .providers.research import ResearchProvider
from .providers.search import SearchProvider
from .providers.vision import VisionProvider
from .schemas import FieldSuggestionPayload, SourceRecord
from .validators import normalize_raw_suggestions

logger = logging.getLogger(__name__)
autofill_logger = logging.getLogger("bunoraa.catalog.autofill")
QUERY_GENERIC_TOKENS = {
    "product",
    "products",
    "details",
    "detail",
    "model",
    "models",
    "wearing",
    "photo",
    "image",
    "images",
    "apparel",
    "fashion",
    "outfit",
    "style",
    "women",
    "woman",
    "men",
    "man",
    "set",
    "black",
    "white",
    "brown",
    "beige",
    "pink",
    "blue",
    "green",
    "red",
    "yellow",
    "purple",
    "grey",
    "gray",
}
RELAXED_QUERY_STOPWORDS = {
    "product",
    "products",
    "details",
    "detail",
    "photo",
    "photos",
    "image",
    "images",
    "upload",
    "file",
    "tmp",
    "temp",
}


def _looks_query_noise(token: str) -> bool:
    token = (token or "").strip().lower()
    if not token:
        return True
    if token in {
        "requirements",
        "requirement",
        "guideline",
        "guidelines",
        "guide",
        "tutorial",
        "policy",
        "policies",
        "help",
        "lounge",
        "seller",
        "sellers",
        "listings",
    }:
        return True
    if token in {"tmp", "temp", "image", "img", "upload", "file"}:
        return True
    if any(token.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic")):
        return True
    if token.startswith("image(") or re.match(r"^img[_-]?\d+$", token):
        return True
    if "image" in token and any(ch.isdigit() for ch in token):
        return True
    if re.fullmatch(r"[0-9a-f]{8,}", token):
        return True
    if re.match(r"^(tmp|temp)[a-z0-9_-]{3,}$", token):
        return True
    letters = sum(ch.isalpha() for ch in token)
    digits = sum(ch.isdigit() for ch in token)
    vowels = sum(ch in "aeiou" for ch in token if ch.isalpha())
    if letters > 0 and digits > 0 and len(token) >= 8 and vowels <= 1:
        return True
    return False


def _query_seed_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&'().+\-_]*", text or "")
    return [
        token
        for token in tokens
        if len(token) >= 3
        and not _looks_query_noise(token)
        and token.lower().strip("._-") not in QUERY_GENERIC_TOKENS
    ]


def _query_seed_tokens_relaxed(text: str, *, limit: int = 12) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&'().+\-_]*", text or "")
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned = token.strip()
        lowered = cleaned.lower().strip("._-")
        if len(lowered) < 3:
            continue
        if _looks_query_noise(lowered):
            continue
        if lowered in RELAXED_QUERY_STOPWORDS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(cleaned)
        if len(deduped) >= limit:
            break
    return deduped


class ProductAutofillEngine:
    """
    Orchestrates asynchronous product autofill generation.
    """

    def __init__(self, job_id: str):
        from apps.catalog.models import ProductAutofillJob

        self.job = ProductAutofillJob.objects.select_related("product", "requested_by").get(id=job_id)
        self.confidence_threshold = float(
            getattr(settings, "PRODUCT_AI_CONFIDENCE_THRESHOLD", 0.8)
        )
        self.max_images = int(getattr(settings, "PRODUCT_AI_MAX_IMAGES", 4))
        self.product_ai_enabled = bool(getattr(settings, "PRODUCT_AI_ENABLED", False))
        self.strict_evidence_mode = bool(getattr(settings, "PRODUCT_AI_STRICT_EVIDENCE_MODE", True))
        self.strict_evidence_hard_fail = bool(
            getattr(settings, "PRODUCT_AI_STRICT_EVIDENCE_HARD_FAIL", False)
        )
        self.fail_on_research_empty = bool(getattr(settings, "PRODUCT_AI_FAIL_ON_RESEARCH_EMPTY", True))
        self.min_web_sources = max(1, int(getattr(settings, "PRODUCT_AI_MIN_WEB_SOURCES", 3) or 3))
        self.min_web_sources_with_nn = max(
            1,
            int(getattr(settings, "PRODUCT_AI_MIN_WEB_SOURCES_WITH_NN", self.min_web_sources) or self.min_web_sources),
        )
        self.min_high_trust_docs = max(0, int(getattr(settings, "PRODUCT_AI_MIN_HIGH_TRUST_DOCS", 1) or 1))
        self.strict_require_high_trust = bool(
            getattr(settings, "PRODUCT_AI_STRICT_REQUIRE_HIGH_TRUST", False)
        )
        self.max_research_latency_seconds = max(
            10,
            int(getattr(settings, "PRODUCT_AI_MAX_RESEARCH_LATENCY_SECONDS", 90) or 90),
        )
        self.nn_enabled = bool(getattr(settings, "PRODUCT_AI_NN_ENABLED", False))
        self.nn_model_id = str(getattr(settings, "PRODUCT_AI_NN_MODEL_ID", "openai/clip-vit-base-patch32") or "").strip()
        self.nn_device = str(getattr(settings, "PRODUCT_AI_NN_DEVICE", "cpu") or "cpu").strip().lower()
        self.nn_timeout_seconds = float(getattr(settings, "PRODUCT_AI_NN_TIMEOUT_SECONDS", 8.0) or 8.0)
        self.nn_confidence_threshold = float(getattr(settings, "PRODUCT_AI_NN_CONFIDENCE_THRESHOLD", 0.26) or 0.26)
        self.nn_model_cache_dir = str(
            getattr(settings, "PRODUCT_AI_NN_MODEL_CACHE_DIR", str(Path(settings.BASE_DIR) / "ml" / "models_data" / "catalog_product_ai"))
            or ""
        ).strip()
        self.search_provider = SearchProvider()
        self.research_provider = ResearchProvider()
        self.deep_research_provider = ProductDeepResearchProvider(
            search_provider=self.search_provider,
            research_provider=self.research_provider,
        )
        self.ocr_provider = OCRProvider()
        self.vision_provider = VisionProvider()
        self.nn_vision_provider = (
            NNVisionProvider(
                model_id=self.nn_model_id,
                device=self.nn_device,
                timeout_seconds=self.nn_timeout_seconds,
                cache_dir=self.nn_model_cache_dir,
            )
            if self.nn_enabled
            else None
        )
        self.pricing_provider = PricingProvider()
        self.personalization_provider = PersonalizationProvider()
        self._local_temp_files: list[str] = []

    def run(self) -> dict[str, Any]:
        if not self.product_ai_enabled:
            self._fail("Product AI is disabled by configuration.")
            return {"status": "failed", "error": "feature_disabled"}

        if self.job.status == self.job.STATUS_COMPLETED:
            return {"status": "completed", "job_id": str(self.job.id), "reused": True}
        if self.job.status == self.job.STATUS_RUNNING:
            return {"status": "running", "job_id": str(self.job.id), "reused": True}

        self._mark_running()
        started_at = time.monotonic()
        try:
            context_hints = self._get_context_hints()
            payload = self.job.input_payload or {}
            requested_temp_images = payload.get("temp_images") if isinstance(payload.get("temp_images"), list) else []
            autofill_logger.info(
                "Autofill engine started job_id=%s product_id=%s allow_external=%s temp_images=%s locale=%s currency=%s",
                self.job.id,
                getattr(self.job, "product_id", "") or "",
                bool(self.job.allow_external),
                len(requested_temp_images),
                self.job.locale or "",
                self.job.currency or "",
            )
            if context_hints:
                autofill_logger.debug(
                    "Autofill engine context hints job_id=%s keys=%s",
                    self.job.id,
                    sorted(context_hints.keys()),
                )

            image_paths = self._collect_image_paths()
            autofill_logger.info(
                "Autofill image collection job_id=%s requested_temp=%s resolved_images=%s",
                self.job.id,
                len(requested_temp_images),
                len(image_paths),
            )
            if requested_temp_images and not image_paths:
                message = (
                    "Uploaded images could not be read for analysis. "
                    "Please retry with a clear JPEG/PNG/WebP image."
                )
                autofill_logger.error(
                    "Autofill engine failed job_id=%s reason=no_readable_uploaded_images temp_paths=%s",
                    self.job.id,
                    requested_temp_images[: self.max_images],
                )
                self._fail(message)
                return {"status": "failed", "job_id": str(self.job.id), "error": message}

            if not image_paths:
                autofill_logger.warning(
                    "Autofill engine continuing without images job_id=%s product_id=%s",
                    self.job.id,
                    getattr(self.job, "product_id", "") or "",
                )

            self._set_progress(15)
            nn_result = self._default_nn_result(status="disabled")
            if self.nn_enabled:
                try:
                    if not self.nn_vision_provider:
                        raise NNVisionLoadError("NN provider is not initialized.")
                    nn_result = self.nn_vision_provider.analyze(image_paths)
                except NNVisionLoadError as exc:
                    error_message = "Neural vision model is unavailable. Check ProductAI NN model configuration."
                    error_summary = self._build_nn_error_summary(
                        error_code="NN_MODEL_UNAVAILABLE",
                        exception=exc,
                    )
                    self._fail(error_message, error_code="NN_MODEL_UNAVAILABLE", summary=error_summary)
                    return {
                        "status": "failed",
                        "job_id": str(self.job.id),
                        "error": error_message,
                        "error_code": "NN_MODEL_UNAVAILABLE",
                    }
                except NNVisionInferenceError as exc:
                    error_message = "Neural vision inference failed. Retry after verifying runtime dependencies and model cache."
                    error_summary = self._build_nn_error_summary(
                        error_code="NN_INFERENCE_FAILED",
                        exception=exc,
                    )
                    self._fail(error_message, error_code="NN_INFERENCE_FAILED", summary=error_summary)
                    return {
                        "status": "failed",
                        "job_id": str(self.job.id),
                        "error": error_message,
                        "error_code": "NN_INFERENCE_FAILED",
                    }

            vision = self.vision_provider.analyze(image_paths)
            vision = self._merge_nn_into_vision(vision=vision, nn_result=nn_result)
            autofill_logger.debug(
                "Autofill vision summary job_id=%s image_count=%s aspect_ratio=%s colors=%s people_present=%s nn_status=%s nn_confidence=%.4f",
                self.job.id,
                vision.get("image_count", 0),
                vision.get("aspect_ratio", ""),
                vision.get("dominant_colors", []),
                bool(vision.get("people_present")),
                nn_result.get("nn_inference_status", "disabled"),
                float(nn_result.get("nn_confidence", 0.0) or 0.0),
            )
            self._set_progress(30)
            ocr = self.ocr_provider.extract(image_paths)
            ocr_text = str(ocr.get("text") or "")
            autofill_logger.debug(
                "Autofill OCR summary job_id=%s text_chars=%s lines=%s sku_candidates=%s",
                self.job.id,
                len(ocr_text),
                len(ocr.get("lines") or []),
                len(ocr.get("sku_candidates") or []),
            )
            self._set_progress(45)

            context_text = self._context_hint_text(context_hints)
            candidate_text = " ".join(
                chunk
                for chunk in [
                    vision.get("candidate_name"),
                    vision.get("scene_summary"),
                    ocr.get("text"),
                    getattr(self.job.product, "name", ""),
                    context_text,
                ]
                if chunk
            )
            similar_products = get_internal_similar_products(self.job.product, candidate_text, limit=6)
            autofill_logger.debug(
                "Autofill similarity summary job_id=%s similar_products=%s candidate_text_chars=%s",
                self.job.id,
                len(similar_products),
                len(candidate_text or ""),
            )
            self._set_progress(55)

            search_results = []
            used_provider = "none"
            research_docs = []
            query = ""
            deep_research_summary: dict[str, Any] = {}
            research_diagnostics: dict[str, Any] = {}
            research_phase_started = time.monotonic()
            if self.job.allow_external and bool(getattr(settings, "PRODUCT_AI_ALLOW_EXTERNAL_DEFAULT", True)):
                query = self._build_search_query(
                    candidate_text,
                    ocr=ocr,
                    vision=vision,
                    context_hints=context_hints,
                    similar_products=similar_products,
                )
                if query:
                    if bool(getattr(settings, "PRODUCT_AI_DEEP_RESEARCH_ENABLED", True)):
                        deep_result = self.deep_research_provider.run(
                            query=query,
                            candidate_text=candidate_text,
                            ocr=ocr,
                            vision=vision,
                            context_hints=context_hints,
                        )
                        search_results = deep_result.get("search_results", [])
                        research_docs = deep_result.get("documents", [])
                        used_provider = deep_result.get("primary_provider", "none")
                        deep_research_summary = {
                            key: value
                            for key, value in deep_result.items()
                            if key not in {"search_results", "documents"}
                        }
                        research_diagnostics = deep_result.get("diagnostics", {}) or {}
                    else:
                        search_results, used_provider = self.search_provider.search(query=query, max_results=10)
                        provider_diag = (
                            self.search_provider.get_last_diagnostics()
                            if hasattr(self.search_provider, "get_last_diagnostics")
                            else {}
                        )
                        if hasattr(self.research_provider, "fetch_documents_with_diagnostics"):
                            research_docs, fetch_diag = self.research_provider.fetch_documents_with_diagnostics(
                                search_results,
                                max_docs=8,
                            )
                        else:
                            research_docs = self.research_provider.fetch_documents(search_results, max_docs=8)
                            fetch_diag = {
                                "attempted_urls": len(search_results),
                                "accepted_docs": len(research_docs),
                                "rejected_docs": max(0, len(search_results) - len(research_docs)),
                                "rejection_reasons": {},
                            }
                        research_diagnostics = {
                            "query_diagnostics": [
                                {
                                    "query": query,
                                    "provider": used_provider,
                                    "status": provider_diag.get("status", "ok" if used_provider != "none" else "none"),
                                    "search_results_before_filter": len(search_results),
                                    "search_results_after_filter": len(search_results),
                                    "provider_attempts": provider_diag.get("attempts", []),
                                }
                            ],
                            "provider_counts": {used_provider: 1} if used_provider and used_provider != "none" else {},
                            "fetch_attempted": int(fetch_diag.get("attempted_urls", 0) or 0),
                            "fetch_success": int(fetch_diag.get("accepted_docs", 0) or 0),
                            "fetch_failed": int(fetch_diag.get("rejected_docs", 0) or 0),
                            "fetch_rejection_reasons": fetch_diag.get("rejection_reasons", {}) or {},
                        }
                else:
                    used_provider = "none"
                    research_diagnostics = {
                        "query_diagnostics": [],
                        "provider_counts": {},
                        "fetch_attempted": 0,
                        "fetch_success": 0,
                        "fetch_failed": 0,
                        "fetch_rejection_reasons": {},
                        "query_generation": "empty",
                    }
            research_duration_ms = int((time.monotonic() - research_phase_started) * 1000)
            validated_source_count = len(research_docs)
            high_trust_docs = sum(1 for doc in research_docs if float(getattr(doc, "trust_score", 0.0) or 0.0) >= 0.65)
            unique_domains = sorted(
                {
                    str(getattr(doc, "domain", "") or "").lower()
                    for doc in research_docs
                    if str(getattr(doc, "domain", "") or "").strip()
                }
            )
            # Keep a transparent snapshot of web evidence before any strict-gate degradation.
            reported_validated_source_count = validated_source_count
            reported_high_trust_docs = high_trust_docs
            reported_unique_domains = list(unique_domains)
            persisted_research_docs = list(research_docs)
            research_diagnostics = dict(research_diagnostics or {})
            research_diagnostics.setdefault("duration_ms", research_duration_ms)
            research_diagnostics.setdefault("validated_source_count", validated_source_count)
            research_diagnostics.setdefault("high_trust_doc_count", high_trust_docs)
            research_diagnostics.setdefault("unique_domains", unique_domains)
            research_diagnostics.setdefault("query_present", bool(query))
            research_diagnostics.setdefault("search_provider", used_provider)
            nn_confidence = float(nn_result.get("nn_confidence", 0.0) or 0.0)
            nn_status = str(nn_result.get("nn_inference_status", "disabled") or "disabled")
            nn_confident = self._is_nn_confident(nn_result)
            effective_min_required_sources = self._effective_min_required_sources(nn_result)
            configured_min_required_sources = self.min_web_sources
            source_activity = self._build_source_activity(
                search_results=search_results,
                research_docs=research_docs,
            )
            research_diagnostics.setdefault("source_activity", source_activity)
            research_diagnostics.setdefault("nn_confidence", nn_confidence)
            research_diagnostics.setdefault("nn_inference_status", nn_status)
            research_diagnostics.setdefault("nn_gate_confident", nn_confident)
            self._update_summary(
                {
                    "strict_mode": self.strict_evidence_mode,
                    "search_provider": used_provider,
                    "search_query": query if self.job.allow_external else "",
                    "search_result_count": len(search_results),
                    "research_docs_count": len(research_docs),
                    "validated_source_count": validated_source_count,
                    "min_required_sources": effective_min_required_sources,
                    "configured_min_required_sources": configured_min_required_sources,
                    "effective_min_required_sources": effective_min_required_sources,
                    "high_trust_doc_count": high_trust_docs,
                    "research_duration_ms": research_duration_ms,
                    "research_diagnostics": research_diagnostics,
                    "source_activity": source_activity,
                    "nn_enabled": self.nn_enabled,
                    "nn_model_id": self.nn_model_id if self.nn_enabled else "",
                    "nn_inference_status": nn_status,
                    "nn_confidence": nn_confidence,
                    "nn_labels": nn_result.get("nn_labels") or [],
                    "nn_caption_like_summary": str(nn_result.get("nn_caption_like_summary") or ""),
                    "nn_confidence_threshold": self.nn_confidence_threshold,
                    "nn_gate_confident": nn_confident,
                    "strict_require_high_trust": self.strict_require_high_trust,
                }
            )
            autofill_logger.info(
                "Autofill research summary job_id=%s provider=%s query_present=%s search_results=%s docs=%s high_trust_docs=%s duration_ms=%s",
                self.job.id,
                used_provider,
                bool(query),
                len(search_results),
                len(research_docs),
                high_trust_docs,
                research_duration_ms,
            )
            autofill_logger.debug(
                "Autofill research diagnostics job_id=%s diagnostics=%s",
                self.job.id,
                research_diagnostics,
            )
            self._set_progress(72)
            strict_gate_failed = False
            strict_gate_passed = True
            strict_gate_enforced = False
            strict_gate_error_code = ""
            strict_gate_error_message = ""

            if self.strict_evidence_mode and self.job.allow_external:
                strict_ok, error_code, error_message = self._evaluate_strict_research_gate(
                    query=query,
                    used_provider=used_provider,
                    search_results=search_results,
                    research_docs=research_docs,
                    research_diagnostics=research_diagnostics,
                    effective_min_sources=effective_min_required_sources,
                )
                if not strict_ok:
                    strict_gate_failed = True
                    strict_gate_passed = False
                    strict_gate_enforced = self.strict_evidence_hard_fail
                    strict_gate_error_code = error_code
                    strict_gate_error_message = error_message
                    strict_summary = {
                        "strict_mode": True,
                        "strict_gate_passed": False,
                        "strict_gate_failed": True,
                        "strict_gate_enforced": strict_gate_enforced,
                        "strict_gate_error_code": error_code,
                        "strict_gate_error_message": error_message,
                        "error_code": error_code,
                        "min_required_sources": effective_min_required_sources,
                        "configured_min_required_sources": configured_min_required_sources,
                        "effective_min_required_sources": effective_min_required_sources,
                        "validated_source_count": validated_source_count,
                        "search_provider": used_provider,
                        "search_query": query,
                        "search_result_count": len(search_results),
                        "research_docs_count": len(research_docs),
                        "high_trust_doc_count": high_trust_docs,
                        "research_diagnostics": research_diagnostics,
                        "deep_research": deep_research_summary,
                        "source_activity": source_activity,
                        "nn_enabled": self.nn_enabled,
                        "nn_model_id": self.nn_model_id if self.nn_enabled else "",
                        "nn_inference_status": nn_status,
                        "nn_confidence": nn_confidence,
                        "nn_labels": nn_result.get("nn_labels") or [],
                        "nn_caption_like_summary": str(nn_result.get("nn_caption_like_summary") or ""),
                        "nn_confidence_threshold": self.nn_confidence_threshold,
                        "nn_gate_confident": nn_confident,
                        "strict_require_high_trust": self.strict_require_high_trust,
                    }
                    if strict_gate_enforced:
                        self._fail(error_message, error_code=error_code, summary=strict_summary)
                        return {
                            "status": "failed",
                            "job_id": str(self.job.id),
                            "error": error_message,
                            "error_code": error_code,
                        }

                    # Soft-fail strict research gate and continue with non-web evidence.
                    autofill_logger.warning(
                        "Autofill strict gate soft-fail job_id=%s error_code=%s message=%s",
                        self.job.id,
                        error_code,
                        error_message,
                    )
                    research_docs = []
                    validated_source_count = 0
                    high_trust_docs = 0
                    unique_domains = []
                    research_diagnostics = dict(research_diagnostics or {})
                    research_diagnostics["validated_source_count_before_soft_fail"] = reported_validated_source_count
                    research_diagnostics["high_trust_doc_count_before_soft_fail"] = reported_high_trust_docs
                    research_diagnostics["unique_domains_before_soft_fail"] = reported_unique_domains
                    research_diagnostics["strict_gate_failed"] = True
                    research_diagnostics["strict_gate_error_code"] = error_code
                    research_diagnostics["strict_gate_error_message"] = error_message
                    research_diagnostics["strict_gate_action"] = "degraded_to_internal_evidence"
                    research_diagnostics["strict_gate_enforced"] = False

            category = getattr(self.job.product, "primary_category", None)
            if not category and similar_products:
                category = getattr(similar_products[0], "primary_category", None)
            hints = self.personalization_provider.get_hints(
                user=self.job.requested_by,
                category=category,
                locale=self.job.locale,
            )

            extracted = build_field_candidates(
                product=self.job.product,
                vision=vision,
                ocr=ocr,
                research_docs=research_docs,
                internal_similar_products=similar_products,
                personalization_hints=hints,
                context_hints=context_hints,
            )
            pricing = self.pricing_provider.estimate(
                product=self.job.product,
                primary_category=category,
                research_docs=research_docs,
                similar_products=similar_products,
                context_hints=context_hints,
                currency=self.job.currency,
            )
            raw_suggestions = {**extracted, **pricing}
            provider_trace = sorted(
                {
                    str((getattr(doc, "metadata", {}) or {}).get("provider") or "").strip()
                    for doc in research_docs
                    if str((getattr(doc, "metadata", {}) or {}).get("provider") or "").strip()
                }
            )
            for payload in raw_suggestions.values():
                if not isinstance(payload, dict):
                    continue
                metadata = dict(payload.get("metadata") or {})
                source_urls = payload.get("source_urls") or []
                if source_urls and not metadata.get("provider_trace"):
                    metadata["provider_trace"] = provider_trace
                if source_urls and "evidence_kind" not in metadata:
                    metadata["evidence_kind"] = "web"
                    metadata["evidence_count"] = len([url for url in source_urls if str(url).strip()])
                payload["metadata"] = metadata

            normalized_suggestions = normalize_raw_suggestions(
                raw_suggestions=raw_suggestions,
                confidence_threshold=self.confidence_threshold,
                context_hints=context_hints,
                strict_mode=self.strict_evidence_mode,
            )
            non_null_suggestions = sum(
                1 for item in normalized_suggestions if item.value not in (None, "", [])
            )
            high_confidence_suggestions = sum(
                1 for item in normalized_suggestions if item.confidence >= self.confidence_threshold
            )
            low_confidence_suggestions = sum(1 for item in normalized_suggestions if item.low_confidence)
            autofill_logger.info(
                "Autofill suggestion summary job_id=%s total=%s non_null=%s high_confidence=%s low_confidence=%s",
                self.job.id,
                len(normalized_suggestions),
                non_null_suggestions,
                high_confidence_suggestions,
                low_confidence_suggestions,
            )
            self._set_progress(88)

            self._persist_sources(
                research_docs=persisted_research_docs,
                search_provider=used_provider,
                vision=vision,
                similar_products=similar_products,
            )
            self._persist_suggestions(normalized_suggestions)
            self._mark_completed(
                summary={
                    "images_analyzed": len(image_paths),
                    "strict_mode": self.strict_evidence_mode,
                    "strict_gate_passed": strict_gate_passed,
                    "strict_gate_failed": strict_gate_failed,
                    "strict_gate_enforced": strict_gate_enforced,
                    "strict_gate_error_code": strict_gate_error_code,
                    "strict_gate_error_message": strict_gate_error_message,
                    "search_provider": used_provider,
                    "search_query": query if self.job.allow_external else "",
                    "search_result_count": len(search_results),
                    "research_docs_count": reported_validated_source_count,
                    "validated_source_count": reported_validated_source_count,
                    "min_required_sources": effective_min_required_sources,
                    "configured_min_required_sources": configured_min_required_sources,
                    "effective_min_required_sources": effective_min_required_sources,
                    "high_trust_doc_count": reported_high_trust_docs,
                    "research_duration_ms": research_duration_ms,
                    "research_diagnostics": research_diagnostics,
                    "error_code": "",
                    "internal_similar_count": len(similar_products),
                    "deep_research": deep_research_summary,
                    "source_activity": source_activity,
                    "nn_enabled": self.nn_enabled,
                    "nn_model_id": self.nn_model_id if self.nn_enabled else "",
                    "nn_inference_status": nn_status,
                    "nn_confidence": nn_confidence,
                    "nn_labels": nn_result.get("nn_labels") or [],
                    "nn_caption_like_summary": str(nn_result.get("nn_caption_like_summary") or ""),
                    "nn_confidence_threshold": self.nn_confidence_threshold,
                    "nn_gate_confident": nn_confident,
                    "strict_require_high_trust": self.strict_require_high_trust,
                    "confidence_threshold": self.confidence_threshold,
                    "context_hint_keys": sorted(context_hints.keys()),
                    "non_null_suggestions": non_null_suggestions,
                    "high_confidence_suggestions": high_confidence_suggestions,
                }
            )
            autofill_logger.info(
                "Autofill engine completed job_id=%s duration_ms=%s",
                self.job.id,
                int((time.monotonic() - started_at) * 1000),
            )
            return {"status": "completed", "job_id": str(self.job.id)}
        except Exception as exc:
            logger.exception("Product autofill failed for job %s", self.job.id)
            autofill_logger.exception(
                "Autofill engine exception job_id=%s error=%s",
                self.job.id,
                str(exc),
            )
            self._fail(str(exc))
            return {"status": "failed", "job_id": str(self.job.id), "error": str(exc)}
        finally:
            self._cleanup_local_files()

    def _build_search_query(
        self,
        candidate_text: str,
        *,
        ocr: dict[str, Any],
        vision: dict[str, Any],
        context_hints: dict[str, Any],
        similar_products: list[Any] | None = None,
    ) -> str:
        similar_products = similar_products or []
        parts = []
        text_tokens = _query_seed_tokens(candidate_text)
        relaxed_text_tokens = _query_seed_tokens_relaxed(candidate_text, limit=10)
        if text_tokens:
            parts.append(" ".join(text_tokens[:8]))
        elif relaxed_text_tokens:
            parts.append(" ".join(relaxed_text_tokens[:6]))
        hint_name = (context_hints.get("name") or "").strip()
        if hint_name:
            hint_tokens = _query_seed_tokens(hint_name)
            if hint_tokens:
                parts.append(" ".join(hint_tokens[:6]))
        primary_category_name = (context_hints.get("primary_category_name") or "").strip()
        if not primary_category_name:
            for similar in similar_products[:3]:
                category = getattr(similar, "primary_category", None)
                category_name = str(getattr(category, "name", "") or "").strip()
                if category_name:
                    primary_category_name = category_name[:60]
                    break
        if primary_category_name:
            parts.append(primary_category_name[:60])
        sku_candidates = ocr.get("sku_candidates") or []
        if sku_candidates:
            parts.append(str(sku_candidates[0])[:40])
        image_name_codes: list[str] = []
        image_names = context_hints.get("image_names")
        if isinstance(image_names, list):
            for raw_name in image_names[:3]:
                stem = Path(str(raw_name or "").strip()).stem.strip()
                if not stem:
                    continue
                for token in _query_seed_tokens(stem):
                    if len(token) < 4:
                        continue
                    if token not in image_name_codes:
                        image_name_codes.append(token)
                if len(image_name_codes) >= 3:
                    break
        if image_name_codes:
            parts.append(f"\"{image_name_codes[0][:50]}\"")
        vision_tokens: list[str] = []
        if vision.get("tokens"):
            vision_tokens = [token for token in vision["tokens"][:4] if not _looks_query_noise(str(token))]
            if vision_tokens:
                parts.append(" ".join(vision_tokens))

        has_strong_anchor = any([hint_name, primary_category_name, sku_candidates, image_name_codes])
        weak_signal_count = len(text_tokens[:3]) + len(relaxed_text_tokens[:3]) + len(vision_tokens[:3])
        # Avoid broad web search only when both strong anchors and weak visual/text signals are missing.
        if not has_strong_anchor and weak_signal_count < 3:
            return ""
        if not parts:
            return ""
        parts.append("product details")
        return " ".join(part for part in parts if part).strip()

    def _evaluate_strict_research_gate(
        self,
        *,
        query: str,
        used_provider: str,
        search_results: list[dict[str, Any]],
        research_docs: list[Any],
        research_diagnostics: dict[str, Any],
        effective_min_sources: int,
    ) -> tuple[bool, str, str]:
        if not query:
            return (
                False,
                "INSUFFICIENT_WEB_SOURCES",
                "Unable to build a product-identifying deep-research query from the provided evidence.",
            )

        blocked = self._diagnostics_show_blocked(research_diagnostics)
        provider_name = (used_provider or "none").strip().lower()
        if self.fail_on_research_empty and provider_name == "none":
            if blocked:
                return (
                    False,
                    "SEARCH_BLOCKED_OR_CAPTCHA",
                    "Deep research was blocked by search providers or CAPTCHA challenges.",
                )
            return (
                False,
                "SEARCH_PROVIDER_UNAVAILABLE",
                "No configured search provider returned usable results for deep research.",
            )

        validated_source_count = len(research_docs or [])
        min_required = max(1, int(effective_min_sources or self.min_web_sources))
        if validated_source_count < min_required:
            return (
                False,
                "INSUFFICIENT_WEB_SOURCES",
                f"Deep research returned {validated_source_count} validated sources; minimum {min_required} required.",
            )

        high_trust_docs = sum(
            1
            for doc in (research_docs or [])
            if float(getattr(doc, "trust_score", 0.0) or 0.0) >= 0.65
        )
        if self.strict_require_high_trust and high_trust_docs < self.min_high_trust_docs:
            return (
                False,
                "INSUFFICIENT_WEB_SOURCES",
                f"Only {high_trust_docs} high-trust sources were found; minimum {self.min_high_trust_docs} required.",
            )

        unique_domains = {
            str(getattr(doc, "domain", "") or "").lower()
            for doc in (research_docs or [])
            if str(getattr(doc, "domain", "") or "").strip()
        }
        if validated_source_count >= 2 and len(unique_domains) < 2:
            return (
                False,
                "INSUFFICIENT_WEB_SOURCES",
                "Deep research sources lacked domain diversity (minimum 2 unique domains required).",
            )

        duration_ms = int(research_diagnostics.get("duration_ms", 0) or 0)
        if duration_ms > (self.max_research_latency_seconds * 1000):
            return (
                False,
                "INSUFFICIENT_WEB_SOURCES",
                f"Deep research exceeded latency budget ({duration_ms}ms > {self.max_research_latency_seconds * 1000}ms).",
            )

        if not search_results:
            return (
                False,
                "INSUFFICIENT_WEB_SOURCES",
                "Deep research returned no candidate search results.",
            )

        return True, "", ""

    @staticmethod
    def _diagnostics_show_blocked(research_diagnostics: dict[str, Any]) -> bool:
        diagnostics = research_diagnostics or {}
        fetch_rejections = diagnostics.get("fetch_rejection_reasons") or {}
        for reason in fetch_rejections:
            if "challenge" in str(reason).lower() or "captcha" in str(reason).lower():
                return True
        for query_diag in diagnostics.get("query_diagnostics") or []:
            attempts = query_diag.get("provider_attempts") or []
            for attempt in attempts:
                if str(attempt.get("status") or "").lower() == "blocked":
                    return True
                reason = str(attempt.get("reason") or "").lower()
                if "captcha" in reason or "challenge" in reason:
                    return True
        return False

    def _default_nn_result(self, *, status: str) -> dict[str, Any]:
        return {
            "nn_enabled": self.nn_enabled,
            "nn_model_id": self.nn_model_id if self.nn_enabled else "",
            "nn_inference_status": status,
            "nn_confidence": 0.0,
            "nn_labels": [],
            "nn_caption_like_summary": "",
            "nn_tokens": [],
        }

    def _build_nn_error_summary(self, *, error_code: str, exception: Exception) -> dict[str, Any]:
        return {
            "strict_mode": self.strict_evidence_mode,
            "error_code": error_code,
            "nn_enabled": self.nn_enabled,
            "nn_model_id": self.nn_model_id if self.nn_enabled else "",
            "nn_inference_status": "failed",
            "nn_confidence": 0.0,
            "nn_labels": [],
            "nn_caption_like_summary": "",
            "nn_confidence_threshold": self.nn_confidence_threshold,
            "nn_error": {
                "exception_type": exception.__class__.__name__,
                "message": str(exception),
                "model_id": self.nn_model_id,
                "cache_dir": self.nn_model_cache_dir,
                "timeout_seconds": self.nn_timeout_seconds,
            },
            "strict_require_high_trust": self.strict_require_high_trust,
            "configured_min_required_sources": self.min_web_sources,
            "effective_min_required_sources": self.min_web_sources,
            "min_required_sources": self.min_web_sources,
        }

    def _is_nn_confident(self, nn_result: dict[str, Any]) -> bool:
        if not self.nn_enabled:
            return False
        status = str((nn_result or {}).get("nn_inference_status") or "").lower()
        if status != "ok":
            return False
        confidence = float((nn_result or {}).get("nn_confidence", 0.0) or 0.0)
        return confidence >= self.nn_confidence_threshold

    def _effective_min_required_sources(self, nn_result: dict[str, Any]) -> int:
        if not self._is_nn_confident(nn_result):
            return self.min_web_sources
        lowered = max(1, int(self.min_web_sources_with_nn or self.min_web_sources))
        return min(self.min_web_sources, lowered)

    @staticmethod
    def _merge_nn_into_vision(vision: dict[str, Any], nn_result: dict[str, Any]) -> dict[str, Any]:
        merged = dict(vision or {})
        nn_payload = dict(nn_result or {})
        merged.update(
            {
                "nn_inference_status": str(nn_payload.get("nn_inference_status") or "disabled"),
                "nn_confidence": float(nn_payload.get("nn_confidence", 0.0) or 0.0),
                "nn_labels": nn_payload.get("nn_labels") or [],
                "nn_caption_like_summary": str(nn_payload.get("nn_caption_like_summary") or ""),
                "nn_tokens": nn_payload.get("nn_tokens") or [],
            }
        )

        tokens = list(merged.get("tokens") or [])
        tokens.extend(str(token).strip() for token in (nn_payload.get("nn_tokens") or []) if str(token).strip())
        merged["tokens"] = list(dict.fromkeys(tokens))[:20]

        top_label = ""
        labels = nn_payload.get("nn_labels") or []
        if isinstance(labels, list) and labels:
            top_label = str((labels[0] or {}).get("label") or "").strip()
        if top_label:
            existing_name = str(merged.get("candidate_name") or "").strip()
            existing_lower = existing_name.lower()
            if not existing_name or existing_lower in {"product", "item"} or existing_lower.endswith(" product"):
                merged["candidate_name"] = top_label.title()

        nn_caption = str(nn_payload.get("nn_caption_like_summary") or "").strip()
        if nn_caption:
            scene_summary = str(merged.get("scene_summary") or "").strip()
            if not scene_summary:
                merged["scene_summary"] = nn_caption
            elif nn_caption.lower() not in scene_summary.lower():
                merged["scene_summary"] = f"{scene_summary} {nn_caption}".strip()

        return merged

    def _get_context_hints(self) -> dict[str, Any]:
        payload = self.job.input_payload or {}
        hints = payload.get("context_hints")
        return hints if isinstance(hints, dict) else {}

    def _context_hint_text(self, context_hints: dict[str, Any]) -> str:
        chunks = []
        for key in ("name", "primary_category_name"):
            value = context_hints.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
        for key in ("category_names", "tag_names", "eco_certification_names"):
            values = context_hints.get(key)
            if not isinstance(values, list):
                continue
            cleaned = [str(item).strip() for item in values if str(item).strip()]
            if cleaned:
                chunks.append(" ".join(cleaned[:8]))
        image_names = context_hints.get("image_names")
        if isinstance(image_names, list):
            image_tokens: list[str] = []
            for raw_name in image_names[:4]:
                name = str(raw_name or "").strip()
                if not name:
                    continue
                stem = Path(name).stem.strip()
                if not stem:
                    continue
                image_tokens.extend(_query_seed_tokens(stem))
            if image_tokens:
                chunks.append(" ".join(list(dict.fromkeys(image_tokens))[:6]))
        return " ".join(chunks).strip()

    def _collect_image_paths(self) -> list[str]:
        paths: list[str] = []
        seen_paths: set[str] = set()
        payload = self.job.input_payload or {}
        temp_paths = payload.get("temp_images") or []
        for stored_path in temp_paths[: self.max_images]:
            local_path = self._materialize_storage_path(stored_path)
            if local_path and self._is_image_readable(local_path) and local_path not in seen_paths:
                paths.append(local_path)
                seen_paths.add(local_path)
            elif stored_path:
                autofill_logger.warning(
                    "Autofill temp image unavailable job_id=%s storage_path=%s",
                    self.job.id,
                    stored_path,
                )

        if self.job.product_id and len(paths) < self.max_images:
            images = self.job.product.images.order_by("ordering", "-is_primary")[: self.max_images]
            for image in images:
                local_path = self._materialize_file_field(image.image)
                if local_path and self._is_image_readable(local_path) and local_path not in seen_paths:
                    paths.append(local_path)
                    seen_paths.add(local_path)
                if len(paths) >= self.max_images:
                    break

        return paths[: self.max_images]

    def _is_image_readable(self, path: str) -> bool:
        if not path:
            return False
        try:
            from PIL import Image

            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as exc:
            autofill_logger.warning(
                "Autofill unreadable image job_id=%s path=%s error=%s",
                self.job.id,
                path,
                str(exc),
            )
            return False

    def _materialize_storage_path(self, path: str) -> str | None:
        if not path:
            return None
        retries = max(1, int(getattr(settings, "PRODUCT_AI_STORAGE_OPEN_RETRIES", 3)))
        delay_seconds = max(0.0, float(getattr(settings, "PRODUCT_AI_STORAGE_OPEN_RETRY_DELAY_SECONDS", 0.25)))
        for attempt in range(1, retries + 1):
            try:
                with default_storage.open(path, "rb") as source:
                    suffix = Path(path).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(source.read())
                        tmp.flush()
                        self._local_temp_files.append(tmp.name)
                        return tmp.name
            except Exception as exc:
                if attempt >= retries:
                    autofill_logger.warning(
                        "Autofill storage open failed job_id=%s path=%s attempts=%s error=%s",
                        self.job.id,
                        path,
                        retries,
                        str(exc),
                    )
                    return None
                time.sleep(delay_seconds * attempt)
        return None

    def _materialize_file_field(self, file_field) -> str | None:
        if not file_field:
            return None
        try:
            if hasattr(file_field, "path") and file_field.path and os.path.exists(file_field.path):
                return file_field.path
        except Exception as exc:
            autofill_logger.debug(
                "Autofill file path lookup failed job_id=%s file=%s error=%s",
                self.job.id,
                getattr(file_field, "name", ""),
                str(exc),
            )
        try:
            with file_field.open("rb") as source:
                suffix = Path(file_field.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(source.read())
                    tmp.flush()
                    self._local_temp_files.append(tmp.name)
                    return tmp.name
        except Exception as exc:
            autofill_logger.warning(
                "Autofill file field open failed job_id=%s file=%s error=%s",
                self.job.id,
                getattr(file_field, "name", ""),
                str(exc),
            )
            return None

    def _persist_sources(self, *, research_docs, search_provider: str, vision: dict[str, Any], similar_products):
        from apps.catalog.models import ProductAutofillSource

        ProductAutofillSource.objects.filter(job=self.job).delete()
        source_records = []

        for doc in research_docs:
            source_records.append(
                ProductAutofillSource(
                    job=self.job,
                    provider=doc.metadata.get("provider") or search_provider or "web",
                    source_type=ProductAutofillSource.SOURCE_WEB,
                    url=doc.url,
                    domain=doc.domain,
                    title=doc.title,
                    snippet=doc.snippet[:1200],
                    trust_score=doc.trust_score,
                    metadata=doc.metadata,
                )
            )

        for product in similar_products:
            source_records.append(
                ProductAutofillSource(
                    job=self.job,
                    provider="internal_similarity",
                    source_type=ProductAutofillSource.SOURCE_INTERNAL,
                    title=product.name,
                    snippet=f"Internal product id={product.id}",
                    trust_score=0.9,
                    metadata={"product_id": str(product.id)},
                )
            )

        source_records.append(
            ProductAutofillSource(
                job=self.job,
                provider="vision",
                source_type=ProductAutofillSource.SOURCE_IMAGE,
                title="Image analysis",
                snippet=f"Analyzed {vision.get('image_count', 0)} images",
                trust_score=0.85,
                metadata={"vision": vision},
            )
        )

        if source_records:
            ProductAutofillSource.objects.bulk_create(source_records, batch_size=100)

    def _persist_suggestions(self, suggestions: list[FieldSuggestionPayload]):
        from apps.catalog.models import ProductFieldSuggestion

        ProductFieldSuggestion.objects.filter(job=self.job).delete()
        records = []
        for suggestion in suggestions:
            payload = suggestion.to_model_payload()
            records.append(ProductFieldSuggestion(job=self.job, **payload))
        if records:
            ProductFieldSuggestion.objects.bulk_create(records, batch_size=100)
        autofill_logger.debug(
            "Autofill suggestions persisted job_id=%s count=%s",
            self.job.id,
            len(records),
        )

    def _update_summary(self, summary_patch: dict[str, Any]):
        if not isinstance(summary_patch, dict) or not summary_patch:
            return
        merged = dict(self.job.summary or {})
        merged.update(summary_patch)
        self.job.summary = merged
        self.job.save(update_fields=["summary", "updated_at"])

    def _build_source_activity(self, *, search_results: list[dict[str, Any]], research_docs: list[Any]) -> list[dict[str, str]]:
        activity: list[dict[str, str]] = []
        seen: set[str] = set()

        def _normalize_domain(raw_domain: str, raw_url: str) -> str:
            domain = str(raw_domain or "").strip().lower()
            if not domain and raw_url:
                try:
                    domain = urlparse(raw_url).netloc.lower()
                except Exception:
                    domain = ""
            if domain.startswith("www."):
                domain = domain[4:]
            return domain[:255]

        def _append_entry(*, url: str, domain: str, title: str, provider: str, phase: str):
            url = str(url or "").strip()
            domain = _normalize_domain(domain, url)
            if not domain and not url:
                return
            key = f"{domain}|{url or ''}"
            if key in seen:
                return
            seen.add(key)
            activity.append(
                {
                    "url": url or (f"https://{domain}/" if domain else ""),
                    "domain": domain,
                    "title": str(title or "").strip()[:160],
                    "provider": str(provider or "").strip()[:60],
                    "phase": phase,
                }
            )

        for doc in research_docs or []:
            metadata = getattr(doc, "metadata", {}) or {}
            _append_entry(
                url=str(getattr(doc, "url", "") or ""),
                domain=str(getattr(doc, "domain", "") or ""),
                title=str(getattr(doc, "title", "") or ""),
                provider=str(metadata.get("provider") or ""),
                phase="validated",
            )
            if len(activity) >= 12:
                return activity

        for result in search_results or []:
            if not isinstance(result, dict):
                continue
            _append_entry(
                url=str(result.get("url") or ""),
                domain=str(result.get("domain") or ""),
                title=str(result.get("title") or ""),
                provider=str(result.get("provider") or ""),
                phase="candidate",
            )
            if len(activity) >= 12:
                break

        return activity

    def _mark_running(self):
        self.job.status = self.job.STATUS_RUNNING
        self.job.started_at = timezone.now()
        self.job.error_message = ""
        self.job.summary = {}
        self.job.completed_at = None
        self.job.progress = 5
        self.job.save(
            update_fields=[
                "status",
                "started_at",
                "error_message",
                "summary",
                "completed_at",
                "progress",
                "updated_at",
            ]
        )

    def _set_progress(self, value: int):
        self.job.progress = max(0, min(100, int(value)))
        self.job.save(update_fields=["progress", "updated_at"])

    def _mark_completed(self, summary: dict[str, Any]):
        self.job.status = self.job.STATUS_COMPLETED
        self.job.progress = 100
        self.job.error_message = ""
        self.job.completed_at = timezone.now()
        self.job.summary = summary
        self.job.save(update_fields=["status", "progress", "error_message", "completed_at", "summary", "updated_at"])

    def _fail(self, message: str, *, error_code: str = "", summary: dict[str, Any] | None = None):
        self.job.status = self.job.STATUS_FAILED
        self.job.error_message = message[:3000]
        self.job.completed_at = timezone.now()
        existing_summary = dict(self.job.summary or {})
        if summary:
            existing_summary.update(summary)
        if error_code:
            existing_summary["error_code"] = error_code
        existing_summary.setdefault("strict_mode", self.strict_evidence_mode)
        existing_summary.setdefault("min_required_sources", self.min_web_sources)
        existing_summary.setdefault("validated_source_count", 0)
        self.job.summary = existing_summary
        self.job.save(update_fields=["status", "error_message", "completed_at", "summary", "updated_at"])
        autofill_logger.error(
            "Autofill job marked failed job_id=%s error_code=%s message=%s",
            self.job.id,
            error_code or "",
            self.job.error_message,
        )

    def _cleanup_local_files(self):
        for path in self._local_temp_files:
            try:
                os.remove(path)
            except OSError:
                continue
        self._local_temp_files = []
