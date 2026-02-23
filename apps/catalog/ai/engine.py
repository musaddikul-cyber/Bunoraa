from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.files.storage import default_storage
from django.utils import timezone

from .providers.extractors import build_field_candidates, get_internal_similar_products
from .providers.deep_research import ProductDeepResearchProvider
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
        if len(token) >= 3 and not _looks_query_noise(token)
    ]


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
        self.search_provider = SearchProvider()
        self.research_provider = ResearchProvider()
        self.deep_research_provider = ProductDeepResearchProvider(
            search_provider=self.search_provider,
            research_provider=self.research_provider,
        )
        self.ocr_provider = OCRProvider()
        self.vision_provider = VisionProvider()
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
            vision = self.vision_provider.analyze(image_paths)
            autofill_logger.debug(
                "Autofill vision summary job_id=%s image_count=%s aspect_ratio=%s colors=%s people_present=%s",
                self.job.id,
                vision.get("image_count", 0),
                vision.get("aspect_ratio", ""),
                vision.get("dominant_colors", []),
                bool(vision.get("people_present")),
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
            if self.job.allow_external and bool(getattr(settings, "PRODUCT_AI_ALLOW_EXTERNAL_DEFAULT", True)):
                query = self._build_search_query(
                    candidate_text,
                    ocr=ocr,
                    vision=vision,
                    context_hints=context_hints,
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
                    else:
                        search_results, used_provider = self.search_provider.search(query=query, max_results=10)
                        research_docs = self.research_provider.fetch_documents(search_results, max_docs=8)
                else:
                    used_provider = "none"
            autofill_logger.info(
                "Autofill research summary job_id=%s provider=%s query_present=%s search_results=%s docs=%s",
                self.job.id,
                used_provider,
                bool(query),
                len(search_results),
                len(research_docs),
            )
            self._set_progress(72)

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

            normalized_suggestions = normalize_raw_suggestions(
                raw_suggestions=raw_suggestions,
                confidence_threshold=self.confidence_threshold,
                context_hints=context_hints,
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
                research_docs=research_docs,
                search_provider=used_provider,
                vision=vision,
                similar_products=similar_products,
            )
            self._persist_suggestions(normalized_suggestions)
            self._mark_completed(
                summary={
                    "images_analyzed": len(image_paths),
                    "search_provider": used_provider,
                    "search_query": query if self.job.allow_external else "",
                    "search_result_count": len(search_results),
                    "research_docs_count": len(research_docs),
                    "internal_similar_count": len(similar_products),
                    "deep_research": deep_research_summary,
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
    ) -> str:
        parts = []
        text_tokens = _query_seed_tokens(candidate_text)
        if text_tokens:
            parts.append(" ".join(text_tokens[:8]))
        hint_name = (context_hints.get("name") or "").strip()
        if hint_name:
            hint_tokens = _query_seed_tokens(hint_name)
            if hint_tokens:
                parts.append(" ".join(hint_tokens[:6]))
        primary_category_name = (context_hints.get("primary_category_name") or "").strip()
        if primary_category_name:
            parts.append(primary_category_name[:60])
        sku_candidates = ocr.get("sku_candidates") or []
        if sku_candidates:
            parts.append(str(sku_candidates[0])[:40])
        if vision.get("tokens"):
            vision_tokens = [token for token in vision["tokens"][:4] if not _looks_query_noise(str(token))]
            if vision_tokens:
                parts.append(" ".join(vision_tokens))

        # Avoid broad web search when we do not have enough product-identifying anchors.
        if not any([hint_name, primary_category_name, sku_candidates]) and len(text_tokens) < 3:
            return ""
        if not parts:
            return ""
        parts.append("product details")
        return " ".join(part for part in parts if part).strip()

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

    def _mark_running(self):
        self.job.status = self.job.STATUS_RUNNING
        self.job.started_at = timezone.now()
        self.job.error_message = ""
        self.job.progress = 5
        self.job.save(update_fields=["status", "started_at", "error_message", "progress", "updated_at"])

    def _set_progress(self, value: int):
        self.job.progress = max(0, min(100, int(value)))
        self.job.save(update_fields=["progress", "updated_at"])

    def _mark_completed(self, summary: dict[str, Any]):
        self.job.status = self.job.STATUS_COMPLETED
        self.job.progress = 100
        self.job.completed_at = timezone.now()
        self.job.summary = summary
        self.job.save(update_fields=["status", "progress", "completed_at", "summary", "updated_at"])

    def _fail(self, message: str):
        self.job.status = self.job.STATUS_FAILED
        self.job.error_message = message[:3000]
        self.job.completed_at = timezone.now()
        self.job.save(update_fields=["status", "error_message", "completed_at", "updated_at"])
        autofill_logger.error(
            "Autofill job marked failed job_id=%s message=%s",
            self.job.id,
            self.job.error_message,
        )

    def _cleanup_local_files(self):
        for path in self._local_temp_files:
            try:
                os.remove(path)
            except OSError:
                continue
        self._local_temp_files = []
