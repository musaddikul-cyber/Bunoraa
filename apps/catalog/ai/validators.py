from __future__ import annotations

import difflib
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from django.utils.text import slugify

from apps.catalog.models import (
    get_active_aspect_ratio_codes,
    get_default_aspect_ratio_code,
)
from core.utils.helpers import generate_sku

from .schemas import AUTOFILL_FIELDS, FieldSuggestionPayload


NULL_IF_LOW_CONFIDENCE_FIELDS = {
    "name",
    "description",
    "short_description",
    "primary_category",
    "categories",
    "tags",
    "weight",
    "length",
    "width",
    "height",
    "shipping_material",
    "carbon_footprint_kg",
    "recycled_content_percentage",
    "sustainability_score",
    "ethical_sourcing_notes",
    "eco_certifications",
    "meta_title",
    "meta_description",
}

FIELD_CONFIDENCE_THRESHOLDS = {
    "name": 0.75,
    "description": 0.72,
    "short_description": 0.72,
    "primary_category": 0.70,
    "categories": 0.65,
    "tags": 0.60,
    "shipping_material": 0.58,
    "weight": 0.62,
    "length": 0.62,
    "width": 0.62,
    "height": 0.62,
    "carbon_footprint_kg": 0.70,
    "recycled_content_percentage": 0.70,
    "sustainability_score": 0.70,
    "ethical_sourcing_notes": 0.72,
    "eco_certifications": 0.66,
    "meta_title": 0.72,
    "meta_description": 0.72,
}


DECIMAL_FIELDS = {"price", "sale_price", "cost", "weight", "length", "width", "height", "recycled_content_percentage"}
INTEGER_FIELDS = {"stock_quantity", "low_stock_threshold"}
TEXT_FIELDS = {
    "name",
    "description",
    "short_description",
    "meta_title",
    "meta_description",
    "ethical_sourcing_notes",
}
TMP_TOKEN_RE = re.compile(r"^(tmp|temp)[a-z0-9_-]{3,}$", re.I)
UUIDISH_RE = re.compile(r"^[0-9a-f]{8,}$", re.I)
NON_PRODUCT_TEXT_RE = re.compile(
    r"\b(requirements?|guidelines?|policy|help(?:\s*center)?|tutorial|forum|community|lounge|documentation)\b",
    re.I,
)
UI_NOISE_TEXT_RE = re.compile(
    r"\b(open\s+media|in\s+modal|skip\s+to|cookie|javascript|sign\s*in|sign\s*up|wishlist|share|"
    r"seller\s+central|product\s+image\s+requirements?)\b",
    re.I,
)


def _allowed_aspect_codes(*, include_code: str | None = None) -> set[str]:
    return get_active_aspect_ratio_codes(include_code=include_code)


def clamp_confidence(value: Any) -> float:
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, casted))


def quantize_decimal(value: Any, places: str = "0.01") -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def clean_text_value(value: Any, max_chars: int = 5000) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    return text[:max_chars]


def text_looks_like_noise(value: Any) -> bool:
    text = clean_text_value(value, max_chars=220)
    if not text:
        return True
    lowered = text.lower()
    if UI_NOISE_TEXT_RE.search(lowered):
        return True
    if NON_PRODUCT_TEXT_RE.search(lowered) and not re.search(
        r"\b(price|size|material|fabric|cotton|linen|silk|kurti|dress|shirt|pant|trouser|set|buy|cart)\b",
        lowered,
    ):
        return True
    if re.search(r"\b\d+\s*/\s*(?:of\s+)?\d+\b", lowered):
        return True
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&'().+\-_]*", text)
    if not tokens:
        return True
    normalized_tokens = [token.strip("._-") for token in tokens if token.strip("._-")]
    if not normalized_tokens:
        return True

    def _token_is_noise(token: str) -> bool:
        if TMP_TOKEN_RE.match(token):
            return True
        compact = token.lower().replace("-", "").replace("_", "")
        if UUIDISH_RE.match(compact):
            return True
        letters = sum(ch.isalpha() for ch in token)
        digits = sum(ch.isdigit() for ch in token)
        vowels = sum(ch.lower() in "aeiou" for ch in token if ch.isalpha())
        if letters > 0 and digits > 0 and len(token) >= 8 and vowels <= 1:
            return True
        return False

    if len(normalized_tokens) == 1 and _token_is_noise(normalized_tokens[0]):
        return True
    if all(_token_is_noise(token) for token in normalized_tokens):
        return True

    alpha_tokens = [
        token for token in tokens
        if any(ch.isalpha() for ch in token)
    ]
    if len(alpha_tokens) == 0:
        return True

    normalized = slugify(text).replace("-", " ").strip()
    words = [word for word in normalized.split() if len(word) >= 2]
    if len(words) >= 10:
        windows = [" ".join(words[index : index + 5]) for index in range(0, max(0, len(words) - 4))]
        seen: set[str] = set()
        repeats = 0
        for window in windows:
            if window in seen:
                repeats += 1
                if repeats >= 2:
                    return True
            else:
                seen.add(window)
    return False


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_norm = slugify(a).replace("-", " ")
    b_norm = slugify(b).replace("-", " ")
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def map_category_value(value: Any):
    from apps.catalog.models import Category

    if value in (None, ""):
        return None, 0.0

    if isinstance(value, dict):
        category_id = value.get("id")
        label = value.get("name", "")
    else:
        category_id = None
        label = str(value)

    if category_id:
        category = Category.objects.filter(id=category_id, is_deleted=False).first()
        if category:
            return category, 1.0

    best = None
    best_score = 0.0
    for category in Category.objects.filter(is_deleted=False).only("id", "name"):
        score = _string_similarity(label, category.name)
        if score > best_score:
            best = category
            best_score = score
    return best, best_score


def map_many_to_many_by_name(model, raw_values: Any, threshold: float = 0.74):
    if not raw_values:
        return [], 0.0

    values = raw_values if isinstance(raw_values, list) else [raw_values]
    matched = []
    confidences = []
    available = list(model.objects.all().only("id", "name"))
    for value in values:
        label = value.get("name") if isinstance(value, dict) else str(value)
        best_item = None
        best_score = 0.0
        for item in available:
            score = _string_similarity(label, item.name)
            if score > best_score:
                best_item = item
                best_score = score
        if best_item and best_score >= threshold:
            matched.append(best_item)
            confidences.append(best_score)
    if not matched:
        return [], 0.0
    return matched, sum(confidences) / len(confidences)


def map_shipping_material_value(value: Any):
    from apps.catalog.models import ShippingMaterial

    if value in (None, ""):
        return None, 0.0

    if isinstance(value, dict):
        material_id = value.get("id")
        label = value.get("name", "")
    else:
        material_id = None
        label = str(value)

    if material_id:
        material = ShippingMaterial.objects.filter(id=material_id).first()
        if material:
            return material, 1.0

    best = None
    best_score = 0.0
    for material in ShippingMaterial.objects.all().only("id", "name"):
        score = _string_similarity(label, material.name)
        if score > best_score:
            best = material
            best_score = score
    return best, best_score


def compute_sustainability_score(carbon_kg: float | None, recycled_pct: Decimal | None, certifications_count: int = 0) -> float | None:
    if carbon_kg is None and recycled_pct is None and certifications_count == 0:
        return None
    recycled = float(recycled_pct or 0) / 100.0
    carbon_score = 1.0
    if carbon_kg is not None:
        carbon_score = max(0.0, 1.0 - min(float(carbon_kg) / 100.0, 1.0))
    cert_bonus = min(certifications_count * 0.1, 0.2)
    score = (0.6 * recycled) + (0.3 * carbon_score) + cert_bonus
    return round(max(0.0, min(1.0, score)), 4)


def _context_hint_text(context_hints: dict[str, Any], key: str, *, max_chars: int = 600) -> str:
    value = context_hints.get(key)
    if not isinstance(value, str):
        return ""
    text = clean_text_value(value, max_chars=max_chars)
    return text or ""


def _context_hint_list(context_hints: dict[str, Any], key: str, *, limit: int = 20) -> list[str]:
    values = context_hints.get(key)
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for item in values[:limit]:
        cleaned = clean_text_value(item, max_chars=180)
        if cleaned:
            out.append(cleaned)
    return out


def normalize_raw_suggestions(
    raw_suggestions: dict[str, dict[str, Any]],
    confidence_threshold: float,
    *,
    context_hints: dict[str, Any] | None = None,
) -> list[FieldSuggestionPayload]:
    """
    Normalize all raw provider output into validated field suggestions.
    """
    from apps.catalog.models import EcoCertification, Tag

    context_hints = context_hints or {}
    normalized: list[FieldSuggestionPayload] = []
    raw = dict(raw_suggestions or {})
    resolved_name_value: str | None = None
    resolved_description_value: str | None = None

    # Hard guarantees
    sku_value = raw.get("sku", {}).get("value")
    if not sku_value:
        raw["sku"] = {
            "value": generate_sku("PRD"),
            "confidence": 0.6,
            "rationale": "Generated fallback SKU because no reliable OCR SKU was found.",
            "source_urls": [],
        }

    default_aspect_code = get_default_aspect_ratio_code()
    if raw.get("aspect_ratio", {}).get("value") not in _allowed_aspect_codes():
        raw["aspect_ratio"] = {
            "value": default_aspect_code,
            "confidence": 0.75,
            "rationale": "Default aspect ratio fallback.",
            "source_urls": [],
        }

    if raw.get("price", {}).get("value") in (None, ""):
        raw["price"] = {
            "value": "10.00",
            "confidence": 0.25,
            "rationale": "Conservative fallback estimate when no strong market comparison was available.",
            "source_urls": [],
            "low_confidence": True,
        }

    for field in AUTOFILL_FIELDS:
        payload = raw.get(field, {})
        value = payload.get("value")
        confidence = clamp_confidence(payload.get("confidence", 0.0))
        rationale = payload.get("rationale", "")
        source_urls = [u for u in (payload.get("source_urls") or []) if isinstance(u, str)]
        metadata = payload.get("metadata") or {}
        low_confidence = bool(payload.get("low_confidence", False))
        is_null = False
        effective_threshold = FIELD_CONFIDENCE_THRESHOLDS.get(field, confidence_threshold)

        if field in DECIMAL_FIELDS:
            value = quantize_decimal(value)
        elif field in INTEGER_FIELDS:
            value = int(value) if value not in (None, "") else None
        elif field in TEXT_FIELDS:
            value = clean_text_value(value)
            if value and text_looks_like_noise(value):
                value = None
                confidence = min(confidence, 0.2)
                low_confidence = True
                is_null = True
                rationale = (rationale + " " if rationale else "") + "Rejected placeholder/noise text."

        if field == "primary_category":
            raw_primary_value = value
            raw_primary_id = ""
            if isinstance(raw_primary_value, dict):
                raw_primary_id = str(raw_primary_value.get("id") or "").strip()
            category, mapped_conf = map_category_value(value)
            if not category:
                hint_category_id = _context_hint_text(context_hints, "primary_category_id", max_chars=64)
                hint_category_name = _context_hint_text(context_hints, "primary_category_name", max_chars=200)
                if hint_category_id:
                    category, mapped_conf = map_category_value({"id": hint_category_id, "name": hint_category_name})
                elif hint_category_name:
                    category, mapped_conf = map_category_value(hint_category_name)
            if category:
                value = str(category.id)
                hint_category_id = _context_hint_text(context_hints, "primary_category_id", max_chars=64)
                if hint_category_id and str(category.id) == str(hint_category_id):
                    confidence = max(confidence, 0.99)
                    rationale = rationale or "Mapped directly from selected category context hint."
                elif raw_primary_id:
                    # A raw AI-proposed UUID is not high-confidence evidence by itself.
                    confidence = max(confidence, min(mapped_conf, 0.62))
                else:
                    confidence = max(confidence, mapped_conf)
                metadata["name"] = category.name
            else:
                value = None

        if field == "categories":
            from apps.catalog.models import Category

            candidates = list(value or [])
            if not candidates:
                hint_ids = _context_hint_list(context_hints, "category_ids", limit=16)
                hint_names = _context_hint_list(context_hints, "category_names", limit=16)
                if hint_ids:
                    candidates = [
                        {
                            "id": hint_id,
                            "name": hint_names[index] if index < len(hint_names) else "",
                        }
                        for index, hint_id in enumerate(hint_ids)
                    ]
                elif hint_names:
                    candidates = hint_names
            categories = []
            confidence_accumulator = []
            hint_ids = set(_context_hint_list(context_hints, "category_ids", limit=20))
            for candidate in candidates:
                cat, mapped_conf = map_category_value(candidate)
                if cat and cat.id not in {c.id for c in categories}:
                    categories.append(cat)
                    candidate_id = ""
                    if isinstance(candidate, dict):
                        candidate_id = str(candidate.get("id") or "").strip()
                    if hint_ids and candidate_id and candidate_id in hint_ids:
                        confidence_accumulator.append(max(mapped_conf, 0.95))
                    elif candidate_id and not hint_ids:
                        confidence_accumulator.append(min(mapped_conf, 0.62))
                    else:
                        confidence_accumulator.append(mapped_conf)
            value = [str(c.id) for c in categories]
            if confidence_accumulator:
                confidence = max(confidence, sum(confidence_accumulator) / len(confidence_accumulator))
            metadata["names"] = list(Category.objects.filter(id__in=value).values_list("name", flat=True))

        if field == "tags":
            tag_candidates = value or []
            if not tag_candidates:
                tag_candidates = _context_hint_list(context_hints, "tag_names", limit=20)
            tag_matches, mapped_conf = map_many_to_many_by_name(Tag, tag_candidates)
            value = [str(tag.id) for tag in tag_matches]
            if tag_matches:
                confidence = max(confidence, mapped_conf)
                metadata["names"] = [tag.name for tag in tag_matches]

        if field == "eco_certifications":
            cert_candidates = value or []
            if not cert_candidates:
                cert_candidates = _context_hint_list(context_hints, "eco_certification_names", limit=16)
            cert_matches, mapped_conf = map_many_to_many_by_name(EcoCertification, cert_candidates)
            value = [str(cert.id) for cert in cert_matches]
            if cert_matches:
                confidence = max(confidence, mapped_conf)
                metadata["names"] = [cert.name for cert in cert_matches]

        if field == "shipping_material":
            material, mapped_conf = map_shipping_material_value(value)
            if material:
                value = str(material.id)
                confidence = max(confidence, mapped_conf)
                metadata["name"] = material.name
            else:
                value = None

        if field == "sale_price":
            price_value = quantize_decimal(raw.get("price", {}).get("value"))
            if value is not None and price_value is not None and value >= price_value:
                value = None

        if field == "cost":
            if value is None:
                price_value = quantize_decimal(raw.get("price", {}).get("value"))
                if price_value is not None:
                    value = (price_value * Decimal("0.65")).quantize(Decimal("0.01"))
                    rationale = rationale or "Estimated from base margin profile."
                    confidence = max(confidence, 0.45)

        if field == "sustainability_score":
            carbon = raw.get("carbon_footprint_kg", {}).get("value")
            recycled = quantize_decimal(raw.get("recycled_content_percentage", {}).get("value"))
            cert_count = len(raw.get("eco_certifications", {}).get("value") or [])
            computed = compute_sustainability_score(float(carbon) if carbon not in (None, "") else None, recycled, cert_count)
            value = computed
            if computed is not None:
                confidence = max(confidence, 0.7)

        if field == "name":
            if not value:
                hint_name = _context_hint_text(context_hints, "name", max_chars=220)
                if hint_name:
                    value = hint_name
                    confidence = max(confidence, 0.9)
                    rationale = rationale or "Used name from current form context."
            if value:
                token_count = len(re.findall(r"[A-Za-z]{2,}", value))
                if token_count == 0:
                    value = None
                    confidence = min(confidence, 0.2)
                    low_confidence = True
                    is_null = True
            resolved_name_value = value

        if field in {"description", "short_description", "meta_description", "ethical_sourcing_notes"} and value:
            word_count = len(re.findall(r"[A-Za-z]{2,}", value))
            if word_count < 4:
                value = None
                confidence = min(confidence, 0.3)
                low_confidence = True
                is_null = True
                rationale = (rationale + " " if rationale else "") + "Text content too short for reliable enrichment."

        if field == "description":
            if not value:
                hint_description = _context_hint_text(context_hints, "description", max_chars=2000)
                if hint_description:
                    value = hint_description
                    confidence = max(confidence, 0.88)
                    rationale = rationale or "Used description from current form context."
            resolved_description_value = value

        if field == "short_description" and not value:
            hint_short = _context_hint_text(context_hints, "short_description", max_chars=600)
            if hint_short:
                value = hint_short
                confidence = max(confidence, 0.85)
                rationale = rationale or "Used short description from current form context."

        if field in {"description", "short_description", "meta_description"} and value and resolved_name_value:
            if slugify(str(value)) == slugify(str(resolved_name_value)):
                value = None
                confidence = min(confidence, 0.25)
                low_confidence = True
                is_null = True
                rationale = (rationale + " " if rationale else "") + "Rejected duplicate name-only content."

        if field == "short_description" and value and not resolved_description_value:
            value = None
            confidence = min(confidence, 0.25)
            low_confidence = True
            is_null = True
            rationale = (rationale + " " if rationale else "") + "Missing reliable long description evidence."

        if field in {"meta_title", "meta_description"} and (not resolved_name_value):
            value = None
            confidence = min(confidence, 0.2)
            low_confidence = True
            is_null = True
            rationale = (rationale + " " if rationale else "") + "SEO fields require a reliable product name."

        if field in NULL_IF_LOW_CONFIDENCE_FIELDS and confidence < effective_threshold:
            value = None if field not in {"tags", "categories", "eco_certifications"} else []
            is_null = True
            rationale = rationale or f"Insufficient evidence for {field} (confidence below threshold)."

        if field == "price" and value is None:
            value = Decimal("10.00")
            low_confidence = True
            confidence = max(confidence, 0.25)
            rationale = rationale or "Fallback price estimate."

        normalized.append(
            FieldSuggestionPayload(
                field_name=field,
                value=value,
                confidence=confidence,
                rationale=rationale,
                source_urls=source_urls,
                metadata=metadata,
                is_null=is_null,
                low_confidence=low_confidence or confidence < effective_threshold,
            )
        )
    return normalized


def is_blank_model_field(product, field_name: str) -> bool:
    value = getattr(product, field_name, None)
    if field_name in {"categories", "tags", "eco_certifications"}:
        return value.count() == 0
    return value in (None, "")


def apply_suggestions_to_product(product, suggestions, force_overwrite: bool = False) -> dict[str, Any]:
    """
    Apply suggestion queryset/list onto a product object.
    """
    from apps.catalog.models import Category, EcoCertification, ShippingMaterial, Tag

    changed_fields: set[str] = set()
    m2m_updates: dict[str, list[str]] = {}
    m2m_changed_fields: set[str] = set()
    applied = 0
    skipped = 0

    for suggestion in suggestions:
        field = suggestion.field_name
        if field not in AUTOFILL_FIELDS:
            continue
        value = suggestion.value_json

        # Never apply null/empty autofill values to model fields. These
        # suggestions are informational (e.g., low-confidence or insufficient
        # evidence) and should not overwrite existing DB values.
        if value in (None, "", []):
            skipped += 1
            continue

        if not force_overwrite and field not in {"categories", "tags", "eco_certifications"} and not is_blank_model_field(product, field):
            skipped += 1
            continue

        if field == "primary_category":
            category = Category.objects.filter(id=value, is_deleted=False).first() if value else None
            if category:
                if str(product.primary_category_id or "") != str(category.id):
                    product.primary_category = category
                    changed_fields.add("primary_category")
                    applied += 1
                else:
                    skipped += 1
            else:
                skipped += 1
            continue

        if field in {"categories", "tags", "eco_certifications"}:
            if not force_overwrite and not is_blank_model_field(product, field):
                skipped += 1
                continue
            ids = [str(v) for v in (value or [])]
            m2m_updates[field] = ids
            continue

        if field == "shipping_material":
            material = ShippingMaterial.objects.filter(id=value).first() if value else None
            if material:
                if str(product.shipping_material_id or "") != str(material.id):
                    product.shipping_material = material
                    changed_fields.add("shipping_material")
                    applied += 1
                else:
                    skipped += 1
            else:
                skipped += 1
            continue

        if field in DECIMAL_FIELDS and value is not None:
            quantized = quantize_decimal(value)
            if quantized is None:
                skipped += 1
                continue
            if getattr(product, field) != quantized:
                setattr(product, field, quantized)
                changed_fields.add(field)
                applied += 1
            else:
                skipped += 1
            continue

        if field in INTEGER_FIELDS and value is not None:
            numeric_value = int(value)
            if getattr(product, field) != numeric_value:
                setattr(product, field, numeric_value)
                changed_fields.add(field)
                applied += 1
            else:
                skipped += 1
            continue

        if field == "aspect_ratio":
            if value not in _allowed_aspect_codes(include_code=str(value) if value else None):
                value = get_default_aspect_ratio_code()
            if getattr(product, field) != value:
                setattr(product, field, value)
                changed_fields.add(field)
                applied += 1
            else:
                skipped += 1
            continue

        if field == "sustainability_score":
            # Derived from material fields; do not directly force if absent.
            continue

        if getattr(product, field) != value:
            setattr(product, field, value)
            changed_fields.add(field)
            applied += 1
        else:
            skipped += 1

    if changed_fields:
        update_fields = sorted(changed_fields | {"updated_at"})
        product.save(update_fields=update_fields)

    if "categories" in m2m_updates:
        categories = Category.objects.filter(id__in=m2m_updates["categories"], is_deleted=False)
        current_ids = {str(v) for v in product.categories.values_list("id", flat=True)}
        target_ids = {str(v) for v in categories.values_list("id", flat=True)}
        if current_ids != target_ids:
            product.categories.set(categories)
            m2m_changed_fields.add("categories")
            applied += 1
        else:
            skipped += 1
        if not product.primary_category_id:
            first_category = categories.first()
            if first_category:
                product.primary_category = first_category
                product.save(update_fields=["primary_category", "updated_at"])
                changed_fields.add("primary_category")
                applied += 1
    if "tags" in m2m_updates:
        tags = Tag.objects.filter(id__in=m2m_updates["tags"])
        current_ids = {str(v) for v in product.tags.values_list("id", flat=True)}
        target_ids = {str(v) for v in tags.values_list("id", flat=True)}
        if current_ids != target_ids:
            product.tags.set(tags)
            m2m_changed_fields.add("tags")
            applied += 1
        else:
            skipped += 1
    if "eco_certifications" in m2m_updates:
        certs = EcoCertification.objects.filter(id__in=m2m_updates["eco_certifications"])
        current_ids = {str(v) for v in product.eco_certifications.values_list("id", flat=True)}
        target_ids = {str(v) for v in certs.values_list("id", flat=True)}
        if current_ids != target_ids:
            product.eco_certifications.set(certs)
            m2m_changed_fields.add("eco_certifications")
            applied += 1
        else:
            skipped += 1

    # Recompute only when sustainability inputs actually changed.
    sustainability_deps = {"carbon_footprint_kg", "recycled_content_percentage", "eco_certifications"}
    if changed_fields.intersection(sustainability_deps) or m2m_changed_fields.intersection(sustainability_deps):
        previous_score = product.sustainability_score
        recomputed_score = product.compute_sustainability_score(save=False)
        if previous_score != recomputed_score:
            product.sustainability_score = recomputed_score
            product.save(update_fields=["sustainability_score", "updated_at"])
            changed_fields.add("sustainability_score")
            applied += 1

    return {
        "applied": applied,
        "skipped": skipped,
        "changed_fields": sorted(changed_fields | m2m_changed_fields),
    }
