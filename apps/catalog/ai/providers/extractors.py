from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from django.db.models import Q
from django.utils.text import slugify

from apps.catalog.models import get_default_aspect_ratio_code

CERTIFICATION_KEYWORDS = {
    "fsc": "FSC",
    "fairtrade": "Fairtrade",
    "gots": "GOTS",
    "grs": "GRS",
    "oekotex": "OEKO-TEX",
}

MATERIAL_KEYWORDS = (
    "paper",
    "cardboard",
    "plastic",
    "recycled",
    "compostable",
    "biodegradable",
    "cotton",
    "jute",
    "wood",
)


NOISE_WORDS = {
    "tmp",
    "temp",
    "image",
    "img",
    "upload",
    "file",
    "untitled",
    "screenshot",
    "copy",
}
UUIDISH_RE = re.compile(r"^[0-9a-f]{8,}$", re.I)
TMP_TOKEN_RE = re.compile(r"^(tmp|temp)[a-z0-9_-]{3,}$", re.I)
NON_PRODUCT_PAGE_RE = re.compile(
    r"\b(requirements?|guidelines?|policy|policies|help(?:\s*center)?|support|how\s+to|tutorial|forum|community|lounge|documentation|seller(?:\s+central)?)\b",
    re.I,
)
PRODUCT_CUE_RE = re.compile(
    r"\b(product|price|sku|model|size|material|fabric|cotton|linen|silk|embroid|kurti|dress|shirt|pant|trouser|set|buy|cart|artisan|handmade)\b",
    re.I,
)
PRODUCT_ENTITY_RE = re.compile(
    r"\b(kurti|dress|shirt|pant|trouser|set|saree|blouse|bottle|bag|lamp|table|chair|sofa|mug|vase|shoe|sandal|watch|jewelry|necklace|earring|ring|bracelet|fabric|cotton|linen|silk|artisan|handmade)\b",
    re.I,
)
APPAREL_TEXT_RE = re.compile(
    r"\b(apparel|fashion|clothing|wear|outfit|dress|kurti|saree|shirt|blouse|top|pant|palazzo|women|men)\b",
    re.I,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_text(value: Any, max_chars: int = 4000) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:max_chars]


def _looks_like_noise_token(token: str) -> bool:
    token = (token or "").strip()
    if not token:
        return True
    lower = token.lower()
    if lower in NOISE_WORDS:
        return True
    if TMP_TOKEN_RE.match(lower):
        return True
    compact = lower.replace("-", "").replace("_", "")
    if UUIDISH_RE.match(compact):
        return True
    letters = sum(ch.isalpha() for ch in token)
    digits = sum(ch.isdigit() for ch in token)
    vowels = sum(ch.lower() in "aeiou" for ch in token if ch.isalpha())
    if letters > 0 and digits > 0 and len(token) >= 8 and vowels <= 1:
        return True
    return False


def _is_meaningful_text(value: Any, min_alpha_tokens: int = 2) -> bool:
    text = _clean_text(value)
    if not text:
        return False
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&'().+\-_/]*", text)
    valid_tokens = [
        token
        for token in raw_tokens
        if any(ch.isalpha() for ch in token) and not _looks_like_noise_token(token)
    ]
    if len(valid_tokens) >= min_alpha_tokens:
        return True
    if len(valid_tokens) == 1:
        token = valid_tokens[0]
        return token.isalpha() and len(token) >= 3
    return False


def _sanitize_name_candidate(value: Any) -> str:
    text = _clean_text(value, max_chars=200)
    if not text:
        return ""
    # Drop trailing site-brand segment often present in page titles.
    text = re.split(r"\s+[|\-]\s+", text, maxsplit=1)[0]
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9&'().+\-_/]*", text):
        token = token.strip("._-/")
        if not token or _looks_like_noise_token(token):
            continue
        tokens.append(token)
    candidate = " ".join(tokens[:10]).strip()
    if not _is_meaningful_text(candidate, min_alpha_tokens=1):
        return ""
    if NON_PRODUCT_PAGE_RE.search(candidate) and not PRODUCT_ENTITY_RE.search(candidate):
        return ""
    return candidate[:160]


def _structured_signal_count(structured: dict[str, Any]) -> int:
    signal_keys = ("names", "sku_candidates", "price_amounts", "category_names", "brand_names", "material_hints")
    return sum(1 for key in signal_keys if structured.get(key))


def _doc_structured_payload(doc: Any) -> dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return {}
    structured = metadata.get("structured") or {}
    return structured if isinstance(structured, dict) else {}


def _doc_is_likely_product_page(doc: Any) -> bool:
    structured = _doc_structured_payload(doc)
    signal_count = _structured_signal_count(structured)
    if signal_count >= 1:
        return True

    combined = _clean_text(
        " ".join(
            [
                str(getattr(doc, "title", "") or ""),
                str(getattr(doc, "snippet", "") or ""),
                str(getattr(doc, "text", "") or "")[:420],
            ]
        ),
        max_chars=900,
    )
    if not _is_meaningful_text(combined, min_alpha_tokens=5):
        return False

    if NON_PRODUCT_PAGE_RE.search(combined) and signal_count == 0:
        return False
    return True


def _filter_research_docs(research_docs: list[Any]) -> list[Any]:
    filtered = [doc for doc in research_docs if _doc_is_likely_product_page(doc)]
    return filtered[:8]


def _compact_description_text(value: Any, *, max_chars: int = 800) -> str:
    text = _clean_text(value, max_chars=max_chars * 3)
    if not text:
        return ""

    sentences = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        cleaned = _clean_text(sentence, max_chars=260)
        if not cleaned:
            continue
        if NON_PRODUCT_PAGE_RE.search(cleaned) and not PRODUCT_CUE_RE.search(cleaned):
            continue
        sentences.append(cleaned)
        if len(" ".join(sentences)) >= max_chars:
            break

    if not sentences:
        fallback = _clean_text(text, max_chars=max_chars)
        return fallback

    compact = " ".join(sentences).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0]


def _collect_structured_signals(research_docs: list[Any]) -> dict[str, list[str]]:
    signals = {
        "names": [],
        "descriptions": [],
        "sku_candidates": [],
        "price_amounts": [],
        "brand_names": [],
        "category_names": [],
        "material_hints": [],
    }

    def _add_unique(bucket: str, value: Any, *, max_chars: int = 240):
        cleaned = _clean_text(value, max_chars=max_chars)
        if cleaned and cleaned not in signals[bucket]:
            signals[bucket].append(cleaned)

    for doc in research_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        structured = metadata.get("structured") or {}
        if not isinstance(structured, dict):
            continue
        for key in signals:
            raw_values = structured.get(key) or []
            if not isinstance(raw_values, list):
                continue
            for value in raw_values:
                _add_unique(key, value)

    for key, values in signals.items():
        signals[key] = values[:12]
    return signals


def _context_field_text(context_hints: dict[str, Any], key: str, *, max_chars: int = 400) -> str:
    value = context_hints.get(key)
    if not isinstance(value, str):
        return ""
    return _clean_text(value, max_chars=max_chars)


def _context_list_text(context_hints: dict[str, Any], key: str, *, limit: int = 12) -> list[str]:
    values = context_hints.get(key)
    if not isinstance(values, list):
        return []
    out = []
    for value in values[:limit]:
        cleaned = _clean_text(value, max_chars=220)
        if cleaned:
            out.append(cleaned)
    return out


def _best_name_from_ocr(ocr: dict[str, Any]) -> tuple[str, float, str]:
    for line in (ocr.get("lines") or [])[:8]:
        candidate = _sanitize_name_candidate(line)
        if candidate and len(candidate) <= 140:
            return candidate, 0.76, "Derived from OCR text extracted from uploaded image."
    return "", 0.0, ""


def _best_name_from_research(research_docs: list[Any]) -> tuple[str, float, str]:
    for doc in _filter_research_docs(research_docs)[:5]:
        candidate = _sanitize_name_candidate(getattr(doc, "title", ""))
        if candidate:
            return candidate, 0.74, "Derived from trusted source title agreement."
    return "", 0.0, ""


def _collect_text(vision: dict[str, Any], ocr: dict[str, Any], research_docs: list[Any]) -> str:
    chunks = []
    candidate_name = _sanitize_name_candidate(vision.get("candidate_name"))
    if candidate_name and _is_meaningful_text(candidate_name, min_alpha_tokens=1):
        chunks.append(candidate_name)
    scene_summary = _compact_description_text(
        _clean_text(vision.get("scene_summary"), max_chars=700),
        max_chars=420,
    )
    if _is_meaningful_text(scene_summary, min_alpha_tokens=4):
        chunks.append(scene_summary)
    ocr_text = _clean_text(ocr.get("text"), max_chars=1600)
    if _is_meaningful_text(ocr_text, min_alpha_tokens=3):
        chunks.append(ocr_text)
    for doc in _filter_research_docs(research_docs):
        structured = _doc_structured_payload(doc)
        include_body_text = _structured_signal_count(structured) >= 1
        for part, limit in (
            (getattr(doc, "title", ""), 220),
            (getattr(doc, "snippet", ""), 320),
        ):
            cleaned = _clean_text(part, max_chars=limit)
            if _is_meaningful_text(cleaned, min_alpha_tokens=3):
                chunks.append(cleaned)
        if include_body_text:
            body = _compact_description_text(getattr(doc, "text", ""), max_chars=520)
            if _is_meaningful_text(body, min_alpha_tokens=6):
                chunks.append(body)
    return " ".join(chunk for chunk in chunks if chunk).strip()


def _find_dimension(text: str, axis: str) -> Decimal | None:
    axis_pattern = {
        "length": r"(?:length|l)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|in|inch|inches)?",
        "width": r"(?:width|w)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|in|inch|inches)?",
        "height": r"(?:height|h)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(cm|mm|in|inch|inches)?",
        "weight": r"(?:weight)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(kg|g|lb|oz)?",
    }
    pattern = axis_pattern.get(axis)
    if not pattern:
        return None
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    value = Decimal(match.group(1))
    unit = (match.group(2) or "").lower()
    if axis in {"length", "width", "height"}:
        if unit in {"in", "inch", "inches"}:
            value *= Decimal("2.54")
        elif unit == "mm":
            value *= Decimal("0.1")
    if axis == "weight":
        if unit == "g":
            value /= Decimal("1000")
        elif unit == "lb":
            value *= Decimal("0.453592")
        elif unit == "oz":
            value *= Decimal("0.0283495")
    return value.quantize(Decimal("0.001"))


def _extract_carbon_kg(text: str) -> float | None:
    match = re.search(r"(?:carbon(?:\s+footprint)?|co2e)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*kg", text, re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_recycled_pct(text: str) -> Decimal | None:
    match = re.search(r"(?:recycled(?:\s+content)?)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
    if not match:
        return None
    return Decimal(match.group(1)).quantize(Decimal("0.01"))


def _best_category_name(full_text: str):
    from apps.catalog.models import Category

    best = None
    best_score = 0.0
    tokens = set(slugify(full_text).split("-"))
    categories = list(Category.objects.filter(is_deleted=False).only("id", "name"))
    for category in categories:
        score = 0.0
        for token in slugify(category.name).split("-"):
            if token and token in tokens:
                score += 0.25
        if score > best_score:
            best = category
            best_score = min(1.0, score)

    if best_score >= 0.7:
        return best, best_score

    if APPAREL_TEXT_RE.search(full_text):
        apparel_category_tokens = {
            "apparel",
            "fashion",
            "clothing",
            "wear",
            "womens",
            "mens",
            "dresses",
            "sarees",
            "tops",
            "bottoms",
        }
        fallback_best = best
        fallback_score = best_score
        for category in categories:
            category_tokens = set(slugify(category.name).split("-"))
            overlap = category_tokens.intersection(apparel_category_tokens)
            if not overlap:
                continue
            score = min(0.82, 0.70 + (0.03 * len(overlap)))
            if score > fallback_score:
                fallback_best = category
                fallback_score = score
        best = fallback_best
        best_score = fallback_score
    return best, best_score


def _match_tags(full_text: str, limit: int = 8):
    from apps.catalog.models import Tag

    text_norm = slugify(full_text)
    matched = []
    for tag in Tag.objects.all().only("id", "name"):
        tag_norm = slugify(tag.name)
        if tag_norm and tag_norm in text_norm:
            matched.append(tag.name)
        if len(matched) >= limit:
            break
    return matched


def _match_certifications(full_text: str):
    from apps.catalog.models import EcoCertification

    text_norm = slugify(full_text).replace("-", "")
    matches = []
    for cert in EcoCertification.objects.all().only("id", "name", "slug"):
        name_token = slugify(cert.name).replace("-", "")
        slug_token = (cert.slug or "").replace("-", "").lower()
        if (name_token and name_token in text_norm) or (slug_token and slug_token in text_norm):
            matches.append(cert.name)
    for keyword, label in CERTIFICATION_KEYWORDS.items():
        if keyword in text_norm and label not in matches:
            matches.append(label)
    return matches


def _shipping_material_hint(full_text: str) -> str | None:
    text_norm = slugify(full_text)
    for keyword in MATERIAL_KEYWORDS:
        if keyword in text_norm:
            return keyword
    return None


def build_field_candidates(
    *,
    product,
    vision: dict[str, Any],
    ocr: dict[str, Any],
    research_docs: list[Any],
    internal_similar_products: list[Any],
    personalization_hints: dict[str, Any],
    context_hints: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    context_hints = context_hints or {}
    filtered_research_docs = _filter_research_docs(research_docs)
    structured = _collect_structured_signals(filtered_research_docs)

    full_text = _collect_text(vision, ocr, filtered_research_docs)
    context_text = " ".join(
        chunk
        for chunk in [
            _context_field_text(context_hints, "name", max_chars=220),
            _context_field_text(context_hints, "short_description", max_chars=600),
            _context_field_text(context_hints, "description", max_chars=1800),
            _context_field_text(context_hints, "primary_category_name", max_chars=180),
            " ".join(_context_list_text(context_hints, "category_names", limit=10)),
            " ".join(_context_list_text(context_hints, "tag_names", limit=10)),
            " ".join(_context_list_text(context_hints, "eco_certification_names", limit=10)),
        ]
        if chunk
    ).strip()
    if context_text:
        full_text = " ".join([context_text, full_text]).strip()
    source_urls = [doc.url for doc in filtered_research_docs][:8]

    name_candidate = _sanitize_name_candidate(context_hints.get("name"))
    name_confidence = 0.92 if name_candidate else 0.0
    name_rationale = "Derived from merchant-provided context hints."
    if not name_candidate and structured.get("names"):
        name_candidate = _sanitize_name_candidate(structured["names"][0])
        name_confidence = 0.84 if name_candidate else 0.0
        name_rationale = "Derived from structured product metadata."
    if not name_candidate:
        name_candidate = _sanitize_name_candidate(vision.get("candidate_name"))
        if name_candidate:
            vision_tokens = [str(token).lower() for token in (vision.get("tokens") or [])]
            apparel_cues = {"apparel", "fashion", "clothing", "outfit", "dress", "kurti", "saree"}
            strong_visual = bool(apparel_cues.intersection(vision_tokens))
            name_confidence = 0.76 if strong_visual else 0.68
            name_rationale = "Derived from visual evidence."
        else:
            name_confidence = 0.0
            name_rationale = ""
    if not name_candidate:
        name_candidate, name_confidence, name_rationale = _best_name_from_ocr(ocr)
    if not name_candidate:
        name_candidate, name_confidence, name_rationale = _best_name_from_research(filtered_research_docs)

    description_parts = []
    evidence_parts = []
    context_description = _compact_description_text(
        _context_field_text(context_hints, "description", max_chars=2000),
        max_chars=900,
    )
    context_short_description = _context_field_text(context_hints, "short_description", max_chars=800)
    structured_description = _compact_description_text(
        _clean_text((structured.get("descriptions") or [""])[0], max_chars=1600),
        max_chars=900,
    )
    if context_description:
        evidence_parts.append(context_description)
    elif structured_description and _is_meaningful_text(structured_description, min_alpha_tokens=6):
        evidence_parts.append(structured_description)
    if _is_meaningful_text(full_text, min_alpha_tokens=6):
        evidence_parts.append(_compact_description_text(full_text, max_chars=900))
    ocr_text = _compact_description_text(_clean_text(ocr.get("text"), max_chars=900), max_chars=700)
    if _is_meaningful_text(ocr_text, min_alpha_tokens=5):
        evidence_parts.append(ocr_text)
    if evidence_parts and personalization_hints.get("description_style"):
        description_parts.append(personalization_hints["description_style"])
    description_parts.extend(evidence_parts[:2])
    description = _compact_description_text("\n\n".join(part for part in description_parts if part).strip(), max_chars=1000)
    if len(description) < 48:
        description = ""
    short_description = context_short_description if context_short_description else ""
    if description and len(description.split()) >= 8:
        short_description = short_description or description[:320].rsplit(" ", 1)[0]
    short_description = _compact_description_text(short_description, max_chars=320) if short_description else ""
    description_confidence = 0.25
    scene_summary = _clean_text(vision.get("scene_summary"), max_chars=700)
    if description:
        if context_description:
            description_confidence = 0.9
        elif structured_description:
            description_confidence = 0.84
        elif filtered_research_docs:
            description_confidence = 0.82
        elif _is_meaningful_text(scene_summary, min_alpha_tokens=5):
            description_confidence = 0.74
        elif _is_meaningful_text(ocr_text, min_alpha_tokens=5):
            description_confidence = 0.70
        else:
            description_confidence = 0.58

    from apps.catalog.models import Category

    category, category_score = _best_category_name(full_text)
    structured_category_names = " ".join(structured.get("category_names") or [])
    if (not category or category_score < 0.7) and structured_category_names:
        structured_category, structured_score = _best_category_name(structured_category_names)
        if structured_category and structured_score >= category_score:
            category = structured_category
            category_score = max(structured_score, 0.74)
    hint_category_id = _context_field_text(context_hints, "primary_category_id", max_chars=64)
    hint_category_name = _context_field_text(context_hints, "primary_category_name", max_chars=180)
    if hint_category_id:
        hinted_category = Category.objects.filter(id=hint_category_id, is_deleted=False).only("id", "name").first()
        if hinted_category:
            category = hinted_category
            category_score = max(category_score, 0.99)
    if not category and hint_category_name:
        category, category_score = _best_category_name(hint_category_name)
    if not category and internal_similar_products:
        first_similar_category = getattr(internal_similar_products[0], "primary_category", None)
        if first_similar_category:
            category = first_similar_category
            category_score = max(category_score, 0.62)

    hint_category_ids = _context_list_text(context_hints, "category_ids", limit=12)
    hint_category_names = _context_list_text(context_hints, "category_names", limit=12)
    categories_value = []
    if hint_category_ids:
        for index, category_id in enumerate(hint_category_ids):
            categories_value.append(
                {
                    "id": category_id,
                    "name": hint_category_names[index] if index < len(hint_category_names) else "",
                }
            )
    elif category:
        categories_value = [{"id": str(category.id), "name": category.name}]

    tag_names = _match_tags(full_text)
    for tag_name in _context_list_text(context_hints, "tag_names", limit=16):
        if tag_name not in tag_names:
            tag_names.append(tag_name)
    cert_names = _match_certifications(full_text)
    for cert_name in _context_list_text(context_hints, "eco_certification_names", limit=16):
        if cert_name not in cert_names:
            cert_names.append(cert_name)
    shipping_material_hint = _shipping_material_hint(full_text)
    if not shipping_material_hint and structured.get("material_hints"):
        shipping_material_hint = structured["material_hints"][0]
    ethical_notes = ""
    if re.search(r"\b(ethically sourced|fair trade|artisan made|handmade)\b", full_text, re.I):
        ethical_notes = "Evidence indicates ethical or artisan sourcing claims in cited sources."

    weight = _find_dimension(full_text, "weight")
    length = _find_dimension(full_text, "length")
    width = _find_dimension(full_text, "width")
    height = _find_dimension(full_text, "height")
    carbon_kg = _extract_carbon_kg(full_text)
    recycled_pct = _extract_recycled_pct(full_text)

    seo_title = f"{name_candidate} | Bunoraa" if name_candidate else ""
    if category and name_candidate:
        seo_title = f"{name_candidate} - {category.name} | Bunoraa"
    seo_description = short_description[:500] if short_description else ""

    sku_candidates = list(ocr.get("sku_candidates") or [])
    for candidate in structured.get("sku_candidates", []):
        if candidate and candidate not in sku_candidates:
            sku_candidates.append(candidate)

    return {
        "name": {
            "value": name_candidate or None,
            "confidence": name_confidence if name_candidate else 0.2,
            "rationale": name_rationale if name_candidate else "No reliable name evidence available.",
            "source_urls": source_urls,
        },
        "sku": {
            "value": (sku_candidates or [None])[0],
            "confidence": 0.86 if sku_candidates else 0.35,
            "rationale": "Structured/OCR-derived SKU candidate." if sku_candidates else "No reliable SKU evidence found.",
            "source_urls": [],
        },
        "description": {
            "value": description or None,
            "confidence": description_confidence if description else 0.25,
            "rationale": "Built from extracted source and merchant style memory.",
            "source_urls": source_urls,
        },
        "short_description": {
            "value": short_description or None,
            "confidence": max(0.2, min(0.8, description_confidence - 0.05)) if short_description else 0.2,
            "rationale": "Condensed summary from description synthesis.",
            "source_urls": source_urls,
        },
        "primary_category": {
            "value": {"id": str(category.id), "name": category.name} if category else None,
            "confidence": category_score,
            "rationale": "Matched against existing taxonomy from extracted terms.",
            "source_urls": source_urls,
        },
        "categories": {
            "value": categories_value,
            "confidence": 0.95 if hint_category_ids else max(0.0, category_score - 0.02),
            "rationale": "Primary category and close taxonomy match.",
            "source_urls": source_urls,
        },
        "tags": {
            "value": tag_names,
            "confidence": 0.82 if _context_list_text(context_hints, "tag_names", limit=1) else (0.76 if tag_names else 0.25),
            "rationale": "Mapped from extracted keywords to existing tags.",
            "source_urls": source_urls,
        },
        "weight": {
            "value": weight,
            "confidence": 0.72 if weight else 0.2,
            "rationale": "Parsed from product specification-like text.",
            "source_urls": source_urls,
        },
        "length": {
            "value": length,
            "confidence": 0.72 if length else 0.2,
            "rationale": "Parsed from product specification-like text.",
            "source_urls": source_urls,
        },
        "width": {
            "value": width,
            "confidence": 0.72 if width else 0.2,
            "rationale": "Parsed from product specification-like text.",
            "source_urls": source_urls,
        },
        "height": {
            "value": height,
            "confidence": 0.72 if height else 0.2,
            "rationale": "Parsed from product specification-like text.",
            "source_urls": source_urls,
        },
        "shipping_material": {
            "value": shipping_material_hint,
            "confidence": 0.74 if shipping_material_hint else 0.22,
            "rationale": "Material cue inferred from description keywords.",
            "source_urls": source_urls,
        },
        "aspect_ratio": {
            "value": vision.get("aspect_ratio") or get_default_aspect_ratio_code(),
            "confidence": 0.9 if vision.get("aspect_ratio") else 0.5,
            "rationale": "Computed from primary image dimensions.",
            "source_urls": [],
        },
        "carbon_footprint_kg": {
            "value": carbon_kg,
            "confidence": 0.75 if carbon_kg is not None else 0.2,
            "rationale": "Direct extraction when explicit carbon data exists.",
            "source_urls": source_urls,
        },
        "recycled_content_percentage": {
            "value": recycled_pct,
            "confidence": 0.75 if recycled_pct is not None else 0.2,
            "rationale": "Direct extraction when explicit recycled content data exists.",
            "source_urls": source_urls,
        },
        "ethical_sourcing_notes": {
            "value": ethical_notes or None,
            "confidence": 0.8 if ethical_notes else 0.25,
            "rationale": "Generated from ethical sourcing terms in trusted sources.",
            "source_urls": source_urls,
        },
        "eco_certifications": {
            "value": cert_names,
            "confidence": 0.84 if _context_list_text(context_hints, "eco_certification_names", limit=1) else (0.78 if cert_names else 0.22),
            "rationale": "Certification labels matched from web evidence.",
            "source_urls": source_urls,
        },
        "meta_title": {
            "value": seo_title or None,
            "confidence": min(0.9, name_confidence + 0.08) if seo_title else 0.2,
            "rationale": "SEO title generated from name/category intent.",
            "source_urls": [],
        },
        "meta_description": {
            "value": seo_description or None,
            "confidence": max(0.2, min(0.85, description_confidence)) if seo_description else 0.2,
            "rationale": "SEO description generated from synthesized attributes.",
            "source_urls": [],
        },
    }


def get_internal_similar_products(product, candidate_text: str, limit: int = 5):
    from apps.catalog.models import Product

    if not candidate_text:
        return []
    tokens = [token for token in slugify(candidate_text).split("-") if len(token) > 2][:6]
    if not tokens:
        return []
    query = Q()
    for token in tokens:
        query |= Q(name__icontains=token) | Q(description__icontains=token)
    qs = Product.objects.filter(is_deleted=False).filter(query)
    if product and product.pk:
        qs = qs.exclude(pk=product.pk)
    return list(qs.select_related("primary_category")[:limit])
