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
    text = re.split(r"\s+[|–—]\s+", text, maxsplit=1)[0]
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9&'().+\-_/]*", text):
        token = token.strip("._-/")
        if not token or _looks_like_noise_token(token):
            continue
        tokens.append(token)
    candidate = " ".join(tokens[:10]).strip()
    if not _is_meaningful_text(candidate, min_alpha_tokens=1):
        return ""
    return candidate[:160]


def _best_name_from_ocr(ocr: dict[str, Any]) -> tuple[str, float, str]:
    for line in (ocr.get("lines") or [])[:8]:
        candidate = _sanitize_name_candidate(line)
        if candidate and len(candidate) <= 140:
            return candidate, 0.76, "Derived from OCR text extracted from uploaded image."
    return "", 0.0, ""


def _best_name_from_research(research_docs: list[Any]) -> tuple[str, float, str]:
    for doc in research_docs[:5]:
        candidate = _sanitize_name_candidate(getattr(doc, "title", ""))
        if candidate:
            return candidate, 0.74, "Derived from trusted source title agreement."
    return "", 0.0, ""


def _collect_text(vision: dict[str, Any], ocr: dict[str, Any], research_docs: list[Any]) -> str:
    chunks = []
    candidate_name = _sanitize_name_candidate(vision.get("candidate_name"))
    if candidate_name and _is_meaningful_text(candidate_name, min_alpha_tokens=1):
        chunks.append(candidate_name)
    ocr_text = _clean_text(ocr.get("text"), max_chars=1600)
    if _is_meaningful_text(ocr_text, min_alpha_tokens=3):
        chunks.append(ocr_text)
    for doc in research_docs:
        for part, limit in (
            (getattr(doc, "title", ""), 220),
            (getattr(doc, "snippet", ""), 420),
            (getattr(doc, "text", ""), 1200),
        ):
            cleaned = _clean_text(part, max_chars=limit)
            if _is_meaningful_text(cleaned, min_alpha_tokens=3):
                chunks.append(cleaned)
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
    for category in Category.objects.filter(is_deleted=False).only("id", "name"):
        score = 0.0
        for token in slugify(category.name).split("-"):
            if token and token in tokens:
                score += 0.25
        if score > best_score:
            best = category
            best_score = min(1.0, score)
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
) -> dict[str, dict[str, Any]]:
    full_text = _collect_text(vision, ocr, research_docs)
    source_urls = [doc.url for doc in research_docs][:8]

    name_candidate = _sanitize_name_candidate(vision.get("candidate_name"))
    name_confidence = 0.60 if name_candidate else 0.0
    name_rationale = "Derived from visual evidence."
    if not name_candidate:
        name_candidate, name_confidence, name_rationale = _best_name_from_ocr(ocr)
    if not name_candidate:
        name_candidate, name_confidence, name_rationale = _best_name_from_research(research_docs)

    description_parts = []
    evidence_parts = []
    if _is_meaningful_text(full_text, min_alpha_tokens=6):
        evidence_parts.append(full_text[:1400])
    ocr_text = _clean_text(ocr.get("text"), max_chars=900)
    if _is_meaningful_text(ocr_text, min_alpha_tokens=5):
        evidence_parts.append(ocr_text)
    if evidence_parts and personalization_hints.get("description_style"):
        description_parts.append(personalization_hints["description_style"])
    description_parts.extend(evidence_parts[:2])
    description = "\n\n".join(part for part in description_parts if part).strip()
    if len(description) < 48:
        description = ""
    short_description = ""
    if description and len(description.split()) >= 8:
        short_description = description[:320].rsplit(" ", 1)[0]
    description_confidence = 0.25
    if description:
        if research_docs:
            description_confidence = 0.82
        elif _is_meaningful_text(ocr_text, min_alpha_tokens=5):
            description_confidence = 0.70
        else:
            description_confidence = 0.58

    category, category_score = _best_category_name(full_text)
    tag_names = _match_tags(full_text)
    cert_names = _match_certifications(full_text)
    shipping_material_hint = _shipping_material_hint(full_text)
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

    return {
        "name": {
            "value": name_candidate or None,
            "confidence": name_confidence if name_candidate else 0.2,
            "rationale": name_rationale if name_candidate else "No reliable name evidence available.",
            "source_urls": source_urls,
        },
        "sku": {
            "value": (ocr.get("sku_candidates") or [None])[0],
            "confidence": 0.78 if ocr.get("sku_candidates") else 0.35,
            "rationale": "OCR-derived SKU candidate." if ocr.get("sku_candidates") else "No reliable OCR SKU found.",
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
            "value": [{"id": str(category.id), "name": category.name}] if category else [],
            "confidence": max(0.0, category_score - 0.02),
            "rationale": "Primary category and close taxonomy match.",
            "source_urls": source_urls,
        },
        "tags": {
            "value": tag_names,
            "confidence": 0.76 if tag_names else 0.25,
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
            "confidence": 0.68 if shipping_material_hint else 0.22,
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
            "confidence": 0.78 if cert_names else 0.22,
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
