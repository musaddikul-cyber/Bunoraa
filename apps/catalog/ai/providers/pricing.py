from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from statistics import median
from typing import Any

from django.conf import settings


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _extract_price_candidates(text: str) -> list[Decimal]:
    content = str(text or "")
    if not content:
        return []

    patterns = (
        r"(?:\$)\s*([0-9]{1,6}(?:[.,][0-9]{1,2})?)",
        r"\b(?:usd|bdt|eur|gbp|inr|cad|aud|taka|tk)\s*([0-9]{1,6}(?:[.,][0-9]{1,2})?)\b",
        r"\b(?:price|sale price|our price|list price|mrp|msrp)\s*[:=]?\s*([0-9]{1,6}(?:[.,][0-9]{1,2})?)\b",
    )

    values: list[Decimal] = []
    for pattern in patterns:
        for raw in re.findall(pattern, content, flags=re.I):
            candidate = str(raw).strip()
            if candidate.count(",") == 1 and "." not in candidate:
                left, right = candidate.split(",", 1)
                if len(right) <= 2:
                    candidate = f"{left}.{right}"
                else:
                    candidate = candidate.replace(",", "")
            else:
                candidate = candidate.replace(",", "")
            parsed = _to_decimal(candidate)
            if parsed is None:
                continue
            if parsed <= 0 or parsed > Decimal("100000"):
                continue
            values.append(parsed)
    return values


def _robust_center(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) <= 2:
        return median(ordered)

    trimmed = ordered[1:-1] if len(ordered) >= 5 else ordered
    return median(trimmed) if trimmed else median(ordered)


def _quantize_price_for_currency(value: Decimal, currency_code: str) -> Decimal:
    code = str(currency_code or "").upper()
    if code in {"BDT", "INR", "PKR", "NPR"}:
        rounded_to_ten = (value / Decimal("10")).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * Decimal("10")
        return rounded_to_ten.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class PricingProvider:
    """
    Estimate pricing/inventory fields from internal + market signals.
    Strict mode removes placeholder values and requires evidence.
    """

    def estimate(
        self,
        *,
        product,
        primary_category,
        research_docs,
        similar_products,
        context_hints=None,
        currency: str = "",
    ) -> dict[str, dict[str, Any]]:
        from apps.catalog.models import CategoryPricingProfile

        strict_mode = bool(getattr(settings, "PRODUCT_AI_STRICT_EVIDENCE_MODE", True))
        allow_heuristic_pricing = bool(getattr(settings, "PRODUCT_AI_ALLOW_HEURISTIC_PRICING", False))
        allow_inventory_defaults = bool(getattr(settings, "PRODUCT_AI_ALLOW_INVENTORY_DEFAULTS", False))

        profile = None
        if primary_category:
            profile = CategoryPricingProfile.objects.filter(category=primary_category, is_active=True).first()
        context_hints = context_hints or {}

        internal_prices: list[Decimal] = []
        for similar in similar_products:
            price = _to_decimal(getattr(similar, "current_price", None) or getattr(similar, "price", None))
            if price:
                internal_prices.append(price)

        market_prices: list[Decimal] = []
        for doc in research_docs:
            market_prices.extend(_extract_price_candidates(getattr(doc, "text", "")))
            market_prices.extend(_extract_price_candidates(getattr(doc, "snippet", "")))
            structured = getattr(doc, "metadata", {}).get("structured", {}) if getattr(doc, "metadata", None) else {}
            for amount in structured.get("price_amounts", [])[:5]:
                parsed = _to_decimal(amount)
                if parsed and parsed > 0:
                    market_prices.append(parsed)

        baseline: Decimal | None = None
        confidence = 0.28
        rationale_parts: list[str] = []

        internal_center = _robust_center(internal_prices)
        if internal_center is not None and (allow_heuristic_pricing or not strict_mode):
            baseline = internal_center
            confidence += 0.24
            rationale_parts.append("internal similar-product pricing")

        market_center = _robust_center(market_prices)
        if market_center is not None:
            if baseline is None:
                baseline = market_center
            else:
                baseline = (baseline + market_center) / Decimal("2")
            confidence += 0.2
            rationale_parts.append("market comparison pricing")

        if baseline is not None and profile and profile.price_floor:
            baseline = max(baseline, profile.price_floor)
            confidence += 0.1
            rationale_parts.append("category price floor")
        elif baseline is not None and product and getattr(product, "price", None):
            existing_price = _to_decimal(getattr(product, "price", None))
            if existing_price and existing_price > 0:
                baseline = max(baseline, existing_price)
                confidence = max(confidence, 0.45)
                rationale_parts.append("existing product baseline")

        if baseline is not None and context_hints.get("name"):
            confidence = min(1.0, confidence + 0.02)
            rationale_parts.append("merchant context hints")

        currency_code = currency or getattr(product, "currency", "")
        if baseline is not None:
            baseline = _quantize_price_for_currency(baseline, currency_code)

        sale_price: Decimal | None = None
        cost: Decimal | None = None
        if baseline is not None:
            min_discount = Decimal(str(getattr(profile, "sale_discount_min_percentage", 5)))
            max_discount = Decimal(str(getattr(profile, "sale_discount_max_percentage", 15)))
            discount = ((min_discount + max_discount) / 2) / Decimal("100")
            sale_price = _quantize_price_for_currency(
                baseline * (Decimal("1.0") - discount),
                currency_code,
            )
            if sale_price <= 0 or sale_price >= baseline:
                sale_price = None

            min_margin = Decimal(str(getattr(profile, "min_margin_percentage", 35))) / Decimal("100")
            cost = _quantize_price_for_currency(
                baseline * (Decimal("1.0") - min_margin),
                currency_code,
            )
            if cost <= 0:
                cost = None

        stock_value: int | None = None
        low_stock_value: int | None = None
        if allow_inventory_defaults or not strict_mode:
            stock_value = int(
                getattr(profile, "stock_default", None)
                or getattr(product, "stock_quantity", 0)
                or 12
            )
            low_stock_value = int(
                getattr(profile, "low_stock_threshold_default", None)
                or getattr(product, "low_stock_threshold", 0)
                or 5
            )

        source_urls = [doc.url for doc in research_docs[:4] if getattr(doc, "url", None)]
        provider_trace = sorted(
            {
                str((getattr(doc, "metadata", {}) or {}).get("provider") or "").strip()
                for doc in research_docs
                if str((getattr(doc, "metadata", {}) or {}).get("provider") or "").strip()
            }
        )
        price_evidence_kind = "web" if source_urls else "none"
        evidence_count = len(source_urls)

        rationale = ""
        if rationale_parts:
            rationale = "Estimated from " + ", ".join(rationale_parts)
        elif baseline is None:
            rationale = "No reliable market evidence was found."

        confidence = max(0.0, min(1.0, confidence))
        if baseline is None:
            confidence = min(confidence, 0.2)

        return {
            "price": {
                "value": baseline,
                "confidence": confidence,
                "rationale": rationale,
                "source_urls": source_urls,
                "metadata": {
                    "evidence_kind": price_evidence_kind,
                    "evidence_count": evidence_count,
                    "provider_trace": provider_trace,
                },
                "low_confidence": confidence < 0.8,
            },
            "sale_price": {
                "value": sale_price,
                "confidence": confidence - 0.03 if sale_price else confidence - 0.15,
                "rationale": "Derived from category discount profile and estimated market position.",
                "source_urls": source_urls[:3],
                "metadata": {
                    "evidence_kind": price_evidence_kind,
                    "evidence_count": min(3, evidence_count),
                    "provider_trace": provider_trace,
                },
            },
            "cost": {
                "value": cost,
                "confidence": max(0.2, confidence - 0.1),
                "rationale": "Reverse-estimated from margin template." if cost is not None else "No reliable cost evidence available.",
                "source_urls": source_urls[:2],
                "metadata": {
                    "evidence_kind": price_evidence_kind,
                    "evidence_count": min(2, evidence_count),
                    "provider_trace": provider_trace,
                },
            },
            "stock_quantity": {
                "value": stock_value,
                "confidence": 0.7 if (profile and stock_value is not None) else (0.5 if stock_value is not None else 0.2),
                "rationale": "Category inventory default." if stock_value is not None else "Inventory defaults disabled in strict mode.",
                "source_urls": [],
                "metadata": {
                    "evidence_kind": "none",
                    "evidence_count": 0,
                    "provider_trace": [],
                },
            },
            "low_stock_threshold": {
                "value": low_stock_value,
                "confidence": 0.7 if (profile and low_stock_value is not None) else (0.5 if low_stock_value is not None else 0.2),
                "rationale": "Category low-stock threshold default." if low_stock_value is not None else "Inventory defaults disabled in strict mode.",
                "source_urls": [],
                "metadata": {
                    "evidence_kind": "none",
                    "evidence_count": 0,
                    "provider_trace": [],
                },
            },
        }
