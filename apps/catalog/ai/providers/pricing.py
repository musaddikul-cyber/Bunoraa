from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from statistics import mean
from typing import Any


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
        r"(?:\$|€|£|₹|৳)\s*([0-9]{1,6}(?:[.,][0-9]{1,2})?)",
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


class PricingProvider:
    """
    Estimate pricing/inventory fields from internal + market comparison signals.
    """

    def estimate(
        self,
        *,
        product,
        primary_category,
        research_docs,
        similar_products,
        context_hints=None,
    ) -> dict[str, dict[str, Any]]:
        from apps.catalog.models import CategoryPricingProfile

        profile = None
        if primary_category:
            profile = CategoryPricingProfile.objects.filter(category=primary_category, is_active=True).first()
        context_hints = context_hints or {}

        internal_prices = []
        for similar in similar_products:
            price = _to_decimal(getattr(similar, "current_price", None) or getattr(similar, "price", None))
            if price:
                internal_prices.append(price)

        market_prices = []
        for doc in research_docs:
            market_prices.extend(_extract_price_candidates(getattr(doc, "text", "")))
            market_prices.extend(_extract_price_candidates(getattr(doc, "snippet", "")))
            structured = getattr(doc, "metadata", {}).get("structured", {}) if getattr(doc, "metadata", None) else {}
            for amount in structured.get("price_amounts", [])[:5]:
                parsed = _to_decimal(amount)
                if parsed and parsed > 0:
                    market_prices.append(parsed)

        baseline = None
        confidence = 0.28
        rationale_parts = []

        if internal_prices:
            baseline = Decimal(str(mean(internal_prices)))
            confidence += 0.24
            rationale_parts.append("internal similar-product pricing")
        if market_prices:
            market_avg = Decimal(str(mean(market_prices)))
            if baseline is None:
                baseline = market_avg
            else:
                baseline = (baseline + market_avg) / 2
            confidence += 0.2
            rationale_parts.append("market comparison pricing")
        if baseline is None:
            baseline = Decimal("10.00")
            rationale_parts.append("fallback floor estimate")

        if profile and profile.price_floor:
            baseline = max(baseline, profile.price_floor)
            confidence += 0.1
            rationale_parts.append("category price floor")
        elif product and getattr(product, "price", None):
            existing_price = _to_decimal(getattr(product, "price", None))
            if existing_price and existing_price > 0:
                baseline = max(baseline, existing_price)
                confidence = max(confidence, 0.45)
                rationale_parts.append("existing product baseline")
        if context_hints.get("name"):
            confidence = min(1.0, confidence + 0.02)
            rationale_parts.append("merchant context hints")

        baseline = baseline.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        min_discount = Decimal(str(getattr(profile, "sale_discount_min_percentage", 5)))
        max_discount = Decimal(str(getattr(profile, "sale_discount_max_percentage", 15)))
        discount = ((min_discount + max_discount) / 2) / Decimal("100")
        sale_price = (baseline * (Decimal("1.0") - discount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if sale_price <= 0 or sale_price >= baseline:
            sale_price = None

        min_margin = Decimal(str(getattr(profile, "min_margin_percentage", 35))) / Decimal("100")
        cost = (baseline * (Decimal("1.0") - min_margin)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if cost <= 0:
            cost = (baseline * Decimal("0.65")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        stock_default = int(
            getattr(profile, "stock_default", None)
            or getattr(product, "stock_quantity", 0)
            or 12
        )
        low_stock_default = int(
            getattr(profile, "low_stock_threshold_default", None)
            or getattr(product, "low_stock_threshold", 0)
            or 5
        )

        rationale = "Estimated from " + ", ".join(rationale_parts)
        confidence = max(0.0, min(1.0, confidence))

        return {
            "price": {
                "value": baseline,
                "confidence": confidence,
                "rationale": rationale,
                "source_urls": [doc.url for doc in research_docs[:4]],
                "low_confidence": confidence < 0.8,
            },
            "sale_price": {
                "value": sale_price,
                "confidence": confidence - 0.03 if sale_price else confidence - 0.15,
                "rationale": "Derived from category discount profile and estimated market position.",
                "source_urls": [doc.url for doc in research_docs[:3]],
            },
            "cost": {
                "value": cost,
                "confidence": max(0.35, confidence - 0.1),
                "rationale": "Reverse-estimated from margin template.",
                "source_urls": [],
            },
            "stock_quantity": {
                "value": stock_default,
                "confidence": 0.7 if profile else 0.5,
                "rationale": "Category inventory default.",
                "source_urls": [],
            },
            "low_stock_threshold": {
                "value": low_stock_default,
                "confidence": 0.7 if profile else 0.5,
                "rationale": "Category low-stock threshold default.",
                "source_urls": [],
            },
        }
