from __future__ import annotations

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
    ) -> dict[str, dict[str, Any]]:
        from apps.catalog.models import CategoryPricingProfile

        profile = None
        if primary_category:
            profile = CategoryPricingProfile.objects.filter(category=primary_category, is_active=True).first()

        internal_prices = []
        for similar in similar_products:
            price = _to_decimal(getattr(similar, "current_price", None) or getattr(similar, "price", None))
            if price:
                internal_prices.append(price)

        market_prices = []
        for doc in research_docs:
            for token in doc.text.split():
                cleaned = token.replace(",", "")
                if cleaned.startswith("$"):
                    try:
                        market_prices.append(Decimal(cleaned.replace("$", "")))
                    except Exception:
                        continue

        baseline = None
        confidence = 0.32
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

        stock_default = int(getattr(profile, "stock_default", 12))
        low_stock_default = int(getattr(profile, "low_stock_threshold_default", 5))

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
