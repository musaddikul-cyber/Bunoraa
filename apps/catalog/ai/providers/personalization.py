from __future__ import annotations

from collections import Counter
from typing import Any

from django.db.models import Q


class PersonalizationProvider:
    """
    Extract merchant-specific style preferences from accepted feedback history.
    """

    def get_hints(self, *, user, category, locale: str) -> dict[str, Any]:
        from apps.catalog.models import ProductAutofillFeedback

        feedback_qs = ProductAutofillFeedback.objects.filter(
            user=user,
            feedback_type__in=[
                ProductAutofillFeedback.TYPE_ACCEPTED,
                ProductAutofillFeedback.TYPE_EDITED,
            ],
        )
        if category:
            feedback_qs = feedback_qs.filter(
                Q(job__product__primary_category=category) | Q(job__product__categories=category)
            ).distinct()

        recent = feedback_qs.order_by("-created_at")[:80]
        text_pool = []
        for item in recent:
            if item.field_name in {"description", "short_description", "meta_description"}:
                value = item.final_value or item.previous_value
                if isinstance(value, str) and value.strip():
                    text_pool.append(value.strip())

        style = self._derive_style(text_pool, locale=locale)
        return {
            "description_style": style,
            "samples_considered": len(text_pool),
        }

    @staticmethod
    def _derive_style(samples: list[str], locale: str) -> str:
        if not samples:
            return ""
        words = []
        for sample in samples:
            words.extend(token.lower() for token in sample.split() if len(token) > 4)
        common = [token for token, _ in Counter(words).most_common(8)]
        if not common:
            return ""
        locale_prefix = f"[{locale}] " if locale else ""
        return locale_prefix + "Use concise commerce copy with keywords: " + ", ".join(common[:6])
