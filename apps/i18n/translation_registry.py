"""
Translation registry for dynamic content models.

Define which models/fields should be auto-translated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TranslationTarget:
    model_label: str  # e.g., "catalog.Product"
    content_type: str  # matches ContentTranslation.content_type
    fields: List[str]


TRANSLATION_TARGETS: List[TranslationTarget] = [
    TranslationTarget(
        model_label="catalog.Category",
        content_type="category",
        fields=["name", "meta_title", "meta_description", "meta_keywords"],
    ),
    TranslationTarget(
        model_label="catalog.Product",
        content_type="product",
        fields=[
            "name",
            "short_description",
            "description",
            "meta_title",
            "meta_description",
            "meta_keywords",
            "ethical_sourcing_notes",
        ],
    ),
    TranslationTarget(
        model_label="pages.Page",
        content_type="page",
        fields=["title", "content", "excerpt", "meta_title", "meta_description"],
    ),
    TranslationTarget(
        model_label="pages.FAQ",
        content_type="page",
        fields=["question", "answer"],
    ),
    TranslationTarget(
        model_label="pages.BlogPost",
        content_type="page",
        fields=["title", "excerpt", "content", "meta_title", "meta_description", "meta_keywords"],
    ),
    TranslationTarget(
        model_label="pages.BlogCategory",
        content_type="category",
        fields=["name", "description"],
    ),
]


def get_translation_targets() -> List[TranslationTarget]:
    return TRANSLATION_TARGETS
