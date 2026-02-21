from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any


AUTOFILL_FIELDS = (
    "name",
    "sku",
    "description",
    "short_description",
    "primary_category",
    "categories",
    "tags",
    "price",
    "sale_price",
    "cost",
    "stock_quantity",
    "low_stock_threshold",
    "weight",
    "length",
    "width",
    "height",
    "shipping_material",
    "aspect_ratio",
    "carbon_footprint_kg",
    "recycled_content_percentage",
    "sustainability_score",
    "ethical_sourcing_notes",
    "eco_certifications",
    "meta_title",
    "meta_description",
)


def to_json_safe(value: Any) -> Any:
    """
    Convert values to JSON-serializable primitives for JSONField persistence.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    return value


@dataclass(slots=True)
class JobContext:
    job_id: str
    product_id: str | None
    locale: str
    currency: str
    allow_external: bool
    confidence_threshold: float
    max_images: int


@dataclass(slots=True)
class SourceRecord:
    provider: str
    source_type: str
    url: str = ""
    domain: str = ""
    title: str = ""
    snippet: str = ""
    trust_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FieldSuggestionPayload:
    field_name: str
    value: Any
    confidence: float
    rationale: str = ""
    source_urls: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_null: bool = False
    low_confidence: bool = False

    def to_model_payload(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value_json": to_json_safe(self.value),
            "display_value": self._display_value(),
            "confidence": float(self.confidence),
            "rationale": self.rationale,
            "source_urls": to_json_safe(self.source_urls),
            "metadata": to_json_safe(self.metadata),
            "is_null_suggestion": self.is_null,
            "low_confidence": self.low_confidence,
        }

    def _display_value(self) -> str:
        if self.value is None:
            return ""
        if isinstance(self.value, list):
            return ", ".join(str(v) for v in self.value)
        return str(self.value)
