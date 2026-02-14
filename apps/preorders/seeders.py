from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from django.utils.text import slugify

from core.seed.base import JSONSeedSpec, SeedContext
from core.seed.registry import register_seed
from apps.preorders.models import (
    PreOrderCategory,
    PreOrderOption,
    PreOrderOptionChoice,
    PreOrderTemplate,
)


DEFAULT_PREORDER_TAXONOMY_PATH = "apps/preorders/data/taxonomy.json"

LEGACY_PREORDER_DATA_FILES: dict[str, str] = {
    "categories": "apps/preorders/data/preorder_categories.json",
    "options": "apps/preorders/data/preorder_options.json",
    "option_choices": "apps/preorders/data/preorder_option_choices.json",
    "templates": "apps/preorders/data/preorder_templates.json",
}


def _parse_json_items(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        if "items" in payload:
            payload = payload["items"]
        elif "data" in payload:
            payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError("Expected a list payload or an object containing an 'items'/'data' list.")
    records: list[dict[str, Any]] = []
    for idx, raw in enumerate(payload, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Expected object record at index {idx}.")
        records.append(dict(raw))
    return records


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _load_legacy_taxonomy(ctx: SeedContext) -> dict[str, list[dict[str, Any]]]:
    sections: dict[str, list[dict[str, Any]]] = {}
    for section_name, section_path in LEGACY_PREORDER_DATA_FILES.items():
        path = ctx.resolve_path(section_path)
        if not path.exists():
            sections[section_name] = []
            continue
        sections[section_name] = _parse_json_items(_load_json_file(path))

    # Legacy option_choices lacked category. Infer category only when unambiguous.
    category_by_option: dict[str, set[str]] = defaultdict(set)
    for option in sections.get("options", []):
        category = (option.get("category") or "").strip()
        option_name = (option.get("name") or "").strip()
        if category and option_name:
            category_by_option[option_name].add(category)

    for choice in sections.get("option_choices", []):
        if choice.get("category"):
            continue
        option_name = (choice.get("option") or "").strip()
        categories = category_by_option.get(option_name, set())
        if len(categories) == 1:
            choice["category"] = next(iter(categories))
            continue
        if not categories:
            raise ValueError(
                f"Legacy option choice '{option_name}:{choice.get('value')}' has unknown option reference."
            )
        raise ValueError(
            f"Legacy option choice '{option_name}:{choice.get('value')}' is ambiguous across categories. "
            "Add 'category' to each choice or migrate to apps/preorders/data/taxonomy.json."
        )

    return sections


def _load_preorder_taxonomy(ctx: SeedContext) -> dict[str, list[dict[str, Any]]]:
    env_path = os.environ.get("SEED_PREORDER_TAXONOMY_PATH") or os.environ.get("PREORDER_TAXONOMY_PATH")
    path = ctx.resolve_path(env_path or DEFAULT_PREORDER_TAXONOMY_PATH)

    if path.exists():
        payload = _load_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Preorder taxonomy must be a JSON object. File: {path}")
        taxonomy = {
            "categories": _parse_json_items(payload.get("categories")),
            "options": _parse_json_items(payload.get("options")),
            "option_choices": _parse_json_items(payload.get("option_choices")),
            "templates": _parse_json_items(payload.get("templates")),
        }
    else:
        taxonomy = _load_legacy_taxonomy(ctx)

    if not any(taxonomy.values()):
        raise FileNotFoundError(
            "No preorder taxonomy data found. Expected apps/preorders/data/taxonomy.json "
            "or legacy preorder_* JSON files."
        )

    _validate_preorder_taxonomy(taxonomy)
    return taxonomy


def _validate_preorder_taxonomy(taxonomy: dict[str, list[dict[str, Any]]]) -> None:
    categories = taxonomy.get("categories", [])
    options = taxonomy.get("options", [])
    option_choices = taxonomy.get("option_choices", [])
    templates = taxonomy.get("templates", [])

    category_slugs: set[str] = set()
    for idx, record in enumerate(categories, start=1):
        name = (record.get("name") or "").strip()
        slug = (record.get("slug") or "").strip() or slugify(name)
        if not name:
            raise ValueError(f"categories[{idx}] is missing 'name'.")
        if not slug:
            raise ValueError(f"categories[{idx}] is missing 'slug'.")
        if slug in category_slugs:
            raise ValueError(f"Duplicate category slug '{slug}'.")
        category_slugs.add(slug)
        record["name"] = name
        record["slug"] = slug

    option_type_values = {value for value, _ in PreOrderOption.OPTION_TYPE_CHOICES}
    option_keys: set[tuple[str, str]] = set()
    option_name_by_category: dict[str, set[str]] = defaultdict(set)
    for idx, record in enumerate(options, start=1):
        category_slug = (record.get("category") or "").strip()
        option_name = (record.get("name") or "").strip()
        option_type = (record.get("option_type") or "").strip()

        if not category_slug:
            raise ValueError(f"options[{idx}] is missing 'category'.")
        if category_slug not in category_slugs:
            raise ValueError(f"options[{idx}] references unknown category '{category_slug}'.")
        if not option_name:
            raise ValueError(f"options[{idx}] is missing 'name'.")
        if option_type and option_type not in option_type_values:
            raise ValueError(
                f"options[{idx}] has invalid option_type '{option_type}'. "
                f"Allowed: {', '.join(sorted(option_type_values))}."
            )

        key = (category_slug, option_name.casefold())
        if key in option_keys:
            raise ValueError(
                f"Duplicate option name '{option_name}' in category '{category_slug}'."
            )
        option_keys.add(key)
        option_name_by_category[category_slug].add(option_name)
        record["category"] = category_slug
        record["name"] = option_name

    choice_keys: set[tuple[str, str, str]] = set()
    for idx, record in enumerate(option_choices, start=1):
        category_slug = (record.get("category") or "").strip()
        option_name = (record.get("option") or "").strip()
        value = (record.get("value") or "").strip()
        display_name = (record.get("display_name") or "").strip()

        if not category_slug:
            raise ValueError(f"option_choices[{idx}] is missing 'category'.")
        if category_slug not in category_slugs:
            raise ValueError(f"option_choices[{idx}] references unknown category '{category_slug}'.")
        if not option_name:
            raise ValueError(f"option_choices[{idx}] is missing 'option'.")
        if (category_slug, option_name.casefold()) not in option_keys:
            raise ValueError(
                f"option_choices[{idx}] references unknown option '{option_name}' in category '{category_slug}'."
            )
        if not value:
            raise ValueError(f"option_choices[{idx}] is missing 'value'.")
        if not display_name:
            raise ValueError(f"option_choices[{idx}] is missing 'display_name'.")

        key = (category_slug, option_name.casefold(), value.casefold())
        if key in choice_keys:
            raise ValueError(
                f"Duplicate option choice value '{value}' for option '{option_name}' in '{category_slug}'."
            )
        choice_keys.add(key)

        record["category"] = category_slug
        record["option"] = option_name
        record["value"] = value
        record["display_name"] = display_name

    template_slugs: set[str] = set()
    for idx, record in enumerate(templates, start=1):
        category_slug = (record.get("category") or "").strip()
        name = (record.get("name") or "").strip()
        slug = (record.get("slug") or "").strip() or slugify(name)
        description = (record.get("description") or "").strip()
        default_options = record.get("default_options", {})

        if not category_slug:
            raise ValueError(f"templates[{idx}] is missing 'category'.")
        if category_slug not in category_slugs:
            raise ValueError(f"templates[{idx}] references unknown category '{category_slug}'.")
        if not name:
            raise ValueError(f"templates[{idx}] is missing 'name'.")
        if not description:
            raise ValueError(f"templates[{idx}] is missing 'description'.")
        if not slug:
            raise ValueError(f"templates[{idx}] is missing 'slug'.")
        if slug in template_slugs:
            raise ValueError(f"Duplicate template slug '{slug}'.")
        if default_options is None:
            default_options = {}
        if not isinstance(default_options, dict):
            raise ValueError(f"templates[{idx}].default_options must be an object.")

        for option_name in default_options.keys():
            if option_name not in option_name_by_category[category_slug]:
                raise ValueError(
                    f"templates[{idx}] references unknown default option '{option_name}' "
                    f"for category '{category_slug}'."
                )

        template_slugs.add(slug)
        record["category"] = category_slug
        record["name"] = name
        record["slug"] = slug
        record["description"] = description
        record["default_options"] = default_options


class PreorderTaxonomySectionSeedSpec(JSONSeedSpec):
    section_key: str = ""

    def __init__(self, *, section_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.section_key = section_key

    def load_records(self, ctx: SeedContext) -> list[dict[str, Any]]:
        taxonomy = _load_preorder_taxonomy(ctx)
        return [dict(record) for record in taxonomy.get(self.section_key, [])]


class PreorderOptionChoiceSeedSpec(PreorderTaxonomySectionSeedSpec):
    def _record_key(self, record: dict[str, Any]) -> Any:
        return (
            (record.get("category") or "").strip(),
            (record.get("option") or "").strip().casefold(),
            (record.get("value") or "").strip().casefold(),
        )

    def _obj_key(self, obj: PreOrderOptionChoice) -> Any:
        return (
            obj.option.category.slug,
            obj.option.name.casefold(),
            obj.value.casefold(),
        )

    def _build_lookup(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "option__category__slug": record["category"],
            "option__name": record["option"],
            "value": record["value"],
        }

    def _build_values(self, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        values = dict(record)
        category_slug = values.pop("category", None)
        option_name = values.pop("option", None)
        if not category_slug or not option_name:
            raise ValueError("Option choice record must include 'category' and 'option'.")
        values["option"] = PreOrderOption.objects.get(category__slug=category_slug, name=option_name)
        return values, {}


register_seed(
    PreorderTaxonomySectionSeedSpec(
        name="preorders.categories",
        app_label="preorders",
        model=PreOrderCategory,
        section_key="categories",
        data_path=DEFAULT_PREORDER_TAXONOMY_PATH,
        key_fields=["slug"],
        update_fields=[
            "name",
            "slug",
            "description",
            "icon",
            "base_price",
            "deposit_percentage",
            "min_production_days",
            "max_production_days",
            "requires_design",
            "requires_approval",
            "allow_rush_order",
            "rush_order_fee_percentage",
            "min_quantity",
            "max_quantity",
            "is_active",
            "order",
        ],
    )
)

register_seed(
    PreorderTaxonomySectionSeedSpec(
        name="preorders.options",
        app_label="preorders",
        model=PreOrderOption,
        section_key="options",
        data_path=DEFAULT_PREORDER_TAXONOMY_PATH,
        key_fields=["category__slug", "name"],
        fk_fields={"category": (PreOrderCategory, "slug")},
        update_fields=[
            "category",
            "name",
            "description",
            "option_type",
            "is_required",
            "min_length",
            "max_length",
            "price_modifier",
            "placeholder",
            "help_text",
            "order",
            "is_active",
        ],
        dependencies=["preorders.categories"],
    )
)

register_seed(
    PreorderOptionChoiceSeedSpec(
        name="preorders.option_choices",
        app_label="preorders",
        model=PreOrderOptionChoice,
        section_key="option_choices",
        data_path=DEFAULT_PREORDER_TAXONOMY_PATH,
        key_fields=["category", "option", "value"],
        update_fields=[
            "option",
            "value",
            "display_name",
            "price_modifier",
            "color_code",
            "order",
            "is_active",
        ],
        dependencies=["preorders.options"],
    )
)

register_seed(
    PreorderTaxonomySectionSeedSpec(
        name="preorders.templates",
        app_label="preorders",
        model=PreOrderTemplate,
        section_key="templates",
        data_path=DEFAULT_PREORDER_TAXONOMY_PATH,
        key_fields=["slug"],
        fk_fields={"category": (PreOrderCategory, "slug")},
        update_fields=[
            "name",
            "slug",
            "description",
            "category",
            "default_quantity",
            "base_price",
            "estimated_days",
            "default_options",
            "is_active",
            "is_featured",
            "order",
        ],
        dependencies=["preorders.categories"],
    )
)
