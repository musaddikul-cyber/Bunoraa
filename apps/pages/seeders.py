from __future__ import annotations

import json
import os
from typing import Any

from django.utils.text import slugify

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.pages.models import (
    SiteSettings,
    SocialLink,
    NewsletterIncentive,
    BlogCategory,
    BlogTag,
    FAQ,
    Page,
)
from apps.i18n.models import Currency


DEFAULT_PAGES_TAXONOMY_PATH = "apps/pages/data/taxonomy.json"

LEGACY_PAGES_DATA_FILES: dict[str, str] = {
    "pages": "apps/pages/data/pages.json",
    "site_settings": "apps/pages/data/site_settings.json",
    "newsletter_incentives": "apps/pages/data/newsletter_incentives.json",
    "blog_categories": "apps/pages/data/blog_categories.json",
    "blog_tags": "apps/pages/data/blog_tags.json",
    "faqs": "apps/pages/data/faqs.json",
    "social_links": "apps/pages/data/social_links.json",
}


def _load_json_file(path):
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


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


def _parse_site_settings(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        if "settings" in payload and isinstance(payload["settings"], dict):
            payload = payload["settings"]
        elif "item" in payload and isinstance(payload["item"], dict):
            payload = payload["item"]
    elif isinstance(payload, list):
        payload = payload[0] if payload else {}

    if not isinstance(payload, dict):
        raise ValueError("Site settings payload must be an object.")

    return dict(payload)


def _load_legacy_taxonomy(ctx: SeedContext) -> dict[str, Any]:
    data: dict[str, Any] = {
        "pages": [],
        "site_settings": {},
        "newsletter_incentives": [],
        "blog_categories": [],
        "blog_tags": [],
        "faqs": [],
        "social_links": [],
    }

    for section_name, section_path in LEGACY_PAGES_DATA_FILES.items():
        path = ctx.resolve_path(section_path)
        if not path.exists():
            continue
        payload = _load_json_file(path)
        if section_name == "site_settings":
            data["site_settings"] = _parse_site_settings(payload)
        else:
            data[section_name] = _parse_json_items(payload)

    return data


def _normalize_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _validate_pages_taxonomy(taxonomy: dict[str, Any]) -> None:
    page_slugs: set[str] = set()
    valid_templates = {choice[0] for choice in Page.TEMPLATE_CHOICES}
    for idx, record in enumerate(taxonomy.get("pages", []), start=1):
        title = str(record.get("title") or "").strip()
        if not title:
            raise ValueError(f"pages[{idx}] is missing 'title'.")
        slug = slugify(str(record.get("slug") or "").strip() or title)
        if not slug:
            raise ValueError(f"pages[{idx}] is missing 'slug'.")
        if slug in page_slugs:
            raise ValueError(f"Duplicate page slug '{slug}'.")
        page_slugs.add(slug)

        template = str(record.get("template") or Page.TEMPLATE_DEFAULT).strip() or Page.TEMPLATE_DEFAULT
        if template not in valid_templates:
            allowed = ", ".join(sorted(valid_templates))
            raise ValueError(
                f"pages[{idx}] has invalid 'template' value '{template}'. Allowed: {allowed}."
            )

        menu_order_raw = record.get("menu_order", 0)
        try:
            menu_order = max(0, int(menu_order_raw or 0))
        except (TypeError, ValueError):
            raise ValueError(f"pages[{idx}] has invalid 'menu_order' value '{menu_order_raw}'.")

        record["title"] = title
        record["slug"] = slug
        record["content"] = str(record.get("content") or "")
        record["excerpt"] = str(record.get("excerpt") or "")
        record["meta_title"] = str(record.get("meta_title") or "")
        record["meta_description"] = str(record.get("meta_description") or "")
        record["template"] = template
        record["show_in_header"] = _coerce_bool(record.get("show_in_header"), default=False)
        record["show_in_footer"] = _coerce_bool(record.get("show_in_footer"), default=False)
        record["menu_order"] = menu_order
        record["is_published"] = _coerce_bool(record.get("is_published"), default=True)

    site_settings = taxonomy.get("site_settings") or {}
    if site_settings:
        currency = _normalize_code(site_settings.get("currency"))
        if currency:
            site_settings["currency"] = currency

    blog_category_slugs: set[str] = set()
    for idx, record in enumerate(taxonomy.get("blog_categories", []), start=1):
        name = str(record.get("name") or "").strip()
        if not name:
            raise ValueError(f"blog_categories[{idx}] is missing 'name'.")
        slug = slugify(str(record.get("slug") or "").strip() or name)
        if not slug:
            raise ValueError(f"blog_categories[{idx}] is missing 'slug'.")
        if slug in blog_category_slugs:
            raise ValueError(f"Duplicate blog category slug '{slug}'.")
        blog_category_slugs.add(slug)
        record["name"] = name
        record["slug"] = slug

    blog_tag_slugs: set[str] = set()
    for idx, record in enumerate(taxonomy.get("blog_tags", []), start=1):
        name = str(record.get("name") or "").strip()
        if not name:
            raise ValueError(f"blog_tags[{idx}] is missing 'name'.")
        slug = slugify(str(record.get("slug") or "").strip() or name)
        if not slug:
            raise ValueError(f"blog_tags[{idx}] is missing 'slug'.")
        if slug in blog_tag_slugs:
            raise ValueError(f"Duplicate blog tag slug '{slug}'.")
        blog_tag_slugs.add(slug)
        record["name"] = name
        record["slug"] = slug

    faq_questions: set[str] = set()
    for idx, record in enumerate(taxonomy.get("faqs", []), start=1):
        question = str(record.get("question") or "").strip()
        if not question:
            raise ValueError(f"faqs[{idx}] is missing 'question'.")
        key = question.casefold()
        if key in faq_questions:
            raise ValueError(f"Duplicate FAQ question '{question}'.")
        faq_questions.add(key)
        record["question"] = question

    social_name_site_keys: set[tuple[int | None, str]] = set()
    for idx, record in enumerate(taxonomy.get("social_links", []), start=1):
        name = str(record.get("name") or "").strip()
        url = str(record.get("url") or "").strip()
        site_id = record.get("site")
        if site_id in ("", None):
            site_id = 1
        try:
            site_id_int = int(site_id)
        except (TypeError, ValueError):
            raise ValueError(f"social_links[{idx}] has invalid 'site' value '{site_id}'.")
        if not name:
            raise ValueError(f"social_links[{idx}] is missing 'name'.")
        if not url:
            raise ValueError(f"social_links[{idx}] is missing 'url'.")
        dedupe_key = (site_id_int, name.casefold())
        if dedupe_key in social_name_site_keys:
            raise ValueError(f"Duplicate social link name '{name}' for site {site_id_int}.")
        social_name_site_keys.add(dedupe_key)
        record["name"] = name
        record["url"] = url
        record["site"] = site_id_int


def _load_pages_taxonomy(ctx: SeedContext) -> dict[str, Any]:
    env_path = os.environ.get("SEED_PAGES_TAXONOMY_PATH") or os.environ.get("PAGES_TAXONOMY_PATH")
    path = ctx.resolve_path(env_path or DEFAULT_PAGES_TAXONOMY_PATH)

    if path.exists():
        payload = _load_json_file(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Pages taxonomy must be a JSON object. File: {path}")
        taxonomy = {
            "pages": _parse_json_items(payload.get("pages")),
            "site_settings": _parse_site_settings(payload.get("site_settings")),
            "newsletter_incentives": _parse_json_items(payload.get("newsletter_incentives")),
            "blog_categories": _parse_json_items(payload.get("blog_categories")),
            "blog_tags": _parse_json_items(payload.get("blog_tags")),
            "faqs": _parse_json_items(payload.get("faqs")),
            "social_links": _parse_json_items(payload.get("social_links")),
        }
    else:
        taxonomy = _load_legacy_taxonomy(ctx)

    has_data = any(
        [
            bool(taxonomy.get("pages")),
            bool(taxonomy.get("site_settings")),
            bool(taxonomy.get("newsletter_incentives")),
            bool(taxonomy.get("blog_categories")),
            bool(taxonomy.get("blog_tags")),
            bool(taxonomy.get("faqs")),
            bool(taxonomy.get("social_links")),
        ]
    )
    if not has_data:
        raise FileNotFoundError(
            "No pages taxonomy data found. Expected apps/pages/data/taxonomy.json "
            "or legacy pages data JSON files."
        )

    _validate_pages_taxonomy(taxonomy)
    return taxonomy


class SiteSettingsSeedSpec(SeedSpec):
    name = "pages.site_settings"
    app_label = "pages"
    kind = "prod"
    dependencies = ["i18n.currencies"]
    description = "Seed SiteSettings singleton"

    def apply(self, ctx: SeedContext) -> SeedResult:
        taxonomy = _load_pages_taxonomy(ctx)
        payload = dict(taxonomy.get("site_settings") or {})
        if not payload:
            return SeedResult()

        valid_fields = {field.name for field in SiteSettings._meta.fields if field.name != "id"}
        payload = {field: value for field, value in payload.items() if field in valid_fields}

        currency_code = _normalize_code(payload.pop("currency", None))
        if currency_code:
            payload["currency"] = Currency.objects.get(code=currency_code)

        result = SeedResult()
        obj = SiteSettings.objects.first()
        if obj:
            changed = False
            for field, value in payload.items():
                if getattr(obj, field) != value:
                    if not ctx.dry_run:
                        setattr(obj, field, value)
                    changed = True
            if changed:
                if not ctx.dry_run:
                    obj.save()
                result.updated += 1
        else:
            if ctx.dry_run:
                result.created += 1
            else:
                SiteSettings.objects.create(**payload)
                result.created += 1
        return result


class PagesTaxonomySectionSeedSpec(JSONSeedSpec):
    section_key: str = ""

    def __init__(self, *, section_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.section_key = section_key

    def load_records(self, ctx: SeedContext) -> list[dict[str, Any]]:
        taxonomy = _load_pages_taxonomy(ctx)
        return [dict(record) for record in taxonomy.get(self.section_key, [])]


register_seed(SiteSettingsSeedSpec())

register_seed(
    PagesTaxonomySectionSeedSpec(
        name="pages.pages",
        app_label="pages",
        model=Page,
        section_key="pages",
        data_path=DEFAULT_PAGES_TAXONOMY_PATH,
        key_fields=["slug"],
        update_fields=[
            "title",
            "slug",
            "content",
            "excerpt",
            "meta_title",
            "meta_description",
            "template",
            "show_in_header",
            "show_in_footer",
            "menu_order",
            "is_published",
        ],
        prune=False,
    )
)

register_seed(
    PagesTaxonomySectionSeedSpec(
        name="pages.newsletter_incentives",
        app_label="pages",
        model=NewsletterIncentive,
        section_key="newsletter_incentives",
        data_path=DEFAULT_PAGES_TAXONOMY_PATH,
        key_fields=["discount_code"],
        update_fields=[
            "title",
            "description",
            "discount_percentage",
            "discount_code",
            "min_order_amount",
            "max_uses",
            "is_active",
            "valid_until",
        ],
    )
)

register_seed(
    PagesTaxonomySectionSeedSpec(
        name="pages.blog_categories",
        app_label="pages",
        model=BlogCategory,
        section_key="blog_categories",
        data_path=DEFAULT_PAGES_TAXONOMY_PATH,
        key_fields=["slug"],
        update_fields=["name", "slug", "description", "icon"],
    )
)

register_seed(
    PagesTaxonomySectionSeedSpec(
        name="pages.blog_tags",
        app_label="pages",
        model=BlogTag,
        section_key="blog_tags",
        data_path=DEFAULT_PAGES_TAXONOMY_PATH,
        key_fields=["slug"],
        update_fields=["name", "slug"],
    )
)

register_seed(
    PagesTaxonomySectionSeedSpec(
        name="pages.faqs",
        app_label="pages",
        model=FAQ,
        section_key="faqs",
        data_path=DEFAULT_PAGES_TAXONOMY_PATH,
        key_fields=["question"],
        update_fields=["question", "answer", "category", "sort_order", "is_active"],
    )
)

register_seed(
    PagesTaxonomySectionSeedSpec(
        name="pages.social_links",
        app_label="pages",
        model=SocialLink,
        section_key="social_links",
        data_path=DEFAULT_PAGES_TAXONOMY_PATH,
        key_fields=["name", "site__pk"],
        update_fields=["name", "url", "order", "is_active", "site"],
        fk_fields={"site": (SiteSettings, "pk")},
        dependencies=["pages.site_settings"],
    )
)
