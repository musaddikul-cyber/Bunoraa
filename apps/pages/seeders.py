from __future__ import annotations

import json
from typing import Any

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.pages.models import (
    SiteSettings,
    SocialLink,
    NewsletterIncentive,
    BlogCategory,
    BlogTag,
    FAQ,
)


def _load_json(ctx: SeedContext, path: str) -> dict[str, Any]:
    p = ctx.resolve_path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


class SiteSettingsSeedSpec(SeedSpec):
    name = "pages.site_settings"
    app_label = "pages"
    kind = "prod"
    description = "Seed SiteSettings singleton"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_json(ctx, "apps/pages/data/site_settings.json")
        payload = data.get("settings") or data.get("item") or {}
        if not payload:
            return SeedResult()

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


register_seed(SiteSettingsSeedSpec())

register_seed(
    JSONSeedSpec(
        name="pages.newsletter_incentives",
        app_label="pages",
        model=NewsletterIncentive,
        data_path="apps/pages/data/newsletter_incentives.json",
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
    JSONSeedSpec(
        name="pages.blog_categories",
        app_label="pages",
        model=BlogCategory,
        data_path="apps/pages/data/blog_categories.json",
        key_fields=["slug"],
        update_fields=["name", "slug", "description", "icon"],
    )
)

register_seed(
    JSONSeedSpec(
        name="pages.blog_tags",
        app_label="pages",
        model=BlogTag,
        data_path="apps/pages/data/blog_tags.json",
        key_fields=["slug"],
        update_fields=["name", "slug"],
    )
)

register_seed(
    JSONSeedSpec(
        name="pages.faqs",
        app_label="pages",
        model=FAQ,
        data_path="apps/pages/data/faqs.json",
        key_fields=["question"],
        update_fields=["question", "answer", "category", "sort_order", "is_active"],
    )
)

register_seed(
    JSONSeedSpec(
        name="pages.social_links",
        app_label="pages",
        model=SocialLink,
        data_path="apps/pages/data/social_links.json",
        key_fields=["name"],
        update_fields=["name", "url", "order", "is_active", "site"],
        fk_fields={"site": (SiteSettings, "pk")},
        dependencies=["pages.site_settings"],
    )
)
