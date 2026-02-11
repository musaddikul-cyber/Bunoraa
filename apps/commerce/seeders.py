from __future__ import annotations

import json
from typing import Any

from core.seed.base import SeedContext, SeedResult, SeedSpec
from core.seed.registry import register_seed
from apps.commerce.models import CartSettings


def _load_json(ctx: SeedContext, path: str) -> dict[str, Any]:
    p = ctx.resolve_path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class CartSettingsSeedSpec(SeedSpec):
    name = "commerce.cart_settings"
    app_label = "commerce"
    kind = "prod"
    description = "Seed CartSettings singleton"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_json(ctx, "apps/commerce/data/cart_settings.json")
        payload = data.get("settings") or data.get("item") or {}
        if not payload:
            return SeedResult()

        result = SeedResult()
        obj = CartSettings.objects.first()
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
                CartSettings.objects.create(**payload)
                result.created += 1
        return result


register_seed(CartSettingsSeedSpec())
