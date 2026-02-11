from __future__ import annotations

import json
import os
from typing import Any

from django.utils import timezone

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.shipping.models import (
    ShippingSettings,
    ShippingCarrier,
    ShippingMethod,
    ShippingZone,
    ShippingRate,
    ShippingRestriction,
)
from apps.i18n.models import Currency


def _load_shipping_data(ctx: SeedContext) -> dict[str, Any]:
    path = ctx.resolve_path("apps/shipping/data/shipping.json")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


class ShippingSectionSeedSpec(JSONSeedSpec):
    section_key: str = ""

    def __init__(self, *, section_key: str, **kwargs):
        super().__init__(**kwargs)
        self.section_key = section_key

    def load_records(self, ctx: SeedContext) -> list[dict[str, Any]]:
        data = _load_shipping_data(ctx)
        records = data.get(self.section_key, [])
        return list(records) if isinstance(records, list) else []


class ShippingSettingsSeedSpec(SeedSpec):
    name = "shipping.settings"
    app_label = "shipping"
    kind = "prod"
    description = "Seed ShippingSettings singleton"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_shipping_data(ctx)
        payload = data.get("settings") or {}
        if not payload:
            return SeedResult()

        threshold = os.environ.get("SEED_SHIPPING_FREE_THRESHOLD")
        if threshold is not None:
            payload["free_shipping_threshold"] = threshold
        handling_days = os.environ.get("SEED_SHIPPING_HANDLING_DAYS")
        if handling_days is not None:
            try:
                payload["handling_days"] = int(handling_days)
            except ValueError:
                payload["handling_days"] = handling_days
        enable_free = os.environ.get("SEED_SHIPPING_ENABLE_FREE_SHIPPING")
        if enable_free is not None:
            payload["enable_free_shipping"] = enable_free in ("1", "true", "True", "yes", "YES")

        result = SeedResult()
        obj = ShippingSettings.objects.first()
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
                ShippingSettings.objects.create(**payload)
                result.created += 1
        return result


class ShippingRestrictionSeedSpec(SeedSpec):
    name = "shipping.restrictions"
    app_label = "shipping"
    kind = "prod"
    dependencies = ["shipping.zones", "shipping.methods"]

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_shipping_data(ctx)
        records = data.get("restrictions", None)
        if records is None:
            return SeedResult()
        records = list(records)

        result = SeedResult()
        desired_keys: set[tuple[Any, Any, str, str]] = set()

        for record in records:
            payload = dict(record)
            zone_name = payload.pop("zone", None)
            method_code = payload.pop("method", None)
            restriction_type = payload.get("restriction_type")
            action = payload.get("action")
            if not restriction_type or not action:
                result.skipped += 1
                continue

            zone = ShippingZone.objects.filter(name=zone_name).first() if zone_name else None
            method = ShippingMethod.objects.filter(code=method_code).first() if method_code else None
            key = (zone.id if zone else None, method.id if method else None, restriction_type, action)
            desired_keys.add(key)

            if ctx.dry_run:
                exists = ShippingRestriction.objects.filter(
                    zone=zone,
                    method=method,
                    restriction_type=restriction_type,
                    action=action,
                ).exists()
                if not exists:
                    result.created += 1
                else:
                    result.updated += 1
                continue

            obj, created = ShippingRestriction.objects.get_or_create(
                zone=zone,
                method=method,
                restriction_type=restriction_type,
                action=action,
                defaults={
                    **payload,
                    "zone": zone,
                    "method": method,
                },
            )
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in payload.items():
                    if getattr(obj, field) != value:
                        setattr(obj, field, value)
                        changed = True
                if changed:
                    obj.save()
                    result.updated += 1

        if ctx.prune and not ctx.dry_run:
            to_prune = []
            for obj in ShippingRestriction.objects.all():
                key = (obj.zone_id, obj.method_id, obj.restriction_type, obj.action)
                if key not in desired_keys:
                    to_prune.append(obj)
            for obj in to_prune:
                if hasattr(obj, "is_deleted"):
                    obj.is_deleted = True
                    if hasattr(obj, "deleted_at"):
                        obj.deleted_at = timezone.now()
                    obj.save(update_fields=[f for f in ["is_deleted", "deleted_at"] if hasattr(obj, f)])
                elif hasattr(obj, "is_active"):
                    obj.is_active = False
                    obj.save(update_fields=["is_active"])
                else:
                    obj.delete()
                result.pruned += 1

        return result


register_seed(ShippingSettingsSeedSpec())

register_seed(
    ShippingSectionSeedSpec(
        name="shipping.carriers",
        app_label="shipping",
        model=ShippingCarrier,
        data_path="apps/shipping/data/shipping.json",
        section_key="carriers",
        key_fields=["code"],
        update_fields=[
            "name",
            "code",
            "website",
            "tracking_url_template",
            "api_enabled",
            "api_key",
            "api_secret",
            "api_account_number",
            "api_endpoint",
            "api_sandbox",
            "is_active",
            "supports_real_time_rates",
            "supports_tracking",
            "supports_label_generation",
        ],
    )
)

register_seed(
    ShippingSectionSeedSpec(
        name="shipping.methods",
        app_label="shipping",
        model=ShippingMethod,
        data_path="apps/shipping/data/shipping.json",
        section_key="methods",
        key_fields=["code"],
        fk_fields={"carrier": (ShippingCarrier, "code")},
        update_fields=[
            "name",
            "code",
            "description",
            "carrier",
            "carrier_service_code",
            "min_delivery_days",
            "max_delivery_days",
            "delivery_time_text",
            "is_active",
            "requires_signature",
            "is_express",
            "sort_order",
            "max_weight",
            "max_dimensions",
        ],
        dependencies=["shipping.carriers"],
    )
)

register_seed(
    ShippingSectionSeedSpec(
        name="shipping.zones",
        app_label="shipping",
        model=ShippingZone,
        data_path="apps/shipping/data/shipping.json",
        section_key="zones",
        key_fields=["name"],
        update_fields=[
            "name",
            "description",
            "countries",
            "states",
            "cities",
            "postal_codes",
            "is_active",
            "is_default",
            "priority",
        ],
    )
)

register_seed(
    ShippingSectionSeedSpec(
        name="shipping.rates",
        app_label="shipping",
        model=ShippingRate,
        data_path="apps/shipping/data/shipping.json",
        section_key="rates",
        key_fields=["zone__name", "method__code"],
        fk_fields={
            "zone": (ShippingZone, "name"),
            "method": (ShippingMethod, "code"),
            "currency": (Currency, "code"),
        },
        update_fields=[
            "zone",
            "method",
            "rate_type",
            "base_rate",
            "per_kg_rate",
            "per_item_rate",
            "free_shipping_threshold",
            "weight_tiers",
            "price_tiers",
            "is_active",
            "currency",
        ],
        dependencies=["shipping.zones", "shipping.methods", "i18n.currencies"],
    )
)

register_seed(ShippingRestrictionSeedSpec())
