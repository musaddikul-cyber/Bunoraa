from __future__ import annotations

import json
import random
from typing import Any

from django.contrib.auth import get_user_model

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.contacts.models import ContactCategory, StoreLocation, ContactSettings, CustomizationRequest
from apps.catalog.models import Product

User = get_user_model()


def _load_json(ctx: SeedContext, path: str) -> dict[str, Any]:
    p = ctx.resolve_path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


class ContactSettingsSeedSpec(SeedSpec):
    name = "contacts.settings"
    app_label = "contacts"
    kind = "prod"
    description = "Seed ContactSettings singleton"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_json(ctx, "apps/contacts/data/contact_settings.json")
        payload = data.get("settings") or data.get("item") or {}
        if not payload:
            return SeedResult()

        result = SeedResult()
        obj = ContactSettings.objects.first()
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
                ContactSettings.objects.create(**payload)
                result.created += 1
        return result


register_seed(ContactSettingsSeedSpec())

register_seed(
    JSONSeedSpec(
        name="contacts.categories",
        app_label="contacts",
        model=ContactCategory,
        data_path="apps/contacts/data/contact_categories.json",
        key_fields=["slug"],
        update_fields=[
            "name",
            "slug",
            "description",
            "email_recipients",
            "auto_response_subject",
            "auto_response_message",
            "order",
            "is_active",
        ],
    )
)

register_seed(
    JSONSeedSpec(
        name="contacts.store_locations",
        app_label="contacts",
        model=StoreLocation,
        data_path="apps/contacts/data/store_locations.json",
        key_fields=["slug"],
        update_fields=[
            "name",
            "slug",
            "address_line1",
            "address_line2",
            "city",
            "state",
            "postal_code",
            "country",
            "latitude",
            "longitude",
            "phone",
            "email",
            "monday_hours",
            "tuesday_hours",
            "wednesday_hours",
            "thursday_hours",
            "friday_hours",
            "saturday_hours",
            "sunday_hours",
            "is_pickup_location",
            "is_returns_location",
            "pickup_fee",
            "min_pickup_time_hours",
            "max_hold_days",
            "description",
            "order",
            "is_active",
            "is_main",
        ],
    )
)


class CustomizationRequestDemoSeedSpec(SeedSpec):
    name = "contacts.demo"
    app_label = "contacts"
    kind = "demo"
    description = "Seed demo customization requests"

    def apply(self, ctx: SeedContext) -> SeedResult:
        result = SeedResult()
        products = list(Product.objects.filter(can_be_customized=True))
        users = list(User.objects.filter(is_superuser=False))

        if not products or not users:
            result.skipped += 1
            return result

        sample_products = random.sample(products, min(5, len(products)))
        for product in sample_products:
            user = random.choice(users)
            message = f"I would like to customize the '{product.name}' with custom colors and initials."

            if ctx.dry_run:
                if not CustomizationRequest.objects.filter(product=product, user=user).exists():
                    result.created += 1
                continue

            obj, created = CustomizationRequest.objects.get_or_create(
                product=product,
                user=user,
                defaults={
                    "name": user.get_full_name() or user.email,
                    "email": user.email,
                    "phone": getattr(user, "phone", "") or "",
                    "message": message,
                    "status": "new",
                },
            )
            if created:
                result.created += 1

        return result


register_seed(CustomizationRequestDemoSeedSpec())
