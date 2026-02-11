from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.i18n.models import (
    Language,
    Currency,
    Timezone,
    Country,
    Division,
    District,
    ExchangeRate,
    I18nSettings,
)


def _load_json(ctx: SeedContext, path: str) -> dict[str, Any]:
    p = ctx.resolve_path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class DivisionSeedSpec(SeedSpec):
    name = "i18n.divisions"
    app_label = "i18n"
    kind = "prod"
    dependencies = ["i18n.countries"]

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_json(ctx, "apps/i18n/data/divisions.json")
        divisions = data.get("divisions", [])
        result = SeedResult()
        desired = set()

        country = Country.objects.filter(code="BD").first()
        if not country:
            return result

        for item in divisions:
            code = item.get("code")
            if not code:
                continue
            desired.add(code)
            defaults = {
                "name": item.get("name", ""),
                "native_name": item.get("native_name", ""),
                "division_type": item.get("division_type", "division"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "is_active": True,
                "is_shipping_available": True,
                "sort_order": item.get("sort_order", 0),
            }
            if ctx.dry_run:
                division = Division.objects.filter(country=country, code=code).first()
                if not division:
                    result.created += 1
                else:
                    changed = False
                    for field, value in defaults.items():
                        if getattr(division, field) != value:
                            changed = True
                            break
                    if not division.is_active:
                        changed = True
                    if changed:
                        result.updated += 1
                continue

            division, created = Division.objects.get_or_create(
                country=country,
                code=code,
                defaults=defaults,
            )
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in defaults.items():
                    if getattr(division, field) != value:
                        setattr(division, field, value)
                        changed = True
                if not division.is_active:
                    division.is_active = True
                    changed = True
                if changed:
                    division.save()
                    result.updated += 1

        if ctx.prune and not ctx.dry_run:
            to_prune = Division.objects.filter(country=country).exclude(code__in=desired)
            if to_prune.exists():
                to_prune.update(is_active=False)
                result.pruned += to_prune.count()

        return result


class DistrictSeedSpec(SeedSpec):
    name = "i18n.districts"
    app_label = "i18n"
    kind = "prod"
    dependencies = ["i18n.divisions"]

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_json(ctx, "apps/i18n/data/districts.json")
        districts = data.get("districts", [])
        result = SeedResult()
        desired = set()

        for item in districts:
            code = item.get("code")
            division_code = item.get("division")
            if not code or not division_code:
                continue
            division = Division.objects.filter(code=division_code).first()
            if not division:
                continue
            key = f"{division_code}:{code}"
            desired.add(key)
            defaults = {
                "name": item.get("name", ""),
                "native_name": item.get("native_name", ""),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "shipping_zone": item.get("shipping_zone", "suburban"),
                "estimated_delivery_days_min": item.get("estimated_delivery_days_min", 2),
                "estimated_delivery_days_max": item.get("estimated_delivery_days_max", 5),
                "is_active": True,
                "is_shipping_available": True,
            }
            if ctx.dry_run:
                district = District.objects.filter(division=division, code=code).first()
                if not district:
                    result.created += 1
                else:
                    changed = False
                    for field, value in defaults.items():
                        if getattr(district, field) != value:
                            changed = True
                            break
                    if not district.is_active:
                        changed = True
                    if changed:
                        result.updated += 1
                continue

            district, created = District.objects.get_or_create(
                division=division,
                code=code,
                defaults=defaults,
            )
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in defaults.items():
                    if getattr(district, field) != value:
                        setattr(district, field, value)
                        changed = True
                if not district.is_active:
                    district.is_active = True
                    changed = True
                if changed:
                    district.save()
                    result.updated += 1

        if ctx.prune and not ctx.dry_run:
            to_prune = []
            for district in District.objects.select_related("division").all():
                key = f"{district.division.code}:{district.code}"
                if key not in desired:
                    to_prune.append(district.id)
            if to_prune:
                District.objects.filter(id__in=to_prune).update(is_active=False)
                result.pruned += len(to_prune)

        return result


class I18nSettingsSeedSpec(SeedSpec):
    name = "i18n.settings"
    app_label = "i18n"
    kind = "prod"
    dependencies = ["i18n.languages", "i18n.currencies", "i18n.timezones"]

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_json(ctx, "apps/i18n/data/i18n_settings.json")
        if not data:
            return SeedResult()

        result = SeedResult()
        defaults = dict(data)
        if "default_language" in defaults:
            defaults["default_language"] = Language.objects.get(code=defaults["default_language"])
        if "fallback_language" in defaults and defaults["fallback_language"]:
            defaults["fallback_language"] = Language.objects.get(code=defaults["fallback_language"])
        if "default_currency" in defaults:
            defaults["default_currency"] = Currency.objects.get(code=defaults["default_currency"])
        if "default_timezone" in defaults and defaults["default_timezone"]:
            defaults["default_timezone"] = Timezone.objects.get(name=defaults["default_timezone"])

        obj = I18nSettings.objects.first()
        if obj:
            changed = False
            for field, value in defaults.items():
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
                I18nSettings.objects.create(**defaults)
                result.created += 1
        return result


register_seed(
    JSONSeedSpec(
        name="i18n.languages",
        app_label="i18n",
        model=Language,
        data_path="apps/i18n/data/languages.json",
        key_fields=["code"],
        update_fields=[
            "name",
            "native_name",
            "is_rtl",
            "flag_code",
            "font_family",
            "locale_code",
            "is_active",
            "is_default",
            "sort_order",
        ],
    )
)

register_seed(
    JSONSeedSpec(
        name="i18n.currencies",
        app_label="i18n",
        model=Currency,
        data_path="apps/i18n/data/currencies.json",
        key_fields=["code"],
        update_fields=[
            "name",
            "symbol",
            "native_symbol",
            "decimal_places",
            "symbol_position",
            "number_system",
            "is_base_currency",
            "is_default",
            "is_active",
            "sort_order",
        ],
    )
)

register_seed(
    JSONSeedSpec(
        name="i18n.timezones",
        app_label="i18n",
        model=Timezone,
        data_path="apps/i18n/data/timezones.json",
        key_fields=["name"],
        update_fields=[
            "display_name",
            "offset",
            "offset_minutes",
            "has_dst",
            "dst_offset",
            "is_active",
            "is_common",
        ],
    )
)

register_seed(
    JSONSeedSpec(
        name="i18n.countries",
        app_label="i18n",
        model=Country,
        data_path="apps/i18n/data/countries.json",
        key_fields=["code"],
        update_fields=[
            "code_alpha3",
            "code_numeric",
            "name",
            "native_name",
            "default_language",
            "default_currency",
            "default_timezone",
            "phone_code",
            "phone_format",
            "address_format",
            "postal_code_format",
            "default_tax_rate",
            "continent",
            "region",
            "is_active",
            "is_shipping_available",
            "is_billing_allowed",
            "sort_order",
        ],
        fk_fields={
            "default_language": (Language, "code"),
            "default_currency": (Currency, "code"),
            "default_timezone": (Timezone, "name"),
        },
        dependencies=["i18n.languages", "i18n.currencies", "i18n.timezones"],
    )
)

register_seed(DivisionSeedSpec())
register_seed(DistrictSeedSpec())

register_seed(
    JSONSeedSpec(
        name="i18n.exchange_rates",
        app_label="i18n",
        model=ExchangeRate,
        data_path="apps/i18n/data/exchange_rates.json",
        key_fields=["from_currency__code", "to_currency__code"],
        fk_fields={
            "from_currency": (Currency, "code"),
            "to_currency": (Currency, "code"),
        },
        update_fields=["from_currency", "to_currency", "rate", "source", "valid_from", "valid_until", "is_active"],
        dependencies=["i18n.currencies"],
    )
)

register_seed(I18nSettingsSeedSpec())
