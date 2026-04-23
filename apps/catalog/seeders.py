from __future__ import annotations

import json
import os
import random
from typing import Any

from django.utils.text import slugify

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.catalog.models import (
    ASPECT_RATIO_DEFAULT_CODE,
    AspectRatioChoice,
    Category,
    Facet,
    CategoryFacet,
    Tag,
    Attribute,
    AttributeValue,
    ProductType,
    ShippingMaterial,
    Badge,
    EcoCertification,
    Option,
    OptionValue,
    Product,
    ProductMakingOf,
    ProductQuestion,
    ProductAnswer,
    SizeChart,
    get_default_aspect_ratio_code,
)
from django.contrib.auth import get_user_model

User = get_user_model()


def _load_taxonomy(ctx: SeedContext) -> dict[str, Any]:
    env_path = os.environ.get("SEED_TAXONOMY_PATH") or os.environ.get("CATALOG_TAXONOMY_PATH")
    if env_path:
        path = ctx.resolve_path(env_path)
    else:
        path = ctx.resolve_path("apps/catalog/data/taxonomy.json")
    if not path.exists():
        return {"categories": []}
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


class CategorySeedSpec(SeedSpec):
    name = "catalog.categories"
    app_label = "catalog"
    kind = "prod"
    dependencies = ["catalog.aspect_ratio_choices"]
    description = "Seed category taxonomy tree"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_taxonomy(ctx)
        tree = data.get("categories", [])
        result = SeedResult()
        desired_keys: set[tuple[str | None, str]] = set()
        taxonomy_default_aspect = str(data.get("default_aspect_ratio") or "").strip() or get_default_aspect_ratio_code()

        def create_node(
            node: dict[str, Any],
            parent: Category | None = None,
            sibling_index: int = 0,
        ) -> Category | None:
            name = node.get("name") or node.get("display_name")
            if not name:
                return None
            slug = node.get("slug") or slugify(name)
            raw_sort_order = node.get("sort_order", sibling_index)
            try:
                sort_order = int(raw_sort_order)
            except (TypeError, ValueError):
                sort_order = sibling_index
            sort_order = max(sort_order, 0)
            defaults = {
                "name": name,
                "sort_order": sort_order,
                "is_active": node.get("is_active", True),
                "is_visible": node.get("is_visible", True),
                "is_deleted": False,
                "meta_title": node.get("meta_title", ""),
                "meta_description": node.get("meta_description", ""),
                "aspect_ratio": node.get("aspect_ratio", taxonomy_default_aspect),
            }
            lookup = {"parent": parent, "slug": slug}
            if ctx.dry_run:
                cat = Category.objects.filter(**lookup).first()
                if not cat:
                    result.created += 1
                else:
                    changed = False
                    for field, value in defaults.items():
                        if getattr(cat, field) != value:
                            changed = True
                            break
                    if cat.is_deleted:
                        changed = True
                    if changed:
                        result.updated += 1
                desired_keys.add((str(parent.id) if parent else None, slug))
                for idx, child in enumerate(node.get("children", []) or []):
                    create_node(child, parent=cat, sibling_index=idx)
                return cat

            cat, created = Category.objects.get_or_create(**lookup, defaults=defaults)
            if created:
                result.created += 1
            else:
                changed = False
                for field, value in defaults.items():
                    if getattr(cat, field) != value:
                        setattr(cat, field, value)
                        changed = True
                if cat.is_deleted:
                    cat.is_deleted = False
                    cat.deleted_at = None
                    changed = True
                if changed:
                    cat.save()
                    result.updated += 1

            desired_keys.add((str(parent.id) if parent else None, slug))

            for idx, child in enumerate(node.get("children", []) or []):
                create_node(child, parent=cat, sibling_index=idx)
            return cat

        for idx, node in enumerate(tree):
            create_node(node, parent=None, sibling_index=idx)

        if ctx.prune and not ctx.dry_run:
            to_prune = []
            for cat in Category.objects.all():
                key = (str(cat.parent_id) if cat.parent_id else None, cat.slug)
                if key not in desired_keys:
                    to_prune.append(cat)
            if to_prune:
                from django.utils import timezone
                now = timezone.now()
                for cat in to_prune:
                    cat.is_deleted = True
                    cat.deleted_at = now
                    cat.save(update_fields=["is_deleted", "deleted_at"])
                result.pruned += len(to_prune)

        return result


class AspectRatioChoiceSeedSpec(SeedSpec):
    name = "catalog.aspect_ratio_choices"
    app_label = "catalog"
    kind = "prod"
    description = "Seed catalog aspect-ratio choices from taxonomy data"

    @staticmethod
    def _extract_from_taxonomy(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        desired: dict[str, dict[str, Any]] = {}
        raw_choices = data.get("aspect_choices") or []
        for idx, item in enumerate(raw_choices):
            if isinstance(item, str):
                code = item.strip()
                payload = {}
            else:
                payload = dict(item or {})
                code = str(payload.get("code") or payload.get("value") or "").strip()
            if not code:
                continue
            label = str(payload.get("label") or code).strip() or code
            sort_order = int(payload.get("sort_order", idx))
            is_active = bool(payload.get("is_active", True))
            is_default = bool(payload.get("is_default", False))
            desired[code] = {
                "label": label,
                "sort_order": sort_order,
                "is_active": is_active,
                "is_default": is_default,
            }

        def collect_category_aspects(nodes: list[dict[str, Any]]):
            for node in nodes or []:
                code = str((node or {}).get("aspect_ratio") or "").strip()
                if code and code not in desired:
                    desired[code] = {
                        "label": code,
                        "sort_order": len(desired) + 100,
                        "is_active": True,
                        "is_default": False,
                    }
                collect_category_aspects((node or {}).get("children") or [])

        collect_category_aspects(data.get("categories") or [])
        return desired

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_taxonomy(ctx)
        desired = self._extract_from_taxonomy(data)
        result = SeedResult()

        if not desired:
            # Ensure at least one valid choice exists.
            desired[ASPECT_RATIO_DEFAULT_CODE] = {
                "label": ASPECT_RATIO_DEFAULT_CODE,
                "sort_order": 0,
                "is_active": True,
                "is_default": True,
            }

        if not any(item.get("is_default") for item in desired.values()):
            preferred_default = str(data.get("default_aspect_ratio") or "").strip()
            if preferred_default and preferred_default in desired:
                desired[preferred_default]["is_default"] = True
            elif ASPECT_RATIO_DEFAULT_CODE in desired:
                desired[ASPECT_RATIO_DEFAULT_CODE]["is_default"] = True
            else:
                first_code = sorted(
                    desired.items(),
                    key=lambda item: (item[1].get("sort_order", 0), item[0]),
                )[0][0]
                desired[first_code]["is_default"] = True

        if ctx.dry_run:
            existing = {obj.code: obj for obj in AspectRatioChoice.objects.all()}
            for code, payload in desired.items():
                obj = existing.get(code)
                if not obj:
                    result.created += 1
                    continue
                changed = any(
                    getattr(obj, field) != payload[field]
                    for field in ("label", "sort_order", "is_active", "is_default")
                )
                if changed:
                    result.updated += 1
            if ctx.prune:
                for code in existing:
                    if code not in desired:
                        result.pruned += 1
            return result

        for code, payload in desired.items():
            obj, created = AspectRatioChoice.objects.get_or_create(
                code=code,
                defaults=payload,
            )
            if created:
                result.created += 1
                continue
            changed = False
            for field in ("label", "sort_order", "is_active", "is_default"):
                new_value = payload[field]
                if getattr(obj, field) != new_value:
                    setattr(obj, field, new_value)
                    changed = True
            if changed:
                obj.save(update_fields=["label", "sort_order", "is_active", "is_default", "updated_at"])
                result.updated += 1

        if ctx.prune:
            stale_qs = AspectRatioChoice.objects.exclude(code__in=list(desired.keys()))
            for stale in stale_qs:
                stale.is_active = False
                stale.is_default = False
                stale.save(update_fields=["is_active", "is_default", "updated_at"])
                result.pruned += 1

        return result


class CategoryFacetSeedSpec(SeedSpec):
    name = "catalog.category_facets"
    app_label = "catalog"
    kind = "prod"
    dependencies = ["catalog.categories", "catalog.facets"]
    description = "Assign facets to categories from taxonomy"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_taxonomy(ctx)
        tree = data.get("categories", [])
        result = SeedResult()
        desired_pairs: set[tuple[str, str]] = set()

        def assign(node: dict[str, Any], parent: Category | None = None) -> None:
            name = node.get("name") or node.get("display_name")
            if not name:
                return
            slug = node.get("slug") or slugify(name)
            category = Category.objects.filter(parent=parent, slug=slug).first()
            if not category:
                return
            facets = node.get("facets", []) or []
            for facet_slug in facets:
                facet = Facet.objects.filter(slug=facet_slug).first()
                if not facet:
                    continue
                if ctx.dry_run:
                    if not CategoryFacet.objects.filter(category=category, facet=facet).exists():
                        result.created += 1
                else:
                    CategoryFacet.objects.get_or_create(category=category, facet=facet)
                desired_pairs.add((str(category.id), str(facet.id)))

            for child in node.get("children", []) or []:
                assign(child, parent=category)

        for node in tree:
            assign(node, parent=None)

        if ctx.prune and not ctx.dry_run:
            to_prune = []
            for cf in CategoryFacet.objects.select_related("category", "facet").all():
                key = (str(cf.category_id), str(cf.facet_id))
                if key not in desired_pairs:
                    to_prune.append(cf)
            if to_prune:
                for cf in to_prune:
                    cf.delete()
                result.pruned += len(to_prune)

        return result


register_seed(AspectRatioChoiceSeedSpec())
register_seed(CategorySeedSpec())

register_seed(
    JSONSeedSpec(
        name="catalog.facets",
        app_label="catalog",
        model=Facet,
        data_path="apps/catalog/data/facets.json",
        key_fields=["slug"],
        update_fields=["name", "type", "values"],
    )
)

register_seed(CategoryFacetSeedSpec())

class TagSeedSpec(JSONSeedSpec):
    def load_records(self, ctx: SeedContext) -> list[dict[str, Any]]:
        override_path = os.environ.get("SEED_TAGS_PATH")
        data_path = override_path or self.data_path
        path = ctx.resolve_path(data_path)
        if not path.exists():
            ctx.log(f"[seed:{self.name}] data file not found: {path}")
            return []
        with path.open("r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            if "items" in data:
                return list(data["items"])
            if "data" in data:
                return list(data["data"])
        if isinstance(data, list):
            return list(data)
        return []


register_seed(
    TagSeedSpec(
        name="catalog.tags",
        app_label="catalog",
        model=Tag,
        data_path="apps/catalog/data/tags.json",
        key_fields=["name"],
        update_fields=["name"],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.attributes",
        app_label="catalog",
        model=Attribute,
        data_path="apps/catalog/data/attributes.json",
        key_fields=["slug"],
        update_fields=["name", "slug"],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.attribute_values",
        app_label="catalog",
        model=AttributeValue,
        data_path="apps/catalog/data/attribute_values.json",
        key_fields=["attribute__slug", "value"],
        fk_fields={"attribute": (Attribute, "slug")},
        update_fields=["attribute", "value"],
        dependencies=["catalog.attributes"],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.product_types",
        app_label="catalog",
        model=ProductType,
        data_path="apps/catalog/data/product_types.json",
        key_fields=["slug"],
        update_fields=["name", "slug", "description"],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.shipping_materials",
        app_label="catalog",
        model=ShippingMaterial,
        data_path="apps/catalog/data/shipping_materials.json",
        key_fields=["name"],
        update_fields=[
            "name",
            "eco_score",
            "notes",
            "packaging_weight",
            "length",
            "width",
            "height",
            "units_per_package",
            "dimensional_weight_divisor",
        ],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.badges",
        app_label="catalog",
        model=Badge,
        data_path="apps/catalog/data/badges.json",
        key_fields=["slug"],
        update_fields=["name", "slug", "css_class", "start", "end", "priority", "is_active", "target_raw"],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.eco_certifications",
        app_label="catalog",
        model=EcoCertification,
        data_path="apps/catalog/data/eco_certifications.json",
        key_fields=["slug"],
        update_fields=["name", "slug", "issuer", "metadata"],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.options",
        app_label="catalog",
        model=Option,
        data_path="apps/catalog/data/options.json",
        key_fields=["slug"],
        update_fields=["name", "slug"],
    )
)

register_seed(
    JSONSeedSpec(
        name="catalog.option_values",
        app_label="catalog",
        model=OptionValue,
        data_path="apps/catalog/data/option_values.json",
        key_fields=["option__slug", "value"],
        fk_fields={"option": (Option, "slug")},
        update_fields=["option", "value"],
        dependencies=["catalog.options"],
    )
)


class CatalogDemoSeedSpec(SeedSpec):
    name = "catalog.demo"
    app_label = "catalog"
    kind = "demo"
    description = "Seed demo catalog data (making-of steps and Q&A)"

    def apply(self, ctx: SeedContext) -> SeedResult:
        result = SeedResult()
        products = list(Product.objects.all())
        users = list(User.objects.filter(is_superuser=False))

        if not products:
            result.skipped += 1
            return result

        sample_products = random.sample(products, min(5, len(products)))
        for product in sample_products:
            steps = [
                "Design",
                "Crafting",
                "Finishing",
            ]
            for idx, label in enumerate(steps, start=1):
                title = f"Step {idx}: {label}"
                if ctx.dry_run:
                    exists = ProductMakingOf.objects.filter(product=product, order=idx).exists()
                    if not exists:
                        result.created += 1
                    continue
                obj, created = ProductMakingOf.objects.get_or_create(
                    product=product,
                    order=idx,
                    defaults={
                        "title": title,
                        "description": f"{label} process for {product.name}.",
                        "image": "",
                        "video_url": "",
                    },
                )
                if created:
                    result.created += 1

        if not users:
            return result

        qa_products = random.sample(products, min(10, len(products)))
        for product in qa_products:
            user = random.choice(users)
            question_text = f"Is {product.name} available in other colors?"
            if ctx.dry_run:
                exists = ProductQuestion.objects.filter(product=product, user=user, question_text=question_text).exists()
                if not exists:
                    result.created += 1
                continue

            question, created = ProductQuestion.objects.get_or_create(
                product=product,
                user=user,
                question_text=question_text,
                defaults={"status": "approved"},
            )
            if created:
                result.created += 1

            if question.status == "approved":
                ProductAnswer.objects.get_or_create(
                    question=question,
                    user=user,
                    answer_text="Yes, some products are available in multiple colors.",
                    defaults={"status": "approved"},
                )

        return result


register_seed(CatalogDemoSeedSpec())


# =============================================================================
# Size Chart Seeder
# =============================================================================

DEFAULT_SIZE_CHARTS = [
    {
        "name": "Men's T-Shirt Size Chart",
        "slug": "mens-tshirt-size-chart",
        "garment_type": "tops",
        "unit": "in",
        "description": "Standard men's t-shirt sizing. Measure across chest at widest point.",
        "columns": ["Size", "Chest (in)", "Length (in)", "Shoulder (in)", "Sleeve (in)"],
        "rows": [
            ["XS", "34-36", "27", "16", "7.5"],
            ["S", "36-38", "28", "17", "8"],
            ["M", "38-40", "29", "18", "8.5"],
            ["L", "40-42", "30", "19", "9"],
            ["XL", "42-44", "31", "20", "9.5"],
            ["XXL", "44-46", "32", "21", "10"],
        ],
        "fit_notes": "For a relaxed fit, size up one. Measurements may vary ±0.5 inches.",
    },
    {
        "name": "Women's Top Size Chart",
        "slug": "womens-top-size-chart",
        "garment_type": "tops",
        "unit": "in",
        "description": "Standard women's top sizing guide.",
        "columns": ["Size", "Bust (in)", "Waist (in)", "Length (in)", "Shoulder (in)"],
        "rows": [
            ["XS", "30-32", "24-26", "24", "14"],
            ["S", "32-34", "26-28", "25", "14.5"],
            ["M", "34-36", "28-30", "26", "15"],
            ["L", "36-38", "30-32", "27", "15.5"],
            ["XL", "38-40", "32-34", "28", "16"],
            ["XXL", "40-42", "34-36", "29", "16.5"],
        ],
        "fit_notes": "Body-skimming fit. If between sizes, we recommend sizing up.",
    },
    {
        "name": "Men's Pants Size Chart",
        "slug": "mens-pants-size-chart",
        "garment_type": "bottoms",
        "unit": "in",
        "description": "Standard men's trousers and pants sizing.",
        "columns": ["Size", "Waist (in)", "Hip (in)", "Inseam (in)", "Outseam (in)"],
        "rows": [
            ["28", "28", "36", "30", "39"],
            ["30", "30", "38", "30", "40"],
            ["32", "32", "40", "31", "41"],
            ["34", "34", "42", "31", "42"],
            ["36", "36", "44", "32", "43"],
            ["38", "38", "46", "32", "44"],
        ],
        "fit_notes": "Measured flat across waistband, doubled. Inseam from crotch to hem.",
    },
    {
        "name": "Women's Pants Size Chart",
        "slug": "womens-pants-size-chart",
        "garment_type": "bottoms",
        "unit": "in",
        "description": "Standard women's trousers, jeans and bottom sizing.",
        "columns": ["Size", "Waist (in)", "Hip (in)", "Inseam (in)", "Rise (in)"],
        "rows": [
            ["XS / 24", "24", "34", "29", "9"],
            ["S / 26", "26", "36", "29", "9.5"],
            ["M / 28", "28", "38", "30", "10"],
            ["L / 30", "30", "40", "30", "10.5"],
            ["XL / 32", "32", "42", "30", "11"],
            ["XXL / 34", "34", "44", "31", "11.5"],
        ],
        "fit_notes": "Mid-rise fit. For high-rise styles, add 1-2 inches to rise.",
    },
    {
        "name": "Dress Size Chart",
        "slug": "dress-size-chart",
        "garment_type": "dresses",
        "unit": "in",
        "description": "Women's dress sizing for all dress styles.",
        "columns": ["Size", "Bust (in)", "Waist (in)", "Hip (in)", "Length (in)"],
        "rows": [
            ["XS / 2", "32", "25", "35", "34"],
            ["S / 4", "34", "27", "37", "35"],
            ["M / 6-8", "36", "29", "39", "36"],
            ["L / 10-12", "38", "31", "41", "37"],
            ["XL / 14", "40", "33", "43", "38"],
            ["XXL / 16", "42", "35", "45", "39"],
        ],
        "fit_notes": "Length measured from shoulder to hem. May vary by style.",
    },
    {
        "name": "Footwear Size Chart",
        "slug": "footwear-size-chart",
        "garment_type": "footwear",
        "unit": "cm",
        "description": "Unisex footwear sizing conversion guide.",
        "columns": ["EU", "US Men", "US Women", "UK", "Foot Length (cm)"],
        "rows": [
            ["36", "4", "6", "3.5", "22.5"],
            ["37", "5", "7", "4.5", "23.5"],
            ["38", "5.5", "7.5", "5", "24"],
            ["39", "6.5", "8.5", "6", "24.5"],
            ["40", "7", "9", "6.5", "25.5"],
            ["41", "8", "10", "7.5", "26"],
            ["42", "9", "11", "8", "27"],
            ["43", "9.5", "11.5", "8.5", "27.5"],
            ["44", "10.5", "12.5", "9.5", "28.5"],
            ["45", "11", "13", "10", "29"],
        ],
        "fit_notes": "Stand on paper, trace foot, measure longest point. If between sizes, go up.",
    },
    {
        "name": "Kids Clothing Size Chart",
        "slug": "kids-clothing-size-chart",
        "garment_type": "kids",
        "unit": "in",
        "description": "Children's clothing size guide by age range.",
        "columns": ["Size / Age", "Height (in)", "Chest (in)", "Waist (in)", "Hip (in)"],
        "rows": [
            ["2-3Y", "36-39", "21", "20", "21"],
            ["4-5Y", "40-43", "22", "21", "22"],
            ["6-7Y", "44-48", "24", "22", "24"],
            ["8-9Y", "49-52", "26", "23", "26"],
            ["10-11Y", "53-57", "28", "24", "28"],
            ["12-13Y", "58-61", "30", "25", "30"],
        ],
        "fit_notes": "Kids grow fast—when in doubt, size up for a comfortable fit.",
    },
    {
        "name": "Accessories Size Chart",
        "slug": "accessories-size-chart",
        "garment_type": "accessories",
        "unit": "in",
        "description": "Sizing guide for belts, hats and accessories.",
        "columns": ["Item", "S", "M", "L", "XL"],
        "rows": [
            ["Belt Length", "30-32", "34-36", "38-40", "42-44"],
            ["Hat Circumference", "21.5", "22", "22.5", "23"],
            ["Scarf Width", "10", "12", "14", "16"],
            ["Glove (palm circ.)", "7-7.5", "7.5-8", "8-8.5", "8.5-9"],
        ],
        "fit_notes": "Measure around the widest part. For belts, measure your existing belt from buckle pin to most-used hole.",
    },
    {
        "name": "Unisex General Size Chart",
        "slug": "unisex-general-size-chart",
        "garment_type": "unisex",
        "unit": "in",
        "description": "General unisex sizing reference.",
        "columns": ["Size", "Chest (in)", "Waist (in)", "Hip (in)"],
        "rows": [
            ["XS", "32-34", "24-26", "34-36"],
            ["S", "34-36", "26-28", "36-38"],
            ["M", "38-40", "30-32", "38-40"],
            ["L", "42-44", "34-36", "42-44"],
            ["XL", "46-48", "38-40", "46-48"],
            ["XXL", "50-52", "42-44", "50-52"],
        ],
        "fit_notes": "Relaxed fit. For a tighter fit, consider sizing down.",
        "is_default": True,
    },
]


class SizeChartSeedSpec(SeedSpec):
    name = "catalog.size_charts"
    app_label = "catalog"
    kind = "prod"
    dependencies = []
    description = "Seed default size charts for common garment types"

    def apply(self, ctx: SeedContext) -> SeedResult:
        result = SeedResult()

        for chart_data in DEFAULT_SIZE_CHARTS:
            slug = chart_data["slug"]
            is_default = chart_data.pop("is_default", False)
            defaults = {
                "name": chart_data["name"],
                "garment_type": chart_data["garment_type"],
                "unit": chart_data["unit"],
                "description": chart_data.get("description", ""),
                "columns": chart_data["columns"],
                "rows": chart_data["rows"],
                "fit_notes": chart_data.get("fit_notes", ""),
                "is_active": True,
                "is_default": is_default,
            }

            if ctx.dry_run:
                exists = SizeChart.objects.filter(slug=slug).exists()
                if not exists:
                    result.created += 1
                continue

            obj, created = SizeChart.objects.update_or_create(
                slug=slug, defaults=defaults
            )
            if created:
                result.created += 1
            else:
                result.updated += 1

        return result


register_seed(SizeChartSeedSpec())
