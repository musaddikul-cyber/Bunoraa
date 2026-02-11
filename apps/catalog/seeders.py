from __future__ import annotations

import json
import os
import random
from typing import Any

from django.utils.text import slugify

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.catalog.models import (
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
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class CategorySeedSpec(SeedSpec):
    name = "catalog.categories"
    app_label = "catalog"
    kind = "prod"
    description = "Seed category taxonomy tree"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_taxonomy(ctx)
        tree = data.get("categories", [])
        result = SeedResult()
        desired_keys: set[tuple[str | None, str]] = set()

        def create_node(node: dict[str, Any], parent: Category | None = None) -> Category | None:
            name = node.get("name") or node.get("display_name")
            if not name:
                return None
            slug = node.get("slug") or slugify(name)
            defaults = {
                "name": name,
                "is_visible": node.get("is_visible", True),
                "is_deleted": False,
                "meta_title": node.get("meta_title", ""),
                "meta_description": node.get("meta_description", ""),
                "aspect_ratio": node.get("aspect_ratio", "1:1"),
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
                for child in node.get("children", []) or []:
                    create_node(child, parent=cat)
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

            for child in node.get("children", []) or []:
                create_node(child, parent=cat)
            return cat

        for node in tree:
            create_node(node, parent=None)

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
        with path.open("r", encoding="utf-8") as fh:
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
