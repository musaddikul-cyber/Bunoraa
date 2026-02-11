from __future__ import annotations

from core.seed.base import JSONSeedSpec
from core.seed.registry import register_seed
from apps.preorders.models import (
    PreOrderCategory,
    PreOrderOption,
    PreOrderOptionChoice,
    PreOrderTemplate,
)


register_seed(
    JSONSeedSpec(
        name="preorders.categories",
        app_label="preorders",
        model=PreOrderCategory,
        data_path="apps/preorders/data/preorder_categories.json",
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
    JSONSeedSpec(
        name="preorders.options",
        app_label="preorders",
        model=PreOrderOption,
        data_path="apps/preorders/data/preorder_options.json",
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
    JSONSeedSpec(
        name="preorders.option_choices",
        app_label="preorders",
        model=PreOrderOptionChoice,
        data_path="apps/preorders/data/preorder_option_choices.json",
        key_fields=["option__name", "value"],
        fk_fields={"option": (PreOrderOption, "name")},
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
    JSONSeedSpec(
        name="preorders.templates",
        app_label="preorders",
        model=PreOrderTemplate,
        data_path="apps/preorders/data/preorder_templates.json",
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
