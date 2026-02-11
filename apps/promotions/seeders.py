from __future__ import annotations

from core.seed.base import JSONSeedSpec
from core.seed.registry import register_seed
from apps.promotions.models import Coupon, Sale
from apps.catalog.models import Category, Product

register_seed(
    JSONSeedSpec(
        name="promotions.coupons",
        app_label="promotions",
        model=Coupon,
        data_path="apps/promotions/data/coupons.json",
        key_fields=["code"],
        update_fields=[
            "code",
            "description",
            "discount_type",
            "discount_value",
            "minimum_order_amount",
            "maximum_discount",
            "usage_limit",
            "usage_limit_per_user",
            "first_order_only",
            "valid_from",
            "valid_until",
            "is_active",
        ],
        m2m_fields={
            "categories": (Category, "slug"),
            "products": (Product, "slug"),
        },
        dependencies=["catalog.categories"],
    )
)

register_seed(
    JSONSeedSpec(
        name="promotions.sales",
        app_label="promotions",
        model=Sale,
        data_path="apps/promotions/data/sales.json",
        key_fields=["slug"],
        update_fields=[
            "name",
            "slug",
            "description",
            "discount_type",
            "discount_value",
            "start_date",
            "end_date",
            "is_active",
        ],
        m2m_fields={
            "categories": (Category, "slug"),
            "products": (Product, "slug"),
        },
        dependencies=["catalog.categories"],
    )
)
