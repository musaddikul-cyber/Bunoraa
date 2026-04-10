from __future__ import annotations

from pathlib import Path

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from core.seed.base import JSONSeedSpec
from core.seed.registry import register_seed
from apps.promotions.models import Banner, Coupon, Sale
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


class BannerSeedSpec(JSONSeedSpec):
    """Seed promotional banners and ensure source images are available in media storage."""

    def load_records(self, ctx):
        records = super().load_records(ctx)
        if not records:
            return records

        source_dir = ctx.base_dir / "static" / "images" / "banners"
        for record in records:
            self._ensure_banner_image(record, "image", source_dir, ctx)
            self._ensure_banner_image(record, "image_mobile", source_dir, ctx)
        return records

    def _ensure_banner_image(self, record, field_name, source_dir: Path, ctx):
        raw_value = (record.get(field_name) or "").strip()
        if not raw_value:
            return

        source_name = Path(raw_value).name
        source_path = source_dir / source_name
        target_path = f"banners/{source_name}"

        if not source_path.exists():
            ctx.log(
                f"[seed:{self.name}] source image not found for {field_name}: {source_path}"
            )
            return

        if not default_storage.exists(target_path):
            with source_path.open("rb") as handle:
                default_storage.save(target_path, ContentFile(handle.read()))

        record[field_name] = target_path


register_seed(
    BannerSeedSpec(
        name="promotions.banners",
        app_label="promotions",
        model=Banner,
        data_path="apps/promotions/data/banners.json",
        key_fields=["title", "position"],
        update_fields=[
            "title",
            "subtitle",
            "image",
            "image_mobile",
            "link_url",
            "link_text",
            "style_height",
            "style_width",
            "style_max_width",
            "style_border_radius",
            "style_border_width",
            "style_border_color",
            "style_background_color",
            "overlay_color",
            "overlay_opacity",
            "text_color",
            "position",
            "sort_order",
            "start_date",
            "end_date",
            "is_active",
        ],
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
