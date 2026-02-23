from django.contrib import admin, messages
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _
from django.db.models import Sum, Count, F, Q
from django.db import OperationalError, transaction
from django.urls import reverse, path
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed, Http404
from django.core.cache import cache
from django.core.files.storage import default_storage
from django.utils import timezone, translation
import csv
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlparse
from django.conf import settings

from core.admin_mixins import (
    ImportExportEnhancedModelAdmin,
    EnhancedTabularInline,
    ExportCSVMixin,
    BulkActivateMixin,
    BulkFeaturedMixin,
    StatusBadgeMixin,
    DateRangeFilter,
    PriceRangeFilter,
    StockFilter,
    format_currency,
    format_number,
    truncate_text,
)

from .models import (
    AspectRatioChoice,
    Category,
    Product,
    ProductVariant,
    ProductImage,
    Badge,
    ShippingMaterial,
    Spotlight,
    ProductMakingOf,
    Product3DAsset,
    CustomerPhoto,
    ProductQuestion, 
    ProductAnswer,   
    Attribute,
    AttributeValue,
    Facet,
    CategoryFacet,
    Tag,
    EcoCertification,
    ProductAutofillJob,
    ProductAutofillSource,
    ProductFieldSuggestion,
    ProductAutofillFeedback,
    CategoryPricingProfile,
)
from .ai.validators import apply_suggestions_to_product
from .tasks import run_product_autofill_job

logger = logging.getLogger(__name__)


# =============================================================================
# INLINES
# =============================================================================

class ProductImageInline(EnhancedTabularInline):
    model = ProductImage
    extra = 1
    fields = ("image", "thumbnail_preview", "alt_text", "is_primary", "ordering")
    readonly_fields = ("thumbnail_preview",)
    ordering = ["ordering"]
    
    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 80px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return "-"
    thumbnail_preview.short_description = "Preview"


class ProductVariantInline(EnhancedTabularInline):
    model = ProductVariant
    extra = 0
    fields = ("sku", "price", "stock_quantity", "stock_status_badge", "is_default")
    readonly_fields = ("stock_status_badge",)
    
    def stock_status_badge(self, obj):
        if obj.stock_quantity <= 0:
            return format_html('<span style="color: #dc2626; font-weight: 600;">Out of Stock</span>')
        elif obj.stock_quantity < 10:
            return format_html('<span style="color: #d97706; font-weight: 600;">Low ({}))</span>', obj.stock_quantity)
        return format_html('<span style="color: #16a34a; font-weight: 600;">In Stock ({})</span>', obj.stock_quantity)
    stock_status_badge.short_description = "Status"


class Product3DAssetInline(EnhancedTabularInline):
    model = Product3DAsset
    extra = 0
    fields = ("file", "file_type", "validated", "is_ar_compatible")


class ProductMakingOfInline(EnhancedTabularInline):
    model = ProductMakingOf
    extra = 0
    fields = ("order", "title", "description", "image", "video_url")
    ordering = ["order"]


class CustomerPhotoInline(EnhancedTabularInline):
    model = CustomerPhoto
    extra = 0
    fields = ("image", "thumbnail_preview", "description", "status")
    readonly_fields = ("thumbnail_preview",)
    ordering = ["-created_at"]

    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 80px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return "-"
    thumbnail_preview.short_description = "Preview"


class ProductAnswerInline(EnhancedTabularInline):
    model = ProductAnswer
    extra = 0
    fields = ("user", "answer_text", "status")
    readonly_fields = ("user",)
    ordering = ["created_at"]


class ProductQuestionInline(EnhancedTabularInline):
    model = ProductQuestion
    extra = 0
    fields = ("user", "question_text", "status")
    readonly_fields = ("user",)
    ordering = ["-created_at"]
    inlines = [ProductAnswerInline] # Nested inline for answers


from .forms import CategoryAdminForm, ProductAdminForm


@admin.register(Category)
class CategoryAdmin(ImportExportEnhancedModelAdmin):
    form = CategoryAdminForm
    list_display = (
        "name",
        "slug",
        "parent",
        "display_path",
        "depth",
        "sort_order",
        "product_count",
        "is_active",
        "is_visible",
        "aspect_ratio",
    )
    search_fields = ("name", "slug")
    list_filter = ("is_active", "is_visible", "is_deleted", "aspect_ratio", "parent", "depth")
    prepopulated_fields = {"slug": ("name",)}
    ordering = ["depth", "sort_order", "name"]
    list_editable = ("sort_order",)
    
    actions = [
        "seed_default_tree",
        "import_taxonomy_sync",
        "import_taxonomy_no_sync",
        "sync_and_save_taxonomy",
        "export_taxonomy_json",
        "export_taxonomy_csv",
        "rebuild_paths",
        "make_enabled",
        "make_disabled",
        "make_visible",
        "make_hidden",
        "export_selected_csv",
    ]

    def display_path(self, obj):
        crumbs = obj.breadcrumbs()
        return " / ".join([c.name for c in crumbs])

    display_path.short_description = "Path"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("parent")

    def seed_default_tree(self, request, queryset):
        """Admin action to run the default category seeder (idempotent)."""
        from django.core.management import call_command
        try:
            call_command("seed_categories")
            self.message_user(request, "Default category tree seeded successfully (idempotent).")
        except Exception as e:
            self.message_user(request, f"Error seeding categories: {e}", level="error")
    seed_default_tree.short_description = "Seed default category tree"

    def import_taxonomy_sync(self, request, queryset):
        """Import taxonomy and sync (prune) categories + facets."""
        from django.core.management import call_command
        try:
            call_command("seed_categories", assign_facets=True, force=True)
            self.message_user(
                request,
                "Imported taxonomy with sync (prune) and facet assignments.",
                level="success",
            )
        except Exception as e:
            self.message_user(request, f"Error importing taxonomy: {e}", level="error")
    import_taxonomy_sync.short_description = "Import taxonomy (sync + prune)"

    def import_taxonomy_no_sync(self, request, queryset):
        """Import taxonomy without pruning existing categories."""
        from django.core.management import call_command
        try:
            call_command("seed_categories", assign_facets=True, no_prune=True)
            self.message_user(
                request,
                "Imported taxonomy without pruning (no sync).",
                level="success",
            )
        except Exception as e:
            self.message_user(request, f"Error importing taxonomy: {e}", level="error")
    import_taxonomy_no_sync.short_description = "Import taxonomy (no prune)"

    def sync_and_save_taxonomy(self, request, queryset):
        """Sync taxonomy and write the current DB state back to taxonomy.json."""
        from django.core.management import call_command
        try:
            call_command("seed_categories", assign_facets=True, force=True)
            out_path = Path(settings.BASE_DIR) / "apps" / "catalog" / "data" / "taxonomy.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            call_command("export_taxonomy", out=str(out_path), format="json")
            self.message_user(
                request,
                f"Synced taxonomy and saved export to {out_path}.",
                level="success",
            )
        except Exception as e:
            self.message_user(request, f"Error syncing/exporting taxonomy: {e}", level="error")
    sync_and_save_taxonomy.short_description = "Sync and save taxonomy.json"

    def export_taxonomy_json(self, request, queryset):
        """Export taxonomy as JSON (download)."""
        from django.core.management import call_command
        fd, path = tempfile.mkstemp(prefix="taxonomy_", suffix=".json")
        os.close(fd)
        try:
            call_command("export_taxonomy", out=path, format="json")
            with open(path, "rb") as fh:
                data = fh.read()
            response = HttpResponse(data, content_type="application/json")
            response["Content-Disposition"] = 'attachment; filename="taxonomy.json"'
            return response
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    export_taxonomy_json.short_description = "Export taxonomy as JSON"

    def export_taxonomy_csv(self, request, queryset):
        """Export taxonomy as CSV (download)."""
        from django.core.management import call_command
        fd, path = tempfile.mkstemp(prefix="taxonomy_", suffix=".csv")
        os.close(fd)
        try:
            call_command("export_taxonomy", out=path, format="csv")
            with open(path, "rb") as fh:
                data = fh.read()
            response = HttpResponse(data, content_type="text/csv")
            response["Content-Disposition"] = 'attachment; filename="taxonomy.csv"'
            return response
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    export_taxonomy_csv.short_description = "Export taxonomy as CSV"

    def rebuild_paths(self, request, queryset):
        """Admin action to rebuild path and depth for selected categories."""
        if queryset.count() == 0:
            # Rebuild entire tree
            fixed = Category.rebuild_all_paths() if hasattr(Category, "rebuild_all_paths") else 0
            self.message_user(request, f"Rebuilt paths for entire tree, fixed {fixed} nodes.")
            return

        fixed = 0
        for cat in queryset:
            if hasattr(cat, "rebuild_subtree"):
                fixed += 1 if cat.rebuild_subtree() else 0
            else:
                # Fallback: manually update depth and path
                cat.depth = cat.calculate_depth() if hasattr(cat, "calculate_depth") else 0
                cat.save(update_fields=["depth"])
                fixed += 1

        self.message_user(request, f"Rebuilt paths for {fixed} selected categories.")
    rebuild_paths.short_description = "Rebuild path/depth for selected categories"

    def make_visible(self, request, queryset):
        """Mark selected categories as visible."""
        updated = queryset.update(is_visible=True)
        self.message_user(request, f"Marked {updated} categories as visible.")
    make_visible.short_description = "Mark selected as visible"

    def make_enabled(self, request, queryset):
        """Mark selected categories as enabled/active."""
        updated = queryset.update(is_active=True)
        self.message_user(request, f"Marked {updated} categories as enabled.")
    make_enabled.short_description = "Mark selected as enabled"

    def make_disabled(self, request, queryset):
        """Mark selected categories as disabled/inactive."""
        updated = queryset.update(is_active=False)
        self.message_user(request, f"Marked {updated} categories as disabled.")
    make_disabled.short_description = "Mark selected as disabled"

    def make_hidden(self, request, queryset):
        """Mark selected categories as hidden."""
        updated = queryset.update(is_visible=False)
        self.message_user(request, f"Marked {updated} categories as hidden.")
    make_hidden.short_description = "Mark selected as hidden"

    def export_selected_csv(self, request, queryset):
        """Export selected categories to CSV."""
        import csv
        import tempfile

        fd, path = tempfile.mkstemp(prefix="categories_", suffix=".csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["id", "name", "slug", "parent_id", "depth", "sort_order", "is_active", "is_visible"]
            )
            for c in queryset.order_by("depth", "sort_order", "name"):
                parent_id = c.parent_id if c.parent_id else ""
                writer.writerow(
                    [c.id, c.name, c.slug, parent_id, c.depth, c.sort_order, c.is_active, c.is_visible]
                )

        self.message_user(request, f"Exported {queryset.count()} categories to {path}")
    export_selected_csv.short_description = "Export selected categories as CSV"

@admin.register(Product)
class ProductAdmin(ImportExportEnhancedModelAdmin, BulkActivateMixin, BulkFeaturedMixin):
    form = ProductAdminForm 
    change_form_template = "admin/catalog/product/change_form.html"
    list_display = (
        "thumbnail_preview", "name", "sku", "primary_category_display", 
        "price_display", "stock_status", "performance_stats", "is_active",
        "is_active_badge", "is_featured_badge", "created_at"
    )
    list_display_links = ("thumbnail_preview", "name")
    search_fields = ("name", "sku", "description")
    list_filter = (
        "is_active", "is_featured", "is_bestseller", "is_new_arrival",
        StockFilter, PriceRangeFilter, "aspect_ratio", "primary_category"
    )
    inlines = [ProductImageInline, ProductVariantInline, Product3DAssetInline, ProductMakingOfInline]
    prepopulated_fields = {"slug": ("name",)}
    date_hierarchy = "created_at"
    list_per_page = 25
    list_editable = ("is_active",)
    save_on_top = False
    filter_horizontal = ("categories", "tags")
    
    # Export fields
    export_fields = ['sku', 'name', 'price', 'sale_price', 'stock_quantity', 
                     'is_active', 'is_featured', 'views_count', 'sales_count']
    
    actions = [
        'export_as_csv', 'export_as_json',
        'activate_selected', 'deactivate_selected',
        'mark_featured', 'unmark_featured',
        'mark_bestseller', 'unmark_bestseller',
        'mark_new_arrival', 'unmark_new_arrival',
        'duplicate_products',
        'recalculate_stats',
    ]

    fieldsets = (
        (_('Basic Information'), {
            "fields": ("name", "slug", "sku", "description", "short_description"),
            "classes": ("wide",)
        }),
        (_('Categories & Tags'), {
            "fields": ("primary_category", "categories", "tags"),
            "classes": ("wide",)
        }),
        (_('Pricing'), {
            "fields": ("price", "sale_price", "cost", "currency"),
            "description": "Set product pricing. Sale price will override regular price when set."
        }),
        (_('Inventory'), {
            "fields": ("stock_quantity", "low_stock_threshold"),
            "description": "Manage inventory levels and tracking."
        }),
        (_('Shipping'), {
            "fields": ("weight", "length", "width", "height", "shipping_material"),
        }),
        (_('Display'), {
            "fields": ("aspect_ratio",),
        }),
        (_('Sustainability'), {
            "fields": (
                "carbon_footprint_kg",
                "recycled_content_percentage",
                "sustainability_score",
                "ethical_sourcing_notes",
                "eco_certifications",
            ),
            "classes": ("collapse",),
        }),
        (_('Mobile & Voice'), {
            "fields": ("is_mobile_optimized", "voice_keywords"),
            "classes": ("collapse",),
        }),
        (_('Status & Flags'), {
            "fields": ("is_active", "is_featured", "is_bestseller", "is_new_arrival", "can_be_customized"),
        }),
        (_('SEO'), {
            "fields": ("meta_title", "meta_description"),
            "classes": ("collapse",),
        }),
    )

    class Media:
        css = {
            "all": ("css/admin/category_tree_widget.css",),
        }
        js = (
            "admin/js/vendor/jquery/jquery.min.js",
            "admin/js/jquery.init.js",
            "js/admin/category_tree_widget.js",
            "js/admin/product_image_live_preview.js",
            "js/admin/product_ai_autofill.js",
        )

    def get_urls(self):
        custom_urls = [
            path(
                "ai/autofill/start/",
                self.admin_site.admin_view(self.ai_autofill_start_view),
                name="catalog_product_ai_autofill_start",
            ),
            path(
                "ai/autofill/<uuid:job_id>/status/",
                self.admin_site.admin_view(self.ai_autofill_status_view),
                name="catalog_product_ai_autofill_status",
            ),
            path(
                "ai/autofill/<uuid:job_id>/apply/",
                self.admin_site.admin_view(self.ai_autofill_apply_view),
                name="catalog_product_ai_autofill_apply",
            ),
            path(
                "ai/autofill/<uuid:job_id>/feedback/",
                self.admin_site.admin_view(self.ai_autofill_feedback_view),
                name="catalog_product_ai_autofill_feedback",
            ),
        ]
        return custom_urls + super().get_urls()

    def render_change_form(self, request, context, add=False, change=False, form_url="", obj=None):
        max_images = int(getattr(settings, "PRODUCT_AI_MAX_IMAGES", 4))
        context = dict(context)
        inline_formsets = list(context.get("inline_admin_formsets") or [])
        top_inline_formsets = []
        remaining_inline_formsets = []
        for inline_formset in inline_formsets:
            inline_model = getattr(getattr(inline_formset, "opts", None), "model", None)
            if inline_model is ProductImage:
                top_inline_formsets.append(inline_formset)
            else:
                remaining_inline_formsets.append(inline_formset)
        context["top_inline_admin_formsets"] = top_inline_formsets
        context["inline_admin_formsets"] = remaining_inline_formsets
        context["product_ai_enabled"] = bool(getattr(settings, "PRODUCT_AI_ENABLED", False))
        context["product_ai_max_images"] = max_images
        context["product_ai_endpoints"] = {
            "start": reverse("admin:catalog_product_ai_autofill_start"),
            "status_template": reverse("admin:catalog_product_ai_autofill_status", kwargs={"job_id": uuid.uuid4()}),
            "apply_template": reverse("admin:catalog_product_ai_autofill_apply", kwargs={"job_id": uuid.uuid4()}),
            "feedback_template": reverse("admin:catalog_product_ai_autofill_feedback", kwargs={"job_id": uuid.uuid4()}),
        }
        return super().render_change_form(request, context, add, change, form_url, obj)

    def get_form(self, request, obj=None, **kwargs):
        base_form = super().get_form(request, obj, **kwargs)

        class RequestAwareProductAdminForm(base_form):
            def __init__(self, *args, **inner_kwargs):
                inner_kwargs["request"] = request
                super().__init__(*args, **inner_kwargs)

        return RequestAwareProductAdminForm

    def _parse_payload(self, request):
        if request.content_type and "application/json" in request.content_type:
            try:
                return json.loads(request.body.decode("utf-8"))
            except Exception:
                return {}
        return request.POST

    def _parse_bool(self, value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _clean_hint_text(self, value, *, max_chars=300):
        if value is None:
            return ""
        text = " ".join(str(value).split()).strip()
        if not text:
            return ""
        return text[:max_chars]

    def _parse_context_hints(self, payload):
        raw_hints = payload.get("context_hints") if hasattr(payload, "get") else None
        if not raw_hints:
            return {}

        parsed = {}
        if isinstance(raw_hints, str):
            try:
                parsed = json.loads(raw_hints)
            except Exception:
                return {}
        elif isinstance(raw_hints, dict):
            parsed = dict(raw_hints)
        else:
            return {}

        if not isinstance(parsed, dict):
            return {}

        scalar_keys = {
            "name": 220,
            "short_description": 500,
            "description": 1500,
            "primary_category_id": 64,
            "primary_category_name": 200,
        }
        list_keys = {
            "image_names": 12,
            "category_ids": 20,
            "category_names": 20,
            "tag_names": 24,
            "eco_certification_names": 16,
        }
        sanitized = {}

        for key, max_chars in scalar_keys.items():
            value = self._clean_hint_text(parsed.get(key), max_chars=max_chars)
            if value:
                sanitized[key] = value

        for key, max_items in list_keys.items():
            raw_values = parsed.get(key)
            if not isinstance(raw_values, list):
                continue
            values = []
            for item in raw_values[:max_items]:
                text = self._clean_hint_text(item, max_chars=180)
                if text:
                    values.append(text)
            if values:
                sanitized[key] = values

        return sanitized

    def _rate_limited(self, request):
        limit = int(getattr(settings, "PRODUCT_AI_START_RATE_LIMIT_PER_MIN", 6))
        key = f"catalog:autofill:start:{request.user.id}"
        current = cache.get(key, 0)
        if current >= limit:
            return True
        cache.set(key, current + 1, timeout=60)
        return False

    def _run_with_sqlite_lock_retry(self, operation, *, context: str):
        """
        Retry short-lived SQLite write-lock conflicts.
        """
        retries = int(getattr(settings, "PRODUCT_AI_SQLITE_LOCK_RETRIES", 3))
        base_backoff = float(getattr(settings, "PRODUCT_AI_SQLITE_LOCK_BACKOFF_SECONDS", 0.15))
        attempt = 0
        while True:
            try:
                return operation()
            except OperationalError as exc:
                message = str(exc).lower()
                if "database is locked" not in message or attempt >= retries:
                    raise
                delay = base_backoff * (2 ** attempt)
                logger.warning(
                    "Retrying %s due to SQLite lock (attempt %s/%s, delay=%.3fs)",
                    context,
                    attempt + 1,
                    retries,
                    delay,
                )
                time.sleep(delay)
                attempt += 1

    def _check_upload_file(self, uploaded):
        allowed_mime = {"image/jpeg", "image/png", "image/webp"}
        max_size_mb = int(getattr(settings, "PRODUCT_AI_MAX_IMAGE_SIZE_MB", 8))
        content_type = getattr(uploaded, "content_type", "").lower()
        if content_type not in allowed_mime:
            return False, f"Unsupported file type: {content_type or 'unknown'}"
        if uploaded.size > (max_size_mb * 1024 * 1024):
            return False, f"Image exceeds {max_size_mb}MB limit."
        if not self._scan_upload(uploaded):
            return False, "File blocked by security scanning hook."
        return True, ""

    def _scan_upload(self, uploaded):
        """
        Hook point for antivirus/file-scanning integration.
        """
        return True

    def _store_temp_upload(self, job_id, uploaded):
        filename = f"{uuid.uuid4()}-{uploaded.name}".replace(" ", "_")
        storage_path = f"catalog/autofill/{job_id}/{filename}"
        return default_storage.save(storage_path, uploaded)

    def _get_job_for_user(self, request, job_id):
        qs = ProductAutofillJob.objects.select_related("product", "requested_by")
        if request.user.is_superuser:
            job = qs.filter(id=job_id).first()
        else:
            job = qs.filter(id=job_id, requested_by=request.user).first()
        if not job:
            raise Http404("Autofill job not found.")
        return job

    def _rediss_ssl_param_missing(self):
        for attr in ("CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
            value = getattr(settings, attr, "")
            if not value:
                continue
            parsed = urlparse(str(value))
            if parsed.scheme != "rediss":
                continue
            params = dict(parse_qsl(parsed.query, keep_blank_values=True))
            if "ssl_cert_reqs" not in params:
                return True
        return False

    def _run_autofill_job_sync(self, job):
        from .ai.engine import ProductAutofillEngine

        ProductAutofillEngine(job_id=str(job.id)).run()
        job.refresh_from_db(fields=["status", "progress", "error_message", "updated_at"])

    def _serialize_suggestion(self, suggestion):
        metadata = suggestion.metadata or {}
        display_value = suggestion.display_value

        if suggestion.field_name in {"primary_category", "shipping_material"}:
            label = metadata.get("name")
            if isinstance(label, str) and label.strip():
                display_value = label.strip()
            elif suggestion.field_name == "primary_category" and suggestion.value_json:
                category = Category.objects.filter(id=suggestion.value_json).only("name").first()
                if category:
                    display_value = category.name
            elif suggestion.field_name == "shipping_material" and suggestion.value_json:
                material = ShippingMaterial.objects.filter(id=suggestion.value_json).only("name").first()
                if material:
                    display_value = material.name

        if suggestion.field_name in {"categories", "tags", "eco_certifications"}:
            labels = metadata.get("names")
            if isinstance(labels, list):
                cleaned = [str(value).strip() for value in labels if str(value).strip()]
                if cleaned:
                    display_value = ", ".join(cleaned)
            elif isinstance(suggestion.value_json, list) and suggestion.value_json:
                ids = [str(value) for value in suggestion.value_json if str(value).strip()]
                model = {
                    "categories": Category,
                    "tags": Tag,
                    "eco_certifications": EcoCertification,
                }.get(suggestion.field_name)
                if model:
                    names = list(model.objects.filter(id__in=ids).values_list("name", flat=True))
                    if names:
                        display_value = ", ".join(names)

        return {
            "field_name": suggestion.field_name,
            "value": suggestion.value_json,
            "display_value": display_value,
            "confidence": suggestion.confidence,
            "is_null": suggestion.is_null_suggestion,
            "low_confidence": suggestion.low_confidence,
            "rationale": suggestion.rationale,
            "source_urls": suggestion.source_urls or [],
            "metadata": suggestion.metadata or {},
            "status": suggestion.status,
        }

    def ai_autofill_start_view(self, request):
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])

        if not getattr(settings, "PRODUCT_AI_ENABLED", False):
            return JsonResponse({"ok": False, "error": "Product AI is disabled."}, status=503)

        if not request.user.has_perm("catalog.change_product"):
            return JsonResponse({"ok": False, "error": "Permission denied."}, status=403)

        if self._rate_limited(request):
            return JsonResponse({"ok": False, "error": "Rate limit exceeded."}, status=429)

        payload = self._parse_payload(request)
        context_hints = self._parse_context_hints(payload)
        product_id = payload.get("product_id") if hasattr(payload, "get") else None
        currency = (payload.get("currency") if hasattr(payload, "get") else None) or getattr(settings, "DEFAULT_CURRENCY", "BDT")
        allow_external = self._parse_bool(payload.get("allow_external") if hasattr(payload, "get") else None, default=getattr(settings, "PRODUCT_AI_ALLOW_EXTERNAL_DEFAULT", True))
        locale = (payload.get("locale") if hasattr(payload, "get") else None) or translation.get_language() or "en"

        product = None
        if product_id:
            product = Product.objects.filter(id=product_id).first()
            if not product:
                return JsonResponse({"ok": False, "error": "Product not found."}, status=404)

        max_images = int(getattr(settings, "PRODUCT_AI_MAX_IMAGES", 4))
        uploads = request.FILES.getlist("images")
        if len(uploads) > max_images:
            return JsonResponse({"ok": False, "error": f"Maximum {max_images} images allowed."}, status=400)

        active_limit = int(getattr(settings, "PRODUCT_AI_MAX_CONCURRENT_JOBS", 2))
        active_jobs = ProductAutofillJob.objects.filter(
            requested_by=request.user,
            status__in=[ProductAutofillJob.STATUS_PENDING, ProductAutofillJob.STATUS_RUNNING],
        ).count()
        if active_jobs >= active_limit:
            return JsonResponse({"ok": False, "error": "Too many active autofill jobs. Please wait."}, status=429)

        with transaction.atomic():
            job = ProductAutofillJob.objects.create(
                product=product,
                requested_by=request.user,
                status=ProductAutofillJob.STATUS_PENDING,
                locale=locale,
                currency=currency,
                allow_external=allow_external,
                force_overwrite=False,
                image_count=0,
                progress=0,
            )

            temp_paths = []
            for uploaded in uploads[:max_images]:
                ok, error = self._check_upload_file(uploaded)
                if not ok:
                    job.status = ProductAutofillJob.STATUS_FAILED
                    job.error_message = error
                    job.save(update_fields=["status", "error_message", "updated_at"])
                    return JsonResponse({"ok": False, "error": error}, status=400)
                temp_paths.append(self._store_temp_upload(job.id, uploaded))

            job.image_count = len(temp_paths)
            job.input_payload = {
                "temp_images": temp_paths,
                "requested_ip": request.META.get("REMOTE_ADDR"),
                "requested_at": timezone.now().isoformat(),
                "context_hints": context_hints,
            }
            job.save(update_fields=["image_count", "input_payload", "updated_at"])

        dispatch_mode = "async"
        if self._rediss_ssl_param_missing():
            logger.warning(
                "Skipping Celery enqueue for autofill job %s because rediss URL is missing ssl_cert_reqs; running sync fallback.",
                job.id,
            )
            dispatch_mode = "sync_fallback"
            try:
                self._run_autofill_job_sync(job)
            except Exception as sync_exc:
                logger.exception("Synchronous fallback failed for job %s: %s", job.id, sync_exc)
                job.status = ProductAutofillJob.STATUS_FAILED
                job.error_message = "Unable to dispatch autofill job; check Celery/Redis configuration."
                job.save(update_fields=["status", "error_message", "updated_at"])
                return JsonResponse(
                    {
                        "ok": False,
                        "error": "Unable to dispatch autofill job. Fix Celery/Redis configuration and retry.",
                    },
                    status=500,
                )
        else:
            try:
                run_product_autofill_job.delay(str(job.id))
            except Exception as exc:
                message = str(exc or "")
                ssl_error = "rediss:// URL must have parameter ssl_cert_reqs"
                if ssl_error in message:
                    logger.warning(
                        "Celery enqueue skipped for autofill job %s due to Redis SSL URL config: %s",
                        job.id,
                        message,
                    )
                else:
                    logger.exception("Failed to enqueue product autofill job %s: %s", job.id, exc)
                dispatch_mode = "sync_fallback"
                try:
                    self._run_autofill_job_sync(job)
                except Exception as sync_exc:
                    logger.exception("Synchronous fallback failed for job %s: %s", job.id, sync_exc)
                    job.status = ProductAutofillJob.STATUS_FAILED
                    job.error_message = "Unable to dispatch autofill job; check Celery/Redis configuration."
                    job.save(update_fields=["status", "error_message", "updated_at"])
                    return JsonResponse(
                        {
                            "ok": False,
                            "error": "Unable to dispatch autofill job. Fix Celery/Redis configuration and retry.",
                        },
                        status=500,
                    )

        return JsonResponse(
            {
                "ok": True,
                "job_id": str(job.id),
                "status": job.status,
                "image_count": job.image_count,
                "dispatch_mode": dispatch_mode,
            }
        )

    def ai_autofill_status_view(self, request, job_id):
        if request.method != "GET":
            return HttpResponseNotAllowed(["GET"])
        job = self._get_job_for_user(request, job_id)
        suggestions = [
            self._serialize_suggestion(item)
            for item in job.suggestions.order_by("field_name")
        ]
        return JsonResponse(
            {
                "ok": True,
                "job_id": str(job.id),
                "status": job.status,
                "progress": job.progress,
                "error_message": job.error_message,
                "summary": job.summary or {},
                "suggestions": suggestions,
            }
        )

    def ai_autofill_apply_view(self, request, job_id):
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        job = self._get_job_for_user(request, job_id)
        if job.status != ProductAutofillJob.STATUS_COMPLETED:
            return JsonResponse({"ok": False, "error": "Job not completed yet."}, status=409)

        payload = self._parse_payload(request)
        force_overwrite = self._parse_bool(payload.get("force_overwrite") if hasattr(payload, "get") else None, default=False)
        suggestions = list(job.suggestions.order_by("field_name"))

        if not job.product_id:
            suggestion_map = {
                item.field_name: item.value_json
                for item in suggestions
                if item.value_json not in (None, "", [])
            }
            return JsonResponse(
                {
                    "ok": True,
                    "mode": "client_apply",
                    "fields": suggestion_map,
                }
            )

        lock_key = f"catalog:autofill:apply:{job.id}"
        lock_timeout = int(getattr(settings, "PRODUCT_AI_APPLY_LOCK_TIMEOUT_SECONDS", 30))
        if not cache.add(lock_key, str(request.user.id), timeout=lock_timeout):
            return JsonResponse(
                {
                    "ok": False,
                    "error": "Apply is already in progress for this job. Please wait a moment and retry.",
                },
                status=409,
            )

        try:
            def _apply_once():
                with transaction.atomic():
                    fresh_job = ProductAutofillJob.objects.select_related("product").get(id=job.id)
                    fresh_suggestions = list(fresh_job.suggestions.order_by("field_name"))
                    result = apply_suggestions_to_product(
                        product=fresh_job.product,
                        suggestions=fresh_suggestions,
                        force_overwrite=force_overwrite,
                    )

                    changed_fields = set(result.get("changed_fields", []))
                    if changed_fields:
                        now = timezone.now()
                        applied_suggestions = []
                        feedback_entries = []
                        for suggestion in fresh_suggestions:
                            if suggestion.field_name not in changed_fields:
                                continue
                            suggestion.status = ProductFieldSuggestion.STATUS_APPLIED
                            suggestion.updated_at = now
                            applied_suggestions.append(suggestion)
                            feedback_entries.append(
                                ProductAutofillFeedback(
                                    job=fresh_job,
                                    suggestion=suggestion,
                                    user=request.user,
                                    field_name=suggestion.field_name,
                                    feedback_type=ProductAutofillFeedback.TYPE_ACCEPTED,
                                    final_value=suggestion.value_json,
                                    metadata={"source": "apply_endpoint", "force_overwrite": force_overwrite},
                                )
                            )
                        if applied_suggestions:
                            ProductFieldSuggestion.objects.bulk_update(applied_suggestions, ["status", "updated_at"])
                        if feedback_entries:
                            ProductAutofillFeedback.objects.bulk_create(feedback_entries)
                    return result

            result = self._run_with_sqlite_lock_retry(_apply_once, context=f"autofill apply job={job.id}")
        except OperationalError as exc:
            if "database is locked" in str(exc).lower():
                return JsonResponse(
                    {
                        "ok": False,
                        "error": "Database is busy. Please retry apply in a few seconds.",
                    },
                    status=503,
                )
            raise
        finally:
            cache.delete(lock_key)

        return JsonResponse(
            {
                "ok": True,
                "mode": "server_apply",
                "result": result,
            }
        )

    def ai_autofill_feedback_view(self, request, job_id):
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        job = self._get_job_for_user(request, job_id)
        payload = self._parse_payload(request)

        items = payload.get("items") if hasattr(payload, "get") else None
        if not items:
            items = [payload]

        created = 0
        for item in items:
            field_name = (item or {}).get("field_name")
            feedback_type = (item or {}).get("feedback_type")
            final_value = (item or {}).get("final_value")
            note = (item or {}).get("note", "")
            if not field_name or feedback_type not in {
                ProductAutofillFeedback.TYPE_ACCEPTED,
                ProductAutofillFeedback.TYPE_REJECTED,
                ProductAutofillFeedback.TYPE_EDITED,
            }:
                continue

            suggestion = ProductFieldSuggestion.objects.filter(job=job, field_name=field_name).first()
            ProductAutofillFeedback.objects.create(
                job=job,
                suggestion=suggestion,
                user=request.user,
                field_name=field_name,
                feedback_type=feedback_type,
                previous_value=suggestion.value_json if suggestion else None,
                final_value=final_value,
                note=note,
                metadata={"source": "manual_feedback"},
            )
            if suggestion:
                if feedback_type == ProductAutofillFeedback.TYPE_REJECTED:
                    suggestion.status = ProductFieldSuggestion.STATUS_REJECTED
                elif feedback_type == ProductAutofillFeedback.TYPE_EDITED:
                    suggestion.status = ProductFieldSuggestion.STATUS_EDITED
                elif feedback_type == ProductAutofillFeedback.TYPE_ACCEPTED:
                    suggestion.status = ProductFieldSuggestion.STATUS_APPLIED
                suggestion.save(update_fields=["status", "updated_at"])
            created += 1

        return JsonResponse({"ok": True, "created": created})

    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'primary_category', 'shipping_material'
        ).prefetch_related('categories', 'images')

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        
        # After saving the product, ensure primary_category is also in categories M2M
        if obj.primary_category and obj.primary_category not in obj.categories.all():
            obj.categories.add(obj.primary_category)
            messages.info(request, _("Primary category automatically added to product categories."))
        
        # Also ensure product_count is updated for the primary category
        if change:
            # If primary category changed, update counts for old and new
            old_primary_category_id = form.initial.get('primary_category')
            if old_primary_category_id and old_primary_category_id != obj.primary_category_id:
                old_primary_category = Category.objects.get(pk=old_primary_category_id)
                old_primary_category.product_count = old_primary_category.products.count()
                old_primary_category.save()
            
            # Update new primary category count
            if obj.primary_category:
                obj.primary_category.product_count = obj.primary_category.products.count()
                obj.primary_category.save()
        else: # New product
            if obj.primary_category:
                obj.primary_category.product_count = obj.primary_category.products.count()
                obj.primary_category.save()
    
    def thumbnail_preview(self, obj):
        # Try to get primary image or first image
        primary_image = obj.images.filter(is_primary=True).first() or obj.images.first()
        if primary_image and primary_image.image:
            return format_html(
                '<img src="{}" style="max-height: 40px; max-width: 60px; object-fit: cover; border-radius: 4px;" />',
                primary_image.image.url
            )
        return format_html('<span style="color: #9ca3af;">No image</span>')
    thumbnail_preview.short_description = ""
    
    def price_display(self, obj):
        if obj.sale_price:
            return format_html(
                '<div><span style="text-decoration: line-through; color: #9ca3af;">{}</span><br/>'
                '<span style="color: #dc2626; font-weight: 600;">{}</span></div>',
                format_currency(obj.price), format_currency(obj.sale_price)
            )
        return format_currency(obj.price)
    price_display.short_description = _("Price")
    price_display.admin_order_field = "price"
    
    def stock_status(self, obj):
        qty = obj.stock_quantity or 0
        threshold = getattr(obj, 'low_stock_threshold', 10) or 10
        
        if qty <= 0:
            return format_html(
                '<span style="display: inline-flex; align-items: center; padding: 2px 8px; '
                'background-color: #fee2e2; color: #991b1b; border-radius: 4px; font-size: 11px; font-weight: 600;">'
                'Out of Stock</span>'
            )
        elif qty <= threshold:
            return format_html(
                '<span style="display: inline-flex; align-items: center; padding: 2px 8px; '
                'background-color: #fef3c7; color: #92400e; border-radius: 4px; font-size: 11px; font-weight: 600;">'
                'Low ({})</span>',
                qty
            )
        return format_html(
            '<span style="display: inline-flex; align-items: center; padding: 2px 8px; '
            'background-color: #dcfce7; color: #166534; border-radius: 4px; font-size: 11px; font-weight: 600;">'
            '{}</span>',
            qty
        )
    stock_status.short_description = _("Stock")
    stock_status.admin_order_field = "stock_quantity"
    
    def performance_stats(self, obj):
        views = obj.views_count or 0
        sales = obj.sales_count or 0
        wishlist = obj.wishlist_count or 0
        
        return format_html(
            '<div style="font-size: 11px; line-height: 1.4;">'
            '<span title="Views">👁️ {}</span> · '
            '<span title="Sales">🛒 {}</span> · '
            '<span title="Wishlist">❤️ {}</span></div>',
            format_number(views), format_number(sales), format_number(wishlist)
        )
    performance_stats.short_description = _("Stats")
    
    def is_active_badge(self, obj):
        if obj.is_active:
            return format_html('<span style="color: #16a34a;">●</span>')
        return format_html('<span style="color: #dc2626;">●</span>')
    is_active_badge.short_description = _("Active")
    is_active_badge.admin_order_field = "is_active"
    
    def is_featured_badge(self, obj):
        if obj.is_featured:
            return format_html('<span style="color: #eab308;">⭐</span>')
        return ""
    is_featured_badge.short_description = _("Featured")
    is_featured_badge.admin_order_field = "is_featured"

    def primary_category_display(self, obj):
        if obj.primary_category:
            return format_html(
                '<a href="{}">{}</a>',
                reverse('admin:catalog_category_change', args=[obj.primary_category.pk]),
                obj.primary_category.name
            )
        return "-"
    primary_category_display.short_description = _("Category")
    primary_category_display.admin_order_field = "primary_category__name"
    
    # Bulk Actions
    def mark_bestseller(self, request, queryset):
        updated = queryset.update(is_bestseller=True)
        self.message_user(request, f'{updated} products marked as bestseller.', messages.SUCCESS)
    mark_bestseller.short_description = _("Mark as bestseller")
    
    def unmark_bestseller(self, request, queryset):
        updated = queryset.update(is_bestseller=False)
        self.message_user(request, f'{updated} products unmarked as bestseller.', messages.SUCCESS)
    unmark_bestseller.short_description = _("Remove bestseller status")
    
    def mark_new_arrival(self, request, queryset):
        updated = queryset.update(is_new_arrival=True)
        self.message_user(request, f'{updated} products marked as new arrival.', messages.SUCCESS)
    mark_new_arrival.short_description = _("Mark as new arrival")
    
    def unmark_new_arrival(self, request, queryset):
        updated = queryset.update(is_new_arrival=False)
        self.message_user(request, f'{updated} products unmarked as new arrival.', messages.SUCCESS)
    unmark_new_arrival.short_description = _("Remove new arrival status")
    
    def duplicate_products(self, request, queryset):
        for product in queryset:
            # Create a copy
            product.pk = None
            product.sku = f"{product.sku}-copy"
            product.slug = f"{product.slug}-copy"
            product.is_active = False
            product.save()
        self.message_user(request, f'{queryset.count()} products duplicated (inactive).', messages.SUCCESS)
    duplicate_products.short_description = _("Duplicate selected products")
    
    def recalculate_stats(self, request, queryset):
        """Recalculate view/sales/wishlist counts from actual data."""
        from apps.analytics.models import ProductView
        from apps.orders.models import OrderItem
        
        updated = 0
        for product in queryset:
            try:
                # Recalculate views
                views = ProductView.objects.filter(product=product).count()
                # Recalculate sales
                sales = OrderItem.objects.filter(
                    product=product, 
                    order__status='delivered'
                ).aggregate(total=Sum('quantity'))['total'] or 0
                
                product.views_count = views
                product.sales_count = sales
                product.save(update_fields=['views_count', 'sales_count'])
                updated += 1
            except Exception:
                pass
        
        self.message_user(request, f'Recalculated stats for {updated} products.', messages.SUCCESS)
    recalculate_stats.short_description = _("Recalculate statistics")


@admin.register(ShippingMaterial)
class ShippingMaterialAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("name", "eco_score", "created_at")
    search_fields = ("name",)


@admin.register(Badge)
class BadgeAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("name", "slug", "is_active", "start", "end", "priority")
    search_fields = ("name", "slug")
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Spotlight)
class SpotlightAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("name", "placement", "product", "category", "start", "end", "priority", "is_active")
    list_filter = ("placement", "is_active")


@admin.register(Product3DAsset)
class Product3DAssetAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("product", "file_type", "validated", "is_ar_compatible", "uploaded_at")
    readonly_fields = ("uploaded_at",)


@admin.register(Attribute)
class AttributeAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("name", "slug")
    search_fields = ("name", "slug")


@admin.register(AttributeValue)
class AttributeValueAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("attribute", "value")
    search_fields = ("value",)


@admin.register(Facet)
class FacetAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("name", "slug", "type")
    search_fields = ("name",)


@admin.register(CategoryFacet)
class CategoryFacetAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("category", "facet")


@admin.register(Tag)
class TagAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("name",)
    search_fields = ("name",)


@admin.register(AspectRatioChoice)
class AspectRatioChoiceAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("code", "label", "sort_order", "is_active", "is_default", "updated_at")
    list_filter = ("is_active", "is_default")
    search_fields = ("code", "label")
    ordering = ("sort_order", "code")


@admin.register(CategoryPricingProfile)
class CategoryPricingProfileAdmin(ImportExportEnhancedModelAdmin):
    list_display = (
        "category",
        "min_margin_percentage",
        "max_margin_percentage",
        "sale_discount_min_percentage",
        "sale_discount_max_percentage",
        "stock_default",
        "low_stock_threshold_default",
        "is_active",
    )
    search_fields = ("category__name",)
    list_filter = ("is_active",)


@admin.register(ProductAutofillJob)
class ProductAutofillJobAdmin(ImportExportEnhancedModelAdmin):
    list_display = (
        "id",
        "product",
        "requested_by",
        "status",
        "progress",
        "locale",
        "currency",
        "image_count",
        "allow_external",
        "created_at",
    )
    list_filter = ("status", "allow_external", "locale", "created_at")
    search_fields = ("id", "product__name", "requested_by__email", "requested_by__username")
    readonly_fields = (
        "id",
        "product",
        "requested_by",
        "status",
        "progress",
        "locale",
        "currency",
        "image_count",
        "allow_external",
        "force_overwrite",
        "input_payload",
        "summary",
        "error_message",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    )


@admin.register(ProductAutofillSource)
class ProductAutofillSourceAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("job", "provider", "source_type", "domain", "trust_score", "fetched_at")
    list_filter = ("provider", "source_type", "fetched_at")
    search_fields = ("job__id", "url", "domain", "title")
    readonly_fields = ("job", "provider", "source_type", "url", "domain", "title", "snippet", "trust_score", "metadata", "fetched_at")


@admin.register(ProductFieldSuggestion)
class ProductFieldSuggestionAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("job", "field_name", "confidence", "status", "is_null_suggestion", "low_confidence", "updated_at")
    list_filter = ("status", "is_null_suggestion", "low_confidence", "field_name")
    search_fields = ("job__id", "field_name", "display_value", "rationale")
    readonly_fields = (
        "job",
        "field_name",
        "value_json",
        "display_value",
        "confidence",
        "is_null_suggestion",
        "low_confidence",
        "rationale",
        "source_urls",
        "metadata",
        "status",
        "created_at",
        "updated_at",
    )


@admin.register(ProductAutofillFeedback)
class ProductAutofillFeedbackAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("job", "user", "field_name", "feedback_type", "created_at")
    list_filter = ("feedback_type", "field_name", "created_at")
    search_fields = ("job__id", "user__email", "field_name", "note")
    readonly_fields = (
        "job",
        "suggestion",
        "user",
        "field_name",
        "feedback_type",
        "previous_value",
        "final_value",
        "note",
        "metadata",
        "created_at",
    )


@admin.register(ProductQuestion)
class ProductQuestionAdmin(ImportExportEnhancedModelAdmin):
    list_display = ('product', 'user', 'question_text', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('product__name', 'user__email', 'question_text')
    readonly_fields = ('product', 'user', 'created_at', 'updated_at')
    ordering = ['-created_at']
    inlines = [ProductAnswerInline]

    actions = ['approve_questions', 'reject_questions']

    def approve_questions(self, request, queryset):
        updated = queryset.update(status='approved')
        self.message_user(request, f'{updated} questions approved.')
    approve_questions.short_description = 'Approve selected questions'

    def reject_questions(self, request, queryset):
        updated = queryset.update(status='rejected')
        self.message_user(request, f'{updated} questions rejected.')
    reject_questions.short_description = 'Reject selected questions'


@admin.register(ProductAnswer)
class ProductAnswerAdmin(ImportExportEnhancedModelAdmin):
    list_display = ('question', 'user', 'answer_text', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('question__question_text', 'user__email', 'answer_text')
    readonly_fields = ('question', 'user', 'created_at', 'updated_at')
    ordering = ['created_at']

    actions = ['approve_answers', 'reject_answers']

    def approve_answers(self, request, queryset):
        updated = queryset.update(status='approved')
        self.message_user(request, f'{updated} answers approved.')
    approve_answers.short_description = 'Approve selected answers'

    def reject_answers(self, request, queryset):
        updated = queryset.update(status='rejected')
        self.message_user(request, f'{updated} answers rejected.')
    reject_answers.short_description = 'Reject selected answers'


# Inline registered for ProductVariant to ensure attribute M2M is manageable
@admin.register(ProductVariant)
class ProductVariantAdmin(ImportExportEnhancedModelAdmin):
    list_display = ("sku", "product", "price", "stock_quantity", "is_default")
    search_fields = ("sku", "product__name")


@admin.register(CustomerPhoto)
class CustomerPhotoAdmin(ImportExportEnhancedModelAdmin):
    list_display = ('product', 'user', 'status', 'created_at', 'thumbnail_preview')
    list_filter = ('status', 'created_at')
    search_fields = ('product__name', 'user__email', 'description')
    readonly_fields = ('product', 'user', 'created_at', 'updated_at', 'thumbnail_preview')
    
    actions = ['approve_photos', 'reject_photos']

    def thumbnail_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height: 80px; max-width: 120px; object-fit: cover; border-radius: 4px;" />',
                obj.image.url
            )
        return "-"
    thumbnail_preview.short_description = "Image Preview"

    def approve_photos(self, request, queryset):
        updated = queryset.update(status='approved')
        self.message_user(request, f'{updated} photos approved.')
    approve_photos.short_description = 'Approve selected photos'

    def reject_photos(self, request, queryset):
        updated = queryset.update(status='rejected')
        self.message_user(request, f'{updated} photos rejected.')
    reject_photos.short_description = 'Reject selected photos'
