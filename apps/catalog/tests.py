import io
import json
import os
import tempfile
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from django.conf import settings
from django.contrib import admin as django_admin
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings, RequestFactory

from apps.catalog.admin import ProductAdmin
from apps.catalog.forms import ProductAdminForm
from apps.catalog.ai.schemas import FieldSuggestionPayload
from apps.catalog.ai.providers.personalization import PersonalizationProvider
from apps.catalog.ai.providers.pricing import PricingProvider
from apps.catalog.ai.providers.research import ResearchProvider
from apps.catalog.ai.providers.research import is_safe_public_url
from apps.catalog.ai.providers.search import SearchProvider
from apps.catalog.ai.validators import apply_suggestions_to_product, normalize_raw_suggestions
from apps.catalog.models import (
    AspectRatioChoice,
    Category,
    CategoryPricingProfile,
    EcoCertification,
    Product,
    ProductAutofillFeedback,
    ProductAutofillJob,
    ProductFieldSuggestion,
    ShippingMaterial,
    Tag,
)

def _image_upload(name: str = "sample.png") -> SimpleUploadedFile:
    bio = io.BytesIO()
    image = Image.new("RGB", (120, 120), color=(120, 40, 50))
    image.save(bio, format="PNG")
    return SimpleUploadedFile(name, bio.getvalue(), content_type="image/png")


class CatalogRegressionTests(TestCase):
    def test_category_tree_static_assets_exist(self):
        css_path = os.path.join(
            settings.BASE_DIR,
            "apps",
            "catalog",
            "static",
            "css",
            "admin",
            "category_tree_widget.css",
        )
        js_path = os.path.join(
            settings.BASE_DIR,
            "apps",
            "catalog",
            "static",
            "js",
            "admin",
            "category_tree_widget.js",
        )
        self.assertTrue(os.path.exists(css_path))
        self.assertTrue(os.path.exists(js_path))

    def test_product_image_live_preview_asset_exists(self):
        js_path = os.path.join(
            settings.BASE_DIR,
            "apps",
            "catalog",
            "static",
            "js",
            "admin",
            "product_image_live_preview.js",
        )
        self.assertTrue(os.path.exists(js_path))

    def test_product_admin_media_includes_live_preview_script(self):
        product_admin = ProductAdmin(Product, django_admin.site)
        media_js = tuple(getattr(product_admin.media, "_js", ()))
        self.assertIn("js/admin/product_image_live_preview.js", media_js)

    def test_search_provider_normalizes_duckduckgo_redirect_urls(self):
        url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fproducts%2Fitem-1"
        normalized = SearchProvider._normalize_result_url(url)
        self.assertEqual(normalized, "https://example.com/products/item-1")

    def test_research_provider_extracts_structured_product_data(self):
        from bs4 import BeautifulSoup

        html = """
        <html>
          <head>
            <script type="application/ld+json">
              {"@context":"https://schema.org","@type":"Product","name":"Eco Bottle","sku":"ECO-100","offers":{"@type":"Offer","price":"19.99"}}
            </script>
            <meta property="product:price:amount" content="18.99" />
          </head>
          <body><main>Reusable insulated bottle</main></body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        structured = ResearchProvider._extract_structured_product_data(soup)
        self.assertIn("Eco Bottle", structured["names"])
        self.assertIn("ECO-100", structured["sku_candidates"])
        self.assertIn("19.99", structured["price_amounts"])

    def test_celery_task_names_are_current(self):
        from core.celery import app

        self.assertEqual(app.conf.beat_schedule["check-low-stock"]["task"], "catalog.check_low_stock")
        self.assertEqual(app.conf.task_routes["ml.training.tasks.*"]["queue"], "ml")

    @patch("apps.catalog.management.commands.seed_categories.run_seed_command")
    def test_seed_categories_includes_aspect_ratio_choices(self, mock_run_seed):
        from apps.catalog.management.commands.seed_categories import Command

        command = Command()
        command.handle(
            force=False,
            confirm_prune=False,
            assign_facets=False,
            file=None,
            dry_run=True,
            no_prune=True,
        )

        called_only = mock_run_seed.call_args.kwargs.get("only", [])
        self.assertIn("catalog.aspect_ratio_choices", called_only)

    def test_export_taxonomy_includes_aspect_choices(self):
        from django.core.management import call_command

        AspectRatioChoice.objects.create(
            code="1:1",
            label="1:1",
            sort_order=0,
            is_default=True,
            is_active=True,
        )

        fd, path = tempfile.mkstemp(prefix="taxonomy_", suffix=".json")
        os.close(fd)
        try:
            call_command("export_taxonomy", out=path, format="json")
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertIn("aspect_choices", payload)
            self.assertIn("default_aspect_ratio", payload)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_product_form_primary_category_includes_ancestors_in_categories(self):
        root = Category.objects.create(name="Root", slug="root")
        parent = Category.objects.create(name="Parent", slug="parent", parent=root)
        child = Category.objects.create(name="Child", slug="child", parent=parent)

        form = ProductAdminForm()
        form.cleaned_data = {
            "primary_category": child,
            "categories": Category.objects.filter(id=child.id),
        }

        cleaned = form.clean()
        selected_ids = set(cleaned["categories"].values_list("id", flat=True))
        self.assertEqual(selected_ids, {root.id, parent.id, child.id})


class ProductAutofillValidationTests(TestCase):
    def setUp(self):
        self.category = Category.objects.create(name="Handmade Home", slug="handmade-home")
        self.tag = Tag.objects.create(name="eco")
        self.cert = EcoCertification.objects.create(name="FSC", slug="fsc")
        self.material = ShippingMaterial.objects.create(name="Recycled Paper", eco_score=8)

    def test_threshold_nulling_and_price_fallback(self):
        raw = {
            "name": {"value": "Weak Name", "confidence": 0.2},
            "price": {"value": None, "confidence": 0.1},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8)
        mapped = {item.field_name: item for item in result}
        self.assertIsNone(mapped["name"].value)
        self.assertIsNotNone(mapped["price"].value)

    def test_noise_tokens_are_rejected_for_text_fields(self):
        raw = {
            "name": {"value": "Tmp3S2Z29Pn", "confidence": 0.95},
            "description": {"value": "Tmp3S2Z29Pn tmp3s2z29pn", "confidence": 0.9},
            "short_description": {"value": "Tmp3S2Z29Pn", "confidence": 0.9},
            "meta_title": {"value": "Tmp3S2Z29Pn | Bunoraa", "confidence": 0.9},
            "meta_description": {"value": "Tmp3S2Z29Pn", "confidence": 0.9},
            "price": {"value": "10.00", "confidence": 0.4},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8)
        mapped = {item.field_name: item for item in result}
        self.assertIsNone(mapped["name"].value)
        self.assertIsNone(mapped["description"].value)
        self.assertIsNone(mapped["short_description"].value)
        self.assertIsNone(mapped["meta_title"].value)
        self.assertIsNone(mapped["meta_description"].value)
        self.assertFalse(mapped["price"].value is None)

    def test_aspect_ratio_fallback_uses_db_default_choice(self):
        AspectRatioChoice.objects.all().delete()
        AspectRatioChoice.objects.create(code="4:3", label="4:3", is_active=True, is_default=True, sort_order=0)
        AspectRatioChoice.objects.create(code="3:2", label="3:2", is_active=True, is_default=False, sort_order=10)

        raw = {
            "aspect_ratio": {"value": "9:16", "confidence": 0.9},
            "price": {"value": "20.00", "confidence": 0.6},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8)
        mapped = {item.field_name: item for item in result}
        self.assertEqual(mapped["aspect_ratio"].value, "4:3")

    def test_aspect_ratio_keeps_active_db_choice(self):
        AspectRatioChoice.objects.all().delete()
        AspectRatioChoice.objects.create(code="4:3", label="4:3", is_active=True, is_default=True, sort_order=0)
        AspectRatioChoice.objects.create(code="3:2", label="3:2", is_active=True, is_default=False, sort_order=10)

        raw = {
            "aspect_ratio": {"value": "3:2", "confidence": 0.9},
            "price": {"value": "20.00", "confidence": 0.6},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8)
        mapped = {item.field_name: item for item in result}
        self.assertEqual(mapped["aspect_ratio"].value, "3:2")

    def test_taxonomy_maps_to_existing_entities_only(self):
        raw = {
            "primary_category": {"value": "Handmade Home", "confidence": 0.3},
            "categories": {"value": ["Handmade Home", "Unknown Cat"], "confidence": 0.3},
            "tags": {"value": ["eco", "nonexistent"], "confidence": 0.3},
            "eco_certifications": {"value": ["FSC", "unknown cert"], "confidence": 0.3},
            "shipping_material": {"value": "Recycled Paper", "confidence": 0.3},
            "price": {"value": "20.00", "confidence": 0.9},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.1)
        mapped = {item.field_name: item for item in result}
        self.assertEqual(mapped["primary_category"].value, str(self.category.id))
        self.assertEqual(mapped["categories"].value, [str(self.category.id)])
        self.assertEqual(mapped["tags"].value, [str(self.tag.id)])
        self.assertEqual(mapped["eco_certifications"].value, [str(self.cert.id)])
        self.assertEqual(mapped["shipping_material"].value, str(self.material.id))

    def test_ssrf_url_guard(self):
        self.assertFalse(is_safe_public_url("http://127.0.0.1/admin"))
        self.assertFalse(is_safe_public_url("http://localhost:8000"))
        self.assertTrue(is_safe_public_url("https://example.com/product"))

    def test_pricing_profile_influences_estimation(self):
        CategoryPricingProfile.objects.create(
            category=self.category,
            min_margin_percentage=40,
            max_margin_percentage=60,
            sale_discount_min_percentage=10,
            sale_discount_max_percentage=20,
            stock_default=22,
            low_stock_threshold_default=6,
            is_active=True,
        )
        similar = [
            Product.objects.create(name="A", slug="a", price=Decimal("30.00"), primary_category=self.category),
            Product.objects.create(name="B", slug="b", price=Decimal("40.00"), primary_category=self.category),
        ]
        provider = PricingProvider()
        estimate = provider.estimate(
            product=None,
            primary_category=self.category,
            research_docs=[],
            similar_products=similar,
        )
        self.assertGreater(Decimal(str(estimate["price"]["value"])), Decimal("0.00"))
        self.assertEqual(int(estimate["stock_quantity"]["value"]), 22)
        self.assertEqual(int(estimate["low_stock_threshold"]["value"]), 6)

    def test_field_suggestion_payload_is_json_safe_for_decimal(self):
        payload = FieldSuggestionPayload(
            field_name="price",
            value=Decimal("19.99"),
            confidence=0.9,
            metadata={"nested": {"cost": Decimal("10.25")}},
        ).to_model_payload()
        self.assertEqual(payload["value_json"], "19.99")
        self.assertEqual(payload["metadata"]["nested"]["cost"], "10.25")

    def test_personalization_uses_feedback_memory(self):
        user = get_user_model().objects.create_user(email="staff1@example.com", password="pass")
        product = Product.objects.create(name="X", slug="x", price=Decimal("10.00"), primary_category=self.category)
        job = ProductAutofillJob.objects.create(
            product=product,
            requested_by=user,
            locale="en",
            currency="USD",
            allow_external=False,
        )
        ProductAutofillFeedback.objects.create(
            job=job,
            user=user,
            field_name="description",
            feedback_type=ProductAutofillFeedback.TYPE_EDITED,
            final_value="Elegant handcrafted premium finish with artisan detail",
        )
        hints = PersonalizationProvider().get_hints(user=user, category=self.category, locale="en")
        self.assertIn("[en]", hints["description_style"])

    def test_context_hints_can_resolve_name_and_categories(self):
        raw = {
            "name": {"value": None, "confidence": 0.1},
            "primary_category": {"value": None, "confidence": 0.0},
            "categories": {"value": [], "confidence": 0.0},
            "price": {"value": "20.00", "confidence": 0.9},
        }
        result = normalize_raw_suggestions(
            raw,
            confidence_threshold=0.8,
            context_hints={
                "name": "Handmade Home Vase",
                "primary_category_id": str(self.category.id),
                "primary_category_name": self.category.name,
                "category_ids": [str(self.category.id)],
                "category_names": [self.category.name],
            },
        )
        mapped = {item.field_name: item for item in result}
        self.assertEqual(mapped["name"].value, "Handmade Home Vase")
        self.assertEqual(mapped["primary_category"].value, str(self.category.id))
        self.assertEqual(mapped["categories"].value, [str(self.category.id)])

    def test_apply_suggestions_skips_null_text_values(self):
        product = Product.objects.create(
            name="Existing",
            slug="existing",
            price=Decimal("15.00"),
            primary_category=self.category,
            ethical_sourcing_notes="Already set",
        )
        suggestions = [
            SimpleNamespace(field_name="ethical_sourcing_notes", value_json=None),
            SimpleNamespace(field_name="name", value_json=None),
        ]
        result = apply_suggestions_to_product(
            product=product,
            suggestions=suggestions,
            force_overwrite=False,
        )
        product.refresh_from_db()
        self.assertEqual(product.ethical_sourcing_notes, "Already set")
        self.assertEqual(product.name, "Existing")
        self.assertEqual(result["applied"], 0)
        self.assertEqual(result["skipped"], 2)


@override_settings(
    PRODUCT_AI_ENABLED=True,
    PRODUCT_AI_MAX_IMAGES=4,
)
class ProductAutofillAdminEndpointTests(TestCase):
    def setUp(self):
        cache.clear()
        self.factory = RequestFactory()
        self.product_admin = ProductAdmin(Product, django_admin.site)
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            email="admin@example.com",
            password="pass",
            is_staff=True,
            is_superuser=True,
        )
        self.category = Category.objects.create(name="Decor", slug="decor")
        self.product = Product.objects.create(
            name="Base Product",
            slug="base-product",
            price=Decimal("25.00"),
            primary_category=self.category,
        )
        self.product.categories.add(self.category)

    @patch("apps.catalog.admin.run_product_autofill_job.delay")
    def test_start_endpoint_creates_job(self, mock_delay):
        request = self.factory.post(
            "/admin/catalog/product/ai/autofill/start/",
            data={
                "product_id": str(self.product.id),
                "currency": "USD",
                "locale": "en",
                "allow_external": "true",
                "context_hints": json.dumps(
                    {
                        "name": "Decor Lamp",
                        "primary_category_id": str(self.category.id),
                        "category_ids": [str(self.category.id)],
                        "tag_names": ["eco"],
                    }
                ),
                "images": _image_upload(),
            },
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_start_view(request)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertTrue(payload["ok"])
        job = ProductAutofillJob.objects.filter(id=payload["job_id"]).first()
        self.assertIsNotNone(job)
        self.assertEqual(job.input_payload["context_hints"]["name"], "Decor Lamp")
        self.assertEqual(job.input_payload["context_hints"]["category_ids"], [str(self.category.id)])
        mock_delay.assert_called_once()

    @patch("apps.catalog.ai.engine.ProductAutofillEngine")
    @patch("apps.catalog.admin.run_product_autofill_job.delay")
    def test_start_endpoint_falls_back_when_celery_enqueue_fails(self, mock_delay, mock_engine):
        mock_delay.side_effect = ValueError("redis ssl config invalid")
        mock_engine.return_value.run.return_value = {"status": "completed"}

        request = self.factory.post(
            "/admin/catalog/product/ai/autofill/start/",
            data={
                "product_id": str(self.product.id),
                "currency": "USD",
                "locale": "en",
                "allow_external": "true",
                "images": _image_upload(),
            },
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_start_view(request)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertEqual(payload["dispatch_mode"], "sync_fallback")
        mock_engine.assert_called_once()

    @override_settings(
        CELERY_BROKER_URL="rediss://redis.example.com:6379/1",
        CELERY_RESULT_BACKEND="rediss://redis.example.com:6379/2",
    )
    @patch("apps.catalog.ai.engine.ProductAutofillEngine")
    @patch("apps.catalog.admin.run_product_autofill_job.delay")
    def test_start_endpoint_skips_enqueue_when_rediss_ssl_param_missing(self, mock_delay, mock_engine):
        mock_engine.return_value.run.return_value = {"status": "completed"}

        request = self.factory.post(
            "/admin/catalog/product/ai/autofill/start/",
            data={
                "product_id": str(self.product.id),
                "currency": "USD",
                "locale": "en",
                "allow_external": "true",
                "images": _image_upload(),
            },
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_start_view(request)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertEqual(payload["dispatch_mode"], "sync_fallback")
        mock_delay.assert_not_called()
        mock_engine.assert_called_once()

    def test_status_endpoint_returns_suggestions(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
        )
        ProductFieldSuggestion.objects.create(
            job=job,
            field_name="name",
            value_json="Decor Bowl",
            display_value="Decor Bowl",
            confidence=0.91,
            rationale="test",
        )
        request = self.factory.get(f"/admin/catalog/product/ai/autofill/{job.id}/status/")
        request.user = self.user
        response = self.product_admin.ai_autofill_status_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], ProductAutofillJob.STATUS_COMPLETED)
        self.assertEqual(len(payload["suggestions"]), 1)

    def test_apply_endpoint_fill_blanks_only(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
        )
        ProductFieldSuggestion.objects.create(
            job=job,
            field_name="name",
            value_json="Overwritten Name",
            display_value="Overwritten Name",
            confidence=0.9,
        )
        ProductFieldSuggestion.objects.create(
            job=job,
            field_name="description",
            value_json="AI generated description",
            display_value="AI generated description",
            confidence=0.9,
        )
        request = self.factory.post(
            f"/admin/catalog/product/ai/autofill/{job.id}/apply/",
            data={"force_overwrite": "false"},
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_apply_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        self.product.refresh_from_db()
        self.assertEqual(self.product.name, "Base Product")
        self.assertEqual(self.product.description, "AI generated description")

    def test_apply_endpoint_for_new_product_returns_client_mode(self):
        job = ProductAutofillJob.objects.create(
            product=None,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
        )
        ProductFieldSuggestion.objects.create(
            job=job,
            field_name="name",
            value_json="New Suggested Product",
            display_value="New Suggested Product",
            confidence=0.9,
        )
        request = self.factory.post(
            f"/admin/catalog/product/ai/autofill/{job.id}/apply/",
            data={"force_overwrite": "false"},
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_apply_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertEqual(payload["mode"], "client_apply")
        self.assertEqual(payload["fields"]["name"], "New Suggested Product")

    def test_feedback_endpoint_updates_status(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
        )
        suggestion = ProductFieldSuggestion.objects.create(
            job=job,
            field_name="description",
            value_json="Desc",
            display_value="Desc",
            confidence=0.8,
        )
        request = self.factory.post(
            f"/admin/catalog/product/ai/autofill/{job.id}/feedback/",
            data=json.dumps(
                {
                    "field_name": "description",
                    "feedback_type": ProductAutofillFeedback.TYPE_REJECTED,
                    "note": "not accurate",
                }
            ),
            content_type="application/json",
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_feedback_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        suggestion.refresh_from_db()
        self.assertEqual(suggestion.status, ProductFieldSuggestion.STATUS_REJECTED)
        self.assertTrue(ProductAutofillFeedback.objects.filter(job=job, field_name="description").exists())
