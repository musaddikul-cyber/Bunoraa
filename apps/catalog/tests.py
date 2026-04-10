import io
import json
import os
import tempfile
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from django.conf import settings
from django.contrib import admin as django_admin
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, SimpleTestCase, override_settings, RequestFactory

from apps.catalog.admin import ProductAdmin
from apps.catalog.ai.engine import ProductAutofillEngine, _query_seed_tokens
from apps.catalog.ai.providers.extractors import build_field_candidates, get_internal_similar_products
from apps.catalog.ai.providers.nn_vision import (
    NNVisionInferenceError,
    NNVisionLoadError,
    NNVisionProvider,
)
from apps.catalog.forms import ProductAdminForm
from apps.catalog.ai.schemas import FieldSuggestionPayload
from apps.catalog.ai.providers.personalization import PersonalizationProvider
from apps.catalog.ai.providers.deep_research import ProductDeepResearchProvider
from apps.catalog.ai.providers.pricing import PricingProvider
from apps.catalog.ai.providers.research import ResearchDocument, ResearchProvider
from apps.catalog.ai.providers.research import is_safe_public_url
from apps.catalog.ai.providers.search import SearchProvider
from apps.catalog.ai.validators import apply_suggestions_to_product, normalize_raw_suggestions
from apps.catalog.api.views import CategoryViewSet, SearchAPIView
from apps.catalog.services import CategoryService
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
from apps.i18n.models import Currency

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

    def test_search_provider_normalizes_bing_redirect_urls(self):
        url = (
            "https://www.bing.com/ck/a?!&&p=demo&ptn=3&ver=2&hsh=4&"
            "u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9wcm9kdWN0cy9waW5rLWt1cnRpLXNldA&ntb=1"
        )
        normalized = SearchProvider._normalize_result_url(url)
        self.assertEqual(normalized, "https://example.com/products/pink-kurti-set")

    def test_query_seed_tokens_drop_upload_filename_noise(self):
        tokens = _query_seed_tokens("image(7).jpg product requirements kurti set")
        lowered = [token.lower() for token in tokens]
        self.assertNotIn("image(7).jpg", lowered)
        self.assertNotIn("requirements", lowered)
        self.assertIn("kurti", lowered)

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

    def test_root_categories_exclude_disabled_by_default(self):
        active = Category.objects.create(name="Active Root", slug="active-root", is_active=True)
        Category.objects.create(name="Disabled Root", slug="disabled-root", is_active=False)

        roots = CategoryService.get_root_categories()
        self.assertIn(active, roots)
        self.assertFalse(roots.filter(slug="disabled-root").exists())

    def test_category_api_list_supports_disabled_filter(self):
        Category.objects.create(name="Active Root", slug="active-root", is_active=True)
        Category.objects.create(name="Disabled Root", slug="disabled-root", is_active=False)

        factory = RequestFactory()
        list_view = CategoryViewSet.as_view({"get": "list"})

        default_response = list_view(factory.get("/api/v1/catalog/categories/"))
        self.assertEqual(default_response.status_code, 200)
        default_slugs = {item["slug"] for item in default_response.data}
        self.assertIn("active-root", default_slugs)
        self.assertNotIn("disabled-root", default_slugs)

        include_disabled_response = list_view(
            factory.get("/api/v1/catalog/categories/", {"include_disabled": "true"})
        )
        self.assertEqual(include_disabled_response.status_code, 200)
        include_disabled_slugs = {item["slug"] for item in include_disabled_response.data}
        self.assertIn("active-root", include_disabled_slugs)
        self.assertIn("disabled-root", include_disabled_slugs)

        disabled_only_response = list_view(
            factory.get("/api/v1/catalog/categories/", {"is_active": "false"})
        )
        self.assertEqual(disabled_only_response.status_code, 200)
        disabled_only_slugs = {item["slug"] for item in disabled_only_response.data}
        self.assertEqual(disabled_only_slugs, {"disabled-root"})

    def test_category_api_list_supports_has_products_filter(self):
        Category.objects.create(
            name="Has Products Root",
            slug="has-products-root",
            is_active=True,
            product_count=3,
        )
        Category.objects.create(
            name="Empty Root",
            slug="empty-root",
            is_active=True,
            product_count=0,
        )

        factory = RequestFactory()
        list_view = CategoryViewSet.as_view({"get": "list"})

        with_products_response = list_view(
            factory.get("/api/v1/catalog/categories/", {"has_products": "true"})
        )
        self.assertEqual(with_products_response.status_code, 200)
        with_products_slugs = {item["slug"] for item in with_products_response.data}
        self.assertIn("has-products-root", with_products_slugs)
        self.assertNotIn("empty-root", with_products_slugs)

        without_products_response = list_view(
            factory.get("/api/v1/catalog/categories/", {"has_products": "false"})
        )
        self.assertEqual(without_products_response.status_code, 200)
        without_products_slugs = {item["slug"] for item in without_products_response.data}
        self.assertIn("empty-root", without_products_slugs)
        self.assertNotIn("has-products-root", without_products_slugs)

    def test_category_api_list_orders_by_sort_order(self):
        Category.objects.create(name="Zulu", slug="zulu", sort_order=30, is_active=True)
        Category.objects.create(name="Alpha", slug="alpha", sort_order=10, is_active=True)
        Category.objects.create(name="Beta", slug="beta", sort_order=10, is_active=True)

        factory = RequestFactory()
        list_view = CategoryViewSet.as_view({"get": "list"})

        default_response = list_view(factory.get("/api/v1/catalog/categories/"))
        self.assertEqual(default_response.status_code, 200)
        self.assertEqual([item["slug"] for item in default_response.data], ["alpha", "beta", "zulu"])

        by_name_desc_response = list_view(
            factory.get("/api/v1/catalog/categories/", {"ordering": "-name"})
        )
        self.assertEqual(by_name_desc_response.status_code, 200)
        self.assertEqual(
            [item["slug"] for item in by_name_desc_response.data],
            ["zulu", "beta", "alpha"],
        )

    def test_search_api_excludes_hidden_and_inactive_categories_from_suggestions(self):
        hidden_direct = Category.objects.create(
            name="Decor Hidden",
            slug="decor-hidden",
            is_visible=False,
        )
        inactive_direct = Category.objects.create(
            name="Decor Inactive",
            slug="decor-inactive",
            is_active=False,
        )
        hidden_parent = Category.objects.create(
            name="Hidden Parent",
            slug="hidden-parent",
            is_visible=False,
        )
        inactive_parent = Category.objects.create(
            name="Inactive Parent",
            slug="inactive-parent",
            is_active=False,
        )
        hidden_descendant = Category.objects.create(
            name="Decor Hidden Descendant",
            slug="decor-hidden-descendant",
            parent=hidden_parent,
        )
        inactive_descendant = Category.objects.create(
            name="Decor Inactive Descendant",
            slug="decor-inactive-descendant",
            parent=inactive_parent,
        )
        public_root = Category.objects.create(name="Public Root", slug="public-root")
        public_descendant = Category.objects.create(
            name="Decor Visible",
            slug="decor-visible",
            parent=public_root,
        )

        factory = RequestFactory()
        view = SearchAPIView.as_view()
        response = view(factory.get("/api/v1/catalog/search/", {"q": "Decor"}))

        self.assertEqual(response.status_code, 200)
        returned_ids = {item["id"] for item in response.data["categories"]}

        self.assertNotIn(str(hidden_direct.id), returned_ids)
        self.assertNotIn(str(inactive_direct.id), returned_ids)
        self.assertNotIn(str(hidden_descendant.id), returned_ids)
        self.assertNotIn(str(inactive_descendant.id), returned_ids)
        self.assertIn(str(public_descendant.id), returned_ids)


class CatalogDeepResearchProviderTests(SimpleTestCase):
    @override_settings(PRODUCT_AI_DEEP_RESEARCH_SEARCH_PROVIDER_ORDER="")
    def test_default_deep_research_provider_order_includes_google_cse(self):
        search_provider = SimpleNamespace(provider_order=[])
        ProductDeepResearchProvider(
            search_provider=search_provider,
            research_provider=SimpleNamespace(),
        )
        self.assertIn("google_cse", search_provider.provider_order)
        self.assertEqual(search_provider.provider_order[0], "searxng")

    @override_settings(PRODUCT_AI_DEEP_RESEARCH_MAX_SUBQUERIES=3)
    def test_product_deep_research_builds_focused_query_plan(self):
        provider = ProductDeepResearchProvider(
            search_provider=SimpleNamespace(provider_order=["duckduckgo"]),
            research_provider=SimpleNamespace(),
        )
        plan = provider._build_query_plan(
            query="pink embroidered kurti palazzo set product details",
            candidate_text="pink embroidered kurti palazzo set",
            ocr={"sku_candidates": ["KRT-2201"]},
            vision={"tokens": ["embroidered", "kurti"]},
            context_hints={"name": "Pink Embroidered Kurti Palazzo Set", "primary_category_name": "Fashion Apparel"},
        )
        self.assertTrue(plan)
        self.assertLessEqual(len(plan), 3)
        self.assertTrue(any("KRT-2201" in query for query in plan))

    @override_settings(PRODUCT_AI_DEEP_RESEARCH_MAX_SUBQUERIES=4)
    def test_query_plan_includes_image_name_code_anchor(self):
        provider = ProductDeepResearchProvider(
            search_provider=SimpleNamespace(provider_order=["duckduckgo"]),
            research_provider=SimpleNamespace(),
        )
        plan = provider._build_query_plan(
            query="women apparel product details",
            candidate_text="women apparel set",
            ocr={"sku_candidates": []},
            vision={"tokens": ["women", "apparel"]},
            context_hints={"image_names": ["BUN-KI-002.JPG"]},
        )
        self.assertTrue(any("bun-ki-002" in item.lower() for item in plan))

    @override_settings(PRODUCT_AI_DEEP_RESEARCH_MAX_SUBQUERIES=4)
    def test_query_plan_ignores_generic_image_filename_anchor(self):
        provider = ProductDeepResearchProvider(
            search_provider=SimpleNamespace(provider_order=["duckduckgo"]),
            research_provider=SimpleNamespace(),
        )
        plan = provider._build_query_plan(
            query="women apparel product details",
            candidate_text="women apparel set",
            ocr={"sku_candidates": []},
            vision={"tokens": ["women", "apparel"]},
            context_hints={"image_names": ["image(7).jpg"]},
        )
        joined = " ".join(plan).lower()
        self.assertNotIn('"image(7)"', joined)

    @override_settings(
        PRODUCT_AI_DEEP_RESEARCH_MAX_SUBQUERIES=2,
        PRODUCT_AI_DEEP_RESEARCH_MAX_RESULTS_PER_QUERY=4,
        PRODUCT_AI_DEEP_RESEARCH_MAX_SEARCH_RESULTS=8,
        PRODUCT_AI_DEEP_RESEARCH_MAX_DOCS=6,
        PRODUCT_AI_DEEP_RESEARCH_MAX_SOURCES=4,
        PRODUCT_AI_DEEP_RESEARCH_MIN_SCORE=0.2,
    )
    def test_product_deep_research_filters_help_pages(self):
        class FakeSearchProvider:
            def __init__(self):
                self.provider_order = ["duckduckgo"]

            def search(self, query, max_results=8):  # noqa: ARG002
                return (
                    [
                        {
                            "url": "https://seller.example.com/help/product-image-requirements",
                            "title": "Listings Lounge: Product Image Requirements",
                            "snippet": "Help center listing image requirements for sellers.",
                        },
                        {
                            "url": "https://shop.example.com/pink-embroidered-kurti-palazzo-set",
                            "title": "Pink Embroidered Kurti Palazzo Set",
                            "snippet": "Buy pink embroidered kurti palazzo set in cotton fabric.",
                        },
                    ],
                    "duckduckgo",
                )

        class FakeResearchProvider:
            def fetch_documents(self, search_results, max_docs=8):  # noqa: ARG002
                return [
                    ResearchDocument(
                        url="https://seller.example.com/help/product-image-requirements",
                        domain="seller.example.com",
                        title="Listings Lounge: Product Image Requirements",
                        snippet="Help center listing image requirements for sellers.",
                        text=(
                            "Product image requirements and listing policies for marketplace uploads. "
                            "Help center guidance for sellers and support workflows."
                        ),
                        trust_score=0.5,
                        metadata={"provider": "duckduckgo", "structured": {}},
                    ),
                    ResearchDocument(
                        url="https://shop.example.com/pink-embroidered-kurti-palazzo-set",
                        domain="shop.example.com",
                        title="Pink Embroidered Kurti Palazzo Set",
                        snippet="Buy pink embroidered kurti palazzo set in cotton fabric.",
                        text=(
                            "Pink embroidered kurti palazzo set with cotton fabric and floral motifs. "
                            "Available sizes S to XL. Price 2199 with in stock inventory."
                        ),
                        trust_score=0.62,
                        metadata={
                            "provider": "duckduckgo",
                            "structured": {
                                "names": ["Pink Embroidered Kurti Palazzo Set"],
                                "price_amounts": ["2199"],
                                "category_names": ["Fashion Apparel"],
                            },
                        },
                    ),
                ]

        provider = ProductDeepResearchProvider(
            search_provider=FakeSearchProvider(),
            research_provider=FakeResearchProvider(),
        )
        result = provider.run(
            query="pink embroidered kurti palazzo set product details",
            candidate_text="pink embroidered kurti palazzo set",
            ocr={"sku_candidates": []},
            vision={"tokens": []},
            context_hints={"primary_category_name": "Fashion Apparel"},
        )
        urls = [doc.url for doc in result["documents"]]
        self.assertIn("https://shop.example.com/pink-embroidered-kurti-palazzo-set", urls)
        self.assertNotIn("https://seller.example.com/help/product-image-requirements", urls)
        self.assertEqual(result["primary_provider"], "duckduckgo")

    def test_rank_documents_rejects_when_no_anchor_overlap(self):
        provider = ProductDeepResearchProvider(
            search_provider=SimpleNamespace(provider_order=["duckduckgo"]),
            research_provider=SimpleNamespace(),
        )
        docs = [
            ResearchDocument(
                url="https://example.com/products/book",
                domain="example.com",
                title="Bestselling Novel Product",
                snippet="Product page with price details",
                text="This product page is about a bestselling novel with hardcover binding.",
                trust_score=0.62,
                metadata={"structured": {}},
            )
        ]
        ranked, rejections = provider._rank_documents(
            docs,
            query_terms=["kurti", "embroidered", "palazzo"],
            reference_terms=[],
        )
        self.assertEqual(ranked, [])
        self.assertEqual(rejections.get("no_anchor_overlap", 0), 1)

    def test_rank_documents_relaxed_mode_keeps_product_doc_without_anchor_overlap(self):
        provider = ProductDeepResearchProvider(
            search_provider=SimpleNamespace(provider_order=["duckduckgo"]),
            research_provider=SimpleNamespace(),
        )
        docs = [
            ResearchDocument(
                url="https://example.com/products/book",
                domain="example.com",
                title="Bestselling Novel Product",
                snippet="Product page with price details",
                text="This product page is about a bestselling novel with hardcover binding.",
                trust_score=0.62,
                metadata={"structured": {}},
            )
        ]
        ranked, rejections = provider._rank_documents(
            docs,
            query_terms=["kurti", "embroidered", "palazzo"],
            reference_terms=[],
            enforce_anchor_overlap=False,
        )
        self.assertEqual(rejections.get("no_anchor_overlap", 0), 0)
        self.assertGreaterEqual(len(ranked), 1)

    def test_backfill_product_docs_adds_product_signal_documents(self):
        provider = ProductDeepResearchProvider(
            search_provider=SimpleNamespace(provider_order=["duckduckgo"]),
            research_provider=SimpleNamespace(),
        )
        docs = [
            ResearchDocument(
                url="https://shop.example.com/products/pink-kurti-set",
                domain="shop.example.com",
                title="Pink Kurti Set",
                snippet="Product price and size details",
                text="Pink kurti set with cotton fabric and in stock sizes.",
                trust_score=0.52,
                metadata={},
            ),
            ResearchDocument(
                url="https://example.com/help/upload-guidelines",
                domain="example.com",
                title="Upload guidelines",
                snippet="Help center article",
                text="Image requirements and listing policy details.",
                trust_score=0.70,
                metadata={},
            ),
        ]
        selected, added = provider._backfill_product_docs(
            selected_docs=[],
            all_docs=docs,
            max_docs=3,
        )
        self.assertEqual(added, 1)
        self.assertEqual(len(selected), 1)
        self.assertIn("pink-kurti-set", selected[0].url)


class SearchProviderHardeningTests(SimpleTestCase):
    @override_settings(PRODUCT_AI_SEARCH_PROVIDER_ORDER="duckduckgo")
    @patch("apps.catalog.ai.providers.search.requests.Session.get")
    def test_search_provider_skips_challenged_duckduckgo_response(self, mock_get):
        mock_get.return_value = SimpleNamespace(
            status_code=202,
            text="bots use DuckDuckGo too",
        )
        provider = SearchProvider()
        results, used_provider = provider.search("pink kurti", max_results=4)
        self.assertEqual(results, [])
        self.assertEqual(used_provider, "none")
        diagnostics = provider.get_last_diagnostics()
        attempts = diagnostics.get("attempts") or []
        self.assertTrue(any((attempt.get("status") == "blocked") for attempt in attempts))


class ResearchProviderHardeningTests(SimpleTestCase):
    @override_settings(PRODUCT_AI_RESEARCH_RESPECT_ROBOTS=False)
    @patch("apps.catalog.ai.providers.research._robots_allows", return_value=False)
    def test_fetch_documents_ignores_robots_when_disabled(self, mock_robots):
        provider = ResearchProvider()
        expected = ResearchDocument(
            url="https://example.com/products/pink-kurti-set",
            domain="example.com",
            title="Pink Kurti Set",
            snippet="Product page",
            text="Pink embroidered kurti set with cotton fabric and floral details.",
            trust_score=0.58,
            metadata={"provider": "bing_html", "content_type": "text/html", "structured": {}},
        )
        with patch.object(provider, "_fetch_one", return_value=(expected, "")) as mock_fetch:
            docs, diagnostics = provider.fetch_documents_with_diagnostics(
                [
                    {
                        "url": expected.url,
                        "title": expected.title,
                        "snippet": expected.snippet,
                        "provider": "bing_html",
                    }
                ],
                max_docs=4,
            )
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].url, expected.url)
        self.assertNotIn("robots_disallowed", diagnostics.get("rejection_reasons") or {})
        mock_fetch.assert_called_once()
        mock_robots.assert_not_called()

    @override_settings(PRODUCT_AI_RESEARCH_RESPECT_ROBOTS=True)
    @patch("apps.catalog.ai.providers.research._robots_allows", return_value=False)
    def test_fetch_documents_respects_robots_when_enabled(self, mock_robots):
        provider = ResearchProvider()
        with patch.object(provider, "_fetch_one") as mock_fetch:
            docs, diagnostics = provider.fetch_documents_with_diagnostics(
                [
                    {
                        "url": "https://example.com/products/pink-kurti-set",
                        "title": "Pink Kurti Set",
                        "snippet": "Product page",
                        "provider": "bing_html",
                    }
                ],
                max_docs=4,
            )
        self.assertEqual(docs, [])
        self.assertEqual((diagnostics.get("rejection_reasons") or {}).get("robots_disallowed"), 1)
        mock_fetch.assert_not_called()
        mock_robots.assert_called_once()

    @patch("apps.catalog.ai.providers.research._robots_allows", return_value=True)
    def test_fetch_one_accepts_short_text_when_product_signal_present(self, mock_robots):
        class FakeResponse:
            status_code = 200
            headers = {"Content-Type": "text/html; charset=utf-8"}
            text = "<html><head><title>Pink Kurti Set</title></head><body><main>Buy now</main></body></html>"
            url = "https://example.com/products/pink-kurti-set"

        provider = ResearchProvider()
        with patch.object(provider.session, "get", return_value=FakeResponse()), patch.object(
            provider, "_extract_main_text", return_value="Buy now"
        ), patch.object(
            provider, "_extract_structured_product_data", return_value={}
        ), patch.object(
            provider, "_trust_score", return_value=0.47
        ):
            doc, reason = provider._fetch_one(
                {
                    "url": "https://example.com/products/pink-kurti-set",
                    "title": "Pink Kurti Set",
                    "snippet": "Cotton kurti set product price and size details.",
                    "provider": "bing_html",
                }
            )
        self.assertEqual(reason, "")
        self.assertIsNotNone(doc)
        self.assertIn("kurti", (doc.text or "").lower())
        mock_robots.assert_not_called()

    @override_settings(PRODUCT_AI_RESEARCH_ALLOW_SNIPPET_FALLBACK=True)
    def test_fetch_one_uses_snippet_fallback_on_http_403(self):
        class FakeResponse:
            status_code = 403
            headers = {"Content-Type": "text/html; charset=utf-8"}
            text = "Access denied"
            url = "https://example.com/products/pink-kurti-set"

        provider = ResearchProvider()
        with patch.object(provider.session, "get", return_value=FakeResponse()):
            doc, reason = provider._fetch_one(
                {
                    "url": "https://example.com/products/pink-kurti-set",
                    "title": "Pink Kurti Set",
                    "snippet": "Product price and size details available now.",
                    "provider": "bing_html",
                }
            )
        self.assertEqual(reason, "")
        self.assertIsNotNone(doc)
        self.assertTrue((doc.metadata or {}).get("snippet_fallback"))
        self.assertEqual((doc.metadata or {}).get("snippet_fallback_reason"), "http_403")

    @override_settings(PRODUCT_AI_RESEARCH_ALLOW_SNIPPET_FALLBACK=False)
    def test_fetch_one_does_not_use_snippet_fallback_when_disabled(self):
        class FakeResponse:
            status_code = 403
            headers = {"Content-Type": "text/html; charset=utf-8"}
            text = "Access denied"
            url = "https://example.com/products/pink-kurti-set"

        provider = ResearchProvider()
        with patch.object(provider.session, "get", return_value=FakeResponse()):
            doc, reason = provider._fetch_one(
                {
                    "url": "https://example.com/products/pink-kurti-set",
                    "title": "Pink Kurti Set",
                    "snippet": "Product price and size details available now.",
                    "provider": "bing_html",
                }
            )
        self.assertIsNone(doc)
        self.assertEqual(reason, "http_403")


@override_settings(
    PRODUCT_AI_ENABLED=True,
    PRODUCT_AI_STRICT_EVIDENCE_MODE=True,
    PRODUCT_AI_MIN_WEB_SOURCES=3,
    PRODUCT_AI_MIN_HIGH_TRUST_DOCS=1,
)
class ProductAutofillEngineStrictGateTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            email="strictgate@example.com",
            password="pass",
            is_staff=True,
            is_superuser=True,
        )
        self.category = Category.objects.create(name="Strict Category", slug="strict-category")
        self.product = Product.objects.create(
            name="Strict Product",
            slug="strict-product",
            price=Decimal("20.00"),
            primary_category=self.category,
        )
        self.job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_PENDING,
            locale="en",
            currency="USD",
            allow_external=True,
        )
        self.engine = ProductAutofillEngine(job_id=str(self.job.id))

    def test_strict_mode_fails_when_provider_is_none_and_blocked(self):
        ok, error_code, _ = self.engine._evaluate_strict_research_gate(
            query="strict product",
            used_provider="none",
            search_results=[],
            research_docs=[],
            research_diagnostics={
                "query_diagnostics": [
                    {"provider_attempts": [{"status": "blocked", "reason": "captcha challenge"}]}
                ]
            },
            effective_min_sources=self.engine.min_web_sources,
        )
        self.assertFalse(ok)
        self.assertEqual(error_code, "SEARCH_BLOCKED_OR_CAPTCHA")

    def test_strict_mode_fails_when_validated_sources_below_minimum(self):
        docs = [
            SimpleNamespace(domain="shop-a.example.com", trust_score=0.9),
            SimpleNamespace(domain="shop-b.example.com", trust_score=0.85),
        ]
        ok, error_code, _ = self.engine._evaluate_strict_research_gate(
            query="strict product",
            used_provider="bing_html",
            search_results=[{"url": "https://shop-a.example.com/item"}],
            research_docs=docs,
            research_diagnostics={"duration_ms": 1200},
            effective_min_sources=self.engine.min_web_sources,
        )
        self.assertFalse(ok)
        self.assertEqual(error_code, "INSUFFICIENT_WEB_SOURCES")


class ProductAutofillNNVisionProviderTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        NNVisionProvider._runtime_cache.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = os.path.join(self.temp_dir.name, "sample.png")
        image = Image.new("RGB", (64, 64), color=(20, 100, 160))
        image.save(self.image_path, format="PNG")

    def tearDown(self):
        NNVisionProvider._runtime_cache.clear()
        self.temp_dir.cleanup()
        super().tearDown()

    def test_lazy_load_runtime_success_path(self):
        provider = NNVisionProvider(
            model_id="openai/clip-vit-base-patch32",
            device="cpu",
            timeout_seconds=8.0,
            cache_dir=self.temp_dir.name,
        )
        mocked_labels = [
            {"label": "dress", "score": 0.42, "tokens": ["dress", "fashion"]},
            {"label": "shirt", "score": 0.20, "tokens": ["shirt", "fashion"]},
        ]
        with patch.object(provider, "_load_runtime", return_value={"runtime": "ok"}) as mocked_load, patch.object(
            provider, "_predict_labels", return_value=mocked_labels
        ):
            first = provider.analyze([self.image_path])
            second = provider.analyze([self.image_path])

        self.assertEqual(mocked_load.call_count, 1)
        self.assertEqual(first["nn_inference_status"], "ok")
        self.assertEqual(second["nn_inference_status"], "ok")
        self.assertEqual(first["nn_labels"][0]["label"], "dress")
        self.assertGreater(first["nn_confidence"], 0.0)

    def test_load_failure_raises_nn_load_error(self):
        provider = NNVisionProvider(
            model_id="openai/clip-vit-base-patch32",
            device="cpu",
            timeout_seconds=8.0,
            cache_dir=self.temp_dir.name,
        )
        with patch.object(provider, "_load_runtime", side_effect=NNVisionLoadError("model unavailable")):
            with self.assertRaises(NNVisionLoadError):
                provider.analyze([self.image_path])


@override_settings(
    PRODUCT_AI_ENABLED=True,
    PRODUCT_AI_STRICT_EVIDENCE_MODE=True,
    PRODUCT_AI_MIN_WEB_SOURCES=3,
    PRODUCT_AI_MIN_WEB_SOURCES_WITH_NN=1,
    PRODUCT_AI_MIN_HIGH_TRUST_DOCS=1,
    PRODUCT_AI_NN_ENABLED=True,
    PRODUCT_AI_NN_CONFIDENCE_THRESHOLD=0.26,
)
class ProductAutofillEngineNNModeTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            email="nnmode@example.com",
            password="pass",
            is_staff=True,
            is_superuser=True,
        )
        self.category = Category.objects.create(name="NN Category", slug="nn-category")
        self.product = Product.objects.create(
            name="NN Product",
            slug="nn-product",
            price=Decimal("25.00"),
            primary_category=self.category,
        )
        self.job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_PENDING,
            locale="en",
            currency="USD",
            allow_external=True,
        )

    def test_effective_min_sources_uses_lower_threshold_when_nn_confident(self):
        engine = ProductAutofillEngine(job_id=str(self.job.id))
        nn_result = {"nn_inference_status": "ok", "nn_confidence": 0.42}
        effective_min = engine._effective_min_required_sources(nn_result)
        self.assertEqual(effective_min, 1)

        docs = [SimpleNamespace(domain="shop-a.example.com", trust_score=0.9)]
        ok, error_code, _ = engine._evaluate_strict_research_gate(
            query="nn product search",
            used_provider="bing_html",
            search_results=[{"url": "https://shop-a.example.com/item"}],
            research_docs=docs,
            research_diagnostics={"duration_ms": 800},
            effective_min_sources=effective_min,
        )
        self.assertTrue(ok)
        self.assertEqual(error_code, "")

    def test_nn_inference_failure_fails_job_with_explicit_error_code(self):
        engine = ProductAutofillEngine(job_id=str(self.job.id))
        with patch.object(engine, "_collect_image_paths", return_value=["/tmp/sample.png"]), patch.object(
            engine.nn_vision_provider,
            "analyze",
            side_effect=NNVisionInferenceError("inference failure"),
        ):
            result = engine.run()

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result.get("error_code"), "NN_INFERENCE_FAILED")
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, ProductAutofillJob.STATUS_FAILED)
        self.assertEqual(self.job.summary.get("error_code"), "NN_INFERENCE_FAILED")
        self.assertEqual(self.job.summary.get("nn_inference_status"), "failed")

    def test_nn_load_failure_fails_job_with_explicit_error_code(self):
        engine = ProductAutofillEngine(job_id=str(self.job.id))
        with patch.object(engine, "_collect_image_paths", return_value=["/tmp/sample.png"]), patch.object(
            engine.nn_vision_provider,
            "analyze",
            side_effect=NNVisionLoadError("model download failed"),
        ):
            result = engine.run()

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result.get("error_code"), "NN_MODEL_UNAVAILABLE")
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, ProductAutofillJob.STATUS_FAILED)
        self.assertEqual(self.job.summary.get("error_code"), "NN_MODEL_UNAVAILABLE")


@override_settings(
    PRODUCT_AI_ENABLED=True,
    PRODUCT_AI_STRICT_EVIDENCE_MODE=True,
    PRODUCT_AI_MIN_WEB_SOURCES=3,
    PRODUCT_AI_MIN_WEB_SOURCES_WITH_NN=1,
    PRODUCT_AI_NN_ENABLED=False,
)
class ProductAutofillEngineNNDisabledTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            email="nndisabled@example.com",
            password="pass",
            is_staff=True,
            is_superuser=True,
        )
        self.category = Category.objects.create(name="NN Off Category", slug="nn-off-category")
        self.product = Product.objects.create(
            name="NN Off Product",
            slug="nn-off-product",
            price=Decimal("18.00"),
            primary_category=self.category,
        )
        self.job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_PENDING,
            locale="en",
            currency="USD",
            allow_external=True,
        )

    def test_effective_min_sources_remains_default_when_nn_disabled(self):
        engine = ProductAutofillEngine(job_id=str(self.job.id))
        self.assertEqual(engine._effective_min_required_sources({"nn_inference_status": "ok", "nn_confidence": 0.95}), 3)

    def test_build_search_query_uses_relaxed_visual_tokens_when_hints_are_weak(self):
        engine = ProductAutofillEngine(job_id=str(self.job.id))
        query = engine._build_search_query(
            "Women's product-style brown beige women apparel",
            ocr={"sku_candidates": []},
            vision={"tokens": ["brown", "beige", "women", "apparel"]},
            context_hints={"image_names": ["image(7).jpg"]},
            similar_products=[],
        )
        self.assertTrue(query)
        self.assertIn("product details", query.lower())

    def test_build_search_query_uses_similar_product_category_fallback(self):
        engine = ProductAutofillEngine(job_id=str(self.job.id))
        query = engine._build_search_query(
            "",
            ocr={"sku_candidates": []},
            vision={"tokens": []},
            context_hints={"image_names": ["image(7).jpg"]},
            similar_products=[self.product],
        )
        self.assertTrue(query)
        self.assertIn("NN Off Category", query)


@override_settings(
    PRODUCT_AI_STRICT_EVIDENCE_MODE=False,
    PRODUCT_AI_ALLOW_PRICE_FALLBACK=True,
    PRODUCT_AI_ALLOW_HEURISTIC_PRICING=True,
    PRODUCT_AI_ALLOW_INVENTORY_DEFAULTS=True,
    PRODUCT_AI_ALLOW_SKU_FALLBACK=True,
)
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

    def test_category_uuid_without_context_is_not_treated_as_high_confidence(self):
        raw = {
            "primary_category": {
                "value": {"id": str(self.category.id), "name": self.category.name},
                "confidence": 0.4,
            },
            "categories": {
                "value": [{"id": str(self.category.id), "name": self.category.name}],
                "confidence": 0.4,
            },
            "price": {"value": "20.00", "confidence": 0.9},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8, context_hints={})
        mapped = {item.field_name: item for item in result}
        self.assertIsNone(mapped["primary_category"].value)
        self.assertEqual(mapped["categories"].value, [])

    def test_context_category_id_keeps_high_confidence_mapping(self):
        raw = {
            "primary_category": {
                "value": {"id": str(self.category.id), "name": self.category.name},
                "confidence": 0.4,
            },
            "categories": {
                "value": [{"id": str(self.category.id), "name": self.category.name}],
                "confidence": 0.4,
            },
            "price": {"value": "20.00", "confidence": 0.9},
        }
        result = normalize_raw_suggestions(
            raw,
            confidence_threshold=0.8,
            context_hints={
                "primary_category_id": str(self.category.id),
                "primary_category_name": self.category.name,
                "category_ids": [str(self.category.id)],
                "category_names": [self.category.name],
            },
        )
        mapped = {item.field_name: item for item in result}
        self.assertEqual(mapped["primary_category"].value, str(self.category.id))
        self.assertEqual(mapped["categories"].value, [str(self.category.id)])

    def test_internal_similarity_ignores_generic_visual_tokens(self):
        Product.objects.create(
            name="Generic Product Name",
            slug="generic-product-name",
            price=Decimal("15.00"),
            primary_category=self.category,
            description="A product with decorative style",
        )
        matches = get_internal_similar_products(
            product=None,
            candidate_text="gray product photo dominant color plain background",
            limit=5,
        )
        self.assertEqual(matches, [])

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

    def test_pricing_rounds_local_currency_to_clean_values(self):
        provider = PricingProvider()
        docs = [
            SimpleNamespace(
                url="https://shop.example.com/pink-kurti",
                text="Price ৳3999.90 sale price ৳3890.40",
                snippet="Buy now for ৳3999.90",
                metadata={"structured": {"price_amounts": ["3999.90", "3890.40"]}},
            )
        ]
        estimate = provider.estimate(
            product=None,
            primary_category=self.category,
            research_docs=docs,
            similar_products=[],
            currency="BDT",
        )
        price = Decimal(str(estimate["price"]["value"]))
        self.assertEqual(price % Decimal("10"), Decimal("0"))

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

    def test_extractors_ignore_non_product_help_pages(self):
        noisy_doc = SimpleNamespace(
            url="https://seller.example.com/help/product-image-requirements",
            title="Listings Lounge: Product Image Requirements",
            snippet="Learn image specs and listing policy updates.",
            text=(
                "Product image requirements and listing policies for marketplace uploads. "
                "Help center guidance for sellers and support workflows."
            ),
            metadata={"structured": {}},
        )

        suggestions = build_field_candidates(
            product=None,
            vision={"candidate_name": "", "aspect_ratio": "1:1"},
            ocr={"text": "", "lines": [], "sku_candidates": []},
            research_docs=[noisy_doc],
            internal_similar_products=[],
            personalization_hints={},
            context_hints={},
        )

        self.assertIsNone(suggestions["name"]["value"])
        self.assertIsNone(suggestions["description"]["value"])

    def test_validator_rejects_ui_navigation_text_dump(self):
        raw = {
            "name": {"value": "Emerald Bloom Embroidered Kurti Palazzo Set", "confidence": 0.92},
            "description": {
                "value": (
                    "Open media 1 in modal Open media 2 in modal Skip to product information "
                    "Emerald Bloom Embroidered Kurti Palazzo Set"
                ),
                "confidence": 0.9,
            },
            "short_description": {"value": "Open media 1 in modal", "confidence": 0.9},
            "price": {"value": "3990.87", "confidence": 0.5},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8)
        mapped = {item.field_name: item for item in result}
        self.assertIsNone(mapped["description"].value)
        self.assertIsNone(mapped["short_description"].value)
        self.assertEqual(mapped["name"].value, "Emerald Bloom Embroidered Kurti Palazzo Set")

    def test_extractors_use_vision_scene_summary_for_apparel_fallback(self):
        fashion_category = Category.objects.create(name="Fashion & Apparel", slug="fashion-apparel")
        suggestions = build_field_candidates(
            product=None,
            vision={
                "candidate_name": "Pink Women's Apparel Set",
                "scene_summary": "Model wearing a pink apparel outfit in a product-style photo.",
                "aspect_ratio": "1:1",
                "tokens": ["pink", "women", "apparel", "fashion", "outfit", "set"],
            },
            ocr={"text": "", "lines": [], "sku_candidates": []},
            research_docs=[],
            internal_similar_products=[],
            personalization_hints={},
            context_hints={},
        )
        self.assertEqual(suggestions["name"]["value"], "Pink Women's Apparel Set")
        self.assertGreaterEqual(suggestions["name"]["confidence"], 0.75)
        self.assertIsNotNone(suggestions["description"]["value"])
        self.assertGreaterEqual(suggestions["description"]["confidence"], 0.72)
        self.assertEqual(suggestions["primary_category"]["value"]["id"], str(fashion_category.id))
        self.assertGreaterEqual(suggestions["primary_category"]["confidence"], 0.70)

    def test_extractors_do_not_pick_internal_category_from_generic_visual_text(self):
        nursery = Category.objects.create(name="Nursery Decor", slug="nursery-decor")
        similar = Product.objects.create(
            name="Nursery Product Example",
            slug="nursery-product-example",
            price=Decimal("30.00"),
            primary_category=nursery,
            description="Home decor item",
        )
        suggestions = build_field_candidates(
            product=None,
            vision={
                "candidate_name": "Gray Product",
                "scene_summary": "Product photo with dominant gray color.",
                "aspect_ratio": "1:1",
                "tokens": ["gray", "product", "photo", "dominant", "color"],
                "apparel_item": False,
            },
            ocr={"text": "", "lines": [], "sku_candidates": []},
            research_docs=[],
            internal_similar_products=[similar],
            personalization_hints={},
            context_hints={},
        )
        self.assertIsNone(suggestions["primary_category"]["value"])

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

    def test_apply_suggestions_skips_unchanged_m2m_updates(self):
        product = Product.objects.create(
            name="Tagged",
            slug="tagged",
            price=Decimal("15.00"),
            primary_category=self.category,
        )
        product.tags.add(self.tag)
        suggestions = [
            SimpleNamespace(field_name="tags", value_json=[str(self.tag.id)]),
        ]
        result = apply_suggestions_to_product(
            product=product,
            suggestions=suggestions,
            force_overwrite=True,
        )
        self.assertEqual(result["applied"], 0)
        self.assertNotIn("tags", result["changed_fields"])

    @override_settings(
        PRODUCT_AI_STRICT_EVIDENCE_MODE=True,
        PRODUCT_AI_ALLOW_PRICE_FALLBACK=False,
        PRODUCT_AI_ALLOW_SKU_FALLBACK=True,
    )
    def test_strict_mode_disables_price_fallback_but_keeps_sku_fallback(self):
        raw = {
            "price": {"value": None, "confidence": 0.1},
            "sku": {"value": None, "confidence": 0.1},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8)
        mapped = {item.field_name: item for item in result}
        self.assertIsNone(mapped["price"].value)
        self.assertTrue(bool(mapped["sku"].value))
        self.assertTrue(mapped["sku"].metadata.get("strict_gate_passed"))

    @override_settings(
        PRODUCT_AI_STRICT_EVIDENCE_MODE=True,
        PRODUCT_AI_ALLOW_HEURISTIC_PRICING=False,
        PRODUCT_AI_ALLOW_INVENTORY_DEFAULTS=False,
    )
    def test_pricing_provider_strict_mode_avoids_inventory_defaults(self):
        provider = PricingProvider()
        estimate = provider.estimate(
            product=None,
            primary_category=self.category,
            research_docs=[],
            similar_products=[],
        )
        self.assertIsNone(estimate["price"]["value"])
        self.assertIsNone(estimate["cost"]["value"])
        self.assertIsNone(estimate["stock_quantity"]["value"])
        self.assertIsNone(estimate["low_stock_threshold"]["value"])

    @override_settings(PRODUCT_AI_STRICT_EVIDENCE_MODE=True)
    def test_strict_evidence_requires_web_or_context_for_name(self):
        raw = {
            "name": {"value": "Visual Guess Name", "confidence": 0.92, "source_urls": []},
            "price": {"value": "20.00", "confidence": 0.9, "source_urls": ["https://shop.example.com/item"]},
        }
        result = normalize_raw_suggestions(raw, confidence_threshold=0.8)
        mapped = {item.field_name: item for item in result}
        self.assertIsNone(mapped["name"].value)
        self.assertEqual(mapped["name"].metadata.get("evidence_kind"), "none")
        self.assertFalse(mapped["name"].metadata.get("strict_gate_passed"))

    @override_settings(PRODUCT_AI_STRICT_EVIDENCE_MODE=True)
    def test_strict_evidence_accepts_context_hint_for_name(self):
        raw = {
            "name": {"value": "Handmade Home Vase", "confidence": 0.9, "source_urls": []},
            "price": {"value": "20.00", "confidence": 0.9, "source_urls": ["https://shop.example.com/item"]},
        }
        result = normalize_raw_suggestions(
            raw,
            confidence_threshold=0.8,
            context_hints={"name": "Handmade Home Vase"},
        )
        mapped = {item.field_name: item for item in result}
        self.assertEqual(mapped["name"].value, "Handmade Home Vase")
        self.assertEqual(mapped["name"].metadata.get("evidence_kind"), "context_hint")
        self.assertTrue(mapped["name"].metadata.get("strict_gate_passed"))


@override_settings(
    PRODUCT_AI_ENABLED=True,
    PRODUCT_AI_MAX_IMAGES=4,
    PRODUCT_AI_FORCE_SYNC_ON_FILESYSTEM_STORAGE=False,
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

    def test_start_endpoint_requires_image_for_new_product(self):
        request = self.factory.post(
            "/admin/catalog/product/ai/autofill/start/",
            data={
                "currency": "USD",
                "locale": "en",
                "allow_external": "true",
            },
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_start_view(request)
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertIn("Upload at least one image", payload.get("error", ""))

    @patch("apps.catalog.admin.run_product_autofill_job.delay")
    def test_start_endpoint_rejects_unreadable_image(self, mock_delay):
        bad_upload = SimpleUploadedFile("broken.png", b"not-an-image", content_type="image/png")
        request = self.factory.post(
            "/admin/catalog/product/ai/autofill/start/",
            data={
                "product_id": str(self.product.id),
                "currency": "USD",
                "locale": "en",
                "allow_external": "true",
                "images": bad_upload,
            },
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_start_view(request)
        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertIn("readable image", payload.get("error", ""))
        mock_delay.assert_not_called()

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

    @override_settings(PRODUCT_AI_FORCE_SYNC_ON_FILESYSTEM_STORAGE=True)
    @patch("apps.catalog.ai.engine.ProductAutofillEngine")
    @patch("apps.catalog.admin.run_product_autofill_job.delay")
    def test_start_endpoint_forces_sync_on_filesystem_storage(self, mock_delay, mock_engine):
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
        self.assertEqual(payload["dispatch_mode"], "sync_fallback_local_storage")
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

    def test_status_endpoint_includes_strict_diagnostics_payload(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_FAILED,
            locale="en",
            currency="USD",
            error_message="Insufficient evidence.",
            summary={
                "error_code": "INSUFFICIENT_WEB_SOURCES",
                "strict_mode": True,
                "min_required_sources": 3,
                "validated_source_count": 1,
                "research_diagnostics": {
                    "fetch_success": 1,
                    "fetch_failed": 4,
                },
            },
        )
        request = self.factory.get(f"/admin/catalog/product/ai/autofill/{job.id}/status/")
        request.user = self.user
        response = self.product_admin.ai_autofill_status_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertEqual(payload["error_code"], "INSUFFICIENT_WEB_SOURCES")
        self.assertTrue(payload["strict_mode"])
        self.assertEqual(payload["min_required_sources"], 3)
        self.assertEqual(payload["validated_source_count"], 1)
        self.assertEqual(payload["research_diagnostics"]["fetch_success"], 1)

    def test_status_endpoint_exposes_soft_failed_strict_gate(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
            summary={
                "strict_mode": True,
                "min_required_sources": 3,
                "validated_source_count": 0,
                "strict_gate_passed": False,
                "strict_gate_failed": True,
                "strict_gate_enforced": False,
                "strict_gate_error_code": "INSUFFICIENT_WEB_SOURCES",
                "strict_gate_error_message": "Deep research returned 0 validated sources; minimum 3 required.",
            },
        )
        request = self.factory.get(f"/admin/catalog/product/ai/autofill/{job.id}/status/")
        request.user = self.user
        response = self.product_admin.ai_autofill_status_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertEqual(payload["status"], ProductAutofillJob.STATUS_COMPLETED)
        self.assertTrue(payload["strict_mode"])
        self.assertTrue(payload["strict_gate_failed"])
        self.assertFalse(payload["strict_gate_enforced"])
        self.assertFalse(payload["strict_gate_passed"])
        self.assertEqual(payload["strict_gate_error_code"], "INSUFFICIENT_WEB_SOURCES")

    def test_status_endpoint_includes_nn_fields_and_effective_threshold(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
            summary={
                "strict_mode": True,
                "configured_min_required_sources": 3,
                "effective_min_required_sources": 1,
                "min_required_sources": 1,
                "validated_source_count": 1,
                "nn_enabled": True,
                "nn_model_id": "openai/clip-vit-base-patch32",
                "nn_inference_status": "ok",
                "nn_confidence": 0.33,
            },
        )
        request = self.factory.get(f"/admin/catalog/product/ai/autofill/{job.id}/status/")
        request.user = self.user
        response = self.product_admin.ai_autofill_status_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertEqual(payload["configured_min_required_sources"], 3)
        self.assertEqual(payload["effective_min_required_sources"], 1)
        self.assertEqual(payload["min_required_sources"], 1)
        self.assertTrue(payload["nn_enabled"])
        self.assertEqual(payload["nn_model_id"], "openai/clip-vit-base-patch32")
        self.assertEqual(payload["nn_inference_status"], "ok")
        self.assertEqual(float(payload["nn_confidence"]), 0.33)

    def test_status_endpoint_includes_source_activity_for_favicon_rendering(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_RUNNING,
            locale="en",
            currency="USD",
            allow_external=True,
            summary={
                "source_activity": [
                    {
                        "url": "https://www.example.com/products/item-1",
                        "domain": "www.example.com",
                        "provider": "searxng",
                        "phase": "candidate",
                    },
                    {
                        "url": "https://shop.example.org/p/2",
                        "domain": "shop.example.org",
                        "provider": "duckduckgo",
                        "phase": "validated",
                    },
                ]
            },
        )
        request = self.factory.get(f"/admin/catalog/product/ai/autofill/{job.id}/status/")
        request.user = self.user
        response = self.product_admin.ai_autofill_status_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertTrue(payload["allow_external"])
        self.assertGreaterEqual(len(payload["source_activity"]), 2)
        first = payload["source_activity"][0]
        self.assertIn("domain", first)
        self.assertIn("url", first)
        self.assertIn("phase", first)
        self.assertEqual(first["domain"], "example.com")

    def test_status_endpoint_resolves_display_names_for_taxonomy_ids(self):
        tag = Tag.objects.create(name="Summer")
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
        )
        ProductFieldSuggestion.objects.create(
            job=job,
            field_name="primary_category",
            value_json=str(self.category.id),
            display_value=str(self.category.id),
            confidence=0.91,
        )
        ProductFieldSuggestion.objects.create(
            job=job,
            field_name="tags",
            value_json=[str(tag.id)],
            display_value=str(tag.id),
            confidence=0.88,
        )

        request = self.factory.get(f"/admin/catalog/product/ai/autofill/{job.id}/status/")
        request.user = self.user
        response = self.product_admin.ai_autofill_status_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        mapped = {item["field_name"]: item for item in payload["suggestions"]}
        self.assertEqual(mapped["primary_category"]["display_value"], self.category.name)
        self.assertEqual(mapped["tags"]["display_value"], "Summer")

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

    def test_apply_endpoint_force_overwrite_updates_existing_field(self):
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
        request = self.factory.post(
            f"/admin/catalog/product/ai/autofill/{job.id}/apply/",
            data={"force_overwrite": "true"},
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_apply_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertGreaterEqual(payload["result"]["applied"], 1)
        self.product.refresh_from_db()
        self.assertEqual(self.product.name, "Overwritten Name")

    def test_apply_endpoint_returns_conflict_when_lock_exists(self):
        job = ProductAutofillJob.objects.create(
            product=self.product,
            requested_by=self.user,
            status=ProductAutofillJob.STATUS_COMPLETED,
            locale="en",
            currency="USD",
        )
        ProductFieldSuggestion.objects.create(
            job=job,
            field_name="description",
            value_json="AI generated description",
            display_value="AI generated description",
            confidence=0.9,
        )
        lock_key = f"catalog:autofill:apply:{job.id}"
        cache.set(lock_key, "busy", timeout=30)
        request = self.factory.post(
            f"/admin/catalog/product/ai/autofill/{job.id}/apply/",
            data={"force_overwrite": "true"},
        )
        request.user = self.user
        response = self.product_admin.ai_autofill_apply_view(request, job_id=job.id)
        self.assertEqual(response.status_code, 409)
        payload = json.loads(response.content.decode("utf-8"))
        self.assertFalse(payload["ok"])

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


class ProductImportExportCompatibilityTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        Currency.objects.get_or_create(
            code="BDT",
            defaults={
                "name": "Bangladeshi Taka",
                "symbol": "Tk",
                "native_symbol": "Tk",
                "is_active": True,
                "is_default": True,
            },
        )

    def test_product_resource_exports_portable_relation_identifiers(self):
        from apps.catalog.admin import ProductResource

        root = Category.objects.create(name="Women", slug="women")
        child = Category.objects.create(name="Kurti", slug="kurti", parent=root)
        tag = Tag.objects.create(name="Festive")
        shipping = ShippingMaterial.objects.create(name="Recycled Mailer")
        eco = EcoCertification.objects.create(name="Global Organic Textile Standard", slug="gots")

        product = Product.objects.create(
            name="Portable Export Product",
            slug="portable-export-product",
            sku="PORTABLE-EXPORT-001",
            price=Decimal("1990.00"),
            currency_id="BDT",
            primary_category=child,
            shipping_material=shipping,
        )
        product.categories.set([child])
        product.tags.set([tag])
        product.eco_certifications.set([eco])

        dataset = ProductResource().export(Product.objects.filter(id=product.id))
        self.assertEqual(len(dataset), 1)
        row = dataset.dict[0]

        self.assertEqual(row["primary_category"], "women/kurti")
        self.assertEqual(row["categories"], "women/kurti")
        self.assertEqual(row["tags"], "Festive")
        self.assertEqual(row["shipping_material"], "Recycled Mailer")
        self.assertEqual(row["eco_certifications"], "gots")

    def test_product_resource_imports_legacy_fixture_shape_and_creates_related_records(self):
        try:
            import tablib
        except Exception:
            self.skipTest("tablib is required for django-import-export dataset tests")

        from apps.catalog.admin import ProductResource

        product_id = str(uuid.uuid4())
        legacy_payload = [
            {
                "model": "catalog.product",
                "pk": product_id,
                "fields": {
                    "name": "Legacy Import Product",
                    "slug": "legacy-import-product",
                    "sku": "LEGACY-IMPORT-001",
                    "price": "1490.00",
                    "currency": "BDT",
                    "primary_category": "women/legacy-kurti",
                    "categories": ["women/legacy-kurti"],
                    "tags": ["Legacy Tag"],
                    "shipping_material": "Legacy Mailer",
                    "eco_certifications": ["legacy-cert"],
                    "is_active": True,
                },
            }
        ]

        dataset = tablib.Dataset().load(json.dumps(legacy_payload), format="json")
        result = ProductResource().import_data(dataset, dry_run=False, raise_errors=True)
        self.assertFalse(result.has_errors())

        product = Product.objects.get(slug="legacy-import-product")
        self.assertEqual(product.sku, "LEGACY-IMPORT-001")
        self.assertEqual(product.primary_category.get_slug_path(include_self=True), "women/legacy-kurti")
        self.assertTrue(product.categories.filter(slug="legacy-kurti").exists())
        self.assertTrue(product.tags.filter(name="Legacy Tag").exists())
        self.assertEqual(product.shipping_material.name, "Legacy Mailer")
        self.assertTrue(product.eco_certifications.filter(slug="legacy-cert").exists())
