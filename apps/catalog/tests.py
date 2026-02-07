from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from .models import Category, Product, ProductVariant, ProductImage, Attribute, AttributeValue, Tag, Facet, CategoryFacet, ShippingMaterial, Badge, ProductBadge
from django.contrib.contenttypes.models import ContentType
from decimal import Decimal
from django.utils import timezone
from django.contrib.auth import get_user_model


class CatalogModelsTests(TestCase):
    def setUp(self):
        # categories
        self.root = Category.objects.create(name="Root Category")
        self.child = Category.objects.create(name="Child Category", parent=self.root)
        self.grand = Category.objects.create(name="Grandchild", parent=self.child)

        # attributes
        self.color = Attribute.objects.create(name="Color", slug="color")
        self.red = AttributeValue.objects.create(attribute=self.color, value="Red")

        # product
        self.prod = Product.objects.create(name="Handmade Bowl", price=Decimal("50.00"), sale_price=Decimal("45.00"))
        self.prod.categories.add(self.child)
        self.prod.tags.add(Tag.objects.create(name="wood"))
        self.prod.attributes.add(self.red)

        self.variant = ProductVariant.objects.create(product=self.prod, sku="HB-RED-1", price=Decimal("48.00"), stock_quantity=5)

    def test_category_paths(self):
        self.assertEqual(self.root.depth, 0)
        self.assertEqual(self.child.depth, 1)
        self.assertIn(self.child, list(self.root.get_descendants()))
        self.assertIn(self.root, list(self.child.get_ancestors()))

    def test_product_pricing_helpers(self):
        self.assertTrue(self.prod.is_on_sale)
        self.assertEqual(self.prod.current_price, Decimal("45.00"))
        self.assertGreater(self.prod.discount_percentage, 0)
        self.assertEqual(self.variant.current_price, Decimal("48.00"))

    def test_product_soft_delete(self):
        self.prod.soft_delete()
        qs = Product.objects.filter(id=self.prod.id)
        self.assertFalse(qs.exists())
        # hard query should still find it
        from catalog.managers import SoftDeleteQuerySet
        all_qs = Product.objects.all_with_deleted()
        self.assertTrue(all_qs.dead().filter(id=self.prod.id).exists())


class CatalogAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        # categories
        self.root = Category.objects.create(name="Root Category")
        self.child = Category.objects.create(name="Child Category", parent=self.root)

        # products
        self.prod1 = Product.objects.create(name="Alpha", price=Decimal("10.00"))
        self.prod1.categories.add(self.root)
        self.prod2 = Product.objects.create(name="Beta", price=Decimal("20.00"))
        self.prod2.categories.add(self.child)

        self.facet = Facet.objects.create(name="Color", slug="color", type="choice", values=["Red","Blue"])
        CategoryFacet.objects.create(category=self.root, facet=self.facet)

    def test_category_list(self):
        url = reverse("category-list")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(len(resp.json()) >= 1)

    def test_products_by_category_includes_children(self):
        url = reverse("product-by-category")
        resp = self.client.get(url, {"category": str(self.root.id)})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Should include Beta (in child)
        self.assertTrue(any(p["name"] == "Beta" for p in data.get("results", data)))

    def test_facets_endpoint(self):
        url = reverse("facet-list")
        resp = self.client.get(url, {"category": str(self.root.id)})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(isinstance(data, list))

    def test_category_children_endpoint(self):
        url = reverse("category-children", kwargs={"pk": str(self.root.id)})
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(isinstance(data, list))


class Phase1FeatureTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.root = Category.objects.create(name="Root")
        self.child = Category.objects.create(name="Child", parent=self.root)
        self.prod = Product.objects.create(name="Quick", price="12.00")
        self.prod.categories.add(self.child)

    def test_aspect_ratio_inheritance_and_quick_view(self):
        # Set category aspect ratio
        self.root.aspect_ratio = "16:9"
        self.root.save()
        # Product default should be 1:1 but quick view should show product's own if set
        url = reverse("product-quick_view", kwargs={"pk": str(self.prod.id)})
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("aspect_ratio", data)

        # set product aspect ratio override
        self.prod.aspect_ratio = "4:3"
        self.prod.save()
        resp = self.client.get(url)
        data = resp.json()
        self.assertEqual(data.get("aspect_ratio"), "4:3")

    def test_shipping_material_and_quick_view(self):
        sm = ShippingMaterial.objects.create(name="Recycled Paper", eco_score=8)
        self.prod.shipping_material = sm
        self.prod.save()
        url = reverse("product-quick_view", kwargs={"pk": str(self.prod.id)})
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("shipping", {}).get("name"), "Recycled Paper")

    def test_badge_scheduling_tasks(self):
        from catalog.tasks import apply_scheduled_badges, remove_expired_badges
        import datetime
        now = timezone.now()
        # badge that applies to the child category
        badge = Badge.objects.create(name="On Sale", slug="on-sale", start=now - datetime.timedelta(hours=1), end=now + datetime.timedelta(hours=1), is_active=True, target_content_type=ContentType.objects.get_for_model(Category), target_object_id=str(self.child.id))
        # run scheduler task
        apply_scheduled_badges()
        self.assertTrue(ProductBadge.objects.filter(product=self.prod, badge=badge).exists())
        # expire badge
        badge.is_active = False
        badge.save()
        remove_expired_badges()
        self.assertFalse(ProductBadge.objects.filter(product=self.prod, badge=badge).exists())

    def test_cost_field_and_visibility(self):
        # model persistence
        p = Product.objects.create(name="Costed", price=Decimal("10.00"), cost=Decimal("6.00"))
        self.assertEqual(p.cost, Decimal("6.00"))

        # anonymous quick_view should not include cost
        url = reverse("product-quick_view", kwargs={"pk": str(p.id)})
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json().get("cost"))

        # staff user should see cost on product detail
        User = get_user_model()
        staff = User.objects.create_user("staff", "staff@example.com", "pass")
        staff.is_staff = True
        staff.save()
        self.client.force_authenticate(staff)
        detail_url = reverse("product-detail", kwargs={"pk": str(p.id)})
        resp = self.client.get(detail_url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Decimal(str(resp.json().get("cost"))), Decimal("6.00"))

    def test_product_price_filter(self):
        url = reverse("product-list")
        resp = self.client.get(url, {"price_max": "15"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # should include only Alpha (10)
        names = [p["name"] for p in data.get("results", data)]
        self.assertIn("Alpha", names)
        self.assertNotIn("Beta", names)

    def test_slug_uniqueness_and_primary_category_behavior(self):
        # product slug uniqueness case-insensitive
        p1 = Product.objects.create(name="Case", slug="UniqueSlug", price="1.00")
        with self.assertRaises(Exception):
            Product.objects.create(name="Case2", slug="uniqueslug", price="2.00")

        # primary category auto set on M2M and category counts updated
        p = Product.objects.create(name="ProdCat", price="4.00")
        cat1 = Category.objects.create(name="C1")
        cat2 = Category.objects.create(name="C2", parent=cat1)
        p.categories.add(cat2)
        p.refresh_from_db()
        self.assertIsNotNone(p.primary_category)
        self.assertEqual(p.primary_category, cat2)
        cat2.refresh_from_db()
        self.assertEqual(cat2.product_count, 1)

        # removing product from category decrements count
        p.categories.remove(cat2)
        cat2.refresh_from_db()
        self.assertEqual(cat2.product_count, 0)

    def test_counters_increment_and_soft_delete_affect_counts(self):
        p = Product.objects.create(name="Metrics", price="10.00")
        self.assertEqual(p.views_count, 0)
        # increment views
        p.increment_views()
        self.assertEqual(p.views_count, 1)
        # quick view returns views_count
        url = reverse("product-quick_view", kwargs={"pk": str(p.id)})
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("views_count"), 1)
        # increment sales
        p.increment_sales(2)
        self.assertEqual(p.sales_count, 2)

        # category product count decremented on soft_delete
        c = Category.objects.create(name="SaleCat")
        p.categories.add(c)
        c.refresh_from_db()
        self.assertEqual(c.product_count, 1)
        p.soft_delete()
        c.refresh_from_db()
        self.assertEqual(c.product_count, 0)

    def test_attribute_filter_param(self):
        # Create attribute and assign to a product
        color = Attribute.objects.create(name="Color", slug="color")
        av = AttributeValue.objects.create(attribute=color, value="Red")
        p = Product.objects.create(name="Red Item", price="5.00")
        p.attributes.add(av)
        url = reverse("product-list")
        resp = self.client.get(url, {"attr_color": "Red"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        names = [p2["name"] for p2 in data.get("results", data)]
        self.assertIn("Red Item", names)
