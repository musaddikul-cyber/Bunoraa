from __future__ import annotations

from decimal import Decimal

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.test import TestCase
from apps.catalog.models import Category, Product
from apps.i18n.models import Currency
from apps.reviews.admin import ReviewAdmin
from apps.reviews.models import Review, ReviewReport
from apps.reviews.services import ReviewService


class ReviewsDomainIntegrationTests(TestCase):
    def setUp(self):
        self.user_model = get_user_model()
        self.author = self.user_model.objects.create_user(email="author@example.com", password="pass12345")
        self.voter = self.user_model.objects.create_user(email="voter@example.com", password="pass12345")
        self.staff = self.user_model.objects.create_user(
            email="staff@example.com",
            password="pass12345",
            is_staff=True,
            is_superuser=True,
        )

        Currency.objects.get_or_create(
            code="BDT",
            defaults={
                "name": "Bangladeshi Taka",
                "symbol": "৳",
                "is_default": True,
                "is_base_currency": True,
            },
        )
        category = Category.objects.create(name="Testing", slug="testing")
        self.product = Product.objects.create(
            name="Test Product",
            slug="test-product",
            sku="TEST-SKU-1",
            price=Decimal("100.00"),
            currency_id="BDT",
        )
        self.product.categories.add(category)

    def test_reviews_admin_registers_reviews_review_model(self):
        self.assertIn(Review, admin.site._registry)
        review_admin = admin.site._registry[Review]
        self.assertIsInstance(review_admin, ReviewAdmin)
        self.assertIn("moderation_status", review_admin.list_editable)
        self.assertIn("is_featured", review_admin.list_editable)

    def test_review_service_end_to_end_create_moderate_vote_report(self):
        review, message = ReviewService.create_review(
            product=self.product,
            user=self.author,
            rating=5,
            title="Excellent",
            body="Works exactly as expected.",
            would_recommend=True,
        )
        self.assertIsNotNone(review)
        self.assertIn("pending", message.lower())
        self.assertEqual(review.moderation_status, Review.MODERATION_PENDING)

        ReviewService.moderate_review(
            review=review,
            moderator=self.staff,
            approve=True,
            notes="Looks good",
        )
        review.refresh_from_db()
        self.assertEqual(review.moderation_status, Review.MODERATION_APPROVED)
        self.assertIsNotNone(review.published_at)

        success, _ = ReviewService.vote_review(review=review, user=self.voter, is_helpful=True)
        self.assertTrue(success)
        review.refresh_from_db()
        self.assertEqual(review.helpful_votes, 1)

        success, _ = ReviewService.report_review(
            review=review,
            reporter=self.voter,
            reason=ReviewReport.REASON_SPAM,
            details="Looks suspicious.",
        )
        self.assertTrue(success)
        review.refresh_from_db()
        self.assertEqual(review.report_count, 1)

    def test_product_reviews_query_uses_reviews_domain(self):
        review = Review.objects.create(
            product=self.product,
            user=self.author,
            rating=4,
            title="Good",
            body="Nice quality.",
            moderation_status=Review.MODERATION_APPROVED,
        )
        reviews = ReviewService.get_product_reviews(product=self.product)
        self.assertTrue(reviews.filter(id=review.id).exists())
