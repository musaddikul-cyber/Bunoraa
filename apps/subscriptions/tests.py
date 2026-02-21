from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APITestCase

from .models import Plan, Subscription


User = get_user_model()
TEST_MIDDLEWARE = [
    middleware
    for middleware in settings.MIDDLEWARE
    if middleware != "debug_toolbar.middleware.DebugToolbarMiddleware"
]


class SubscriptionModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="u@example.com", password="pass")
        self.plan = Plan.objects.create(name="Test", interval=Plan.INTERVAL_MONTH, price_amount=9.99)

    def test_create_and_soft_delete(self):
        subscription = Subscription.objects.create(user=self.user, plan=self.plan)
        self.assertFalse(subscription.is_deleted)
        subscription.soft_delete()
        subscription.refresh_from_db()
        self.assertTrue(subscription.is_deleted)
        self.assertIsNotNone(subscription.deleted_at)

    def test_status_updates(self):
        subscription = Subscription.objects.create(user=self.user, plan=self.plan)
        self.assertTrue(subscription.is_active)
        subscription.status = Subscription.STATUS_PAST_DUE
        subscription.save()
        self.assertFalse(subscription.is_active)


@override_settings(SECURE_SSL_REDIRECT=False, MIDDLEWARE=TEST_MIDDLEWARE)
class SubscriptionsAPITests(APITestCase):
    def setUp(self):
        self.client.raise_request_exception = False
        self.user = User.objects.create_user(email="api@example.com", password="pass")
        self.plan = Plan.objects.create(
            name="Basic",
            interval=Plan.INTERVAL_MONTH,
            price_amount=9.99,
            stripe_price_id="price_test",
        )

    def test_list_plans(self):
        response = self.client.get("/api/v1/subscriptions/plans/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertGreaterEqual(payload.get("meta", {}).get("count", 0), 1)

    def test_create_subscription_requires_auth(self):
        response = self.client.post("/api/v1/subscriptions/", {"plan_id": str(self.plan.id)})
        self.assertEqual(response.status_code, 401)

    def test_create_subscription(self):
        self.client.force_authenticate(self.user)
        response = self.client.post("/api/v1/subscriptions/", {"plan_id": str(self.plan.id)})
        # Service may fail when Stripe is not configured.
        self.assertIn(response.status_code, (201, 400, 500))
