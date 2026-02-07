from django.test import TestCase
from django.contrib.auth import get_user_model
from ..models import Plan, Subscription
from django.utils import timezone

User = get_user_model()


class SubscriptionModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="u@example.com", password="pass")
        self.plan = Plan.objects.create(name="Test", interval=Plan.INTERVAL_MONTH, price_amount=9.99)

    def test_create_and_soft_delete(self):
        s = Subscription.objects.create(user=self.user, plan=self.plan)
        self.assertFalse(s.is_deleted)
        s.soft_delete()
        s.refresh_from_db()
        self.assertTrue(s.is_deleted)
        self.assertIsNotNone(s.deleted_at)

    def test_status_updates(self):
        s = Subscription.objects.create(user=self.user, plan=self.plan)
        self.assertTrue(s.is_active)
        s.status = Subscription.STATUS_PAST_DUE
        s.save()
        self.assertFalse(s.is_active)

