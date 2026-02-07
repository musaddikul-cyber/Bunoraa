from celery import shared_task
from django.utils import timezone
from datetime import timedelta
from .models import Subscription
from .services import SubscriptionService
from apps.payments.models import RecurringCharge
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def sync_subscriptions_with_stripe(self, limit=500):
    """Daily job to sync subscriptions with Stripe to ensure statuses and dates are accurate."""
    qs = Subscription.objects.exclude(stripe_subscription_id__isnull=True).exclude(stripe_subscription_id__exact="").order_by("-updated_at")[:limit]
    for s in qs:
        try:
            SubscriptionService.sync_with_stripe(s)
        except Exception as exc:
            logger.exception("Failed to sync subscription %s: %s", s.id, exc)


@shared_task(bind=True)
def create_upcoming_recurring_charges(self, horizon_days=1):
    """Create RecurringCharge entries for subscriptions with impending billing events.

    This helps the payments.run_recurring_billing task to pick up due charges.
    """
    horizon = timezone.now() + timedelta(days=horizon_days)
    subs = Subscription.objects.filter(next_billing_at__lte=horizon, status=Subscription.STATUS_ACTIVE, is_deleted=False)
    for s in subs:
        # Skip if a pending charge already exists for subscription and billing time
        exists = RecurringCharge.objects.filter(subscription=s, status="pending", attempt_at__lte=s.next_billing_at).exists()
        if exists:
            continue
        try:
            RecurringCharge.objects.create(
                subscription=s,
                amount=s.plan.price_amount,
                currency=s.plan.currency,
                status="pending",
                attempt_at=s.next_billing_at or timezone.now(),
                stripe_subscription_id=s.stripe_subscription_id,
                metadata={"auto_generated": True},
            )
        except Exception as exc:
            logger.exception("Failed to create recurring charge for subscription %s: %s", s.id, exc)