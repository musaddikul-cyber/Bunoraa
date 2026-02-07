from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Subscription
from .services import SubscriptionService
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Subscription)
def sync_subscription_on_save(sender, instance: Subscription, created, **kwargs):
    """Keep local subscription in sync with Stripe after changes.

    Runs synchronously (fast) — if this grows heavier switch to a Celery task.
    """
    try:
        # If we just created a subscription with stripe_subscription_id, ensure we sync fields
        if created or instance.stripe_subscription_id:
            SubscriptionService.sync_with_stripe(instance)
    except Exception as exc:
        logger.exception("Failed to sync subscription %s with Stripe: %s", instance, exc)