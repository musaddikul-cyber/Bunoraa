"""
Signal handlers for Analytics app.
Tracks user interactions and logs operational events.
"""
import logging
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model

logger = logging.getLogger('bunoraa.analytics')

User = get_user_model()


@receiver(post_save, sender='analytics.PageView')
def on_page_view_created(sender, instance, created, **kwargs):
    """Handle new page view creation."""
    if created:
        logger.debug(f"Page view recorded: {instance.path}")


@receiver(post_save, sender='analytics.ProductView')
def on_product_view_created(sender, instance, created, **kwargs):
    """Track product view events."""
    if created:
        # Product counters are updated in AnalyticsService.track_product_view;
        # this signal only logs to avoid duplicate increments.
        logger.debug("Product view recorded for product_id=%s", instance.product_id)


@receiver(post_save, sender='analytics.SearchQuery')
def on_search_query_created(sender, instance, created, **kwargs):
    """Track search queries for analytics."""
    if created:
        logger.debug("Search query recorded: %s", instance.query)


@receiver(post_save, sender='analytics.CartEvent')
def on_cart_event_created(sender, instance, created, **kwargs):
    """Track cart events for funnel analysis."""
    if created:
        logger.debug(f"Cart event: {instance.event_type}")
