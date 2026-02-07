"""
Signal handlers for Analytics app.
Tracks user interactions and page views for analytics.
"""
import logging
from django.db.models.signals import post_save, post_delete
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
    """Track product view for recommendation engine."""
    if created:
        try:
            # Increment product view count
            product = instance.product
            if product:
                from apps.products.models import Product
                Product.objects.filter(pk=product.pk).update(
                    view_count=product.view_count + 1
                )
        except Exception as e:
            logger.warning(f"Failed to update product view count: {e}")


@receiver(post_save, sender='analytics.SearchQuery')
def on_search_query_created(sender, instance, created, **kwargs):
    """Track search queries for analytics and suggestions."""
    if created:
        logger.debug(f"Search query recorded: {instance.query}")
        
        # Update search trends if needed
        try:
            from .models import SearchTrend
            trend, _ = SearchTrend.objects.get_or_create(
                query=instance.query.lower().strip(),
                defaults={'count': 0}
            )
            trend.count += 1
            trend.save(update_fields=['count'])
        except Exception:
            pass  # SearchTrend may not exist


@receiver(post_save, sender='analytics.CartEvent')
def on_cart_event_created(sender, instance, created, **kwargs):
    """Track cart events for funnel analysis."""
    if created:
        logger.debug(f"Cart event: {instance.event_type}")
