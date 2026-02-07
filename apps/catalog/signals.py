import logging
from django.db import transaction
from django.db.models import F, Value, Q, Avg, Count
from django.db.models.signals import pre_save, post_save, post_delete, m2m_changed
from django.dispatch import receiver

from .models import (
    Product,
    ProductVariant,
    Review,
    Category,
    CategoryFacet,
    StockAlert,
    _update_category_product_counts,
    _clear_category_facets_cache,
)

logger = logging.getLogger(__name__)


@receiver(pre_save, sender=Product)
def product_pre_save(sender, instance, **kwargs):
    # capture previous state for change detection
    if instance.pk:
        prev = Product.objects.filter(pk=instance.pk).values("is_deleted", "is_active").first()
        instance._prev_is_deleted = prev.get("is_deleted") if prev else False
        instance._prev_is_active = prev.get("is_active") if prev else False
    else:
        instance._prev_is_deleted = False
        instance._prev_is_active = False


@receiver(post_save, sender=Product)
def product_post_save(sender, instance, created, **kwargs):
    try:
        category_ids = list(instance.categories.values_list("id", flat=True)) if instance.pk else []
        if created:
            if instance.is_active and not instance.is_deleted and category_ids:
                _update_category_product_counts(category_ids, 1)
            return

        if getattr(instance, "_prev_is_deleted", False) and not instance.is_deleted:
            if instance.is_active and category_ids:
                _update_category_product_counts(category_ids, 1)

        if not getattr(instance, "_prev_is_active", False) and instance.is_active:
            if not instance.is_deleted and category_ids:
                _update_category_product_counts(category_ids, 1)

        if not getattr(instance, "_prev_is_deleted", False) and instance.is_deleted:
            if category_ids:
                _update_category_product_counts(category_ids, -1)
    except Exception as e:
        logger.exception("Error in product_post_save signal: %s", e)


@receiver(m2m_changed, sender=Product.categories.through)
def product_categories_changed(sender, instance, action, pk_set, **kwargs):
    try:
        if action == "pre_clear":
            instance._pre_clear_category_ids = set(instance.categories.values_list("id", flat=True))
            return
        if action == "post_clear":
            old = getattr(instance, "_pre_clear_category_ids", set())
            if old and instance.is_active and not instance.is_deleted:
                _update_category_product_counts(old, -1)
            instance._pre_clear_category_ids = set()
            return

        if not pk_set:
            return
        if action == "post_add":
            if instance.is_active and not instance.is_deleted:
                _update_category_product_counts(pk_set, 1)
        elif action == "post_remove":
            if instance.is_active and not instance.is_deleted:
                _update_category_product_counts(pk_set, -1)
    except Exception as e:
        logger.exception("Error in product_categories_changed signal: %s", e)


@receiver(post_save, sender=Product)
def product_stock_alert_check(sender, instance, **kwargs):
    try:
        if instance.stock_quantity <= instance.low_stock_threshold:
            exists = StockAlert.objects.filter(product=instance, variant__isnull=True, threshold=instance.low_stock_threshold, notified=False).exists()
            if not exists:
                StockAlert.objects.create(product=instance, threshold=instance.low_stock_threshold)
    except Exception as e:
        logger.exception("Error in product_stock_alert_check: %s", e)


@receiver(post_save, sender=ProductVariant)
def variant_stock_alert_check(sender, instance, **kwargs):
    try:
        product = instance.product
        if instance.stock_quantity <= product.low_stock_threshold:
            exists = StockAlert.objects.filter(product=product, variant=instance, threshold=product.low_stock_threshold, notified=False).exists()
            if not exists:
                StockAlert.objects.create(product=product, variant=instance, threshold=product.low_stock_threshold)
    except Exception as e:
        logger.exception("Error in variant_stock_alert_check: %s", e)


def _recalc_product_ratings(product_id):
    agg = Review.objects.filter(product_id=product_id).aggregate(
        approved_count=Count("id", filter=Q(moderation_status="approved")),
        avg_rating=Avg("rating", filter=Q(moderation_status="approved")),
        total_reviews=Count("id"),
    )
    approved = agg.get("approved_count") or 0
    avg = float(agg.get("avg_rating") or 0.0)
    total = agg.get("total_reviews") or 0
    Product.objects.filter(pk=product_id).update(rating_count=approved, average_rating=avg, reviews_count=total)


@receiver(post_save, sender=Review)
def review_saved(sender, instance, **kwargs):
    try:
        _recalc_product_ratings(instance.product_id)
    except Exception as e:
        logger.exception("Error in review_saved signal: %s", e)


@receiver(post_delete, sender=Review)
def review_deleted(sender, instance, **kwargs):
    try:
        _recalc_product_ratings(instance.product_id)
    except Exception as e:
        logger.exception("Error in review_deleted signal: %s", e)


@receiver(post_save, sender=CategoryFacet)
@receiver(post_delete, sender=CategoryFacet)
def categoryfacet_changed(sender, instance, **kwargs):
    try:
        _clear_category_facets_cache(instance.category_id)
    except Exception as e:
        logger.exception("Error clearing category facet cache: %s", e)


@receiver(m2m_changed, sender=Product.categories.through)
def ensure_primary_category(sender, instance, action, pk_set, **kwargs):
    """When categories are added to a product, ensure primary_category is set to one of them if not already set."""
    if action == "post_add":
        if not instance.primary_category and pk_set:
            # pick the first category by depth
            cat = Category.objects.filter(id__in=pk_set).order_by("depth").first()
            if cat:
                instance.primary_category = cat
                instance.save(update_fields=["primary_category"])