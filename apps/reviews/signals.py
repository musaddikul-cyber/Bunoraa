"""
Reviews signals
"""
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.db.models import Avg

from .models import Review


@receiver(post_save, sender=Review)
def update_product_rating_on_save(sender, instance, **kwargs):
    """Update product average rating when review is saved."""
    if instance.status == Review.STATUS_APPROVED:
        update_product_rating(instance.product)


@receiver(post_delete, sender=Review)
def update_product_rating_on_delete(sender, instance, **kwargs):
    """Update product average rating when review is deleted."""
    update_product_rating(instance.product)


def update_product_rating(product):
    """Calculate and update product average rating."""
    from apps.catalog.models import Product
    
    avg_rating = Review.objects.filter(
        product=product,
        status=Review.STATUS_APPROVED,
        is_deleted=False
    ).aggregate(avg=Avg('rating'))['avg']
    
    review_count = Review.objects.filter(
        product=product,
        status=Review.STATUS_APPROVED,
        is_deleted=False
    ).count()
    
    Product.objects.filter(id=product.id).update(
        average_rating=avg_rating or 0,
        review_count=review_count
    )
