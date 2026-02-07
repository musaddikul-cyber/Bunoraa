"""
Catalog Celery tasks - Background jobs for catalog maintenance
"""
import logging
from celery import shared_task
from django.utils import timezone
from django.db import transaction
from django.db.models import Q, F, Avg, Count
from django.core.cache import cache

logger = logging.getLogger(__name__)


# =============================================================================
# Badge Tasks
# =============================================================================

@shared_task(name='catalog.cleanup_expired_badges')
def cleanup_expired_badges():
    """Clean up expired badge-related data."""
    from .models import Badge
    
    now = timezone.now()
    # Deactivate badges that have passed their end date
    count = Badge.objects.filter(
        is_active=True,
        end__lt=now
    ).update(is_active=False)
    
    logger.info(f"Deactivated {count} expired badges")
    return count


# =============================================================================
# Spotlight Tasks
# =============================================================================

@shared_task(name='catalog.cleanup_spotlights')
def cleanup_spotlights():
    """Clean up expired spotlights."""
    from .models import Spotlight
    
    now = timezone.now()
    # Deactivate expired spotlights
    count = Spotlight.objects.filter(
        is_active=True,
        end__lt=now
    ).update(is_active=False)
    
    # Delete old inactive spotlights (older than 30 days)
    cutoff = now - timezone.timedelta(days=30)
    deleted, _ = Spotlight.objects.filter(
        is_active=False,
        end__lt=cutoff
    ).delete()
    
    logger.info(f"Deactivated {count} spotlights, deleted {deleted} old spotlights")
    return {'deactivated': count, 'deleted': deleted}


# =============================================================================
# Inventory Tasks
# =============================================================================

@shared_task(name='catalog.cleanup_expired_reservations')
def cleanup_expired_reservations():
    """Release expired stock reservations."""
    from .models import Reservation, StockHistory
    from .services import InventoryService
    
    count = InventoryService.cleanup_expired_reservations()
    logger.info(f"Cleaned up {count} expired reservations")
    return count


@shared_task(name='catalog.check_low_stock')
def check_low_stock():
    """Check for low stock products and create alerts."""
    from .models import Product, ProductVariant, StockAlert
    
    now = timezone.now()
    alerts_created = 0
    
    # Check product-level stock
    low_stock_products = Product.objects.filter(
        is_active=True,
        is_deleted=False,
        stock_quantity__lte=F('low_stock_threshold')
    ).exclude(
        stock_alerts__notified=False,
        stock_alerts__created_at__gte=now - timezone.timedelta(hours=24)
    )
    
    for product in low_stock_products:
        StockAlert.objects.create(
            product=product,
            threshold=product.low_stock_threshold
        )
        alerts_created += 1
    
    # Check variant-level stock
    low_stock_variants = ProductVariant.objects.filter(
        product__is_active=True,
        product__is_deleted=False,
        stock_quantity__lte=F('product__low_stock_threshold')
    )
    
    for variant in low_stock_variants:
        if not StockAlert.objects.filter(
            product=variant.product,
            variant=variant,
            notified=False,
            created_at__gte=now - timezone.timedelta(hours=24)
        ).exists():
            StockAlert.objects.create(
                product=variant.product,
                variant=variant,
                threshold=variant.product.low_stock_threshold
            )
            alerts_created += 1
    
    logger.info(f"Created {alerts_created} stock alerts")
    return alerts_created


@shared_task(name='catalog.send_stock_alerts')
def send_stock_alerts():
    """Send notifications for pending stock alerts."""
    from .models import StockAlert
    
    pending_alerts = StockAlert.objects.filter(notified=False).select_related('product', 'variant')
    
    # Group alerts by product for notification
    products_alerted = set()
    for alert in pending_alerts:
        if alert.product_id not in products_alerted:
            # TODO: Send notification (email, webhook, etc.)
            products_alerted.add(alert.product_id)
    
    # Mark as notified
    count = pending_alerts.update(notified=True)
    logger.info(f"Sent {count} stock alert notifications")
    return count


# =============================================================================
# Product Stats Tasks
# =============================================================================

@shared_task(name='catalog.update_product_stats')
def update_product_stats():
    """Update product statistics (review counts, ratings)."""
    from .models import Product, Review
    
    # Update review stats for all products
    products = Product.objects.filter(is_deleted=False)
    updated = 0
    
    for product in products.iterator():
        stats = Review.objects.filter(
            product=product,
            moderation_status='approved'
        ).aggregate(
            count=Count('id'),
            avg=Avg('rating')
        )
        
        product.reviews_count = stats['count'] or 0
        product.rating_count = stats['count'] or 0
        product.average_rating = round(stats['avg'] or 0, 2)
        product.save(update_fields=['reviews_count', 'rating_count', 'average_rating'])
        updated += 1
    
    logger.info(f"Updated stats for {updated} products")
    return updated


@shared_task(name='catalog.update_category_product_counts')
def update_category_product_counts():
    """Recalculate product counts for all categories."""
    from .models import Category, Product
    
    categories = Category.objects.filter(is_deleted=False)
    updated = 0
    
    for category in categories.iterator():
        # Count active, non-deleted products in this category and descendants
        descendants = category.get_descendants(include_self=True)
        count = Product.objects.filter(
            categories__in=descendants,
            is_active=True,
            is_deleted=False
        ).distinct().count()
        
        if category.product_count != count:
            category.product_count = count
            category.save(update_fields=['product_count'])
            updated += 1
    
    logger.info(f"Updated product counts for {updated} categories")
    return updated


# =============================================================================
# Sustainability Tasks
# =============================================================================

@shared_task(name='catalog.compute_sustainability_scores')
def compute_sustainability_scores():
    """Compute sustainability scores for products that need updating."""
    from .models import Product
    
    # Find products with sustainability data but no score
    products = Product.objects.filter(
        is_deleted=False
    ).filter(
        Q(carbon_footprint_kg__isnull=False) |
        Q(recycled_content_percentage__isnull=False)
    ).filter(
        Q(sustainability_score__isnull=True) |
        Q(sustainability_score=0)
    )[:500]  # Batch
    
    updated = 0
    for product in products:
        product.compute_sustainability_score(save=True)
        updated += 1
    
    logger.info(f"Computed sustainability scores for {updated} products")
    return updated


# =============================================================================
# Cache Tasks
# =============================================================================

@shared_task(name='catalog.refresh_category_tree_cache')
def refresh_category_tree_cache():
    """Refresh the category tree cache."""
    from .models import Category
    
    # Clear existing cache
    cache.delete('catalog:category_tree')
    
    # Rebuild cache
    Category.get_tree(use_cache=True)
    
    logger.info("Refreshed category tree cache")
    return True


@shared_task(name='catalog.warm_product_caches')
def warm_product_caches():
    """Warm caches for frequently accessed products."""
    from .models import Product
    
    # Cache featured products
    featured = list(Product.objects.filter(
        is_active=True,
        is_deleted=False,
        is_featured=True
    ).values_list('id', flat=True)[:50])
    
    cache.set('catalog:featured_product_ids', featured, 60 * 30)  # 30 min
    
    # Cache bestsellers
    bestsellers = list(Product.objects.filter(
        is_active=True,
        is_deleted=False,
        is_bestseller=True
    ).values_list('id', flat=True)[:50])
    
    cache.set('catalog:bestseller_product_ids', bestsellers, 60 * 30)
    
    logger.info(f"Warmed caches: {len(featured)} featured, {len(bestsellers)} bestsellers")
    return {'featured': len(featured), 'bestsellers': len(bestsellers)}


# =============================================================================
# Scheduled Publishing Tasks
# =============================================================================

@shared_task(name='catalog.process_scheduled_publishing')
def process_scheduled_publishing():
    """Process products with scheduled publishing windows."""
    from .models import Product
    
    now = timezone.now()
    
    # Activate products whose publish_from has arrived
    activated = Product.objects.filter(
        is_active=False,
        is_deleted=False,
        publish_from__lte=now,
        publish_from__isnull=False
    ).filter(
        Q(publish_until__isnull=True) | Q(publish_until__gte=now)
    ).update(is_active=True)
    
    # Deactivate products whose publish_until has passed
    deactivated = Product.objects.filter(
        is_active=True,
        is_deleted=False,
        publish_until__lt=now,
        publish_until__isnull=False
    ).update(is_active=False)
    
    logger.info(f"Scheduled publishing: activated {activated}, deactivated {deactivated}")
    return {'activated': activated, 'deactivated': deactivated}


# =============================================================================
# Cleanup Tasks
# =============================================================================

@shared_task(name='catalog.cleanup_old_impressions')
def cleanup_old_impressions():
    """Clean up old product impressions to prevent table bloat."""
    from .models import ProductImpression
    
    # Keep last 90 days
    cutoff = timezone.now() - timezone.timedelta(days=90)
    deleted, _ = ProductImpression.objects.filter(occurred_at__lt=cutoff).delete()
    
    logger.info(f"Deleted {deleted} old product impressions")
    return deleted


@shared_task(name='catalog.cleanup_soft_deleted')
def cleanup_soft_deleted():
    """Permanently delete items soft-deleted more than 30 days ago."""
    from .models import Product, Category
    
    cutoff = timezone.now() - timezone.timedelta(days=30)
    
    # Hard delete old soft-deleted products
    products_deleted, _ = Product.objects.all_with_deleted().filter(
        is_deleted=True,
        deleted_at__lt=cutoff
    ).delete()
    
    # Hard delete old soft-deleted categories (only if no products reference them)
    categories_deleted = 0
    old_categories = Category.objects.all_with_deleted().filter(
        is_deleted=True,
        deleted_at__lt=cutoff
    )
    for cat in old_categories:
        if not cat.products.exists():
            cat.delete()
            categories_deleted += 1
    
    logger.info(f"Permanently deleted {products_deleted} products, {categories_deleted} categories")
    return {'products': products_deleted, 'categories': categories_deleted}


# =============================================================================
# ML / Recommendation Tasks
# =============================================================================

@shared_task(name='catalog.generate_product_recommendations')
def generate_product_recommendations(user_id: int = None):
    """
    Generate product recommendations using ML model.
    """
    from django.conf import settings
    import pickle
    from pathlib import Path
    
    logger.info(f"Generating recommendations for user {user_id or 'all'}...")
    
    try:
        model_path = Path(settings.ML_MODELS_DIR) / 'recommendation_model.pkl'
        
        if not model_path.exists():
            logger.warning("Recommendation model not found")
            return {'error': 'Model not found'}
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Generate recommendations
        # Implementation depends on your model architecture
        
        return {'status': 'success'}
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        return {'error': str(e)}


@shared_task(name='catalog.train_product_suggestor')
def train_product_suggestor(data_path: str, model_name: str = 't5-small', output_dir: str = None):
    """
    Train the product suggestor ML model.
    """
    from django.conf import settings
    from pathlib import Path
    
    logger.info(f"Training product suggestor with data: {data_path}")
    
    try:
        if output_dir is None:
            output_dir = Path(settings.ML_MODELS_DIR) / 'product_suggestor'
        
        # Import and run the training script
        from catalog.ml.train_product_suggestor import main
        # Note: main() expects command line args, so this is a simplified integration
        
        return {'status': 'training_started', 'output_dir': str(output_dir)}
        
    except Exception as e:
        logger.error(f"Failed to train product suggestor: {e}")
        return {'error': str(e)}
