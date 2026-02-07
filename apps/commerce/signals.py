"""
Commerce signals - Signal handlers for cart, checkout, and wishlist
"""
import logging
from django.db.models.signals import post_save, pre_save, post_delete
from django.dispatch import receiver, Signal
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Signals
# =============================================================================

# Cart signals
cart_updated = Signal()  # Sender: Cart instance, kwargs: action, item
cart_item_added = Signal()  # Sender: CartItem instance
cart_item_removed = Signal()  # Sender: product, kwargs: cart, variant
cart_cleared = Signal()  # Sender: Cart instance

# Wishlist signals
wishlist_item_added = Signal()  # Sender: WishlistItem instance
wishlist_item_removed = Signal()  # Sender: product, kwargs: wishlist
wishlist_shared = Signal()  # Sender: WishlistShare instance

# Checkout signals
checkout_started = Signal()  # Sender: CheckoutSession instance
checkout_step_completed = Signal()  # Sender: CheckoutSession instance, kwargs: step
checkout_completed = Signal()  # Sender: CheckoutSession instance, kwargs: order
checkout_abandoned = Signal()  # Sender: CheckoutSession instance


# =============================================================================
# Cart Signal Handlers
# =============================================================================

@receiver(post_save, sender='commerce.CartItem')
def on_cart_item_saved(sender, instance, created, **kwargs):
    """Handle cart item save."""
    if created:
        logger.info(f"Cart item added: {instance.product.name} to cart {instance.cart.id}")
        cart_item_added.send(sender=instance.__class__, instance=instance)
        cart_updated.send(sender=instance.cart, action='item_added', item=instance)
    else:
        logger.debug(f"Cart item updated: {instance.product.name}")
        cart_updated.send(sender=instance.cart, action='item_updated', item=instance)


@receiver(post_delete, sender='commerce.CartItem')
def on_cart_item_deleted(sender, instance, **kwargs):
    """Handle cart item deletion."""
    logger.info(f"Cart item removed: {instance.product.name} from cart {instance.cart.id}")
    cart_item_removed.send(
        sender=instance.product.__class__,
        instance=instance.product,
        cart=instance.cart,
        variant=instance.variant
    )
    cart_updated.send(sender=instance.cart, action='item_removed', item=instance)


# =============================================================================
# Wishlist Signal Handlers
# =============================================================================

@receiver(post_save, sender='commerce.WishlistItem')
def on_wishlist_item_saved(sender, instance, created, **kwargs):
    """Handle wishlist item save."""
    if created:
        logger.info(f"Wishlist item added: {instance.product.name} to wishlist {instance.wishlist.id}")
        wishlist_item_added.send(sender=instance.__class__, instance=instance)


@receiver(post_delete, sender='commerce.WishlistItem')
def on_wishlist_item_deleted(sender, instance, **kwargs):
    """Handle wishlist item deletion."""
    logger.info(f"Wishlist item removed: {instance.product.name}")
    wishlist_item_removed.send(
        sender=instance.product.__class__,
        instance=instance.product,
        wishlist=instance.wishlist
    )


@receiver(post_save, sender='commerce.WishlistShare')
def on_wishlist_share_created(sender, instance, created, **kwargs):
    """Handle wishlist share creation."""
    if created:
        logger.info(f"Wishlist share created: {instance.share_token[:8]}...")
        wishlist_shared.send(sender=instance.__class__, instance=instance)


# =============================================================================
# Checkout Signal Handlers
# =============================================================================

@receiver(post_save, sender='commerce.CheckoutSession')
def on_checkout_session_saved(sender, instance, created, **kwargs):
    """Handle checkout session save."""
    from .models import CheckoutSession
    
    if created:
        logger.info(f"Checkout session started: {instance.id}")
        checkout_started.send(sender=instance.__class__, instance=instance)
    else:
        # Check for step changes
        if hasattr(instance, '_original_step') and instance._original_step != instance.current_step:
            checkout_step_completed.send(
                sender=instance.__class__,
                instance=instance,
                step=instance.current_step
            )
            
            # Check for completion
            if instance.current_step == CheckoutSession.STEP_COMPLETED:
                logger.info(f"Checkout completed: {instance.id}")
                checkout_completed.send(sender=instance.__class__, instance=instance)
            
            # Check for abandonment
            elif instance.current_step == CheckoutSession.STEP_ABANDONED:
                logger.info(f"Checkout abandoned: {instance.id}")
                checkout_abandoned.send(sender=instance.__class__, instance=instance)


@receiver(pre_save, sender='commerce.CheckoutSession')
def on_checkout_session_pre_save(sender, instance, **kwargs):
    """Store original step for comparison."""
    if instance.pk:
        try:
            original = sender.objects.get(pk=instance.pk)
            instance._original_step = original.current_step
        except sender.DoesNotExist:
            instance._original_step = None
    else:
        instance._original_step = None


# =============================================================================
# Signal Receivers for External Events
# =============================================================================

@receiver(checkout_abandoned)
def handle_abandoned_checkout(sender, instance, **kwargs):
    """Handle abandoned checkout."""
    logger.info(f"Processing abandoned checkout: {instance.id}")
    
    # Schedule recovery email task
    try:
        from .tasks import send_abandoned_cart_email
        send_abandoned_cart_email.apply_async(
            args=[str(instance.id)],
            countdown=60 * 60  # 1 hour delay
        )
    except Exception as e:
        logger.error(f"Failed to schedule abandoned cart email: {e}")
