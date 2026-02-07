"""
Signal handlers for Promotions app.
Handles promotional events and coupon tracking.
"""
import logging
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.utils import timezone

logger = logging.getLogger('bunoraa.promotions')


@receiver(pre_save, sender='promotions.Coupon')
def on_coupon_pre_save(sender, instance, **kwargs):
    """Validate coupon before save."""
    # Auto-generate code if not provided
    if not instance.code:
        import random
        import string
        instance.code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        
        # Ensure unique
        while sender.objects.filter(code=instance.code).exists():
            instance.code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


@receiver(post_save, sender='promotions.Coupon')
def on_coupon_saved(sender, instance, created, **kwargs):
    """Handle coupon creation/update."""
    if created:
        logger.info(f"Coupon created: {instance.code}")
    
    # Clear coupon cache
    try:
        from django.core.cache import cache
        cache.delete(f'coupon_{instance.code}')
        cache.delete('active_coupons')
    except Exception:
        pass


@receiver(post_save, sender='promotions.CouponUsage')
def on_coupon_usage_saved(sender, instance, created, **kwargs):
    """Track coupon usage."""
    if created:
        logger.info(f"Coupon used: {instance.coupon.code} by user {instance.user_id}")
        
        # Update coupon usage count
        try:
            coupon = instance.coupon
            coupon.times_used = (coupon.times_used or 0) + 1
            coupon.save(update_fields=['times_used'])
            
            # Check if max uses reached
            if coupon.max_uses and coupon.times_used >= coupon.max_uses:
                logger.info(f"Coupon {coupon.code} reached maximum uses")
        except Exception:
            pass


@receiver(post_save, sender='promotions.FlashSale')
def on_flash_sale_saved(sender, instance, created, **kwargs):
    """Handle flash sale creation."""
    if created:
        logger.info(f"Flash sale created: {instance.name}")
    
    # Schedule notifications if upcoming
    if instance.start_date > timezone.now():
        try:
            from apps.notifications.tasks import schedule_flash_sale_notification
            schedule_flash_sale_notification.delay(instance.id)
        except Exception:
            pass
