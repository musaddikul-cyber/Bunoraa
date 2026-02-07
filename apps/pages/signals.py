"""
Signal handlers for Pages app.
Handles CMS page events.
"""
import logging
from django.db.models.signals import post_save, post_delete, pre_save
from django.dispatch import receiver
from django.utils.text import slugify

logger = logging.getLogger('bunoraa.pages')


@receiver(pre_save, sender='pages.Page')
def on_page_pre_save(sender, instance, **kwargs):
    """Handle page pre-save operations."""
    # Auto-generate slug
    if not instance.slug and instance.title:
        instance.slug = slugify(instance.title)
        
        # Ensure unique slug
        base_slug = instance.slug
        counter = 1
        while sender.objects.filter(slug=instance.slug).exclude(pk=instance.pk).exists():
            instance.slug = f"{base_slug}-{counter}"
            counter += 1


@receiver(post_save, sender='pages.Page')
def on_page_saved(sender, instance, created, **kwargs):
    """Handle page creation/update."""
    if created:
        logger.info(f"Page created: {instance.title}")
    else:
        logger.debug(f"Page updated: {instance.title}")
    
    # Clear page cache
    try:
        from django.core.cache import cache
        cache.delete(f'page_{instance.slug}')
        cache.delete('all_pages')
    except Exception:
        pass


@receiver(post_delete, sender='pages.Page')
def on_page_deleted(sender, instance, **kwargs):
    """Handle page deletion."""
    logger.info(f"Page deleted: {instance.title}")
    
    try:
        from django.core.cache import cache
        cache.delete(f'page_{instance.slug}')
        cache.delete('all_pages')
    except Exception:
        pass


@receiver(post_save, sender='pages.SiteSettings')
def on_site_settings_saved(sender, instance, created, **kwargs):
    """Handle site settings update."""
    logger.info("Site settings updated")
    
    # Clear all site settings cache
    try:
        from django.core.cache import cache
        cache.delete('site_settings')
        cache.delete('theme_settings')
    except Exception:
        pass


@receiver(post_save, sender='pages.Banner')
def on_banner_saved(sender, instance, created, **kwargs):
    """Handle banner creation/update."""
    # Clear banner cache for this position
    try:
        from django.core.cache import cache
        cache.delete(f'banners_{instance.position}')
        cache.delete('all_banners')
    except Exception:
        pass
