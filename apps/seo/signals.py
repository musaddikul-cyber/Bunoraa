"""
Signal handlers for SEO app.
Handles automatic SEO metadata generation and sitemap updates.
"""
import logging
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

logger = logging.getLogger('bunoraa.seo')


@receiver(post_save, sender='catalog.Product')
def on_product_saved_update_seo(sender, instance, created, **kwargs):
    """Auto-generate SEO metadata for products."""
    try:
        from django.contrib.contenttypes.models import ContentType
        from .models import SEOMetadata, SitemapEntry
        
        ct = ContentType.objects.get_for_model(instance)
        
        # Create or update SEO metadata
        seo, seo_created = SEOMetadata.objects.get_or_create(
            content_type=ct,
            object_id=instance.pk,
            defaults={
                'title': f"{instance.name} | Bunoraa",
                'description': instance.description[:160] if instance.description else '',
                'canonical_url': f'/products/{instance.slug}/',
            }
        )
        
        # Update sitemap entry
        if instance.is_active:
            SitemapEntry.objects.update_or_create(
                url=f'/products/{instance.slug}/',
                defaults={
                    'priority': 0.8,
                    'changefreq': 'weekly',
                    'is_active': True,
                }
            )
        else:
            SitemapEntry.objects.filter(url=f'/products/{instance.slug}/').update(is_active=False)
            
    except Exception as e:
        logger.warning(f"Failed to update SEO for product {instance.pk}: {e}")


@receiver(post_save, sender='catalog.Category')
def on_category_saved_update_seo(sender, instance, created, **kwargs):
    """Auto-generate SEO metadata for categories."""
    try:
        from django.contrib.contenttypes.models import ContentType
        from .models import SEOMetadata, SitemapEntry
        
        ct = ContentType.objects.get_for_model(instance)
        
        SEOMetadata.objects.get_or_create(
            content_type=ct,
            object_id=instance.pk,
            defaults={
                'title': f"{instance.name} | Bunoraa",
                'description': instance.description[:160] if instance.description else f"Shop {instance.name} products at Bunoraa",
                'canonical_url': f'/categories/{instance.slug}/',
            }
        )
        
        if instance.is_active:
            SitemapEntry.objects.update_or_create(
                url=f'/categories/{instance.slug}/',
                defaults={
                    'priority': 0.9,
                    'changefreq': 'daily',
                    'is_active': True,
                }
            )
            
    except Exception as e:
        logger.warning(f"Failed to update SEO for category {instance.pk}: {e}")


@receiver(post_save, sender='pages.Page')
def on_page_saved_update_seo(sender, instance, created, **kwargs):
    """Auto-generate SEO metadata for CMS pages."""
    try:
        from django.contrib.contenttypes.models import ContentType
        from .models import SEOMetadata, SitemapEntry
        
        ct = ContentType.objects.get_for_model(instance)
        
        SEOMetadata.objects.get_or_create(
            content_type=ct,
            object_id=instance.pk,
            defaults={
                'title': f"{instance.title} | Bunoraa",
                'description': instance.meta_description or '',
                'canonical_url': f'/{instance.slug}/',
            }
        )
        
        if instance.is_published:
            SitemapEntry.objects.update_or_create(
                url=f'/{instance.slug}/',
                defaults={
                    'priority': 0.7,
                    'changefreq': 'monthly',
                    'is_active': True,
                }
            )
            
    except Exception as e:
        logger.warning(f"Failed to update SEO for page {instance.pk}: {e}")


@receiver(post_delete)
def on_object_deleted_cleanup_seo(sender, instance, **kwargs):
    """Clean up SEO metadata when objects are deleted."""
    # Only handle our main content models
    model_name = sender.__name__.lower()
    if model_name not in ['product', 'category', 'page']:
        return
    
    try:
        from django.contrib.contenttypes.models import ContentType
        from .models import SEOMetadata, SitemapEntry
        
        ct = ContentType.objects.get_for_model(instance)
        SEOMetadata.objects.filter(content_type=ct, object_id=instance.pk).delete()
        
        # Remove sitemap entry
        slug = getattr(instance, 'slug', None)
        if slug:
            url_patterns = [
                f'/products/{slug}/',
                f'/categories/{slug}/',
                f'/{slug}/',
            ]
            SitemapEntry.objects.filter(url__in=url_patterns).delete()
            
    except Exception as e:
        logger.warning(f"Failed to cleanup SEO for {sender.__name__} {instance.pk}: {e}")
