"""
Sitemap configuration
"""
from django.contrib.sitemaps import Sitemap
from django.urls import reverse
from django.contrib.sites.shortcuts import get_current_site


class StaticViewSitemap(Sitemap):
    """Sitemap for static pages."""
    priority = 0.5
    changefreq = 'weekly'
    
    def items(self):
        return ['home', 'catalog:catalog_list']
    
    def location(self, item):
        return reverse(item)


class ProductSitemap(Sitemap):
    """Sitemap for products, including images for image sitemap support."""
    changefreq = 'daily'
    priority = 0.8
    limit = 50000  # Google sitemap limit
    
    def items(self):
        from apps.catalog.models import Product
        return Product.objects.filter(is_active=True, is_deleted=False)
    
    def lastmod(self, obj):
        return obj.updated_at
    
    def location(self, obj):
        return reverse('catalog:product_detail', kwargs={'slug': obj.slug})

    def images(self, obj):
        """Include images in the sitemap for better indexing"""
        imgs = []
        for img in obj.images.all()[:5]:
            if img.image:
                imgs.append({
                    'location': img.image.url,
                    'title': obj.name,
                    'caption': (obj.short_description[:250] if obj.short_description else ''),
                })
        return imgs


class CategorySitemap(Sitemap):
    """Sitemap for categories."""
    changefreq = 'weekly'
    priority = 0.7
    
    def items(self):
        from apps.catalog.models import Category
        return Category.objects.filter(is_visible=True, is_deleted=False)
    
    def lastmod(self, obj):
        return obj.updated_at
    
    def location(self, obj):
        return reverse('catalog:category_detail', kwargs={'slug': obj.slug})


class BlogSitemap(Sitemap):
    """Sitemap for blog posts."""
    changefreq = 'weekly'
    priority = 0.6
    
    def items(self):
        from apps.pages.models import Page
        return Page.objects.filter(status='published', is_deleted=False)
    
    def lastmod(self, obj):
        return obj.updated_at
    
    def location(self, obj):
        return reverse('pages:page_detail', kwargs={'slug': obj.slug})

