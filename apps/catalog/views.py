"""
Catalog views - Web/Template views for catalog
"""
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, TemplateView
from django.views import View
from django.db import models
from django.http import JsonResponse, Http404
from django.db.models import Q, F, Prefetch
from django.core.paginator import Paginator
from django.contrib import messages
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

from .models import (
    Category, Product, ProductImage, ProductVariant, Collection, Bundle,
    Review, Badge, Spotlight, Tag, Facet
)
from .services import (
    CategoryService, ProductService, CollectionService, ReviewService,
    BadgeService, SpotlightService, BundleService,
)


# =============================================================================
# Category Views
# =============================================================================

class CategoryListView(ListView):
    """Display all top-level categories."""
    model = Category
    template_name = 'catalog/category_list.html'
    context_object_name = 'categories'
    
    def get_queryset(self):
        return CategoryService.get_root_categories()
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['category_tree'] = CategoryService.get_category_tree()
        return context


class CategoryDetailView(DetailView):
    """Display category with its products."""
    model = Category
    template_name = 'catalog/category_detail.html'
    context_object_name = 'category'
    slug_url_kwarg = 'slug'
    
    def get_object(self, queryset=None):
        # Support nested slug paths (e.g., electronics/smartphones)
        slug_path = self.kwargs.get('slug', '')
        
        # Try direct slug first
        category = CategoryService.get_category_by_slug(slug_path)
        if category:
            return category
        
        # Try slug path
        category = CategoryService.get_category_by_path(slug_path)
        if category:
            return category
        
        raise Http404("Category not found")
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        category = self.object
        
        # Get filter parameters
        page = int(self.request.GET.get('page', 1))
        sort = self.request.GET.get('sort', '-created_at')
        min_price = self.request.GET.get('min_price')
        max_price = self.request.GET.get('max_price')
        in_stock = self.request.GET.get('in_stock')
        
        # Build attribute filters
        attributes = {}
        for key, value in self.request.GET.items():
            if key.startswith('attr_'):
                attributes[key[5:]] = value
        
        # Get products
        products_data = ProductService.get_product_list(
            categories=category,
            min_price=float(min_price) if min_price else None,
            max_price=float(max_price) if max_price else None,
            in_stock=in_stock == 'true' if in_stock else None,
            attributes=attributes if attributes else None,
            sort=sort,
            page=page,
            page_size=24
        )
        
        context.update({
            'products': products_data['products'],
            'total_products': products_data['total'],
            'page_obj': {
                'number': products_data['page'],
                'has_next': products_data['has_next'],
                'has_previous': products_data['has_previous'],
                'num_pages': products_data['total_pages'],
            },
            'breadcrumbs': CategoryService.get_breadcrumbs(category),
            'subcategories': category.children.filter(is_visible=True, is_deleted=False),
            'facets': CategoryService.get_category_facets(category, self.request.GET),
            'price_range': ProductService.get_price_range(category),
            'current_sort': sort,
        })
        
        return context


# =============================================================================
# Product Views
# =============================================================================

class ProductListView(ListView):
    """Display all products with filtering."""
    model = Product
    template_name = 'catalog/product_list.html'
    context_object_name = 'products'
    paginate_by = 24
    
    def get_queryset(self):
        # Get filter parameters
        search = self.request.GET.get('q')
        sort = self.request.GET.get('sort', '-created_at')
        min_price = self.request.GET.get('min_price')
        max_price = self.request.GET.get('max_price')
        in_stock = self.request.GET.get('in_stock')
        tags = self.request.GET.getlist('tag')
        
        result = ProductService.get_product_list(
            search=search,
            tags=tags if tags else None,
            min_price=float(min_price) if min_price else None,
            max_price=float(max_price) if max_price else None,
            in_stock=in_stock == 'true' if in_stock else None,
            sort=sort,
            page=1,
            page_size=1000  # Get all for pagination
        )
        
        return result['products']
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'search_query': self.request.GET.get('q', ''),
            'current_sort': self.request.GET.get('sort', '-created_at'),
            'price_range': ProductService.get_price_range(),
            'popular_tags': Tag.objects.annotate(
                product_count=models.Count('products')
            ).order_by('-product_count')[:20],
        })
        return context


class ProductDetailView(DetailView):
    """Display product details."""
    model = Product
    template_name = 'catalog/product_detail.html'
    context_object_name = 'product'
    slug_url_kwarg = 'slug'
    
    def get_object(self, queryset=None):
        slug = self.kwargs.get('slug')
        product = ProductService.get_product_by_slug(slug)
        if not product:
            raise Http404("Product not found")
        return product
    
    def get(self, request, *args, **kwargs):
        response = super().get(request, *args, **kwargs)
        
        # Record view
        ProductService.record_view(
            self.object,
            user=request.user if request.user.is_authenticated else None,
            session_key=request.session.session_key
        )
        
        return response
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        product = self.object
        
        # Get primary image and gallery
        images = product.images.all().order_by('ordering')
        primary_image = images.filter(is_primary=True).first() or images.first()
        
        # Get variants grouped by option
        variants = product.variants.prefetch_related('option_values__option').all()
        variant_options = {}
        for variant in variants:
            for ov in variant.option_values.all():
                if ov.option.name not in variant_options:
                    variant_options[ov.option.name] = []
                if ov.value not in [v['value'] for v in variant_options[ov.option.name]]:
                    variant_options[ov.option.name].append({
                        'value': ov.value,
                        'variant_id': str(variant.id),
                        'in_stock': variant.stock_quantity > 0,
                        'price': str(variant.current_price)
                    })
        
        # Get review summary
        review_summary = ReviewService.get_review_summary(product)
        
        # Get recent reviews
        reviews_data = ReviewService.get_product_reviews(product, page_size=5)
        
        context.update({
            'primary_image': primary_image,
            'gallery_images': images,
            'variants': variants,
            'variant_options': variant_options,
            'has_variants': variants.exists(),
            'related_products': ProductService.get_related_products(product),
            'badges': BadgeService.get_product_badges(product),
            'review_summary': review_summary,
            'recent_reviews': reviews_data['reviews'],
            'breadcrumbs': CategoryService.get_breadcrumbs(product.primary_category) if product.primary_category else [],
            '3d_assets': product.assets_3d.filter(is_primary=True).first(),
        })
        
        return context


class ProductQuickView(View):
    """AJAX endpoint for product quick view modal."""
    
    def get(self, request, slug):
        product = ProductService.get_product_by_slug(slug)
        if not product:
            return JsonResponse({'error': 'Product not found'}, status=404)
        
        primary_image = product.images.filter(is_primary=True).first() or product.images.first()
        
        data = {
            'id': str(product.id),
            'name': product.name,
            'slug': product.slug,
            'price': str(product.price),
            'sale_price': str(product.sale_price) if product.sale_price else None,
            'current_price': str(product.current_price),
            'short_description': product.short_description,
            'in_stock': product.is_in_stock(),
            'image': primary_image.image.url if primary_image else None,
            'url': f'/products/{product.slug}/',
        }
        
        return JsonResponse(data)


# =============================================================================
# Collection Views
# =============================================================================

class CollectionListView(ListView):
    """Display all collections."""
    model = Collection
    template_name = 'catalog/collection_list.html'
    context_object_name = 'collections'
    
    def get_queryset(self):
        return CollectionService.get_active_collections()


class CollectionDetailView(DetailView):
    """Display collection with its products."""
    model = Collection
    template_name = 'catalog/collection_detail.html'
    context_object_name = 'collection'
    slug_url_kwarg = 'slug'
    
    def get_object(self, queryset=None):
        slug = self.kwargs.get('slug')
        collection = CollectionService.get_collection_by_slug(slug)
        if not collection:
            raise Http404("Collection not found")
        return collection
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['products'] = CollectionService.get_collection_products(self.object)
        return context


# =============================================================================
# Bundle Views
# =============================================================================

class BundleListView(ListView):
    """Display all bundles."""
    model = Bundle
    template_name = 'catalog/bundle_list.html'
    context_object_name = 'bundles'
    
    def get_queryset(self):
        return BundleService.get_active_bundles()


class BundleDetailView(DetailView):
    """Display bundle details."""
    model = Bundle
    template_name = 'catalog/bundle_detail.html'
    context_object_name = 'bundle'
    slug_url_kwarg = 'slug'
    
    def get_object(self, queryset=None):
        slug = self.kwargs.get('slug')
        bundle = BundleService.get_bundle_by_slug(slug)
        if not bundle:
            raise Http404("Bundle not found")
        return bundle
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        bundle = self.object
        context.update({
            'items': bundle.bundle_items.select_related('product').prefetch_related('product__images'),
            'total_price': bundle.price(),
            'savings': BundleService.calculate_bundle_savings(bundle),
        })
        return context


# =============================================================================
# Search Views
# =============================================================================

class SearchView(TemplateView):
    """Product search page."""
    template_name = 'search/results.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        query = self.request.GET.get('q', '').strip()
        page = int(self.request.GET.get('page', 1))
        
        if query:
            products_data = ProductService.get_product_list(
                search=query,
                page=page,
                page_size=24
            )
            context.update({
                'query': query,
                'products': products_data['products'],
                'total': products_data['total'],
                'page_obj': {
                    'number': products_data['page'],
                    'has_next': products_data['has_next'],
                    'has_previous': products_data['has_previous'],
                    'num_pages': products_data['total_pages'],
                },
            })
        else:
            context['products'] = []
            context['query'] = ''
        
        return context


# =============================================================================
# Homepage/Featured Views  
# =============================================================================

class HomepageView(TemplateView):
    """Homepage with featured content."""
    template_name = 'catalog/homepage.html'
    
    @method_decorator(cache_page(60 * 5))  # Cache for 5 minutes
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        context.update({
            'featured_products': ProductService.get_featured_products(limit=8),
            'new_arrivals': ProductService.get_new_arrivals(limit=8),
            'bestsellers': ProductService.get_bestsellers(limit=8),
            'on_sale': ProductService.get_on_sale_products(limit=8),
            'featured_categories': CategoryService.get_featured_categories(limit=6),
            'spotlights': SpotlightService.get_active_spotlights(placement='home', limit=5),
            'collections': CollectionService.get_active_collections(limit=4),
        })
        
        return context


# =============================================================================
# Review Views
# =============================================================================

class ProductReviewsView(ListView):
    """Display all reviews for a product."""
    template_name = 'catalog/product_reviews.html'
    context_object_name = 'reviews'
    paginate_by = 10
    
    def get_queryset(self):
        self.product = get_object_or_404(Product, slug=self.kwargs['slug'], is_active=True, is_deleted=False)
        reviews_data = ReviewService.get_product_reviews(
            self.product,
            page=int(self.request.GET.get('page', 1)),
            page_size=self.paginate_by
        )
        return reviews_data['reviews']
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'product': self.product,
            'review_summary': ReviewService.get_review_summary(self.product),
        })
        return context


class CreateReviewView(View):
    """Create a new review."""
    
    def post(self, request, slug):
        if not request.user.is_authenticated:
            messages.error(request, 'You must be logged in to write a review.')
            return redirect('login')
        
        product = get_object_or_404(Product, slug=slug, is_active=True, is_deleted=False)
        
        try:
            review = ReviewService.create_review(
                product=product,
                user=request.user,
                rating=int(request.POST.get('rating', 5)),
                title=request.POST.get('title', ''),
                body=request.POST.get('body', ''),
            )
            messages.success(request, 'Your review has been submitted and is pending approval.')
        except ValueError as e:
            messages.error(request, str(e))
        
        return redirect('product-detail', slug=slug)


# =============================================================================
# Utility imports for templates
# =============================================================================
