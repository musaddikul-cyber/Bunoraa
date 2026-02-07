"""
Catalog services - Comprehensive business logic layer for catalog operations
Consolidates functionality from categories and products apps
"""
import logging
from collections import Counter
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple

from django.db import transaction, models
from django.db.models import Q, F, Avg, Count, Sum, Prefetch
from django.core.paginator import Paginator
from django.core.cache import cache
from django.utils import timezone
from django.utils.text import slugify

from .models import (
    Category, Product, ProductImage, ProductVariant, Tag, Attribute, AttributeValue,
    Collection, CollectionItem, Review, Badge, Spotlight, Bundle, BundleItem,
    Facet, CategoryFacet, StockHistory, Reservation, StockAlert,
    DigitalAsset, ProductPrice, Currency, EcoCertification
)

logger = logging.getLogger(__name__)


# =============================================================================
# Category Services
# =============================================================================

class CategoryService:
    """Service class for category operations."""
    
    TREE_CACHE_KEY = "catalog:category_tree"
    TREE_CACHE_TTL = 60 * 60  # 1 hour
    
    @classmethod
    def get_category_tree(cls, parent_id=None, max_depth=None, use_cache=True) -> List[Dict]:
        """
        Get category tree structure.
        If parent_id is provided, returns subtree starting from that category.
        """
        cache_key = f"{cls.TREE_CACHE_KEY}:{parent_id or 'root'}:{max_depth or 'all'}"
        
        if use_cache:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        queryset = Category.objects.filter(
            is_visible=True,
            is_deleted=False,
            parent_id=parent_id
        ).order_by('path')
        
        def build_tree(categories, current_depth=0):
            result = []
            for category in categories:
                node = {
                    'id': str(category.id),
                    'name': category.name,
                    'slug': category.slug,
                    'depth': category.depth,
                    'path': category.path,
                    'image': category.image.url if category.image else None,
                    'icon': category.icon,
                    'product_count': category.product_count,
                    'children': []
                }
                
                if max_depth is None or current_depth < max_depth:
                    children = category.children.filter(
                        is_visible=True,
                        is_deleted=False
                    ).order_by('path')
                    node['children'] = build_tree(children, current_depth + 1)
                
                result.append(node)
            return result
        
        tree = build_tree(queryset)
        
        if use_cache:
            cache.set(cache_key, tree, cls.TREE_CACHE_TTL)
        
        return tree
    
    @classmethod
    def clear_tree_cache(cls):
        """Clear all category tree caches."""
        cache.delete_pattern(f"{cls.TREE_CACHE_KEY}:*")
    
    @classmethod
    def get_root_categories(cls):
        """Get all root (top-level) categories."""
        return Category.objects.filter(
            parent__isnull=True,
            is_visible=True,
            is_deleted=False
        ).order_by('path')
    
    @classmethod
    def get_featured_categories(cls, limit=6):
        """Get categories with most products for homepage."""
        return Category.objects.filter(
            is_visible=True,
            is_deleted=False
        ).order_by('-product_count')[:limit]
    
    @classmethod
    def get_category_by_slug(cls, slug: str) -> Optional[Category]:
        """Get category by slug with related data."""
        try:
            return Category.objects.get(
                slug=slug,
                is_visible=True,
                is_deleted=False
            )
        except Category.DoesNotExist:
            return None
    
    @classmethod
    def get_category_by_path(cls, path: str) -> Optional[Category]:
        """Get category by its slug path (e.g., 'electronics/smartphones')."""
        slugs = path.strip('/').split('/')
        current = None
        for slug in slugs:
            try:
                current = Category.objects.get(
                    slug=slug,
                    parent=current,
                    is_visible=True,
                    is_deleted=False
                )
            except Category.DoesNotExist:
                return None
        return current
    
    @classmethod
    def get_category_products(cls, category: Category, include_descendants=True, limit=None):
        """Get products in a category with optional descendant inclusion."""
        if include_descendants:
            categories = category.get_descendants(include_self=True)
            qs = Product.objects.filter(
                categories__in=categories,
                is_active=True,
                is_deleted=False
            ).distinct()
        else:
            qs = Product.objects.filter(
                categories=category,
                is_active=True,
                is_deleted=False
            )
        
        qs = qs.select_related('primary_category').prefetch_related('images', 'categories', 'tags')
        
        if limit:
            return qs[:limit]
        return qs
    
    @classmethod
    def get_category_facets(cls, category: Category, query_params=None) -> List[Dict[str, Any]]:
        """
        Get facets available for filtering in a category with selected values marked.
        
        Args:
            category: The category to get facets for
            query_params: QueryDict from request.GET (optional)
            
        Returns:
            List of facet dicts with selected values marked
        """
        facets = Facet.objects.filter(
            category_facets__category=category
        ).distinct()
        
        result = []
        
        for facet in facets:
            facet_dict = {
                'id': facet.id,
                'name': facet.name,
                'slug': facet.slug,
                'type': facet.type,
                'values': []
            }
            
            # Get the key name for this facet in query params
            selected_values = []
            if query_params:
                param_key = f'attr_{facet.slug}'
                selected_values = query_params.getlist(param_key)
            
            # Add facet values with selection info
            if facet.values:
                for value_item in facet.values:
                    value_str = value_item.get('value') if isinstance(value_item, dict) else str(value_item)
                    value_dict = {
                        'value': value_str,
                        'display_value': value_item.get('display_value', value_item.get('value')) if isinstance(value_item, dict) else str(value_item),
                        'is_selected': value_str in selected_values
                    }
                    facet_dict['values'].append(value_dict)
            
            result.append(facet_dict)
        
        return result
    
    @classmethod
    @transaction.atomic
    def create_category(cls, name: str, parent: Category = None, **kwargs) -> Category:
        """Create a new category."""
        slug = kwargs.pop('slug', None) or slugify(name)
        
        # Ensure unique slug among siblings
        base_slug = slug
        counter = 1
        while Category.objects.filter(slug=slug, parent=parent).exists():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        category = Category.objects.create(
            name=name,
            slug=slug,
            parent=parent,
            **kwargs
        )
        
        cls.clear_tree_cache()
        return category
    
    @classmethod
    @transaction.atomic
    def update_category(cls, category: Category, **data) -> Category:
        """Update a category."""
        for field, value in data.items():
            if hasattr(category, field):
                setattr(category, field, value)
        category.save()
        cls.clear_tree_cache()
        return category
    
    @classmethod
    @transaction.atomic
    def move_category(cls, category: Category, new_parent: Optional[Category]) -> Category:
        """Move a category to a new parent."""
        if new_parent and new_parent.id == category.id:
            raise ValueError("Category cannot be its own parent")
        
        # Check for circular reference
        if new_parent:
            ancestors = new_parent.get_ancestors(include_self=True)
            if category in ancestors:
                raise ValueError("Cannot move category under its own descendant")
        
        category.parent = new_parent
        category.save()
        cls.clear_tree_cache()
        return category
    
    @classmethod
    def search_categories(cls, query: str, limit=10):
        """Search categories by name."""
        return Category.objects.filter(
            Q(name__icontains=query) | Q(slug__icontains=query),
            is_visible=True,
            is_deleted=False
        ).order_by('path')[:limit]
    
    @classmethod
    def get_breadcrumbs(cls, category: Category) -> List[Dict]:
        """Get breadcrumb trail for a category (excluding the category itself)."""
        ancestors = category.get_ancestors(include_self=False)
        return [
            {
                'id': str(c.id),
                'name': c.name,
                'slug': c.slug,
                'url': c.get_absolute_url()
            }
            for c in ancestors
        ]


# =============================================================================
# Product Services
# =============================================================================

class ProductService:
    """Service class for product operations."""
    
    @classmethod
    def get_product_list(
        cls,
        categories=None,
        tags=None,
        min_price=None,
        max_price=None,
        in_stock=None,
        is_featured=None,
        is_on_sale=None,
        is_new_arrival=None,
        search=None,
        attributes=None,
        sort='-created_at',
        page=1,
        page_size=20
    ) -> Dict[str, Any]:
        """Get filtered and paginated product list."""
        queryset = Product.objects.filter(is_active=True, is_deleted=False)
        
        # Category filter
        if categories:
            if isinstance(categories, (list, tuple)):
                # Get all descendant categories
                all_cats = []
                for cat in categories:
                    if isinstance(cat, Category):
                        all_cats.extend(cat.get_descendants(include_self=True))
                    else:
                        try:
                            c = Category.objects.get(pk=cat)
                            all_cats.extend(c.get_descendants(include_self=True))
                        except Category.DoesNotExist:
                            pass
                if all_cats:
                    queryset = queryset.filter(categories__in=all_cats)
            elif isinstance(categories, Category):
                descendants = categories.get_descendants(include_self=True)
                queryset = queryset.filter(categories__in=descendants)
        
        # Tag filter
        if tags:
            if isinstance(tags, (list, tuple)):
                queryset = queryset.filter(tags__id__in=tags)
            else:
                queryset = queryset.filter(tags__id=tags)
        
        # Price filters
        if min_price is not None:
            queryset = queryset.filter(price__gte=min_price)
        if max_price is not None:
            queryset = queryset.filter(price__lte=max_price)
        
        # Stock filter
        if in_stock is not None:
            if in_stock:
                queryset = queryset.filter(
                    Q(allow_backorder=True) |
                    Q(stock_quantity__gt=0) |
                    Q(variants__stock_quantity__gt=0)
                )
            else:
                queryset = queryset.filter(
                    stock_quantity=0,
                    allow_backorder=False
                ).exclude(variants__stock_quantity__gt=0)
        
        # Featured filter
        if is_featured is not None:
            queryset = queryset.filter(is_featured=is_featured)
        
        # Sale filter
        if is_on_sale is not None and is_on_sale:
            queryset = queryset.filter(
                sale_price__isnull=False,
                sale_price__lt=F('price')
            )
        
        # New arrival filter
        if is_new_arrival is not None:
            queryset = queryset.filter(is_new_arrival=is_new_arrival)
        
        # Attribute filters
        if attributes:
            for attr_slug, value in attributes.items():
                queryset = queryset.filter(
                    attributes__attribute__slug=attr_slug,
                    attributes__value__iexact=value
                )
        
        # Search
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(description__icontains=search) |
                Q(short_description__icontains=search) |
                Q(sku__icontains=search) |
                Q(tags__name__icontains=search)
            )
        
        # Distinct (important for M2M filters)
        queryset = queryset.distinct()
        
        # Prefetch related
        queryset = queryset.select_related('primary_category').prefetch_related(
            'images', 'categories', 'tags', 'variants'
        )
        
        # Sorting
        valid_sorts = {
            'price': 'price',
            '-price': '-price',
            'name': 'name',
            '-name': '-name',
            'created_at': 'created_at',
            '-created_at': '-created_at',
            'sales_count': 'sales_count',
            '-sales_count': '-sales_count',
            'views_count': 'views_count',
            '-views_count': '-views_count',
            'rating': 'average_rating',
            '-rating': '-average_rating',
        }
        order_by = valid_sorts.get(sort, '-created_at')
        queryset = queryset.order_by(order_by)
        
        # Pagination
        paginator = Paginator(queryset, page_size)
        page_obj = paginator.get_page(page)
        
        # Attach primary image to each product for easy template access
        for product in page_obj:
            product.primary_image = product.images.filter(is_primary=True).first() or product.images.first()

        return {
            'products': list(page_obj),
            'total': paginator.count,
            'page': page_obj.number,
            'page_size': page_size,
            'total_pages': paginator.num_pages,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
        }
    
    @classmethod
    def get_product_by_slug(cls, slug: str) -> Optional[Product]:
        """Get product by slug with related data."""
        try:
            return Product.objects.select_related(
                'primary_category', 'product_type', 'shipping_material'
            ).prefetch_related(
                'images', 'categories', 'tags', 'variants', 'attributes',
                'variants__option_values', 'eco_certifications'
            ).get(slug=slug, is_active=True, is_deleted=False)
        except Product.DoesNotExist:
            return None
    
    @classmethod
    def get_product_by_id(cls, product_id) -> Optional[Product]:
        """Get product by ID."""
        try:
            return Product.objects.select_related('primary_category').prefetch_related(
                'images', 'categories', 'tags', 'variants'
            ).get(pk=product_id, is_active=True, is_deleted=False)
        except Product.DoesNotExist:
            return None
    
    @classmethod
    def get_featured_products(cls, limit=8):
        """Get featured products for homepage."""
        products = Product.objects.filter(
            is_active=True,
            is_deleted=False,
            is_featured=True
        ).select_related('primary_category').prefetch_related(
            'images', 'categories'
        ).order_by('-created_at')[:limit]
        
        for product in products:
            product.primary_image = product.images.filter(is_primary=True).first() or product.images.first()
        return products
    
    @classmethod
    def get_new_arrivals(cls, limit=8):
        """Get new arrival products."""
        products = Product.objects.filter(
            is_active=True,
            is_deleted=False,
            is_new_arrival=True
        ).select_related('primary_category').prefetch_related(
            'images', 'categories'
        ).order_by('-created_at')[:limit]

        for product in products:
            product.primary_image = product.images.filter(is_primary=True).first() or product.images.first()
        return products
    
    @classmethod
    def get_bestsellers(cls, limit=8):
        """Get bestselling products."""
        products = Product.objects.filter(
            is_active=True,
            is_deleted=False,
            is_bestseller=True
        ).select_related('primary_category').prefetch_related(
            'images', 'categories'
        ).order_by('-sales_count')[:limit]

        for product in products:
            product.primary_image = product.images.filter(is_primary=True).first() or product.images.first()
        return products
    
    @classmethod
    def get_on_sale_products(cls, limit=8):
        """Get products on sale."""
        products = Product.objects.filter(
            is_active=True,
            is_deleted=False,
            sale_price__isnull=False,
            sale_price__lt=F('price')
        ).select_related('primary_category').prefetch_related(
            'images', 'categories'
        ).order_by('-created_at')[:limit]
        
        for product in products:
            product.primary_image = product.images.filter(is_primary=True).first() or product.images.first()
        return products
    
    @classmethod
    def get_related_products(cls, product: Product, limit=4):
        """Get related products."""
        # First try explicit related products
        related = list(product.related_products.filter(
            is_active=True,
            is_deleted=False
        ).select_related('primary_category').prefetch_related('images')[:limit])
        
        if len(related) < limit:
            # Fill with products from same categories
            category_ids = product.categories.values_list('id', flat=True)
            exclude_ids = [product.pk] + [p.pk for p in related]
            
            additional = Product.objects.filter(
                categories__id__in=category_ids,
                is_active=True,
                is_deleted=False
            ).exclude(
                pk__in=exclude_ids
            ).select_related('primary_category').prefetch_related(
                'images'
            ).distinct()[:limit - len(related)]
            
            related.extend(additional)
        
        # Attach primary image to each product for easy template access
        for p in related:
            p.primary_image = p.images.filter(is_primary=True).first() or p.images.first()

        return related
    
    @classmethod
    def search_products(cls, query: str, limit=20):
        """Full-text search for products."""
        products = Product.objects.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(short_description__icontains=query) |
            Q(sku__icontains=query) |
            Q(tags__name__icontains=query),
            is_active=True,
            is_deleted=False
        ).select_related('primary_category').prefetch_related(
            'images'
        ).distinct().order_by('-views_count')[:limit]
        
        for product in products:
            product.primary_image = product.images.filter(is_primary=True).first() or product.images.first()
        return products
    
    @classmethod
    @transaction.atomic
    def create_product(cls, name: str, price: Decimal, **kwargs) -> Product:
        """Create a new product."""
        slug = kwargs.pop('slug', None) or slugify(name)
        
        # Ensure unique slug
        base_slug = slug
        counter = 1
        while Product.objects.filter(slug=slug).exists():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        categories = kwargs.pop('categories', [])
        tags = kwargs.pop('tags', [])
        
        product = Product.objects.create(
            name=name,
            slug=slug,
            price=price,
            **kwargs
        )
        
        if categories:
            product.categories.set(categories)
        if tags:
            product.tags.set(tags)
        
        return product
    
    @classmethod
    @transaction.atomic
    def update_product(cls, product: Product, **data) -> Product:
        """Update a product."""
        categories = data.pop('categories', None)
        tags = data.pop('tags', None)
        
        for field, value in data.items():
            if hasattr(product, field):
                setattr(product, field, value)
        
        product.save()
        
        if categories is not None:
            product.categories.set(categories)
        if tags is not None:
            product.tags.set(tags)
        
        return product
    
    @classmethod
    def record_view(cls, product: Product, user=None, session_key=None):
        """Record a product view and increment view counter."""
        from .models import ProductImpression
        
        product.increment_views()
        
        ProductImpression.objects.create(
            product=product,
            user=user if user and user.is_authenticated else None,
            session_key=session_key
        )
    
    @classmethod
    def get_price_range(cls, category: Category = None) -> Dict[str, Decimal]:
        """Get min and max prices, optionally filtered by category."""
        qs = Product.objects.filter(is_active=True, is_deleted=False)
        
        if category:
            descendants = category.get_descendants(include_self=True)
            qs = qs.filter(categories__in=descendants)
        
        result = qs.aggregate(
            min_price=models.Min('price'),
            max_price=models.Max('price')
        )
        
        return {
            'min': result['min_price'] or Decimal('0'),
            'max': result['max_price'] or Decimal('0')
        }


# =============================================================================
# Inventory Services
# =============================================================================

class InventoryService:
    """Service class for inventory management."""
    
    @classmethod
    @transaction.atomic
    def adjust_stock(
        cls,
        product: Product,
        quantity_change: int,
        reason: str = 'manual',
        variant: ProductVariant = None,
        note: str = ''
    ) -> StockHistory:
        """Adjust stock for a product or variant."""
        if variant:
            variant.stock_quantity = max(0, variant.stock_quantity + quantity_change)
            variant.save(update_fields=['stock_quantity'])
        else:
            product.stock_quantity = max(0, product.stock_quantity + quantity_change)
            product.save(update_fields=['stock_quantity'])
        
        history = StockHistory.objects.create(
            product=product,
            variant=variant,
            change=quantity_change,
            reason=reason,
            note=note
        )
        
        return history
    
    @classmethod
    @transaction.atomic
    def reserve_stock(
        cls,
        product: Product,
        quantity: int,
        user=None,
        variant: ProductVariant = None,
        expires_minutes: int = 30
    ) -> Reservation:
        """Reserve stock for a limited time."""
        expires_at = timezone.now() + timezone.timedelta(minutes=expires_minutes)
        
        reservation = Reservation.objects.create(
            product=product,
            variant=variant,
            user=user,
            quantity=quantity,
            expires_at=expires_at
        )
        
        # Deduct from available stock
        cls.adjust_stock(
            product, -quantity, 'reservation', variant,
            note=f'Reservation {reservation.id}'
        )
        
        return reservation
    
    @classmethod
    @transaction.atomic
    def release_reservation(cls, reservation: Reservation):
        """Release a reservation and return stock."""
        if reservation.expires_at > timezone.now():
            cls.adjust_stock(
                reservation.product,
                reservation.quantity,
                'reservation',
                reservation.variant,
                note=f'Released reservation {reservation.id}'
            )
        reservation.delete()
    
    @classmethod
    def get_available_stock(cls, product: Product, variant: ProductVariant = None) -> int:
        """Get available stock accounting for reservations."""
        if variant:
            base_stock = variant.stock_quantity
        else:
            base_stock = product.stock_quantity
        
        # Subtract active reservations
        reserved = Reservation.objects.filter(
            product=product,
            variant=variant,
            expires_at__gt=timezone.now()
        ).aggregate(total=Sum('quantity'))['total'] or 0
        
        return max(0, base_stock - reserved)
    
    @classmethod
    def get_low_stock_products(cls, limit=50):
        """Get products with low stock."""
        return Product.objects.filter(
            is_active=True,
            is_deleted=False,
            stock_quantity__lte=F('low_stock_threshold')
        ).order_by('stock_quantity')[:limit]
    
    @classmethod
    def cleanup_expired_reservations(cls):
        """Remove expired reservations and return stock."""
        expired = Reservation.objects.filter(expires_at__lte=timezone.now())
        count = 0
        
        for reservation in expired:
            cls.adjust_stock(
                reservation.product,
                reservation.quantity,
                'reservation',
                reservation.variant,
                note=f'Expired reservation {reservation.id}'
            )
            reservation.delete()
            count += 1
        
        return count


# =============================================================================
# Collection Services
# =============================================================================

class CollectionService:
    """Service class for collection operations."""
    
    @classmethod
    def get_active_collections(cls, placement: str = None, limit: int = None):
        """Get active/visible collections."""
        now = timezone.now()
        qs = Collection.objects.filter(
            is_visible=True
        ).filter(
            Q(visible_from__isnull=True) | Q(visible_from__lte=now)
        ).filter(
            Q(visible_until__isnull=True) | Q(visible_until__gte=now)
        ).order_by('ordering', 'name')
        
        if limit:
            qs = qs[:limit]
        
        return qs
    
    @classmethod
    def get_collection_by_slug(cls, slug: str) -> Optional[Collection]:
        """Get collection by slug."""
        try:
            return Collection.objects.get(slug=slug, is_visible=True)
        except Collection.DoesNotExist:
            return None
    
    @classmethod
    def get_collection_products(cls, collection: Collection, limit: int = None):
        """Get products in a collection."""
        return collection.get_products(limit=limit)
    
    @classmethod
    @transaction.atomic
    def add_product_to_collection(
        cls,
        collection: Collection,
        product: Product,
        ordering: int = 0
    ) -> CollectionItem:
        """Add a product to a collection."""
        item, created = CollectionItem.objects.get_or_create(
            collection=collection,
            product=product,
            defaults={'ordering': ordering}
        )
        if not created:
            item.ordering = ordering
            item.save(update_fields=['ordering'])
        return item
    
    @classmethod
    def remove_product_from_collection(cls, collection: Collection, product: Product):
        """Remove a product from a collection."""
        CollectionItem.objects.filter(collection=collection, product=product).delete()


# =============================================================================
# Review Services
# =============================================================================

class ReviewService:
    """Service class for review operations."""
    
    @classmethod
    def get_product_reviews(
        cls,
        product: Product,
        status: str = 'approved',
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """Get paginated reviews for a product."""
        qs = Review.objects.filter(
            product=product,
            moderation_status=status
        ).select_related('user').prefetch_related('images').order_by('-created_at')
        
        paginator = Paginator(qs, page_size)
        page_obj = paginator.get_page(page)
        
        return {
            'reviews': list(page_obj),
            'total': paginator.count,
            'page': page_obj.number,
            'total_pages': paginator.num_pages,
        }
    
    @classmethod
    def get_review_summary(cls, product: Product) -> Dict[str, Any]:
        """Get review summary statistics for a product."""
        reviews = Review.objects.filter(
            product=product,
            moderation_status='approved'
        )
        
        stats = reviews.aggregate(
            total=Count('id'),
            average=Avg('rating'),
            five_star=Count('id', filter=Q(rating=5)),
            four_star=Count('id', filter=Q(rating=4)),
            three_star=Count('id', filter=Q(rating=3)),
            two_star=Count('id', filter=Q(rating=2)),
            one_star=Count('id', filter=Q(rating=1)),
        )
        
        return {
            'total': stats['total'] or 0,
            'average': round(stats['average'] or 0, 1),
            'distribution': {
                5: stats['five_star'] or 0,
                4: stats['four_star'] or 0,
                3: stats['three_star'] or 0,
                2: stats['two_star'] or 0,
                1: stats['one_star'] or 0,
            }
        }
    
    @classmethod
    @transaction.atomic
    def create_review(
        cls,
        product: Product,
        user,
        rating: int,
        title: str = '',
        body: str = '',
        verified_purchase: bool = False
    ) -> Review:
        """Create a new review."""
        # Check if user already reviewed this product
        if Review.objects.filter(product=product, user=user).exists():
            raise ValueError("You have already reviewed this product")
        
        review = Review.objects.create(
            product=product,
            user=user,
            rating=rating,
            title=title,
            body=body,
            verified_purchase=verified_purchase,
            moderation_status='pending'
        )
        
        return review
    
    @classmethod
    @transaction.atomic
    def approve_review(cls, review: Review):
        """Approve a review and update product stats."""
        review.moderation_status = 'approved'
        review.save(update_fields=['moderation_status', 'updated_at'])
        
        # Update product review stats
        cls._update_product_review_stats(review.product)
    
    @classmethod
    @transaction.atomic
    def reject_review(cls, review: Review):
        """Reject a review."""
        review.moderation_status = 'rejected'
        review.save(update_fields=['moderation_status', 'updated_at'])
    
    @classmethod
    def _update_product_review_stats(cls, product: Product):
        """Update product's review statistics."""
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
        product.save(update_fields=['reviews_count', 'rating_count', 'average_rating', 'updated_at'])


# =============================================================================
# Badge & Spotlight Services
# =============================================================================

class BadgeService:
    """Service class for badge operations."""
    
    @classmethod
    def get_active_badges(cls):
        """Get all currently active badges."""
        now = timezone.now()
        return Badge.objects.filter(
            is_active=True
        ).filter(
            Q(start__isnull=True) | Q(start__lte=now)
        ).filter(
            Q(end__isnull=True) | Q(end__gte=now)
        ).order_by('-priority')
    
    @classmethod
    def get_product_badges(cls, product: Product) -> List[Badge]:
        """Get badges applicable to a product."""
        badges = cls.get_active_badges()
        return [b for b in badges if b.applies_to(product)]


class SpotlightService:
    """Service class for spotlight operations."""
    
    @classmethod
    def get_active_spotlights(cls, placement: str = 'home', limit: int = 5):
        """Get active spotlights for a placement."""
        now = timezone.now()
        return Spotlight.objects.filter(
            is_active=True,
            placement=placement
        ).filter(
            Q(start__isnull=True) | Q(start__lte=now)
        ).filter(
            Q(end__isnull=True) | Q(end__gte=now)
        ).select_related('product', 'category').order_by('-priority')[:limit]


# =============================================================================
# Bundle Services
# =============================================================================

class BundleService:
    """Service class for bundle operations."""
    
    @classmethod
    def get_active_bundles(cls, limit: int = None):
        """Get active bundles."""
        qs = Bundle.objects.filter(is_active=True).prefetch_related(
            'bundle_items__product'
        ).order_by('name')
        
        if limit:
            qs = qs[:limit]
        return qs
    
    @classmethod
    def get_bundle_by_slug(cls, slug: str) -> Optional[Bundle]:
        """Get bundle by slug."""
        try:
            return Bundle.objects.prefetch_related(
                'bundle_items__product__images'
            ).get(slug=slug, is_active=True)
        except Bundle.DoesNotExist:
            return None
    
    @classmethod
    def calculate_bundle_savings(cls, bundle: Bundle) -> Decimal:
        """Calculate savings from buying the bundle vs individual items."""
        individual_total = sum(
            item.product.current_price * item.quantity
            for item in bundle.bundle_items.select_related('product').all()
        )
        return max(Decimal('0'), individual_total - bundle.price())


# =============================================================================
# Product Filter Services (merged from filters.py)
# =============================================================================

import django_filters


class ProductFilter(django_filters.FilterSet):
    """Django-filter based product filtering."""
    
    price_min = django_filters.NumberFilter(field_name="price", lookup_expr="gte")
    price_max = django_filters.NumberFilter(field_name="price", lookup_expr="lte")
    category = django_filters.CharFilter(method="filter_category")
    tags = django_filters.CharFilter(method="filter_tags")

    class Meta:
        model = Product
        fields = ["price_min", "price_max", "category", "tags"]

    def filter_category(self, qs, name, value):
        """Filter by category id or slug, including descendants."""
        try:
            cat = Category.objects.get(Q(id=value) | Q(slug=value))
        except Category.DoesNotExist:
            return qs.none()
        descendants = cat.get_descendants(include_self=True)
        return qs.filter(categories__in=descendants).distinct()

    def filter_tags(self, qs, name, value):
        """Filter by comma-separated tag names."""
        tags = [t.strip() for t in value.split(",") if t.strip()]
        return qs.filter(tags__name__in=tags).distinct()


class ProductFilterService:
    """Service class for advanced product filtering."""
    
    @classmethod
    def apply_attribute_filters(cls, qs, params: dict):
        """
        Apply attribute-based filters.
        Uses queryparams like 'attr_color=red', 'attr_size=large'
        """
        attribute_filters = {k: v for k, v in params.items() if k.startswith("attr_")}
        
        for key, val in attribute_filters.items():
            attr_slug = key[len("attr_"):]
            # Find AttributeValue with matching attribute slug and value
            avs = AttributeValue.objects.filter(
                Q(value__iexact=val) & Q(attribute__slug__iexact=attr_slug)
            )
            if not avs.exists():
                # Also try just matching the value
                avs = AttributeValue.objects.filter(value__iexact=val)
            
            if not avs.exists():
                return qs.none()
            
            qs = qs.filter(attributes__in=avs).distinct()
        
        return qs
    
    @classmethod
    def apply_price_range_filter(cls, qs, min_price=None, max_price=None, currency='BDT'):
        """Apply price range filter with currency support."""
        if min_price is not None:
            qs = qs.filter(price__gte=Decimal(str(min_price)))
        if max_price is not None:
            qs = qs.filter(price__lte=Decimal(str(max_price)))
        return qs
    
    @classmethod
    def apply_stock_filter(cls, qs, in_stock_only=False, include_backorder=True):
        """Filter by stock availability."""
        if in_stock_only:
            if include_backorder:
                qs = qs.filter(
                    Q(allow_backorder=True) |
                    Q(stock_quantity__gt=0) |
                    Q(variants__stock_quantity__gt=0)
                )
            else:
                qs = qs.filter(
                    Q(stock_quantity__gt=0) |
                    Q(variants__stock_quantity__gt=0)
                )
        return qs.distinct()
    
    @classmethod
    def apply_rating_filter(cls, qs, min_rating=None):
        """Filter by minimum rating."""
        if min_rating is not None:
            qs = qs.filter(average_rating__gte=Decimal(str(min_rating)))
        return qs
    
    @classmethod
    def apply_date_filter(cls, qs, days=None, is_new_arrival=None):
        """Filter by date/new arrival status."""
        if is_new_arrival:
            qs = qs.filter(is_new_arrival=True)
        elif days:
            cutoff = timezone.now() - timezone.timedelta(days=days)
            qs = qs.filter(created_at__gte=cutoff)
        return qs
    
    @classmethod
    def apply_sale_filter(cls, qs, on_sale_only=False):
        """Filter for products on sale."""
        if on_sale_only:
            qs = qs.filter(
                sale_price__isnull=False,
                sale_price__lt=F('price')
            )
        return qs
    
    @classmethod
    def apply_shipping_filter(cls, qs, free_shipping_only=False, bangladesh_only=True):
        """Filter by shipping options (Bangladesh focused)."""
        if free_shipping_only:
            qs = qs.filter(free_shipping=True)
        return qs
    
    @classmethod
    def build_filtered_queryset(cls, params: dict, base_queryset=None):
        """
        Build a fully filtered queryset from request parameters.
        Supports Bangladesh-specific filtering.
        """
        if base_queryset is None:
            base_queryset = Product.objects.filter(is_active=True, is_deleted=False)
        
        qs = base_queryset
        
        # Category filter
        if params.get('category'):
            try:
                cat = Category.objects.get(
                    Q(id=params['category']) | Q(slug=params['category'])
                )
                descendants = cat.get_descendants(include_self=True)
                qs = qs.filter(categories__in=descendants)
            except (Category.DoesNotExist, ValueError):
                pass
        
        # Price range (in BDT)
        qs = cls.apply_price_range_filter(
            qs,
            min_price=params.get('price_min') or params.get('min_price'),
            max_price=params.get('price_max') or params.get('max_price'),
            currency='BDT'
        )
        
        # Stock filter
        if params.get('in_stock') in ('true', '1', True):
            qs = cls.apply_stock_filter(qs, in_stock_only=True)
        
        # Rating filter
        if params.get('min_rating'):
            qs = cls.apply_rating_filter(qs, min_rating=params.get('min_rating'))
        
        # Sale filter
        if params.get('on_sale') in ('true', '1', True):
            qs = cls.apply_sale_filter(qs, on_sale_only=True)
        
        # New arrivals
        if params.get('new_arrivals') in ('true', '1', True):
            qs = cls.apply_date_filter(qs, is_new_arrival=True)
        
        # Free shipping (Bangladesh)
        if params.get('free_shipping') in ('true', '1', True):
            qs = cls.apply_shipping_filter(qs, free_shipping_only=True)
        
        # Attribute filters
        qs = cls.apply_attribute_filters(qs, params)
        
        # Tags filter
        if params.get('tags'):
            tags = [t.strip() for t in params['tags'].split(",") if t.strip()]
            if tags:
                qs = qs.filter(tags__name__in=tags)
        
        # Search query
        if params.get('q') or params.get('search'):
            query = params.get('q') or params.get('search')
            qs = qs.filter(
                Q(name__icontains=query) |
                Q(description__icontains=query) |
                Q(short_description__icontains=query) |
                Q(sku__icontains=query) |
                Q(tags__name__icontains=query)
            )
        
        return qs.distinct()
    
    @classmethod
    def get_available_filters(cls, category: Category = None) -> dict:
        """
        Get available filter options for a category.
        Useful for building filter UI.
        """
        base_qs = Product.objects.filter(is_active=True, is_deleted=False)
        
        if category:
            descendants = category.get_descendants(include_self=True)
            base_qs = base_qs.filter(categories__in=descendants)
        
        # Get price range
        price_stats = base_qs.aggregate(
            min_price=models.Min('price'),
            max_price=models.Max('price')
        )
        
        # Get available attributes
        attributes = {}
        product_ids = base_qs.values_list('id', flat=True)[:1000]  # Limit for performance
        attr_values = AttributeValue.objects.filter(
            products__id__in=product_ids
        ).select_related('attribute').distinct()
        
        for av in attr_values:
            attr_name = av.attribute.name
            if attr_name not in attributes:
                attributes[attr_name] = {
                    'slug': av.attribute.slug,
                    'values': []
                }
            if av.value not in attributes[attr_name]['values']:
                attributes[attr_name]['values'].append(av.value)
        
        # Get available tags
        tags = list(
            Tag.objects.filter(
                products__id__in=product_ids
            ).values('name', 'slug').distinct()[:50]
        )
        
        return {
            'price_range': {
                'min': float(price_stats['min_price'] or 0),
                'max': float(price_stats['max_price'] or 0),
                'currency': 'BDT',
                'currency_symbol': '৳',
            },
            'attributes': attributes,
            'tags': tags,
            'has_on_sale': base_qs.filter(
                sale_price__isnull=False,
                sale_price__lt=F('price')
            ).exists(),
            'has_free_shipping': base_qs.filter(free_shipping=True).exists(),
        }
