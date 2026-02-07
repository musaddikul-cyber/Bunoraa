"""
Promotions services
"""
from decimal import Decimal
from django.db import models
from django.utils import timezone
from django.db.models import Q

from .models import Coupon, Banner, Sale


class CouponService:
    """Service for coupon operations."""
    
    @staticmethod
    def validate_coupon(code, user=None, subtotal=Decimal('0')):
        """
        Validate a coupon code.
        
        Args:
            code: Coupon code string
            user: Optional user
            subtotal: Cart subtotal
            
        Returns:
            Tuple (coupon, is_valid, message)
        """
        try:
            coupon = Coupon.objects.get(code__iexact=code.strip())
        except Coupon.DoesNotExist:
            return None, False, "Invalid coupon code"
        
        is_valid, message = coupon.can_use(user=user, subtotal=subtotal)
        
        return coupon, is_valid, message
    
    @staticmethod
    def apply_coupon(coupon, subtotal, cart_items=None):
        """
        Apply coupon to cart.
        
        Args:
            coupon: Coupon instance
            subtotal: Cart subtotal
            cart_items: Optional list of cart items for product/category restrictions
            
        Returns:
            Decimal discount amount
        """
        # Check product/category restrictions
        if cart_items and (coupon.products.exists() or coupon.categories.exists()):
            applicable_subtotal = Decimal('0')
            
            for item in cart_items:
                is_applicable = False
                
                # Check if product is in coupon products
                if coupon.products.filter(id=item.product_id).exists():
                    is_applicable = True
                
                # Check if product is in coupon categories
                if not is_applicable and coupon.categories.exists():
                    product_category_ids = item.product.categories.values_list('id', flat=True)
                    if coupon.categories.filter(id__in=product_category_ids).exists():
                        is_applicable = True
                
                if is_applicable:
                    applicable_subtotal += item.line_total
            
            if applicable_subtotal == 0:
                return Decimal('0')
            
            return coupon.calculate_discount(applicable_subtotal)
        
        return coupon.calculate_discount(subtotal)
    
    @staticmethod
    def get_available_coupons(user=None):
        """
        Get coupons available for user.
        
        Args:
            user: Optional user
            
        Returns:
            QuerySet of coupons
        """
        now = timezone.now()
        
        queryset = Coupon.objects.filter(
            is_active=True
        ).filter(
            Q(valid_from__isnull=True) | Q(valid_from__lte=now)
        ).filter(
            Q(valid_until__isnull=True) | Q(valid_until__gte=now)
        ).filter(
            Q(usage_limit__isnull=True) | Q(times_used__lt=models.F('usage_limit'))
        )
        
        if user:
            # Include public coupons and user-specific coupons
            queryset = queryset.filter(
                Q(users__isnull=True) | Q(users=user)
            ).distinct()
        else:
            # Only public coupons
            queryset = queryset.filter(users__isnull=True)
        
        return queryset


class BannerService:
    """Service for banner operations."""
    
    @staticmethod
    def get_active_banners(position=None):
        """
        Get active banners.
        
        Args:
            position: Optional position filter
            
        Returns:
            QuerySet of banners
        """
        now = timezone.now()
        
        queryset = Banner.objects.filter(
            is_active=True
        ).filter(
            Q(start_date__isnull=True) | Q(start_date__lte=now)
        ).filter(
            Q(end_date__isnull=True) | Q(end_date__gte=now)
        )
        
        if position:
            queryset = queryset.filter(position=position)
        
        return queryset.order_by('sort_order')
    
    @staticmethod
    def get_hero_banners():
        """Get hero banners for homepage."""
        return BannerService.get_active_banners(Banner.POSITION_HOME_HERO)
    
    @staticmethod
    def get_secondary_banners():
        """Get secondary banners for homepage."""
        return BannerService.get_active_banners(Banner.POSITION_HOME_SECONDARY)


class SaleService:
    """Service for sale operations."""
    
    @staticmethod
    def get_active_sales():
        """Get currently active sales."""
        now = timezone.now()
        
        return Sale.objects.filter(
            is_active=True,
            start_date__lte=now,
            end_date__gte=now
        )
    
    @staticmethod
    def get_sale_products(sale):
        """
        Get all products in a sale.
        
        Args:
            sale: Sale instance
            
        Returns:
            QuerySet of products
        """
        from apps.products.models import Product
        
        # Direct products
        product_ids = set(sale.products.values_list('id', flat=True))
        
        # Products from categories
        for category in sale.categories.all():
            category_products = Product.objects.filter(
                categories=category,
                is_active=True,
                is_deleted=False
            ).values_list('id', flat=True)
            product_ids.update(category_products)
        
        return Product.objects.filter(
            id__in=product_ids,
            is_active=True,
            is_deleted=False
        )
    
    @staticmethod
    def get_product_sale(product):
        """
        Get active sale for a product.
        
        Args:
            product: Product instance
            
        Returns:
            Sale instance or None
        """
        now = timezone.now()
        
        # Check direct product sales
        sale = Sale.objects.filter(
            is_active=True,
            start_date__lte=now,
            end_date__gte=now,
            products=product
        ).first()
        
        if sale:
            return sale
        
        # Check category sales
        product_categories = product.categories.all()
        sale = Sale.objects.filter(
            is_active=True,
            start_date__lte=now,
            end_date__gte=now,
            categories__in=product_categories
        ).first()
        
        return sale
