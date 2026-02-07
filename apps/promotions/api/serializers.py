"""
Promotions API serializers
"""
from rest_framework import serializers
from ..models import Coupon, Banner, Sale


class CouponSerializer(serializers.ModelSerializer):
    """Serializer for coupon."""
    is_valid = serializers.ReadOnlyField()
    
    class Meta:
        model = Coupon
        fields = [
            'id', 'code', 'description', 'discount_type', 'discount_value',
            'minimum_order_amount', 'maximum_discount',
            'valid_from', 'valid_until', 'is_valid'
        ]


class CouponValidateSerializer(serializers.Serializer):
    """Serializer for coupon validation."""
    code = serializers.CharField(max_length=50)
    subtotal = serializers.DecimalField(
        max_digits=10, decimal_places=2,
        required=False, default=0
    )


class CouponValidateResponseSerializer(serializers.Serializer):
    """Serializer for coupon validation response."""
    is_valid = serializers.BooleanField()
    message = serializers.CharField()
    coupon = CouponSerializer(allow_null=True)
    discount = serializers.DecimalField(
        max_digits=10, decimal_places=2,
        allow_null=True
    )


class BannerSerializer(serializers.ModelSerializer):
    """Serializer for banner."""
    
    class Meta:
        model = Banner
        fields = [
            'id', 'title', 'subtitle', 'image', 'image_mobile',
            'link_url', 'link_text', 'position'
        ]


class SaleSerializer(serializers.ModelSerializer):
    """Serializer for sale."""
    is_running = serializers.ReadOnlyField()
    product_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Sale
        fields = [
            'id', 'name', 'slug', 'description',
            'discount_type', 'discount_value',
            'banner_image', 'start_date', 'end_date',
            'is_running', 'product_count'
        ]
    
    def get_product_count(self, obj):
        from ..services import SaleService
        return SaleService.get_sale_products(obj).count()


class SaleDetailSerializer(SaleSerializer):
    """Detailed sale serializer with products."""
    products = serializers.SerializerMethodField()
    
    class Meta(SaleSerializer.Meta):
        fields = SaleSerializer.Meta.fields + ['products']
    
    def get_products(self, obj):
        from apps.products.api.serializers import ProductListSerializer
        from ..services import SaleService
        
        products = SaleService.get_sale_products(obj)[:20]  # Limit to 20
        return ProductListSerializer(products, many=True).data
