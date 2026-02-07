"""
Commerce API Serializers
"""
from rest_framework import serializers
from decimal import Decimal

from apps.i18n.api.serializers import CurrencyConversionMixin

from ..models import (
    Cart, CartItem, Wishlist, WishlistItem, WishlistShare,
    CheckoutSession, SharedCart
)


# =============================================================================
# Cart Serializers
# =============================================================================

class CartItemSerializer(CurrencyConversionMixin, serializers.ModelSerializer):
    """Serializer for cart items."""
    
    product_id = serializers.UUIDField(source='product.id', read_only=True)
    product_name = serializers.CharField(source='product.name', read_only=True)
    product_slug = serializers.CharField(source='product.slug', read_only=True)
    product_image = serializers.SerializerMethodField()
    variant_id = serializers.UUIDField(source='variant.id', read_only=True, allow_null=True)
    variant_name = serializers.SerializerMethodField()
    unit_price = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    total = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    in_stock = serializers.SerializerMethodField()
    
    class Meta:
        model = CartItem
        fields = [
            'id', 'product_id', 'product_name', 'product_slug', 'product_image',
            'variant_id', 'variant_name', 'quantity', 'unit_price', 'total',
            'price_at_add', 'in_stock', 'gift_wrap', 'gift_message', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

    currency_fields = ['unit_price', 'total', 'price_at_add']

    def get_currency_source_code(self, instance, data):
        return getattr(instance.cart, 'currency', None) or data.get('currency') or 'BDT'
    
    def get_product_image(self, obj):
        image = obj.product.images.filter(is_primary=True).first() or obj.product.images.first()
        if image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(image.image.url)
            return image.image.url
        return None
    
    def get_variant_name(self, obj):
        return str(obj.variant) if obj.variant else None
    
    def get_in_stock(self, obj):
        return obj.product.is_in_stock()


class CartSerializer(CurrencyConversionMixin, serializers.ModelSerializer):
    """Serializer for shopping cart."""
    
    items = CartItemSerializer(many=True, read_only=True)
    item_count = serializers.IntegerField(read_only=True)
    subtotal = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    discount_amount = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    total = serializers.DecimalField(max_digits=12, decimal_places=2, read_only=True)
    coupon_code = serializers.SerializerMethodField()
    
    class Meta:
        model = Cart
        fields = [
            'id', 'items', 'item_count', 'subtotal', 'discount_amount', 'total',
            'coupon_code', 'currency', 'updated_at'
        ]
        read_only_fields = ['id', 'updated_at']

    currency_fields = ['subtotal', 'discount_amount', 'total']
    
    def get_coupon_code(self, obj):
        """Get coupon code, handling cases where coupon might be None."""
        if obj.coupon:
            return obj.coupon.code
        return None


class AddToCartSerializer(serializers.Serializer):
    """Serializer for adding items to cart."""
    
    product_id = serializers.UUIDField()
    variant_id = serializers.UUIDField(required=False, allow_null=True)
    quantity = serializers.IntegerField(min_value=1, default=1)


class UpdateCartItemSerializer(serializers.Serializer):
    """Serializer for updating cart item."""
    
    quantity = serializers.IntegerField(min_value=0)
    gift_wrap = serializers.BooleanField(required=False)
    gift_message = serializers.CharField(required=False, allow_blank=True, max_length=500)


class CartGiftOptionsSerializer(serializers.Serializer):
    """Serializer for cart-level gift options."""

    is_gift = serializers.BooleanField(required=False, default=False)
    gift_message = serializers.CharField(required=False, allow_blank=True, max_length=200)
    gift_wrap = serializers.BooleanField(required=False, default=False)


class ApplyCouponSerializer(serializers.Serializer):
    """Serializer for applying coupon."""
    
    coupon_code = serializers.CharField(max_length=50)


class LockPricesSerializer(serializers.Serializer):
    """Serializer for price lock requests."""
    duration_hours = serializers.IntegerField(min_value=1, max_value=168, required=False)


class ShareCartSerializer(serializers.Serializer):
    """Serializer for cart share requests."""
    name = serializers.CharField(max_length=200, required=False, allow_blank=True)
    permission = serializers.ChoiceField(
        choices=SharedCart.PERMISSION_CHOICES,
        required=False,
        default=SharedCart.PERMISSION_VIEW
    )
    expires_days = serializers.IntegerField(min_value=1, max_value=365, required=False, default=7)
    password = serializers.CharField(max_length=128, required=False, allow_blank=True)


# =============================================================================
# Wishlist Serializers
# =============================================================================

class WishlistItemSerializer(serializers.ModelSerializer):
    """Serializer for wishlist items."""
    
    product_id = serializers.UUIDField(source='product.id', read_only=True)
    product_name = serializers.CharField(source='product.name', read_only=True)
    product_slug = serializers.CharField(source='product.slug', read_only=True)
    product_image = serializers.SerializerMethodField()
    current_price = serializers.SerializerMethodField()
    price_change = serializers.SerializerMethodField()
    price_change_percent = serializers.SerializerMethodField()
    in_stock = serializers.SerializerMethodField()
    variant = serializers.SerializerMethodField()
    
    class Meta:
        model = WishlistItem
        fields = [
            'id', 'product_id', 'product_name', 'product_slug', 'product_image',
            'variant', 'price_at_add', 'current_price', 'price_change',
            'price_change_percent', 'in_stock', 'notes', 'priority', 'added_at',
            'desired_quantity', 'notify_on_sale', 'notify_on_restock', 'notify_on_price_drop', 'target_price'
        ]
        read_only_fields = ['id', 'added_at', 'variant', 'price_change', 'price_change_percent', 'current_price', 'in_stock']
    
    def get_variant(self, obj):
        """Serialize the variant field."""
        if obj.variant:
            return {
                'id': str(obj.variant.id),
                'name': obj.variant.name,
                'sku': obj.variant.sku,
            }
        return None
    
    def get_product_image(self, obj):
        try:
            image = obj.product.images.filter(is_primary=True).first() or obj.product.images.first()
            if image:
                request = self.context.get('request')
                if request:
                    return request.build_absolute_uri(image.image.url)
                return image.image.url
        except Exception:
            pass
        return None
    
    def get_current_price(self, obj):
        try:
            return str(obj.product.current_price)
        except Exception:
            return "0.00"
    
    def get_price_change(self, obj):
        """Calculate price change from when added to current price."""
        try:
            if obj.price_at_add:
                return float(obj.product.current_price) - float(obj.price_at_add)
        except Exception:
            pass
        return 0
    
    def get_price_change_percent(self, obj):
        try:
            if obj.price_at_add and obj.price_at_add > 0:
                current = float(obj.product.current_price)
                added = float(obj.price_at_add)
                change = current - added
                return round((change / added) * 100, 1)
        except Exception:
            pass
        return 0
    
    def get_in_stock(self, obj):
        try:
            return obj.product.is_in_stock()
        except Exception:
            return False


class WishlistSerializer(serializers.ModelSerializer):
    """Serializer for wishlist."""
    
    items = WishlistItemSerializer(many=True, read_only=True)
    item_count = serializers.IntegerField(read_only=True)
    total_value = serializers.SerializerMethodField()
    
    class Meta:
        model = Wishlist
        fields = ['id', 'items', 'item_count', 'total_value', 'updated_at']
        read_only_fields = ['id', 'updated_at']
    
    def get_total_value(self, obj):
        total = sum(item.product.current_price for item in obj.items.all())
        return str(total)


class AddToWishlistSerializer(serializers.Serializer):
    """Serializer for adding items to wishlist."""
    
    product_id = serializers.UUIDField()
    variant_id = serializers.UUIDField(required=False, allow_null=True)
    notes = serializers.CharField(required=False, allow_blank=True, default='')
    notify_on_sale = serializers.BooleanField(default=True)
    notify_on_stock = serializers.BooleanField(default=True)


class WishlistShareSerializer(serializers.ModelSerializer):
    """Serializer for wishlist share links."""
    
    share_url = serializers.SerializerMethodField()
    
    class Meta:
        model = WishlistShare
        fields = [
            'id', 'share_token', 'share_url', 'expires_at', 'allow_purchase',
            'view_count', 'created_at'
        ]
        read_only_fields = ['id', 'share_token', 'view_count', 'created_at']
    
    def get_share_url(self, obj):
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(f'/wishlist/shared/{obj.share_token}/')
        return f'/wishlist/shared/{obj.share_token}/'


class CreateWishlistShareSerializer(serializers.Serializer):
    """Serializer for creating wishlist share."""
    
    expires_days = serializers.IntegerField(min_value=1, max_value=365, default=30)
    allow_purchase = serializers.BooleanField(default=False)


# =============================================================================
# Checkout Serializers
# =============================================================================

class CheckoutSessionSerializer(serializers.ModelSerializer):
    """Serializer for checkout session."""
    
    cart_summary = serializers.SerializerMethodField()
    pickup_location = serializers.SerializerMethodField()
    
    class Meta:
        model = CheckoutSession
        fields = [
            'id', 'current_step', 'email',
            'shipping_first_name', 'shipping_last_name', 'shipping_company',
            'shipping_email', 'shipping_phone', 'shipping_address_line_1',
            'shipping_address_line_2', 'shipping_city', 'shipping_state',
            'shipping_postal_code', 'shipping_country',
            'billing_first_name', 'billing_last_name', 'billing_company',
            'billing_address_line_1', 'billing_address_line_2', 'billing_city',
            'billing_state', 'billing_postal_code', 'billing_country',
            'billing_same_as_shipping',
            'shipping_method', 'shipping_cost', 'payment_method',
            'payment_fee_amount', 'payment_fee_label',
            'subtotal', 'discount_amount', 'tax_amount', 'total',
            'coupon_code', 'order_notes', 'delivery_instructions',
            'is_gift', 'gift_message', 'gift_wrap', 'gift_wrap_cost',
            'pickup_location', 'cart_summary', 'created_at', 'expires_at'
        ]
        read_only_fields = [
            'id', 'current_step', 'subtotal', 'discount_amount', 'tax_amount',
            'total', 'created_at', 'expires_at'
        ]
    
    def get_cart_summary(self, obj):
        if obj.cart:
            return {
                'item_count': obj.cart.item_count,
                'subtotal': str(obj.cart.subtotal),
                'total': str(obj.cart.total),
            }
        return None

    def get_pickup_location(self, obj):
        location = getattr(obj, 'pickup_location', None)
        if not location:
            return None
        return {
            'id': str(location.id),
            'name': location.name,
            'full_address': location.full_address,
            'phone': location.phone,
            'email': location.email,
            'pickup_fee': str(location.pickup_fee),
            'min_pickup_time_hours': location.min_pickup_time_hours,
            'max_hold_days': location.max_hold_days,
        }


class CheckoutShippingInfoSerializer(serializers.Serializer):
    """Serializer for checkout shipping information."""
    
    email = serializers.EmailField()
    shipping_first_name = serializers.CharField(max_length=100)
    shipping_last_name = serializers.CharField(max_length=100)
    shipping_company = serializers.CharField(max_length=200, required=False, allow_blank=True)
    shipping_phone = serializers.CharField(max_length=20)
    shipping_address_line_1 = serializers.CharField(max_length=255)
    shipping_address_line_2 = serializers.CharField(max_length=255, required=False, allow_blank=True)
    shipping_city = serializers.CharField(max_length=100)
    shipping_state = serializers.CharField(max_length=100, required=False, allow_blank=True)
    shipping_postal_code = serializers.CharField(max_length=20)
    shipping_country = serializers.CharField(max_length=100, default='Bangladesh')
    save_address = serializers.BooleanField(required=False, default=False)


class CheckoutShippingMethodSerializer(serializers.Serializer):
    """Serializer for selecting shipping method."""

    shipping_type = serializers.ChoiceField(
        choices=[
            ('delivery', 'Delivery'),
            ('pickup', 'Pickup'),
        ],
        required=False,
        default='delivery'
    )
    shipping_method = serializers.CharField(required=False, allow_blank=True)
    shipping_rate_id = serializers.CharField(required=False, allow_blank=True)
    pickup_location_id = serializers.CharField(required=False, allow_blank=True)
    delivery_instructions = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        shipping_type = attrs.get('shipping_type') or 'delivery'
        if shipping_type == 'pickup':
            return attrs

        if not attrs.get('shipping_rate_id') and not attrs.get('shipping_method'):
            raise serializers.ValidationError({
                'shipping_rate_id': 'Please select a shipping method.'
            })
        return attrs


class CheckoutPaymentMethodSerializer(serializers.Serializer):
    """Serializer for selecting payment method."""

    payment_method = serializers.CharField()
    billing_same_as_shipping = serializers.BooleanField(required=False)
    billing_first_name = serializers.CharField(required=False, allow_blank=True)
    billing_last_name = serializers.CharField(required=False, allow_blank=True)
    billing_company = serializers.CharField(required=False, allow_blank=True)
    billing_address_line_1 = serializers.CharField(required=False, allow_blank=True)
    billing_address_line_2 = serializers.CharField(required=False, allow_blank=True)
    billing_city = serializers.CharField(required=False, allow_blank=True)
    billing_state = serializers.CharField(required=False, allow_blank=True)
    billing_postal_code = serializers.CharField(required=False, allow_blank=True)
    billing_country = serializers.CharField(required=False, allow_blank=True)
