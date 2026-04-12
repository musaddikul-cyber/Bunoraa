"""
Promotions API serializers
"""
from urllib.parse import urlparse

from django.conf import settings
from rest_framework import serializers
from ..models import Coupon, Banner, Sale


class CouponSerializer(serializers.ModelSerializer):
    """Serializer for coupon."""
    is_valid = serializers.ReadOnlyField()
    currency = serializers.CharField(source='currency_id', read_only=True)
    
    class Meta:
        model = Coupon
        fields = [
            'id', 'code', 'description', 'discount_type', 'discount_value', 'currency',
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
    link_url = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    class Meta:
        model = Banner
        fields = [
            'id', 'title', 'subtitle', 'image', 'image_mobile',
            'link_url', 'link_text', 'position',
            'style_height', 'style_width', 'style_max_width',
            'style_border_radius', 'style_border_width',
            'style_border_color', 'style_background_color',
            'overlay_color', 'overlay_opacity', 'text_color',
            'content_vertical_position', 'content_horizontal_alignment',
            'button_alignment',
            'title_font_size', 'subtitle_font_size',
            'button_font_size', 'button_padding', 'button_min_height'
        ]

    def _normalize_origin(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

    def _resolve_site_origin(self) -> str:
        request = self.context.get("request")
        if request is not None:
            for header in ("HTTP_ORIGIN", "HTTP_REFERER"):
                candidate = self._normalize_origin(request.META.get(header, ""))
                if candidate:
                    return candidate
            try:
                host = request.get_host()
                if host:
                    scheme = "https" if request.is_secure() else "http"
                    return f"{scheme}://{host}"
            except Exception:
                pass

        configured = (
            getattr(settings, "NEXT_FRONTEND_ORIGIN", "")
            or getattr(settings, "NEXT_PUBLIC_SITE_URL", "")
            or getattr(settings, "SITE_URL", "")
        )
        normalized = self._normalize_origin(configured)
        if normalized:
            return normalized
        return "http://localhost:3000"

    def validate_link_url(self, value):
        raw = (value or "").strip()
        if not raw:
            return ""

        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            if parsed.scheme not in {"http", "https"}:
                raise serializers.ValidationError("Only http/https URLs are allowed.")
            return raw

        site_origin = self._resolve_site_origin()
        path = raw if raw.startswith("/") else f"/{raw}"
        normalized = f"{site_origin}{path}"
        normalized_parsed = urlparse(normalized)
        if normalized_parsed.scheme not in {"http", "https"} or not normalized_parsed.netloc:
            raise serializers.ValidationError("Enter a valid URL or path.")
        return normalized


class SaleSerializer(serializers.ModelSerializer):
    """Serializer for sale."""
    is_running = serializers.ReadOnlyField()
    product_count = serializers.SerializerMethodField()
    currency = serializers.CharField(source='currency_id', read_only=True)
    
    class Meta:
        model = Sale
        fields = [
            'id', 'name', 'slug', 'description',
            'discount_type', 'discount_value', 'currency',
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
