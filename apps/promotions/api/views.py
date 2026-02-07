"""
Promotions API views
"""
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from ..models import Coupon, Banner, Sale
from ..services import CouponService, BannerService, SaleService

logger = logging.getLogger(__name__)
from .serializers import (
    CouponSerializer,
    CouponValidateSerializer,
    BannerSerializer,
    SaleSerializer,
    SaleDetailSerializer,
)
from apps.commerce.services import CartService
from apps.commerce.api.serializers import CartSerializer


class CouponViewSet(viewsets.ViewSet):
    """
    ViewSet for coupon operations.
    
    Endpoints:
    - POST /api/v1/promotions/coupons/validate/ - Validate coupon code
    - POST /api/v1/promotions/coupons/apply/ - Apply coupon to current cart
    """
    permission_classes = [AllowAny]

    def _get_cart(self, request):
        """Get or create cart for current user/session."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key
        return CartService.get_or_create_cart(user=user, session_key=session_key)

    def _get_currency_context(self, request):
        try:
            from apps.i18n.services import CurrencyService, CurrencyConversionService
            user_currency = CurrencyService.get_user_currency(request=request)
            default_currency = CurrencyService.get_default_currency()
            return user_currency, default_currency, CurrencyConversionService
        except Exception:
            return None, None, None

    def _format_minimum_order_message(self, coupon, user_currency, default_currency, conversion_service, message):
        try:
            from decimal import Decimal
            if coupon and coupon.minimum_order_amount:
                display_amount = coupon.minimum_order_amount
                display_currency = user_currency or default_currency
                if user_currency and default_currency and user_currency.code != default_currency.code and conversion_service:
                    display_amount = conversion_service.convert_by_code(
                        Decimal(str(coupon.minimum_order_amount)),
                        default_currency.code,
                        user_currency.code,
                        round_result=True
                    )
                if display_currency:
                    return f"Minimum order amount is {display_currency.format_amount(display_amount)}"
        except Exception:
            pass
        return message
    
    @action(detail=False, methods=['post'], url_path='validate')
    def validate(self, request):
        """Validate a coupon code."""
        serializer = CouponValidateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'data': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        code = serializer.validated_data['code']
        subtotal = serializer.validated_data.get('subtotal', 0)

        # Normalize subtotal to default currency for validation
        user_currency, default_currency, conversion_service = self._get_currency_context(request)
        try:
            if user_currency and default_currency and user_currency.code != default_currency.code and conversion_service:
                from decimal import Decimal
                subtotal = conversion_service.convert_by_code(
                    Decimal(str(subtotal)), user_currency.code, default_currency.code, round_result=False
                )
        except Exception:
            pass
        
        user = request.user if request.user.is_authenticated else None
        
        coupon, is_valid, message = CouponService.validate_coupon(
            code=code,
            user=user,
            subtotal=subtotal
        )
        
        discount = None
        if coupon and is_valid:
            discount = str(coupon.calculate_discount(subtotal))

        # Replace minimum order message with user-currency formatted value
        if coupon and not is_valid:
            message = self._format_minimum_order_message(
                coupon, user_currency, default_currency, conversion_service, message
            )
        
        return Response({
            'success': is_valid,
            'message': message,
            'data': {
                'is_valid': is_valid,
                'coupon': CouponSerializer(coupon).data if coupon else None,
                'discount': discount
            }
        })

    @action(detail=False, methods=['post'], url_path='apply')
    def apply(self, request):
        """Apply coupon code to the current cart."""
        serializer = CouponValidateSerializer(data=request.data)

        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'data': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)

        code = serializer.validated_data['code']

        cart = self._get_cart(request)
        if not cart or not cart.items.exists():
            return Response({
                'success': False,
                'message': 'Your cart is empty.',
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)

        user_currency, default_currency, conversion_service = self._get_currency_context(request)
        user = request.user if request.user.is_authenticated else None

        coupon, is_valid, message = CouponService.validate_coupon(
            code=code,
            user=user,
            subtotal=cart.subtotal
        )

        if not is_valid:
            message = self._format_minimum_order_message(
                coupon, user_currency, default_currency, conversion_service, message
            )
            return Response({
                'success': False,
                'message': message,
                'data': {
                    'is_valid': False,
                    'coupon': CouponSerializer(coupon).data if coupon else None
                }
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            CartService.apply_coupon(cart, code)
            cart.refresh_from_db()  # Refresh cart to ensure coupon is loaded
        except Exception as e:
            logger.error(f"Error applying coupon {code} to cart: {str(e)}", exc_info=True)
            return Response({
                'success': False,
                'message': str(e),
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            'success': True,
            'message': 'Coupon applied',
            'cart': CartSerializer(cart, context={'request': request}).data
        })


class BannerViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for banners.
    
    Endpoints:
    - GET /api/v1/banners/ - List active banners
    - GET /api/v1/banners/{id}/ - Get banner detail
    - GET /api/v1/banners/hero/ - Get hero banners
    - GET /api/v1/banners/secondary/ - Get secondary banners
    """
    permission_classes = [AllowAny]
    serializer_class = BannerSerializer
    
    def get_queryset(self):
        return BannerService.get_active_banners()
    
    def list(self, request):
        """List active banners."""
        position = request.query_params.get('position')
        queryset = BannerService.get_active_banners(position=position)
        serializer = self.get_serializer(queryset, many=True)
        
        return Response({
            'success': True,
            'message': 'Banners retrieved',
            'data': serializer.data
        })
    
    @action(detail=False, methods=['get'], url_path='hero')
    def hero(self, request):
        """Get hero banners."""
        banners = BannerService.get_hero_banners()
        serializer = self.get_serializer(banners, many=True)
        
        return Response({
            'success': True,
            'message': 'Hero banners retrieved',
            'data': serializer.data
        })
    
    @action(detail=False, methods=['get'], url_path='secondary')
    def secondary(self, request):
        """Get secondary banners."""
        banners = BannerService.get_secondary_banners()
        serializer = self.get_serializer(banners, many=True)
        
        return Response({
            'success': True,
            'message': 'Secondary banners retrieved',
            'data': serializer.data
        })


class SaleViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for sales.
    
    Endpoints:
    - GET /api/v1/sales/ - List active sales
    - GET /api/v1/sales/{slug}/ - Get sale detail with products
    """
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    
    def get_queryset(self):
        return SaleService.get_active_sales()
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return SaleDetailSerializer
        return SaleSerializer
    
    def list(self, request):
        """List active sales."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        
        return Response({
            'success': True,
            'message': 'Sales retrieved',
            'data': serializer.data
        })
    
    def retrieve(self, request, slug=None):
        """Get sale detail with products."""
        sale = self.get_object()
        serializer = self.get_serializer(sale)
        
        return Response({
            'success': True,
            'message': 'Sale retrieved',
            'data': serializer.data
        })
    
    @action(detail=True, methods=['get'], url_path='products')
    def products(self, request, slug=None):
        """Get all products in a sale with pagination."""
        sale = self.get_object()
        products = SaleService.get_sale_products(sale)
        
        # Paginate
        page = self.paginate_queryset(products)
        if page is not None:
            from apps.products.api.serializers import ProductListSerializer
            serializer = ProductListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        from apps.products.api.serializers import ProductListSerializer
        serializer = ProductListSerializer(products, many=True)
        
        return Response({
            'success': True,
            'message': 'Sale products retrieved',
            'data': serializer.data
        })
