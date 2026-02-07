"""
Commerce API Views
"""
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.core.exceptions import ValidationError
from django.urls import reverse
from django.conf import settings
from urllib.parse import urlparse
from decimal import Decimal
from core.pagination import StandardResultsSetPagination
from ..models import (
    Cart, CartItem, Wishlist, WishlistItem, WishlistShare,
    CheckoutSession, CartSettings
)
from ..services import (
    CartService,
    WishlistService,
    CheckoutService,
    EnhancedCartService,
    CartException,
    InsufficientStockException,
)
from .serializers import (
    CartSerializer, CartItemSerializer, AddToCartSerializer, UpdateCartItemSerializer,
    ApplyCouponSerializer, LockPricesSerializer, ShareCartSerializer,
    CartGiftOptionsSerializer,
    WishlistSerializer, WishlistItemSerializer, AddToWishlistSerializer,
    WishlistShareSerializer, CreateWishlistShareSerializer,
    CheckoutSessionSerializer, CheckoutShippingInfoSerializer,
    CheckoutShippingMethodSerializer, CheckoutPaymentMethodSerializer
)
from apps.i18n.services import CurrencyService, CurrencyConversionService
from apps.shipping.models import ShippingRate
from apps.shipping.services import ShippingZoneService
from apps.accounts.services import AddressService


# =============================================================================
# Permission Classes
# =============================================================================

class IsOwnerOrReadOnly(permissions.BasePermission):
    """Allow owners to edit, others to read."""
    
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        
        if hasattr(obj, 'user'):
            return obj.user == request.user
        return False


# =============================================================================
# Cart ViewSet
# =============================================================================

class CartViewSet(viewsets.ViewSet):
    """ViewSet for cart operations."""
    
    permission_classes = [permissions.AllowAny]
    
    def _get_cart(self, request):
        """Get cart for current user/session."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        if not session_key:
            request.session.create()
            session_key = request.session.session_key
        
        return CartService.get_or_create_cart(user=user, session_key=session_key)
    
    def list(self, request):
        """Get current cart."""
        cart = self._get_cart(request)
        serializer = CartSerializer(cart, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'])
    def add(self, request):
        """Add item to cart."""
        from apps.catalog.models import Product, ProductVariant
        
        serializer = AddToCartSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            product = Product.objects.get(
                id=serializer.validated_data['product_id'],
                is_active=True,
                is_deleted=False
            )
            
            variant = None
            if serializer.validated_data.get('variant_id'):
                variant = ProductVariant.objects.get(id=serializer.validated_data['variant_id'])
            
            cart = self._get_cart(request)
            item = CartService.add_item(
                cart=cart,
                product=product,
                quantity=serializer.validated_data['quantity'],
                variant=variant
            )
            
            return Response({
                'success': True,
                'message': f'{product.name} added to cart',
                'item': CartItemSerializer(item, context={'request': request}).data,
                'cart': CartSerializer(cart, context={'request': request}).data
            }, status=status.HTTP_201_CREATED)
            
        except Product.DoesNotExist:
            return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'success': False,
                'message': str(e),
                'errors': None,
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'], url_path='update/(?P<item_id>[^/.]+)')
    def update_item(self, request, item_id=None):
        """Update cart item quantity."""
        serializer = UpdateCartItemSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    'success': False,
                    'message': 'Invalid quantity.',
                    'errors': serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            cart = self._get_cart(request)
            CartService.update_item_quantity(
                cart=cart,
                item_id=item_id,
                quantity=serializer.validated_data['quantity']
            )

            return Response({
                'success': True,
                'message': 'Cart updated',
                'cart': CartSerializer(cart, context={'request': request}).data
            })
        except InsufficientStockException as e:
            return Response(
                {
                    'success': False,
                    'message': str(e),
                    'errors': {'quantity': [str(e)]},
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        except CartException as e:
            return Response(
                {
                    'success': False,
                    'message': str(e),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            return Response(
                {
                    'success': False,
                    'message': str(e),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
    
    @action(detail=False, methods=['post'], url_path='remove/(?P<item_id>[^/.]+)')
    def remove_item(self, request, item_id=None):
        """Remove item from cart."""
        cart = self._get_cart(request)
        removed = CartService.remove_item(cart, item_id)
        
        if removed:
            return Response({
                'success': True,
                'message': 'Item removed',
                'cart': CartSerializer(cart, context={'request': request}).data
            })
        
        return Response({'error': 'Item not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=False, methods=['post'])
    def clear(self, request):
        """Clear all items from cart."""
        cart = self._get_cart(request)
        CartService.clear_cart(cart)
        
        return Response({
            'success': True,
            'message': 'Cart cleared',
            'cart': CartSerializer(cart, context={'request': request}).data
        })
    
    @action(detail=False, methods=['post'])
    def apply_coupon(self, request):
        """Apply coupon to cart."""
        serializer = ApplyCouponSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid coupon code.',
                'errors': serializer.errors,
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            cart = self._get_cart(request)
            CartService.apply_coupon(cart, serializer.validated_data['coupon_code'])
            
            return Response({
                'success': True,
                'message': 'Coupon applied',
                'cart': CartSerializer(cart, context={'request': request}).data
            })
            
        except Exception as e:
            return Response({
                'success': False,
                'message': str(e),
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'], url_path='gift')
    def gift(self, request):
        """Update cart-level gift options."""
        serializer = CartGiftOptionsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cart = self._get_cart(request)
        if not cart or not cart.items.exists():
            return Response({
                'success': False,
                'message': 'Cart is empty.',
                'error': 'Cart is empty.'
            }, status=status.HTTP_400_BAD_REQUEST)

        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key

        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key,
            request=request
        )

        is_gift = serializer.validated_data.get('is_gift', False)
        gift_message = (serializer.validated_data.get('gift_message') or '').strip()
        gift_wrap = serializer.validated_data.get('gift_wrap', False)

        if not is_gift:
            gift_message = ''
            gift_wrap = False

        gift_wrap_cost = Decimal('0')
        gift_wrap_amount = Decimal('0')
        gift_wrap_label = 'Gift Wrap'
        gift_wrap_enabled = False

        try:
            settings = CartSettings.get_settings()
            gift_wrap_enabled = bool(settings.gift_wrap_enabled)
            gift_wrap_label = settings.gift_wrap_label or gift_wrap_label
            gift_wrap_amount = Decimal(str(settings.gift_wrap_amount or 0))
            if not gift_wrap_enabled:
                gift_wrap = False
            if gift_wrap and gift_wrap_enabled:
                gift_wrap_cost = gift_wrap_amount
        except Exception:
            gift_wrap = False
            gift_wrap_cost = Decimal('0')

        checkout_session.is_gift = is_gift
        checkout_session.gift_message = gift_message
        checkout_session.gift_wrap = gift_wrap
        checkout_session.gift_wrap_cost = gift_wrap_cost
        checkout_session.save(update_fields=[
            'is_gift',
            'gift_message',
            'gift_wrap',
            'gift_wrap_cost',
        ])

        # Keep snapshot totals in sync if available
        try:
            from apps.commerce.views import sync_checkout_snapshot
            sync_checkout_snapshot(request, cart, checkout_session)
        except Exception:
            pass

        from_code = getattr(cart, 'currency', None) or 'BDT'
        target_currency = CurrencyService.get_user_currency(request=request) or CurrencyService.get_default_currency()

        display_gift_wrap_amount = gift_wrap_amount
        display_gift_wrap_cost = gift_wrap_cost

        if target_currency and target_currency.code != from_code:
            try:
                display_gift_wrap_amount = CurrencyConversionService.convert_by_code(
                    gift_wrap_amount, from_code, target_currency.code, round_result=True
                )
                display_gift_wrap_cost = CurrencyConversionService.convert_by_code(
                    gift_wrap_cost, from_code, target_currency.code, round_result=True
                )
            except Exception:
                display_gift_wrap_amount = gift_wrap_amount
                display_gift_wrap_cost = gift_wrap_cost

        formatted_gift_wrap_amount = (
            target_currency.format_amount(display_gift_wrap_amount) if target_currency else str(display_gift_wrap_amount)
        )
        formatted_gift_wrap_cost = (
            target_currency.format_amount(display_gift_wrap_cost) if target_currency else str(display_gift_wrap_cost)
        )

        return Response({
            'success': True,
            'message': 'Gift options updated',
            'gift_state': {
                'is_gift': is_gift,
                'gift_message': gift_message,
                'gift_wrap': gift_wrap,
                'gift_wrap_cost': str(display_gift_wrap_cost),
            },
            'gift_wrap_amount': str(display_gift_wrap_amount),
            'formatted_gift_wrap_amount': formatted_gift_wrap_amount,
            'formatted_gift_wrap_cost': formatted_gift_wrap_cost,
            'gift_wrap_label': gift_wrap_label,
            'gift_wrap_enabled': gift_wrap_enabled,
        })

    @action(detail=False, methods=['post'], url_path='validate')
    def validate_cart(self, request):
        """Validate cart items and totals."""
        cart = self._get_cart(request)
        if not cart or not cart.items.exists():
            return Response({
                'success': False,
                'message': 'Cart is empty.',
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)

        validation = CartService.validate_cart(cart)
        return Response({
            'success': True,
            'message': 'Cart validation completed',
            'data': validation
        })

    @action(detail=False, methods=['post'], url_path='lock-prices')
    def lock_prices(self, request):
        """Lock prices for all cart items."""
        serializer = LockPricesSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cart = self._get_cart(request)
        if not cart or not cart.items.exists():
            return Response({
                'success': False,
                'message': 'Cart is empty.',
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)

        duration = serializer.validated_data.get('duration_hours')
        locked_count = CartService.lock_all_prices(cart, duration_hours=duration)

        return Response({
            'success': True,
            'message': f'Locked prices for {locked_count} item(s)',
            'data': {
                'locked_count': locked_count,
                'cart': CartSerializer(cart, context={'request': request}).data
            }
        })

    @action(detail=False, methods=['post'], url_path='share')
    def share(self, request):
        """Create a share link for the current cart."""
        serializer = ShareCartSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        cart = self._get_cart(request)
        if not cart or not cart.items.exists():
            return Response({
                'success': False,
                'message': 'Cart is empty.',
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)

        share = EnhancedCartService.create_share_link(
            cart=cart,
            name=serializer.validated_data.get('name', ''),
            permission=serializer.validated_data.get('permission'),
            expires_days=serializer.validated_data.get('expires_days'),
            password=serializer.validated_data.get('password') or None,
            created_by=request.user if request.user.is_authenticated else None
        )

        share_url = request.build_absolute_uri(
            reverse('commerce:shared_cart', kwargs={'token': share.share_token})
        )

        return Response({
            'success': True,
            'message': 'Share link created',
            'data': {
                'share_url': share_url,
                'share_token': share.share_token
            }
        })
    
    @action(detail=False, methods=['post'])
    def remove_coupon(self, request):
        """Remove coupon from cart."""
        cart = self._get_cart(request)
        CartService.remove_coupon(cart)
        
        return Response({
            'success': True,
            'message': 'Coupon removed',
            'cart': CartSerializer(cart, context={'request': request}).data
        })
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get cart summary."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key

        cart = self._get_cart(request)
        session_key = request.session.session_key

        checkout_session = None
        if cart and cart.items.exists():
            checkout_session = CheckoutService.get_or_create_session(
                cart=cart,
                user=user,
                session_key=session_key,
                request=request,
            )

        try:
            from apps.commerce.views import build_checkout_cart_summary

            summary = build_checkout_cart_summary(request, cart, checkout_session)
        except Exception:
            summary = CartService.get_cart_summary(cart)

        currency_obj = summary.get('currency') if isinstance(summary, dict) else None
        if currency_obj and hasattr(currency_obj, 'code'):
            summary['currency_code'] = summary.get('currency_code') or currency_obj.code
            summary['currency_symbol'] = (
                summary.get('currency_symbol') or getattr(currency_obj, 'native_symbol', None) or currency_obj.symbol
            )
            summary['currency'] = summary['currency_code']

        return Response(summary)


# =============================================================================
# Wishlist ViewSet
# =============================================================================

class WishlistViewSet(viewsets.ViewSet):
    """ViewSet for wishlist operations."""
    
    permission_classes = [permissions.IsAuthenticated]
    
    def list(self, request):
        """Get current user's wishlist."""
        wishlist = WishlistService.get_or_create_wishlist(request.user)
        items = (
            wishlist.items.select_related('product', 'variant')
            .prefetch_related('product__images')
            .order_by('-priority', '-added_at')
        )
        
        # Apply pagination
        paginator = StandardResultsSetPagination()
        paginated_items = paginator.paginate_queryset(items, request)
        
        if paginated_items is None:
            paginated_items = []
        
        item_serializer = WishlistItemSerializer(paginated_items, many=True, context={'request': request})
        return paginator.get_paginated_response(item_serializer.data)
    
    def create(self, request): # Added create method
        """Add item to wishlist (maps to POST /wishlist/)."""
        from apps.catalog.models import Product, ProductVariant
        
        serializer = AddToWishlistSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            product = Product.objects.get(
                id=serializer.validated_data['product_id'],
                is_active=True,
                is_deleted=False
            )
            
            variant = None
            if serializer.validated_data.get('variant_id'):
                variant = ProductVariant.objects.get(id=serializer.validated_data['variant_id'])
            
            wishlist = WishlistService.get_or_create_wishlist(request.user)
            item = WishlistService.add_item(
                wishlist=wishlist,
                product=product,
                variant=variant,
                notes=serializer.validated_data.get('notes', '')
            )
            
            return Response({
                'success': True,
                'message': f'{product.name} added to wishlist',
                'item': WishlistItemSerializer(item, context={'request': request}).data,
                'wishlist': WishlistSerializer(wishlist, context={'request': request}).data
            }, status=status.HTTP_201_CREATED)
            
        except Product.DoesNotExist:
            return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
        # Removed the broad 'except Exception as e' to allow DRF's validation errors to propagate
    
    @action(detail=False, methods=['post'])
    def add(self, request):
        """Add item to wishlist (alternative method, can be removed if create is sufficient)."""
        # This method duplicates the create logic. It can be kept for backward compatibility
        # or removed if all clients can switch to POST /wishlist/.
        return self.create(request)
    
    @action(detail=False, methods=['post'], url_path='remove/(?P<item_id>[^/.]+)')
    def remove_item(self, request, item_id=None):
        """Remove item from wishlist."""
        wishlist = WishlistService.get_or_create_wishlist(request.user)
        removed = WishlistService.remove_item(wishlist, item_id)
        
        if removed:
            return Response({
                'success': True,
                'message': 'Item removed',
                'wishlist': WishlistSerializer(wishlist, context={'request': request}).data
            })
        
        return Response({'error': 'Item not found'}, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=False, methods=['post'], url_path='move-to-cart/(?P<item_id>[^/.]+)')
    def move_to_cart(self, request, item_id=None):
        """Move wishlist item to cart."""
        try:
            wishlist = WishlistService.get_or_create_wishlist(request.user)
            item = wishlist.items.get(id=item_id)
            
            if not request.session.session_key:
                request.session.create()
            
            cart = CartService.get_or_create_cart(
                user=request.user,
                session_key=request.session.session_key
            )
            
            cart_item = WishlistService.move_to_cart(item, cart)
            
            return Response({
                'success': True,
                'message': 'Item moved to cart',
                'wishlist': WishlistSerializer(wishlist, context={'request': request}).data,
                'cart': CartSerializer(cart, context={'request': request}).data
            })
            
        except WishlistItem.DoesNotExist:
            return Response({'error': 'Item not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def share(self, request):
        """Create shareable wishlist link."""
        serializer = CreateWishlistShareSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        wishlist = WishlistService.get_or_create_wishlist(request.user)
        share = WishlistService.create_share_link(
            wishlist,
            expires_days=serializer.validated_data['expires_days'],
            allow_purchase=serializer.validated_data['allow_purchase']
        )
        
        return Response({
            'success': True,
            'share': WishlistShareSerializer(share, context={'request': request}).data
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=False, methods=['get'])
    def price_drops(self, request):
        """Get items with price drops."""
        wishlist = WishlistService.get_or_create_wishlist(request.user)
        items = WishlistService.get_items_with_price_drops(wishlist)
        
        return Response({
            'items': WishlistItemSerializer(items, many=True, context={'request': request}).data
        })
    
    @action(detail=False, methods=['post'])
    def toggle(self, request):
        """Toggle product in wishlist."""
        from apps.catalog.models import Product
        
        product_id = request.data.get('product_id')
        
        try:
            product = Product.objects.get(id=product_id)
            wishlist = WishlistService.get_or_create_wishlist(request.user)
            
            existing = wishlist.items.filter(product=product).first()
            
            if existing:
                WishlistService.remove_item(wishlist, existing.id)
                return Response({
                    'success': True,
                    'in_wishlist': False,
                    'message': 'Removed from wishlist'
                })
            else:
                WishlistService.add_item(wishlist, product)
                return Response({
                    'success': True,
                    'in_wishlist': True,
                    'message': 'Added to wishlist'
                })
                
        except Product.DoesNotExist:
            return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)


class SharedWishlistView(APIView):
    """View a shared wishlist."""
    
    permission_classes = [permissions.AllowAny]
    
    def get(self, request, token):
        """Get shared wishlist by token."""
        wishlist = WishlistService.get_shared_wishlist(token)
        
        if not wishlist:
            return Response({'error': 'Wishlist not found or expired'}, status=status.HTTP_404_NOT_FOUND)
        
        return Response({
            'wishlist': WishlistSerializer(wishlist, context={'request': request}).data
        })


# =============================================================================
# Checkout ViewSet
# =============================================================================

class CheckoutViewSet(viewsets.ViewSet):
    """ViewSet for checkout operations."""
    
    permission_classes = [permissions.AllowAny]
    
    def _get_checkout_session(self, request):
        """Get or create checkout session."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        if not session_key:
            request.session.create()
            session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart or not cart.items.exists():
            return None
        
        return CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key,
            request=request
        )
    
    def list(self, request):
        """Get current checkout session."""
        checkout_session = self._get_checkout_session(request)
        
        if not checkout_session:
            return Response({
                'success': False,
                'message': 'No active cart.',
                'error': 'No active cart.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(
            CheckoutSessionSerializer(checkout_session, context={'request': request}).data
        )
    
    @action(detail=False, methods=['post'])
    def shipping_info(self, request):
        """Update shipping information."""
        checkout_session = self._get_checkout_session(request)
        
        if not checkout_session:
            return Response({
                'success': False,
                'message': 'No active cart.',
                'error': 'No active cart.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = CheckoutShippingInfoSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        save_address = data.pop('save_address', False)
        address_saved = False
        address_save_error = None

        CheckoutService.update_shipping_info(checkout_session, data)

        if save_address and request.user and request.user.is_authenticated:
            full_name = f"{data.get('shipping_first_name', '').strip()} {data.get('shipping_last_name', '').strip()}".strip()
            if not full_name:
                full_name = request.user.get_full_name() or request.user.email

            address_payload = {
                'address_type': 'shipping',
                'full_name': full_name,
                'phone': data.get('shipping_phone', ''),
                'address_line_1': data.get('shipping_address_line_1', ''),
                'address_line_2': data.get('shipping_address_line_2', ''),
                'city': data.get('shipping_city', ''),
                'state': data.get('shipping_state', ''),
                'postal_code': data.get('shipping_postal_code', ''),
                'country': data.get('shipping_country', ''),
            }

            has_existing = request.user.addresses.filter(is_deleted=False).exists()
            if not has_existing:
                address_payload['is_default'] = True

            try:
                address = AddressService.create_address(user=request.user, **address_payload)
                checkout_session.saved_shipping_address = address
                checkout_session.save(update_fields=['saved_shipping_address'])
                address_saved = True
            except ValidationError as exc:
                address_saved = False
                address_save_error = exc.messages[0] if hasattr(exc, 'messages') and exc.messages else str(exc)
            except Exception as exc:
                address_saved = False
                address_save_error = str(exc)
        
        return Response({
            'success': True,
            'message': 'Shipping information saved',
            'checkout': CheckoutSessionSerializer(checkout_session, context={'request': request}).data,
            'address_saved': address_saved if save_address else None,
            'address_save_error': address_save_error if save_address else None,
        })
    
    @action(detail=False, methods=['post'])
    def shipping_method(self, request):
        """Select shipping method."""
        checkout_session = self._get_checkout_session(request)
        
        if not checkout_session:
            return Response({
                'success': False,
                'message': 'No active cart.',
                'error': 'No active cart.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = CheckoutShippingMethodSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        shipping_type = data.get('shipping_type') or 'delivery'
        delivery_instructions = (data.get('delivery_instructions') or '').strip()

        if delivery_instructions:
            checkout_session.delivery_instructions = delivery_instructions
            checkout_session.save(update_fields=['delivery_instructions'])

        if shipping_type == 'pickup':
            pickup_location_id = data.get('pickup_location_id')
            if not pickup_location_id:
                return Response({
                    'success': False,
                    'message': 'Select a pickup location to continue.',
                    'error': 'Select a pickup location to continue.'
                }, status=status.HTTP_400_BAD_REQUEST)

            location = None
            try:
                from apps.contacts.models import StoreLocation

                location = StoreLocation.objects.filter(
                    pk=pickup_location_id,
                    is_active=True,
                    is_pickup_location=True
                ).first()
            except Exception:
                location = None

            if not location:
                return Response({
                    'success': False,
                    'message': 'Pickup location is unavailable. Please choose another.',
                    'error': 'Pickup location is unavailable. Please choose another.'
                }, status=status.HTTP_400_BAD_REQUEST)

            checkout_session.pickup_location = location
            checkout_session.shipping_rate = None
            CheckoutService.select_shipping_method(
                checkout_session, CheckoutSession.SHIPPING_PICKUP
            )
            try:
                from decimal import Decimal

                checkout_session.shipping_cost = Decimal(str(location.pickup_fee or 0))
                checkout_session.save(update_fields=['pickup_location', 'shipping_cost'])
                checkout_session.calculate_totals()
            except Exception:
                pass
            try:
                from apps.commerce.views import sync_checkout_snapshot
                sync_checkout_snapshot(request, checkout_session.cart, checkout_session)
            except Exception:
                pass

            return Response({
                'success': True,
                'message': 'Pickup selected',
                'checkout': CheckoutSessionSerializer(checkout_session, context={'request': request}).data
            })

        rate_id = data.get('shipping_rate_id') or data.get('shipping_method')
        rate = None
        if rate_id:
            try:
                rate = ShippingRate.objects.select_related('method').get(pk=rate_id)
            except (ShippingRate.DoesNotExist, ValueError, TypeError):
                rate = None

        country_value = (checkout_session.shipping_country or '').strip()
        country_code = CheckoutService.normalize_country_code(country_value) or country_value
        state = (checkout_session.shipping_state or '').strip() or None
        city = (checkout_session.shipping_city or '').strip() or None
        postal_code = (checkout_session.shipping_postal_code or '').strip() or None

        if not rate and rate_id:
            zone = ShippingZoneService.find_zone_for_location(
                country_code or None,
                state,
                postal_code,
                city=city
            )
            if zone:
                rate = ShippingRate.objects.select_related('method').filter(
                    zone=zone,
                    method_id=rate_id,
                    is_active=True,
                    method__is_active=True
                ).first()

            if not rate and zone:
                rate = ShippingRate.objects.select_related('method').filter(
                    zone=zone,
                    method__code=rate_id,
                    is_active=True,
                    method__is_active=True
                ).first()

        if rate:
            method_code = rate.method.code or rate.method.name or checkout_session.shipping_method
            CheckoutService.select_shipping_method(checkout_session, method_code, shipping_rate=rate)
        else:
            method = data.get('shipping_method')
            if not method:
                return Response({
                    'success': False,
                    'message': 'Please select a shipping method.',
                    'error': 'Please select a shipping method.'
                }, status=status.HTTP_400_BAD_REQUEST)
            CheckoutService.select_shipping_method(checkout_session, method)

        try:
            from apps.commerce.views import sync_checkout_snapshot
            sync_checkout_snapshot(request, checkout_session.cart, checkout_session)
        except Exception:
            pass

        return Response({
            'success': True,
            'message': 'Shipping method selected',
            'checkout': CheckoutSessionSerializer(checkout_session, context={'request': request}).data
        })
    
    @action(detail=False, methods=['post'])
    def payment_method(self, request):
        """Select payment method."""
        checkout_session = self._get_checkout_session(request)
        
        if not checkout_session:
            return Response({
                'success': False,
                'message': 'No active cart.',
                'error': 'No active cart.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = CheckoutPaymentMethodSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        CheckoutService.select_payment_method(checkout_session, data['payment_method'])

        billing_same = data.get('billing_same_as_shipping')
        if billing_same is not None:
            checkout_session.billing_same_as_shipping = bool(billing_same)

        if not checkout_session.billing_same_as_shipping:
            checkout_session.billing_first_name = data.get('billing_first_name', '')
            checkout_session.billing_last_name = data.get('billing_last_name', '')
            checkout_session.billing_company = data.get('billing_company', '')
            checkout_session.billing_address_line_1 = data.get('billing_address_line_1', '')
            checkout_session.billing_address_line_2 = data.get('billing_address_line_2', '')
            checkout_session.billing_city = data.get('billing_city', '')
            checkout_session.billing_state = data.get('billing_state', '')
            checkout_session.billing_postal_code = data.get('billing_postal_code', '')
            checkout_session.billing_country = data.get('billing_country', '')
        elif billing_same is True:
            checkout_session.billing_first_name = ''
            checkout_session.billing_last_name = ''
            checkout_session.billing_company = ''
            checkout_session.billing_address_line_1 = ''
            checkout_session.billing_address_line_2 = ''
            checkout_session.billing_city = ''
            checkout_session.billing_state = ''
            checkout_session.billing_postal_code = ''
            checkout_session.billing_country = ''

        checkout_session.save()
  
        # Sync fee/currency snapshot for payment selection
        try:
            from apps.commerce.views import sync_checkout_snapshot
            sync_checkout_snapshot(request, checkout_session.cart, checkout_session)
        except Exception:
            pass
        
        return Response({
            'success': True,
            'message': 'Payment method selected',
            'checkout': CheckoutSessionSerializer(checkout_session, context={'request': request}).data
        })
    
    @action(detail=False, methods=['post'])
    def complete(self, request):
        """Complete checkout and place order."""
        checkout_session = self._get_checkout_session(request)
        
        if not checkout_session:
            return Response({
                'success': False,
                'message': 'No active cart.',
                'error': 'No active cart.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            order_notes = (request.data.get('order_notes') or '').strip() if hasattr(request, 'data') else ''
            if order_notes:
                checkout_session.order_notes = order_notes

            terms_accepted = request.data.get('terms_accepted') if hasattr(request, 'data') else None
            if terms_accepted is not None:
                accepted = str(terms_accepted).lower() in {'1', 'true', 'yes', 'on'}
                if not accepted:
                    return Response({
                        'success': False,
                        'message': 'Please accept the terms and conditions.',
                        'error': 'Please accept the terms and conditions.'
                    }, status=status.HTTP_400_BAD_REQUEST)

            if order_notes:
                checkout_session.save(update_fields=['order_notes'])

            order = CheckoutService.complete_checkout(checkout_session)

            payment_redirect_url = None
            payment_instructions = None

            try:
                from apps.payments.models import PaymentGateway
                gateway = PaymentGateway.objects.filter(
                    code=checkout_session.payment_method,
                    is_active=True
                ).first()

                if gateway and gateway.code not in {PaymentGateway.CODE_COD, 'cod'}:
                    payment_instructions = gateway.instructions or None

                    origin = request.headers.get('Origin') or request.headers.get('Referer')
                    base_url = getattr(settings, 'SITE_URL', '').rstrip('/')
                    if origin:
                        try:
                            parsed = urlparse(origin)
                            if parsed.scheme and parsed.netloc:
                                base_url = f"{parsed.scheme}://{parsed.netloc}"
                        except Exception:
                            pass

                    if not base_url:
                        try:
                            base_url = request.build_absolute_uri('/').rstrip('/')
                        except Exception:
                            base_url = ''
                    base_url = base_url.rstrip('/')
                    success_url = f"{base_url}/checkout/success?order_id={order.id}&order_number={order.order_number}"
                    fail_url = f"{base_url}/checkout?payment=failed&order_id={order.id}&order_number={order.order_number}"
                    cancel_url = f"{base_url}/checkout?payment=cancelled&order_id={order.id}&order_number={order.order_number}"
                    ipn_url = request.build_absolute_uri(reverse('payments:gateway-ipn'))

                    if gateway.code == 'sslcommerz' or gateway.ssl_store_id:
                        from apps.payments.bangladesh_gateways import SSLCommerzService
                        result = SSLCommerzService(gateway=gateway).init_transaction(
                            order,
                            success_url=success_url,
                            fail_url=fail_url,
                            cancel_url=cancel_url,
                            ipn_url=ipn_url,
                        )
                        payment_redirect_url = result.get('redirect_url')
                    elif gateway.code == 'bkash':
                        from apps.payments.bangladesh_gateways import BkashService
                        result = BkashService(gateway=gateway).create_payment(
                            order,
                            callback_url=success_url,
                        )
                        payment_redirect_url = result.get('bkash_url')
                    elif gateway.code == 'nagad':
                        from apps.payments.bangladesh_gateways import NagadService
                        result = NagadService(gateway=gateway).init_payment(
                            order,
                            callback_url=success_url,
                        )
                        payment_redirect_url = (
                            result.get('redirect_url') or result.get('payment_url')
                        )
            except Exception:
                payment_redirect_url = None
                payment_instructions = payment_instructions or None
            
            # Return basic order info - clients should use orders API for full details
            return Response({
                'success': True,
                'message': 'Order placed successfully',
                'order_number': order.order_number,
                'order_id': str(order.id),
                'total': str(order.total),
                'payment_status': order.payment_status,
                'payment_redirect_url': payment_redirect_url,
                'payment_instructions': payment_instructions,
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response({
                'success': False,
                'message': str(e),
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
