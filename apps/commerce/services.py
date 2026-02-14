"""
Commerce services - Business logic for cart, checkout, and wishlist
"""
import logging
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta

from django.db import transaction
from django.conf import settings
from django.utils import timezone
from django.core.exceptions import ValidationError

from .models import (
    Cart, CartItem, CartSettings,
    Wishlist, WishlistItem, WishlistShare,
    CheckoutSession, CheckoutEvent,
    SavedForLater, SharedCart, SessionWishlist, SessionWishlistItem,
    WishlistCollection, WishlistCollectionItem, PriceAlert,
    AbandonedCart, ItemReservation, CartAnalytics
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class CommerceException(Exception):
    """Base exception for commerce operations."""
    pass


class CartException(CommerceException):
    """Cart-related exception."""
    pass


class InsufficientStockException(CartException):
    """Raised when not enough stock available."""
    pass


class CheckoutException(CommerceException):
    """Checkout-related exception."""
    pass


# =============================================================================
# Cart Services
# =============================================================================

class CartService:
    """Service class for cart operations."""
    
    @classmethod
    def get_or_create_cart(cls, user=None, session_key=None) -> Cart:
        """Get or create a cart for user or session."""
        if user and user.is_authenticated:
            carts = Cart.objects.filter(user=user).order_by('-updated_at', '-created_at')
            cart = carts.first()
            if not cart:
                cart = Cart.objects.create(user=user)
            elif carts.count() > 1:
                duplicates = carts.exclude(pk=cart.pk)
                logger.warning(
                    "Found %s duplicate carts for user_id=%s; merging into cart_id=%s",
                    duplicates.count(),
                    getattr(user, "id", None),
                    cart.id,
                )
                for duplicate in duplicates:
                    cart.merge_from_session(duplicate)
            
            # Merge session cart if exists
            if session_key:
                session_carts = Cart.objects.filter(
                    session_key=session_key,
                    user__isnull=True,
                ).order_by('-updated_at', '-created_at')
                for session_cart in session_carts:
                    cart.merge_from_session(session_cart)
            
            return cart
        elif session_key:
            carts = Cart.objects.filter(
                session_key=session_key,
                user__isnull=True,
            ).order_by('-updated_at', '-created_at')
            cart = carts.first()
            if not cart:
                cart = Cart.objects.create(session_key=session_key)
            elif carts.count() > 1:
                duplicates = carts.exclude(pk=cart.pk)
                logger.warning(
                    "Found %s duplicate guest carts for session_key=%s; merging into cart_id=%s",
                    duplicates.count(),
                    session_key,
                    cart.id,
                )
                for duplicate in duplicates:
                    cart.merge_from_session(duplicate)
            return cart
        else:
            raise CartException("Either user or session_key is required.")
    
    @classmethod
    def get_cart(cls, user=None, session_key=None) -> Optional[Cart]:
        """Get existing cart without creating."""
        if user and user.is_authenticated:
            return Cart.objects.filter(user=user).order_by('-updated_at', '-created_at').first()
        elif session_key:
            return Cart.objects.filter(
                session_key=session_key,
                user__isnull=True,
            ).order_by('-updated_at', '-created_at').first()
        return None
    
    @classmethod
    @transaction.atomic
    def add_item(cls, cart: Cart, product, quantity=1, variant=None) -> CartItem:
        """Add an item to the cart."""
        # Validate product
        if not product.is_active or product.is_deleted:
            raise CartException("This product is not available.")
        
        # Check stock
        available = variant.stock_quantity if variant else product.stock_quantity
        if not product.allow_backorder and available < quantity:
            raise InsufficientStockException(f"Only {available} items available.")
        
        # Get or create cart item
        item, created = CartItem.objects.get_or_create(
            cart=cart,
            product=product,
            variant=variant,
            defaults={
                'quantity': quantity,
                'price_at_add': variant.current_price if variant else product.current_price
            }
        )
        
        if not created:
            new_quantity = item.quantity + quantity
            
            # Check stock for new quantity
            if not product.allow_backorder and available < new_quantity:
                raise InsufficientStockException(f"Cannot add more. Only {available} items available.")
            
            item.quantity = new_quantity
            item.save()
        
        # Update product cart count
        product.increment_cart(1)
        
        return item
    
    @classmethod
    @transaction.atomic
    def update_item_quantity(cls, cart: Cart, item_id, quantity: int) -> CartItem:
        """Update cart item quantity."""
        try:
            item = cart.items.get(pk=item_id)
        except CartItem.DoesNotExist:
            raise CartException("Item not found in cart.")
        
        if quantity <= 0:
            return cls.remove_item(cart, item_id)
        
        # Check stock
        available = item.variant.stock_quantity if item.variant else item.product.stock_quantity
        if not item.product.allow_backorder and available < quantity:
            raise InsufficientStockException(f"Only {available} items available.")
        
        item.quantity = quantity
        item.save()
        
        return item
    
    @classmethod
    @transaction.atomic
    def remove_item(cls, cart: Cart, item_id) -> bool:
        """Remove item from cart."""
        try:
            item = cart.items.get(pk=item_id)
            product = item.product
            item.delete()
            
            # Update product cart count
            product.increment_cart(-1)
            
            return True
        except CartItem.DoesNotExist:
            return False
    
    @classmethod
    def clear_cart(cls, cart: Cart):
        """Remove all items from cart."""
        for item in cart.items.all():
            item.product.increment_cart(-1)
        cart.clear()
    
    @classmethod
    def apply_coupon(cls, cart: Cart, coupon_code: str) -> bool:
        """Apply a coupon to the cart."""
        from apps.promotions.services import CouponService

        user = cart.user if cart.user and getattr(cart.user, 'is_authenticated', False) else None
        coupon, is_valid, message = CouponService.validate_coupon(
            code=coupon_code,
            user=user,
            subtotal=cart.subtotal
        )

        if not coupon or not is_valid:
            raise CartException(message or "Invalid coupon code.")

        cart.coupon = coupon
        cart.save()
        return True
    
    @classmethod
    def remove_coupon(cls, cart: Cart):
        """Remove coupon from cart."""
        cart.coupon = None
        cart.save()

    @classmethod
    def validate_cart(cls, cart: Cart) -> Dict[str, Any]:
        """
        Validate cart items for availability, stock, and pricing.
        Delegates to ComprehensiveCartService for the full validation ruleset.
        """
        return ComprehensiveCartService.validate_cart(cart)

    @classmethod
    def lock_all_prices(cls, cart: Cart, duration_hours: int = 24) -> int:
        """
        Lock prices for all cart items for a given duration.
        Delegates to ComprehensiveCartService implementation.
        """
        return ComprehensiveCartService.lock_all_prices(cart, duration_hours=duration_hours)
    
    @classmethod
    def get_cart_summary(cls, cart: Cart) -> Dict[str, Any]:
        """Get cart summary with all details."""
        items = []
        for item in cart.items.select_related('product', 'variant').prefetch_related('product__images'):
            primary_image = item.product.images.filter(is_primary=True).first() or item.product.images.first()
            items.append({
                'id': str(item.id),
                'product_id': str(item.product.id),
                'product_name': item.product.name,
                'product_slug': item.product.slug,
                'variant_id': str(item.variant.id) if item.variant else None,
                'variant_name': str(item.variant) if item.variant else None,
                'quantity': item.quantity,
                'unit_price': str(item.unit_price),
                'total': str(item.total),
                'image': primary_image.image.url if primary_image else None,
                'in_stock': item.product.is_in_stock(),
            })
        
        return {
            'id': str(cart.id),
            'items': items,
            'item_count': cart.item_count,
            'subtotal': str(cart.subtotal),
            'discount_amount': str(cart.discount_amount),
            'total': str(cart.total),
            'coupon_code': cart.coupon.code if cart.coupon else None,
            'currency': cart.currency,
        }


# =============================================================================
# Wishlist Services
# =============================================================================

class WishlistService:
    """Service class for wishlist operations."""
    
    @classmethod
    def get_or_create_wishlist(cls, user) -> Wishlist:
        """Get or create a wishlist for a user."""
        wishlist, created = Wishlist.objects.get_or_create(user=user)
        return wishlist
    
    @classmethod
    @transaction.atomic
    def add_item(cls, wishlist: Wishlist, product, variant=None, notes='') -> WishlistItem:
        """Add an item to the wishlist."""
        item, created = WishlistItem.objects.get_or_create(
            wishlist=wishlist,
            product=product,
            variant=variant,
            defaults={
                'notes': notes,
                'price_at_add': product.current_price
            }
        )
        
        if created:
            # Update product wishlist count
            product.increment_wishlist(1)
        
        return item
    
    @classmethod
    @transaction.atomic
    def remove_item(cls, wishlist: Wishlist, item_id) -> bool:
        """Remove item from wishlist."""
        try:
            item = wishlist.items.get(pk=item_id)
            product = item.product
            item.delete()
            
            # Update product wishlist count
            product.increment_wishlist(-1)
            
            return True
        except WishlistItem.DoesNotExist:
            return False
    
    @classmethod
    def move_to_cart(cls, wishlist_item: WishlistItem, cart: Cart) -> CartItem:
        """Move a wishlist item to cart."""
        cart_item = CartService.add_item(
            cart=cart,
            product=wishlist_item.product,
            variant=wishlist_item.variant,
            quantity=1
        )
        cls.remove_item(wishlist_item.wishlist, wishlist_item.id)
        return cart_item
    
    @classmethod
    def create_share_link(cls, wishlist: Wishlist, expires_days=30, allow_purchase=False) -> WishlistShare:
        """Create a shareable link for the wishlist."""
        import secrets
        
        expires_at = timezone.now() + timedelta(days=expires_days) if expires_days else None
        
        share = WishlistShare.objects.create(
            wishlist=wishlist,
            share_token=secrets.token_urlsafe(32),
            expires_at=expires_at,
            allow_purchase=allow_purchase
        )
        
        return share
    
    @classmethod
    def get_shared_wishlist(cls, share_token: str) -> Optional[Wishlist]:
        """Get a wishlist by its share token."""
        try:
            share = WishlistShare.objects.get(share_token=share_token)
            if share.is_valid:
                share.view_count += 1
                share.last_viewed_at = timezone.now()
                share.save(update_fields=['view_count', 'last_viewed_at'])
                return share.wishlist
        except WishlistShare.DoesNotExist:
            pass
        return None
    
    @classmethod
    def get_items_with_price_drops(cls, wishlist: Wishlist) -> List[WishlistItem]:
        """Get wishlist items that have dropped in price."""
        return [
            item for item in wishlist.items.select_related('product')
            if item.price_change < 0
        ]


# =============================================================================
# Checkout Services
# =============================================================================

class CheckoutService:
    """Service class for checkout operations."""
    
    DEFAULT_SHIPPING_COSTS = {
        CheckoutSession.SHIPPING_STANDARD: Decimal('60.00'),
        CheckoutSession.SHIPPING_EXPRESS: Decimal('120.00'),
        CheckoutSession.SHIPPING_OVERNIGHT: Decimal('200.00'),
        CheckoutSession.SHIPPING_PICKUP: Decimal('0.00'),
        CheckoutSession.SHIPPING_FREE: Decimal('0.00'),
    }

    @staticmethod
    def normalize_country_code(value: str) -> str:
        """Normalize country input to ISO alpha-2 code when possible."""
        code = (value or '').strip()
        if not code:
            return ''
        if len(code) == 2 and code.isalpha():
            return code.upper()

        try:
            from apps.i18n.models import Country
            country = (
                Country.objects.filter(code__iexact=code).first()
                or Country.objects.filter(code_alpha3__iexact=code).first()
                or Country.objects.filter(name__iexact=code).first()
                or Country.objects.filter(native_name__iexact=code).first()
            )
            if country:
                return country.code
        except Exception:
            pass

        return code
    
    @classmethod
    def get_or_create_session(cls, cart: Cart, user=None, session_key=None, request=None) -> CheckoutSession:
        """Get or create checkout session for cart."""
        active_steps = [
            CheckoutSession.STEP_CART,
            CheckoutSession.STEP_INFORMATION,
            CheckoutSession.STEP_SHIPPING,
            CheckoutSession.STEP_PAYMENT,
            CheckoutSession.STEP_REVIEW,
        ]
        
        filters = {'cart': cart, 'current_step__in': active_steps}
        
        if user:
            filters['user'] = user
        else:
            filters['session_key'] = session_key
            filters['user__isnull'] = True
        
        checkout_session = CheckoutSession.objects.filter(**filters).first()
        
        if checkout_session:
            if checkout_session.is_expired:
                checkout_session.current_step = CheckoutSession.STEP_ABANDONED
                checkout_session.save()
                checkout_session = None
            else:
                checkout_session.extend_expiry(hours=48)
                return checkout_session
        
        # Create new session
        checkout_session = CheckoutSession.objects.create(
            user=user,
            session_key=session_key if not user else None,
            cart=cart,
            expires_at=timezone.now() + timedelta(hours=48)
        )
        
        # Pre-fill from user profile
        if user:
            cls._prefill_from_user(checkout_session, user)
        
        # Capture analytics
        if request:
            cls._capture_analytics(checkout_session, request)
        
        # Log event
        cls.log_event(checkout_session, CheckoutEvent.EVENT_STARTED, {
            'cart_id': str(cart.id),
            'item_count': cart.item_count
        })
        
        return checkout_session

    @classmethod
    def get_active_session(cls, cart: Cart, user=None, session_key=None) -> Optional[CheckoutSession]:
        """Get active checkout session without creating a new one."""
        if not cart:
            return None

        active_steps = [
            CheckoutSession.STEP_CART,
            CheckoutSession.STEP_INFORMATION,
            CheckoutSession.STEP_SHIPPING,
            CheckoutSession.STEP_PAYMENT,
            CheckoutSession.STEP_REVIEW,
        ]

        filters = {'cart': cart, 'current_step__in': active_steps}

        if user:
            filters['user'] = user
        else:
            filters['session_key'] = session_key
            filters['user__isnull'] = True

        return CheckoutSession.objects.filter(**filters).first()
    
    @classmethod
    def _prefill_from_user(cls, checkout_session: CheckoutSession, user):
        """Pre-fill checkout from user profile."""
        checkout_session.email = user.email
        checkout_session.shipping_first_name = user.first_name
        checkout_session.shipping_last_name = user.last_name
        checkout_session.shipping_email = user.email
        checkout_session.shipping_phone = getattr(user, 'phone', '') or ''
        
        # Try to get default shipping address
        try:
            default_address = user.addresses.filter(
                is_default=True,
                address_type__in=['shipping', 'both'],
                is_deleted=False
            ).first()
            
            if default_address:
                checkout_session.shipping_address_line_1 = default_address.address_line_1
                checkout_session.shipping_address_line_2 = default_address.address_line_2 or ''
                checkout_session.shipping_city = default_address.city
                checkout_session.shipping_state = default_address.state or ''
                checkout_session.shipping_postal_code = default_address.postal_code
                checkout_session.shipping_country = default_address.country or ''
                checkout_session.saved_shipping_address = default_address
        except Exception:
            pass
        
        checkout_session.save()
    
    @classmethod
    def _capture_analytics(cls, checkout_session: CheckoutSession, request):
        """Capture analytics data from request."""
        checkout_session.user_agent = request.META.get('HTTP_USER_AGENT', '')[:500]
        checkout_session.ip_address = cls._get_client_ip(request)
        checkout_session.referrer = request.META.get('HTTP_REFERER', '')[:200]
        
        # UTM parameters
        checkout_session.utm_source = request.GET.get('utm_source', '')[:100]
        checkout_session.utm_medium = request.GET.get('utm_medium', '')[:100]
        checkout_session.utm_campaign = request.GET.get('utm_campaign', '')[:100]
        
        checkout_session.save()
    
    @classmethod
    def _get_client_ip(cls, request):
        """Get client IP from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')
    
    @classmethod
    def log_event(cls, checkout_session: CheckoutSession, event_type: str, data: dict = None):
        """Log a checkout event."""
        CheckoutEvent.objects.create(
            checkout_session=checkout_session,
            event_type=event_type,
            data=data
        )
    
    @classmethod
    @transaction.atomic
    def update_shipping_info(cls, checkout_session: CheckoutSession, data: dict) -> CheckoutSession:
        """Update shipping information."""
        shipping_fields = [
            'shipping_first_name', 'shipping_last_name', 'shipping_company',
            'shipping_email', 'shipping_phone', 'shipping_address_line_1',
            'shipping_address_line_2', 'shipping_city', 'shipping_state',
            'shipping_postal_code', 'shipping_country'
        ]
        
        for field in shipping_fields:
            if field in data:
                setattr(checkout_session, field, data[field])
        
        if data.get('email'):
            checkout_session.email = data['email']
        
        checkout_session.current_step = CheckoutSession.STEP_SHIPPING
        checkout_session.save()
        
        cls.log_event(checkout_session, CheckoutEvent.EVENT_INFO_SUBMITTED)
        
        return checkout_session
    
    @classmethod
    @transaction.atomic
    def select_shipping_method(cls, checkout_session: CheckoutSession, method: str, shipping_rate=None) -> CheckoutSession:
        """Select shipping method."""
        checkout_session.shipping_method = method
        
        if shipping_rate:
            checkout_session.shipping_rate = shipping_rate
            try:
                cart = checkout_session.cart
                subtotal = Decimal(str(cart.subtotal or 0)) if cart else Decimal('0')
                item_count = cart.item_count if cart else 1
                checkout_session.shipping_cost = shipping_rate.calculate_rate(
                    subtotal=subtotal,
                    weight=Decimal('0'),
                    item_count=item_count
                )
            except Exception:
                checkout_session.shipping_cost = getattr(shipping_rate, 'base_rate', Decimal('0.00'))
        else:
            checkout_session.shipping_cost = cls.DEFAULT_SHIPPING_COSTS.get(method, Decimal('60.00'))
        
        checkout_session.current_step = CheckoutSession.STEP_PAYMENT
        checkout_session.calculate_totals()
        checkout_session.save()
        
        cls.log_event(checkout_session, CheckoutEvent.EVENT_SHIPPING_SELECTED, {
            'method': method,
            'cost': str(checkout_session.shipping_cost)
        })
        
        return checkout_session
    
    @classmethod
    @transaction.atomic
    def select_payment_method(cls, checkout_session: CheckoutSession, method: str) -> CheckoutSession:
        """Select payment method."""
        checkout_session.payment_method = method
        checkout_session.current_step = CheckoutSession.STEP_REVIEW
        checkout_session.save()
        
        cls.log_event(checkout_session, CheckoutEvent.EVENT_PAYMENT_INITIATED, {'method': method})
        
        return checkout_session
    
    @classmethod
    @transaction.atomic
    def complete_checkout(cls, checkout_session: CheckoutSession):
        """Complete checkout and create order using orders app."""
        from apps.orders.services import OrderService
        
        if checkout_session.current_step == CheckoutSession.STEP_COMPLETED:
            raise CheckoutException("Checkout already completed")
        
        # Validate cart items
        cart = checkout_session.cart
        if not cart.items.exists():
            raise CheckoutException("Cart is empty")
        
        # Create order using orders app
        order = OrderService.create_order_from_checkout(checkout_session)
        
        # Update checkout session
        checkout_session.current_step = CheckoutSession.STEP_COMPLETED
        checkout_session.completed_at = timezone.now()
        checkout_session.save()
        
        # Clear cart
        CartService.clear_cart(cart)

        # Reset checkout session after completion
        cls.reset_checkout_session(checkout_session)
        
        # Log event
        cls.log_event(checkout_session, CheckoutEvent.EVENT_ORDER_CREATED, {
            'order_id': str(order.id),
            'order_number': order.order_number
        })
        
        return order

    @staticmethod
    def reset_checkout_session(checkout_session: CheckoutSession):
        """Clear sensitive checkout data after order placement."""
        if not checkout_session:
            return
        checkout_session.email = ''
        checkout_session.shipping_first_name = ''
        checkout_session.shipping_last_name = ''
        checkout_session.shipping_company = ''
        checkout_session.shipping_email = ''
        checkout_session.shipping_phone = ''
        checkout_session.shipping_address_line_1 = ''
        checkout_session.shipping_address_line_2 = ''
        checkout_session.shipping_city = ''
        checkout_session.shipping_state = ''
        checkout_session.shipping_postal_code = ''
        checkout_session.shipping_country = ''
        checkout_session.saved_shipping_address = None

        checkout_session.billing_same_as_shipping = True
        checkout_session.billing_first_name = ''
        checkout_session.billing_last_name = ''
        checkout_session.billing_company = ''
        checkout_session.billing_address_line_1 = ''
        checkout_session.billing_address_line_2 = ''
        checkout_session.billing_city = ''
        checkout_session.billing_state = ''
        checkout_session.billing_postal_code = ''
        checkout_session.billing_country = ''
        checkout_session.saved_billing_address = None

        checkout_session.shipping_method = CheckoutSession.SHIPPING_STANDARD
        checkout_session.shipping_rate = None
        checkout_session.shipping_cost = Decimal('0')
        checkout_session.pickup_location = None

        checkout_session.payment_method = CheckoutSession.PAYMENT_COD
        checkout_session.payment_fee_amount = Decimal('0')
        checkout_session.payment_fee_label = ''

        checkout_session.coupon = None
        checkout_session.coupon_code = ''
        checkout_session.discount_amount = Decimal('0')

        checkout_session.is_gift = False
        checkout_session.gift_message = ''
        checkout_session.gift_wrap = False
        checkout_session.gift_wrap_cost = Decimal('0')

        checkout_session.order_notes = ''
        checkout_session.delivery_instructions = ''

        checkout_session.tax_rate = Decimal('0')
        checkout_session.tax_amount = Decimal('0')
        checkout_session.subtotal = Decimal('0')
        checkout_session.total = Decimal('0')

        checkout_session.current_step = CheckoutSession.STEP_COMPLETED
        checkout_session.expires_at = timezone.now()
        checkout_session.save(update_fields=[
            'email',
            'shipping_first_name',
            'shipping_last_name',
            'shipping_company',
            'shipping_email',
            'shipping_phone',
            'shipping_address_line_1',
            'shipping_address_line_2',
            'shipping_city',
            'shipping_state',
            'shipping_postal_code',
            'shipping_country',
            'saved_shipping_address',
            'billing_same_as_shipping',
            'billing_first_name',
            'billing_last_name',
            'billing_company',
            'billing_address_line_1',
            'billing_address_line_2',
            'billing_city',
            'billing_state',
            'billing_postal_code',
            'billing_country',
            'saved_billing_address',
            'shipping_method',
            'shipping_rate',
            'shipping_cost',
            'pickup_location',
            'payment_method',
            'payment_fee_amount',
            'payment_fee_label',
            'coupon',
            'coupon_code',
            'discount_amount',
            'is_gift',
            'gift_message',
            'gift_wrap',
            'gift_wrap_cost',
            'order_notes',
            'delivery_instructions',
            'tax_rate',
            'tax_amount',
            'subtotal',
            'total',
            'current_step',
            'expires_at',
        ])


# =============================================================================
# Enhanced Cart Service
# =============================================================================

class EnhancedCartService:
    """Comprehensive cart service with advanced features."""
    
    CART_CACHE_TTL = 300  # 5 minutes
    RESERVATION_DURATION_MINUTES = 30
    ABANDONED_THRESHOLD_HOURS = 1
    
    @classmethod
    def get_or_create_cart(cls, user=None, session_key=None, merge_session=True) -> Cart:
        """Get or create cart with session merging."""
        if user and user.is_authenticated:
            carts = Cart.objects.filter(user=user).order_by('-updated_at', '-created_at')
            cart = carts.first()
            if not cart:
                cart = Cart.objects.create(user=user)
            elif carts.count() > 1:
                duplicates = carts.exclude(pk=cart.pk)
                logger.warning(
                    "EnhancedCartService: found %s duplicate carts for user_id=%s; merging into cart_id=%s",
                    duplicates.count(),
                    getattr(user, "id", None),
                    cart.id,
                )
                for duplicate in duplicates:
                    cart.merge_from_session(duplicate)
            
            # Merge session cart if exists and requested
            if merge_session and session_key:
                cls._merge_session_cart(cart, session_key)
            
            # Ensure analytics exists
            CartAnalytics.objects.get_or_create(cart=cart)
            
            return cart
        
        elif session_key:
            carts = Cart.objects.filter(
                session_key=session_key,
                user__isnull=True,
            ).order_by('-updated_at', '-created_at')
            cart = carts.first()
            if not cart:
                cart = Cart.objects.create(session_key=session_key)
            elif carts.count() > 1:
                duplicates = carts.exclude(pk=cart.pk)
                logger.warning(
                    "EnhancedCartService: found %s duplicate guest carts for session_key=%s; merging into cart_id=%s",
                    duplicates.count(),
                    session_key,
                    cart.id,
                )
                for duplicate in duplicates:
                    cart.merge_from_session(duplicate)
            CartAnalytics.objects.get_or_create(cart=cart)
            return cart
        
        raise ValueError("Either user or session_key is required")
    
    @classmethod
    def _merge_session_cart(cls, user_cart: Cart, session_key: str):
        """Merge session cart into user cart."""
        session_cart = (
            Cart.objects.filter(session_key=session_key, user__isnull=True)
            .order_by('-updated_at', '-created_at')
            .first()
        )
        if session_cart:
            user_cart.merge_from_session(session_cart)
            
            # Also merge saved for later items
            SavedForLater.objects.filter(
                session_key=session_key,
                user__isnull=True
            ).update(user=user_cart.user, session_key=None)
    
    @classmethod
    @transaction.atomic
    def add_item(
        cls, 
        cart: Cart, 
        product, 
        quantity: int = 1, 
        variant=None,
        reserve_stock: bool = False,
        gift_wrap: bool = False,
        gift_message: str = ''
    ) -> CartItem:
        """Add item to cart with advanced options."""
        from apps.catalog.models import Product
        
        # Validate product
        if not product.is_active or product.is_deleted:
            raise ValueError("Product is not available")
        
        # Check stock
        available = variant.stock_quantity if variant else product.stock_quantity
        if not product.allow_backorder and available < quantity:
            raise ValueError(f"Only {available} items available")
        
        # Get current price
        current_price = variant.current_price if variant else product.current_price
        
        # Get or create cart item
        item, created = CartItem.objects.get_or_create(
            cart=cart,
            product=product,
            variant=variant,
            defaults={
                'quantity': quantity,
                'price_at_add': current_price,
                'gift_wrap': gift_wrap,
                'gift_message': gift_message,
            }
        )
        
        if not created:
            new_qty = item.quantity + quantity
            if not product.allow_backorder and available < new_qty:
                raise ValueError(f"Cannot add more. Only {available} available.")
            item.quantity = new_qty
            if gift_wrap:
                item.gift_wrap = True
                item.gift_message = gift_message
            item.save()
        
        # Create stock reservation if requested
        if reserve_stock:
            cls._create_reservation(cart, item)
        
        # Update analytics
        cls._record_analytics(cart, 'add')
        
        # Clear cart cache
        cls._invalidate_cart_cache(cart)
        
        return item
    
    @classmethod
    def _create_reservation(cls, cart: Cart, item: CartItem):
        """Create stock reservation for cart item."""
        ItemReservation.objects.update_or_create(
            cart=cart,
            cart_item=item,
            defaults={
                'product': item.product,
                'variant': item.variant,
                'quantity': item.quantity,
                'expires_at': timezone.now() + timedelta(minutes=cls.RESERVATION_DURATION_MINUTES),
                'is_active': True,
            }
        )
    
    @classmethod
    @transaction.atomic
    def update_quantity(cls, cart: Cart, item_id, quantity: int) -> Optional[CartItem]:
        """Update cart item quantity."""
        try:
            item = cart.items.get(pk=item_id)
        except CartItem.DoesNotExist:
            return None
        
        if quantity <= 0:
            cls.remove_item(cart, item_id)
            return None
        
        # Check stock
        available = item.variant.stock_quantity if item.variant else item.product.stock_quantity
        if not item.product.allow_backorder and available < quantity:
            raise ValueError(f"Only {available} items available")
        
        item.quantity = quantity
        item.save()
        
        # Update reservation if exists
        if hasattr(item, 'reservation') and item.reservation.is_active:
            item.reservation.quantity = quantity
            item.reservation.save(update_fields=['quantity'])
        
        cls._record_analytics(cart, 'quantity')
        cls._invalidate_cart_cache(cart)
        
        return item
    
    @classmethod
    @transaction.atomic
    def remove_item(cls, cart: Cart, item_id) -> bool:
        """Remove item from cart."""
        try:
            item = cart.items.get(pk=item_id)
            
            # Release reservation if exists
            if hasattr(item, 'reservation'):
                item.reservation.release()
            
            item.delete()
            cls._record_analytics(cart, 'remove')
            cls._invalidate_cart_cache(cart)
            return True
        except CartItem.DoesNotExist:
            return False
    
    @classmethod
    @transaction.atomic
    def save_for_later(cls, cart: Cart, item_id) -> Optional[SavedForLater]:
        """Move cart item to saved for later."""
        try:
            item = cart.items.get(pk=item_id)
            
            saved = SavedForLater.objects.create(
                user=cart.user,
                session_key=cart.session_key if not cart.user else None,
                product=item.product,
                variant=item.variant,
                quantity=item.quantity,
                price_at_save=item.unit_price,
            )
            
            # Remove from cart
            cls.remove_item(cart, item_id)
            cls._record_analytics(cart, 'save_later')
            
            return saved
        except CartItem.DoesNotExist:
            return None
    
    @classmethod
    @transaction.atomic
    def move_from_saved_to_cart(cls, cart: Cart, saved_id) -> Optional[CartItem]:
        """Move saved item back to cart."""
        try:
            saved = SavedForLater.objects.get(pk=saved_id)
            
            # Verify ownership
            if cart.user and saved.user != cart.user:
                return None
            if not cart.user and saved.session_key != cart.session_key:
                return None
            
            item = cls.add_item(
                cart=cart,
                product=saved.product,
                variant=saved.variant,
                quantity=saved.quantity,
            )
            
            saved.delete()
            return item
        except SavedForLater.DoesNotExist:
            return None
    
    @classmethod
    def get_saved_for_later(cls, user=None, session_key=None) -> List[SavedForLater]:
        """Get saved for later items."""
        if user and user.is_authenticated:
            return list(SavedForLater.objects.filter(
                user=user
            ).select_related('product', 'variant'))
        elif session_key:
            return list(SavedForLater.objects.filter(
                session_key=session_key,
                user__isnull=True
            ).select_related('product', 'variant'))
        return []
    
    @classmethod
    def enhanced_apply_coupon(cls, cart: Cart, coupon_code: str) -> Dict[str, Any]:
        """Apply coupon to cart with analytics."""
        from apps.promotions.services import CouponService

        user = cart.user if cart.user and getattr(cart.user, 'is_authenticated', False) else None
        coupon, is_valid, message = CouponService.validate_coupon(
            code=coupon_code,
            user=user,
            subtotal=cart.subtotal
        )

        if not coupon or not is_valid:
            cls._record_analytics(cart, 'coupon_fail')
            return {'success': False, 'error': message or 'Invalid coupon code'}

        cart.coupon = coupon
        cart.save()

        cls._record_analytics(cart, 'coupon_apply')
        cls._invalidate_cart_cache(cart)

        return {
            'success': True,
            'coupon': coupon.code,
            'discount': str(cart.discount_amount),
        }
    
    @classmethod
    def create_share_link(
        cls, 
        cart: Cart, 
        name: str = '',
        permission: str = 'view',
        expires_days: int = 7,
        password: str = None,
        created_by=None
    ) -> SharedCart:
        """Create a shareable link for the cart."""
        import hashlib
        
        share = SharedCart(
            cart=cart,
            name=name,
            permission=permission,
            expires_at=timezone.now() + timedelta(days=expires_days) if expires_days else None,
            created_by=created_by,
        )
        
        if password:
            share.password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        share.save()
        return share
    
    @classmethod
    def get_shared_cart(cls, share_token: str, password: str = None) -> Optional[Cart]:
        """Get cart by share token."""
        import hashlib
        
        try:
            share = SharedCart.objects.select_related('cart').get(share_token=share_token)
            
            if not share.is_valid:
                return None
            
            # Check password if set
            if share.password_hash:
                if not password:
                    return None
                if hashlib.sha256(password.encode()).hexdigest() != share.password_hash:
                    return None
            
            share.record_view()
            return share.cart
        except SharedCart.DoesNotExist:
            return None
    
    @classmethod
    def get_cart_summary(cls, cart: Cart, include_saved: bool = True) -> Dict[str, Any]:
        """Get comprehensive cart summary."""
        from django.core.cache import cache
        
        cache_key = f'cart_summary_{cart.id}'
        summary = cache.get(cache_key)
        
        if summary is None:
            items = []
            for item in cart.items.select_related(
                'product', 'variant'
            ).prefetch_related('product__images'):
                primary_image = (
                    item.product.images.filter(is_primary=True).first() or 
                    item.product.images.first()
                )
                
                items.append({
                    'id': str(item.id),
                    'product_id': str(item.product.id),
                    'product_name': item.product.name,
                    'product_slug': item.product.slug,
                    'variant_id': str(item.variant.id) if item.variant else None,
                    'variant_name': str(item.variant) if item.variant else None,
                    'quantity': item.quantity,
                    'unit_price': str(item.unit_price),
                    'price_at_add': str(item.price_at_add),
                    'total': str(item.total),
                    'image': primary_image.image.url if primary_image else None,
                    'in_stock': item.product.is_in_stock(),
                    'available_quantity': (
                        item.variant.stock_quantity if item.variant 
                        else item.product.stock_quantity
                    ),
                    'gift_wrap': item.gift_wrap,
                    'gift_message': item.gift_message,
                    'has_reservation': hasattr(item, 'reservation') and item.reservation.is_active,
                })
            
            summary = {
                'id': str(cart.id),
                'items': items,
                'item_count': cart.item_count,
                'subtotal': str(cart.subtotal),
                'discount_amount': str(cart.discount_amount),
                'total': str(cart.total),
                'coupon_code': cart.coupon.code if cart.coupon else None,
                'currency': cart.currency,
            }
            
            cache.set(cache_key, summary, cls.CART_CACHE_TTL)
        
        # Add saved for later if requested (not cached as it changes independently)
        if include_saved:
            saved_items = cls.get_saved_for_later(
                user=cart.user, 
                session_key=cart.session_key
            )
            summary['saved_for_later'] = [
                {
                    'id': str(s.id),
                    'product_id': str(s.product.id),
                    'product_name': s.product.name,
                    'product_slug': s.product.slug,
                    'quantity': s.quantity,
                    'price_at_save': str(s.price_at_save),
                    'current_price': str(s.current_price),
                    'price_change': str(s.price_change),
                    'in_stock': s.product.is_in_stock(),
                }
                for s in saved_items
            ]
        
        return summary
    
    @classmethod
    def _record_analytics(cls, cart: Cart, event_type: str):
        """Record cart analytics."""
        try:
            analytics, _ = CartAnalytics.objects.get_or_create(cart=cart)
            analytics.record_event(event_type)
        except Exception as e:
            logger.warning(f"Failed to record cart analytics: {e}")
    
    @classmethod
    def _invalidate_cart_cache(cls, cart: Cart):
        """Invalidate cart cache."""
        from django.core.cache import cache
        cache.delete(f'cart_summary_{cart.id}')
    
    @classmethod
    def check_abandoned_carts(cls):
        """Check for and mark abandoned carts."""
        threshold = timezone.now() - timedelta(hours=cls.ABANDONED_THRESHOLD_HOURS)
        
        abandoned_carts = Cart.objects.filter(
            updated_at__lt=threshold,
            items__isnull=False,
        ).exclude(
            abandoned_tracking__isnull=False
        ).distinct()
        
        count = 0
        for cart in abandoned_carts:
            if cart.item_count > 0:
                cls._mark_as_abandoned(cart)
                count += 1
        
        return count
    
    @classmethod
    def _mark_as_abandoned(cls, cart: Cart):
        """Mark cart as abandoned."""
        email = cart.user.email if cart.user else ''
        
        # Create snapshot of items
        items_snapshot = [
            {
                'product_id': str(item.product.id),
                'product_name': item.product.name,
                'quantity': item.quantity,
                'price': str(item.unit_price),
            }
            for item in cart.items.all()
        ]
        
        AbandonedCart.objects.create(
            cart=cart,
            user=cart.user,
            email=email,
            cart_value=cart.total,
            item_count=cart.item_count,
            items_snapshot=items_snapshot,
        )


# =============================================================================
# Enhanced Wishlist Service
# =============================================================================

class EnhancedWishlistService:
    """Comprehensive wishlist service with advanced features."""
    
    @classmethod
    def get_or_create_wishlist(cls, user=None, session_key=None):
        """Get wishlist for user or session."""
        if user and user.is_authenticated:
            wishlist, _ = Wishlist.objects.get_or_create(user=user)
            
            # Merge session wishlist
            if session_key:
                cls._merge_session_wishlist(wishlist, session_key)
            
            return wishlist, 'user'
        
        elif session_key:
            session_wishlist, _ = SessionWishlist.objects.get_or_create(
                session_key=session_key
            )
            return session_wishlist, 'session'
        
        raise ValueError("Either user or session_key is required")
    
    @classmethod
    def _merge_session_wishlist(cls, user_wishlist: Wishlist, session_key: str):
        """Merge session wishlist into user wishlist."""
        try:
            session_wishlist = SessionWishlist.objects.get(session_key=session_key)
            session_wishlist.merge_into_user_wishlist(user_wishlist)
        except SessionWishlist.DoesNotExist:
            pass
    
    @classmethod
    @transaction.atomic
    def add_item(
        cls, 
        wishlist, 
        product, 
        variant=None, 
        notes: str = '',
        collection_id=None
    ):
        """Add item to wishlist (user or session)."""
        if isinstance(wishlist, Wishlist):
            item, created = WishlistItem.objects.get_or_create(
                wishlist=wishlist,
                product=product,
                variant=variant,
                defaults={
                    'notes': notes,
                    'price_at_add': product.current_price,
                }
            )
            
            # Add to collection if specified
            if created and collection_id:
                try:
                    collection = wishlist.collections.get(pk=collection_id)
                    WishlistCollectionItem.objects.create(
                        collection=collection,
                        wishlist_item=item
                    )
                except WishlistCollection.DoesNotExist:
                    pass
            
            return item, created
        
        else:  # SessionWishlist
            item, created = SessionWishlistItem.objects.get_or_create(
                wishlist=wishlist,
                product=product,
                variant=variant,
                defaults={
                    'notes': notes,
                    'price_at_add': product.current_price,
                }
            )
            return item, created
    
    @classmethod
    @transaction.atomic
    def remove_item(cls, wishlist, item_id) -> bool:
        """Remove item from wishlist."""
        if isinstance(wishlist, Wishlist):
            try:
                item = wishlist.items.get(pk=item_id)
                item.delete()
                return True
            except WishlistItem.DoesNotExist:
                return False
        else:
            try:
                item = wishlist.items.get(pk=item_id)
                item.delete()
                return True
            except SessionWishlistItem.DoesNotExist:
                return False
    
    @classmethod
    def toggle_item(cls, wishlist, product, variant=None) -> Tuple[bool, bool]:
        """Toggle item in wishlist. Returns (is_in_wishlist, was_added)."""
        if isinstance(wishlist, Wishlist):
            item = wishlist.items.filter(product=product, variant=variant).first()
        else:
            item = wishlist.items.filter(product=product, variant=variant).first()
        
        if item:
            item.delete()
            return False, False
        else:
            cls.add_item(wishlist, product, variant)
            return True, True
    
    @classmethod
    def move_to_cart(cls, wishlist, item_id, cart: Cart) -> Optional[CartItem]:
        """Move wishlist item to cart."""
        if isinstance(wishlist, Wishlist):
            try:
                item = wishlist.items.get(pk=item_id)
                cart_item = EnhancedCartService.add_item(
                    cart=cart,
                    product=item.product,
                    variant=item.variant,
                )
                cls.remove_item(wishlist, item_id)
                return cart_item
            except WishlistItem.DoesNotExist:
                return None
        else:
            try:
                item = wishlist.items.get(pk=item_id)
                cart_item = EnhancedCartService.add_item(
                    cart=cart,
                    product=item.product,
                    variant=item.variant,
                )
                cls.remove_item(wishlist, item_id)
                return cart_item
            except SessionWishlistItem.DoesNotExist:
                return None
    
    @classmethod
    def move_all_to_cart(cls, wishlist, cart: Cart) -> int:
        """Move all wishlist items to cart."""
        count = 0
        for item in list(wishlist.items.all()):
            try:
                EnhancedCartService.add_item(
                    cart=cart,
                    product=item.product,
                    variant=item.variant,
                )
                item.delete()
                count += 1
            except Exception:
                pass
        return count
    
    @classmethod
    def create_collection(
        cls, 
        wishlist: Wishlist, 
        name: str, 
        description: str = '',
        emoji: str = '',
        color: str = '',
        is_public: bool = False
    ) -> WishlistCollection:
        """Create a wishlist collection."""
        return WishlistCollection.objects.create(
            wishlist=wishlist,
            name=name,
            description=description,
            emoji=emoji,
            color=color,
            is_public=is_public,
        )
    
    @classmethod
    def add_to_collection(cls, collection: WishlistCollection, item: WishlistItem):
        """Add wishlist item to collection."""
        WishlistCollectionItem.objects.get_or_create(
            collection=collection,
            wishlist_item=item
        )
    
    @classmethod
    def remove_from_collection(cls, collection: WishlistCollection, item: WishlistItem):
        """Remove wishlist item from collection."""
        WishlistCollectionItem.objects.filter(
            collection=collection,
            wishlist_item=item
        ).delete()
    
    @classmethod
    def get_items_with_price_drops(cls, wishlist) -> List:
        """Get wishlist items with price drops."""
        if isinstance(wishlist, Wishlist):
            items = wishlist.items.select_related('product', 'variant')
        else:
            items = wishlist.items.select_related('product', 'variant')
        
        return [
            item for item in items
            if item.price_at_add and item.product.current_price < item.price_at_add
        ]
    
    @classmethod
    def get_items_back_in_stock(cls, wishlist) -> List:
        """Get wishlist items that are back in stock."""
        if isinstance(wishlist, Wishlist):
            items = wishlist.items.select_related('product', 'variant')
        else:
            items = wishlist.items.select_related('product', 'variant')
        
        return [
            item for item in items
            if item.product.is_in_stock() and item.notify_on_restock
        ]
    
    @classmethod
    def check_item_in_wishlist(cls, wishlist, product_id, variant_id=None) -> bool:
        """Check if product is in wishlist."""
        filters = {'product_id': product_id}
        if variant_id:
            filters['variant_id'] = variant_id
        
        return wishlist.items.filter(**filters).exists()
    
    @classmethod
    def get_wishlist_summary(cls, wishlist) -> Dict[str, Any]:
        """Get wishlist summary."""
        items = []
        
        if isinstance(wishlist, Wishlist):
            queryset = wishlist.items.select_related(
                'product', 'variant'
            ).prefetch_related('product__images', 'collection_memberships__collection')
        else:
            queryset = wishlist.items.select_related(
                'product', 'variant'
            ).prefetch_related('product__images')
        
        for item in queryset:
            primary_image = (
                item.product.images.filter(is_primary=True).first() or 
                item.product.images.first()
            )
            
            item_data = {
                'id': str(item.id),
                'product_id': str(item.product.id),
                'product_name': item.product.name,
                'product_slug': item.product.slug,
                'variant_id': str(item.variant.id) if item.variant else None,
                'variant_name': str(item.variant) if item.variant else None,
                'current_price': str(item.product.current_price),
                'price_at_add': str(item.price_at_add) if item.price_at_add else None,
                'image': primary_image.image.url if primary_image else None,
                'in_stock': item.product.is_in_stock(),
                'is_on_sale': item.product.is_on_sale,
                'added_at': item.added_at.isoformat() if hasattr(item, 'added_at') else None,
                'notes': item.notes,
            }
            
            # Add price change for user wishlists
            if hasattr(item, 'price_change'):
                item_data['price_change'] = str(item.price_change)
            
            # Add collections for user wishlists
            if isinstance(wishlist, Wishlist) and hasattr(item, 'collection_memberships'):
                item_data['collections'] = [
                    {'id': str(m.collection.id), 'name': m.collection.name}
                    for m in item.collection_memberships.all()
                ]
            
            items.append(item_data)
        
        summary = {
            'id': str(wishlist.id),
            'item_count': wishlist.item_count,
            'items': items,
        }
        
        # Add collections for user wishlists
        if isinstance(wishlist, Wishlist):
            summary['collections'] = [
                {
                    'id': str(c.id),
                    'name': c.name,
                    'emoji': c.emoji,
                    'color': c.color,
                    'item_count': c.item_count,
                    'is_public': c.is_public,
                }
                for c in wishlist.collections.all()
            ]
            summary['total_value'] = str(wishlist.total_value)
        
        return summary


# =============================================================================
# Price Alert Service
# =============================================================================

class PriceAlertService:
    """Service for managing price alerts."""
    
    @classmethod
    def create_alert(
        cls,
        product,
        user=None,
        email: str = None,
        variant=None,
        alert_type: str = 'any',
        target_price: Decimal = None,
        percentage_drop: int = None,
        expires_days: int = 90
    ) -> PriceAlert:
        """Create a price alert."""
        if not user and not email:
            raise ValueError("Either user or email is required")
        
        current_price = variant.current_price if variant else product.current_price
        
        return PriceAlert.objects.create(
            user=user,
            email=email or (user.email if user else ''),
            product=product,
            variant=variant,
            alert_type=alert_type,
            target_price=target_price,
            percentage_drop=percentage_drop,
            price_at_creation=current_price,
            lowest_price_seen=current_price,
            expires_at=timezone.now() + timedelta(days=expires_days) if expires_days else None,
        )
    
    @classmethod
    def check_alerts_for_product(cls, product) -> List[PriceAlert]:
        """Check and trigger alerts for a product."""
        triggered = []
        current_price = product.current_price
        
        alerts = PriceAlert.objects.filter(
            product=product,
            is_active=True,
            is_triggered=False,
        )
        
        for alert in alerts:
            if alert.check_and_trigger(current_price):
                triggered.append(alert)
        
        return triggered
    
    @classmethod
    def get_user_alerts(cls, user) -> List[PriceAlert]:
        """Get all active alerts for a user."""
        return list(PriceAlert.objects.filter(
            user=user,
            is_active=True
        ).select_related('product', 'variant'))
    
    @classmethod
    def deactivate_alert(cls, alert_id, user=None) -> bool:
        """Deactivate a price alert."""
        filters = {'pk': alert_id}
        if user:
            filters['user'] = user
        
        return PriceAlert.objects.filter(**filters).update(is_active=False) > 0


# =============================================================================
# Comprehensive Cart Operations Service
# =============================================================================

class ComprehensiveCartService:
    """
    Comprehensive cart service with all advanced features.
    Designed for Bangladesh e-commerce with BDT default.
    """
    
    # Bangladesh-specific defaults
    DEFAULT_CURRENCY = 'BDT'
    FREE_SHIPPING_THRESHOLD = Decimal('3000.00')  # ৳3,000
    STANDARD_SHIPPING_COST = Decimal('60.00')  # ৳60 for Dhaka
    OUTSIDE_DHAKA_SHIPPING = Decimal('120.00')  # ৳120 outside Dhaka
    COD_FEE = Decimal('0.00')  # Cash on delivery fee
    
    # Delivery estimates (in days)
    DELIVERY_ESTIMATE_DHAKA = (1, 2)  # 1-2 days
    DELIVERY_ESTIMATE_METRO = (2, 3)  # 2-3 days for divisional cities
    DELIVERY_ESTIMATE_OTHER = (3, 5)  # 3-5 days for other areas
    
    @classmethod
    def get_delivery_estimate(cls, division: str = 'Dhaka') -> Dict[str, Any]:
        """Get delivery estimate based on location."""
        metro_cities = ['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Sylhet', 'Rangpur']
        
        if division == 'Dhaka':
            min_days, max_days = cls.DELIVERY_ESTIMATE_DHAKA
            shipping_cost = cls.STANDARD_SHIPPING_COST
        elif division in metro_cities:
            min_days, max_days = cls.DELIVERY_ESTIMATE_METRO
            shipping_cost = cls.OUTSIDE_DHAKA_SHIPPING
        else:
            min_days, max_days = cls.DELIVERY_ESTIMATE_OTHER
            shipping_cost = cls.OUTSIDE_DHAKA_SHIPPING
        
        today = timezone.now().date()
        estimated_min = today + timedelta(days=min_days)
        estimated_max = today + timedelta(days=max_days)
        
        return {
            'min_days': min_days,
            'max_days': max_days,
            'estimated_delivery_min': estimated_min.isoformat(),
            'estimated_delivery_max': estimated_max.isoformat(),
            'shipping_cost': str(shipping_cost),
            'formatted_shipping': f'৳{shipping_cost:,.0f}',
            'free_shipping_eligible': False,  # Will be calculated with cart total
        }
    
    @classmethod
    def calculate_shipping(cls, cart: Cart, division: str = 'Dhaka') -> Dict[str, Any]:
        """Calculate shipping cost with free shipping threshold."""
        estimate = cls.get_delivery_estimate(division)
        shipping_cost = Decimal(estimate['shipping_cost'])
        
        # Check free shipping eligibility
        if cart.subtotal >= cls.FREE_SHIPPING_THRESHOLD:
            shipping_cost = Decimal('0')
            estimate['free_shipping_eligible'] = True
            estimate['shipping_cost'] = '0'
            estimate['formatted_shipping'] = 'Free'
        else:
            amount_for_free = cls.FREE_SHIPPING_THRESHOLD - cart.subtotal
            estimate['amount_for_free_shipping'] = str(amount_for_free)
            estimate['formatted_amount_for_free'] = f'৳{amount_for_free:,.0f}'
        
        return estimate
    
    @classmethod
    @transaction.atomic
    def bulk_add_items(cls, cart: Cart, items: List[Dict]) -> Dict[str, Any]:
        """
        Bulk add items to cart.
        items: List of {'product_id': ..., 'variant_id': ..., 'quantity': ...}
        """
        from apps.catalog.models import Product, ProductVariant
        
        added = []
        errors = []
        
        for item_data in items:
            try:
                product = Product.objects.get(
                    id=item_data['product_id'],
                    is_active=True,
                    is_deleted=False
                )
                
                variant = None
                if item_data.get('variant_id'):
                    variant = ProductVariant.objects.get(id=item_data['variant_id'])
                
                quantity = item_data.get('quantity', 1)
                
                item = EnhancedCartService.add_item(
                    cart=cart,
                    product=product,
                    quantity=quantity,
                    variant=variant,
                    gift_wrap=item_data.get('gift_wrap', False),
                    gift_message=item_data.get('gift_message', ''),
                )
                
                added.append({
                    'product_id': str(product.id),
                    'product_name': product.name,
                    'quantity': quantity,
                    'success': True,
                })
                
            except Product.DoesNotExist:
                errors.append({
                    'product_id': item_data.get('product_id'),
                    'error': 'Product not found',
                })
            except Exception as e:
                errors.append({
                    'product_id': item_data.get('product_id'),
                    'error': str(e),
                })
        
        return {
            'added': added,
            'errors': errors,
            'added_count': len(added),
            'error_count': len(errors),
            'cart': EnhancedCartService.get_cart_summary(cart),
        }
    
    @classmethod
    @transaction.atomic
    def bulk_remove_items(cls, cart: Cart, item_ids: List[str]) -> Dict[str, Any]:
        """Bulk remove items from cart."""
        removed = []
        errors = []
        
        for item_id in item_ids:
            try:
                if EnhancedCartService.remove_item(cart, item_id):
                    removed.append(item_id)
                else:
                    errors.append({'id': item_id, 'error': 'Item not found'})
            except Exception as e:
                errors.append({'id': item_id, 'error': str(e)})
        
        return {
            'removed': removed,
            'errors': errors,
            'cart': EnhancedCartService.get_cart_summary(cart),
        }
    
    @classmethod
    def validate_cart(cls, cart: Cart) -> Dict[str, Any]:
        """
        Validate all cart items for availability, stock, and pricing.
        Returns validation result with any issues found.
        """
        issues = []
        warnings = []
        valid_items = []
        
        settings = CartSettings.get_settings()
        
        for item in cart.items.select_related('product', 'variant'):
            product = item.product
            variant = item.variant
            
            # Check product availability
            if not product.is_active or product.is_deleted:
                issues.append({
                    'item_id': str(item.id),
                    'product_name': product.name,
                    'type': 'unavailable',
                    'message': f'{product.name} is no longer available',
                })
                continue
            
            # Check stock
            available = variant.stock_quantity if variant else product.stock_quantity
            if available < item.quantity and not product.allow_backorder:
                if available == 0:
                    issues.append({
                        'item_id': str(item.id),
                        'product_name': product.name,
                        'type': 'out_of_stock',
                        'message': f'{product.name} is out of stock',
                    })
                else:
                    warnings.append({
                        'item_id': str(item.id),
                        'product_name': product.name,
                        'type': 'limited_stock',
                        'message': f'Only {available} of {product.name} available',
                        'available': available,
                    })
            
            # Check for price changes
            current_price = variant.current_price if variant else product.current_price
            if item.price_at_add != current_price:
                change = current_price - item.price_at_add
                change_type = 'price_increase' if change > 0 else 'price_decrease'
                warnings.append({
                    'item_id': str(item.id),
                    'product_name': product.name,
                    'type': change_type,
                    'message': f'Price changed from ৳{item.price_at_add:,.0f} to ৳{current_price:,.0f}',
                    'old_price': str(item.price_at_add),
                    'new_price': str(current_price),
                    'change': str(change),
                })
            
            # Check quantity limits
            if item.quantity > settings.max_quantity_per_item:
                warnings.append({
                    'item_id': str(item.id),
                    'product_name': product.name,
                    'type': 'quantity_limit',
                    'message': f'Maximum {settings.max_quantity_per_item} per item',
                    'max_quantity': settings.max_quantity_per_item,
                })
            
            valid_items.append(str(item.id))
        
        # Check cart limits
        if cart.item_count > settings.max_items_per_cart:
            warnings.append({
                'type': 'cart_limit',
                'message': f'Cart exceeds maximum of {settings.max_items_per_cart} items',
            })
        
        # Check minimum order
        if settings.minimum_order_amount > 0 and cart.total < settings.minimum_order_amount:
            issues.append({
                'type': 'minimum_order',
                'message': settings.minimum_order_message or f'Minimum order amount is ৳{settings.minimum_order_amount:,.0f}',
                'minimum': str(settings.minimum_order_amount),
                'current': str(cart.total),
            })
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'valid_items': valid_items,
            'issue_count': len(issues),
            'warning_count': len(warnings),
        }
    
    @classmethod
    def export_cart(cls, cart: Cart, format: str = 'json') -> Dict[str, Any]:
        """Export cart for sharing or backup."""
        items = []
        for item in cart.items.select_related('product', 'variant'):
            items.append({
                'product_id': str(item.product.id),
                'product_sku': item.product.sku,
                'product_name': item.product.name,
                'variant_id': str(item.variant.id) if item.variant else None,
                'variant_name': str(item.variant) if item.variant else None,
                'quantity': item.quantity,
                'unit_price': str(item.unit_price),
                'gift_wrap': item.gift_wrap,
                'gift_message': item.gift_message,
                'customer_note': item.customer_note,
            })
        
        export_data = {
            'export_date': timezone.now().isoformat(),
            'currency': cart.currency,
            'items': items,
            'item_count': cart.item_count,
            'subtotal': str(cart.subtotal),
            'coupon_code': cart.coupon.code if cart.coupon else None,
        }
        
        return export_data
    
    @classmethod
    @transaction.atomic
    def import_cart(cls, cart: Cart, import_data: Dict, merge: bool = False) -> Dict[str, Any]:
        """Import cart from exported data."""
        from apps.catalog.models import Product, ProductVariant
        
        if not merge:
            cart.clear()
        
        items_data = import_data.get('items', [])
        return cls.bulk_add_items(cart, [
            {
                'product_id': item.get('product_id'),
                'variant_id': item.get('variant_id'),
                'quantity': item.get('quantity', 1),
                'gift_wrap': item.get('gift_wrap', False),
                'gift_message': item.get('gift_message', ''),
            }
            for item in items_data
        ])
    
    @classmethod
    def get_comprehensive_summary(cls, cart: Cart, division: str = 'Dhaka') -> Dict[str, Any]:
        """Get comprehensive cart summary with all details."""
        # Get base summary
        summary = EnhancedCartService.get_cart_summary(cart, include_saved=True)
        
        # Add shipping calculation
        shipping = cls.calculate_shipping(cart, division)
        summary['shipping'] = shipping
        
        # Add validation
        validation = cls.validate_cart(cart)
        summary['validation'] = validation
        
        # Get settings
        settings = CartSettings.get_settings()
        
        # Calculate final totals
        subtotal = Decimal(summary['subtotal'])
        discount = Decimal(summary['discount_amount'])
        shipping_cost = Decimal(shipping['shipping_cost'])
        
        # Gift wrap cost
        gift_wrap_items = cart.items.filter(gift_wrap=True).count()
        gift_wrap_cost = Decimal('0')
        if gift_wrap_items > 0 and settings.gift_wrap_enabled:
            gift_wrap_cost = settings.gift_wrap_amount * gift_wrap_items
        
        total = subtotal - discount + shipping_cost + gift_wrap_cost
        
        summary.update({
            'gift_wrap_cost': str(gift_wrap_cost),
            'formatted_gift_wrap_cost': f'৳{gift_wrap_cost:,.0f}' if gift_wrap_cost > 0 else '৳0',
            'shipping_cost': str(shipping_cost),
            'formatted_shipping_cost': shipping['formatted_shipping'],
            'grand_total': str(total),
            'formatted_grand_total': f'৳{total:,.0f}',
            'formatted_subtotal': f'৳{subtotal:,.0f}',
            'formatted_discount': f'-৳{discount:,.0f}' if discount > 0 else '৳0',
            'free_shipping_threshold': str(cls.FREE_SHIPPING_THRESHOLD),
            'is_checkout_ready': validation['is_valid'] and cart.item_count > 0,
        })
        
        # Add COD fee info
        summary['cod_fee'] = str(cls.COD_FEE)
        summary['formatted_cod_fee'] = f'৳{cls.COD_FEE:,.0f}'
        
        return summary
    
    @classmethod
    def lock_all_prices(cls, cart: Cart, duration_hours: int = 24) -> int:
        """Lock prices for all items in cart."""
        settings = CartSettings.get_settings()
        if not settings.price_lock_enabled:
            return 0
        
        duration = duration_hours or settings.price_lock_duration_hours
        count = 0
        
        for item in cart.items.all():
            item.lock_price(duration)
            count += 1
        
        return count
    
    @classmethod
    def get_recurring_items(cls, cart: Cart) -> List[CartItem]:
        """Get items marked for recurring purchase."""
        return list(cart.items.filter(is_recurring=True).select_related('product', 'variant'))


# =============================================================================
# Comprehensive Wishlist Operations Service
# =============================================================================

class ComprehensiveWishlistService:
    """
    Comprehensive wishlist service with all advanced features.
    Supports both user and session wishlists.
    """
    
    @classmethod
    def get_unified_wishlist(cls, user=None, session_key=None) -> Tuple[Any, str]:
        """Get wishlist for user or session with unified interface."""
        if user and user.is_authenticated:
            wishlist, _ = Wishlist.objects.get_or_create(user=user)
            
            # Merge session wishlist if exists
            if session_key:
                EnhancedWishlistService._merge_session_wishlist(wishlist, session_key)
            
            return wishlist, 'user'
        
        elif session_key:
            wishlist, _ = SessionWishlist.objects.get_or_create(session_key=session_key)
            return wishlist, 'session'
        
        raise ValueError("Either user or session_key is required")
    
    @classmethod
    @transaction.atomic
    def add_item_advanced(
        cls,
        wishlist,
        product,
        variant=None,
        notes: str = '',
        priority: int = 2,
        desired_quantity: int = 1,
        target_price: Decimal = None,
        notify_on_sale: bool = True,
        notify_on_restock: bool = True,
        notify_on_price_drop: bool = True,
        collection_id=None,
    ) -> Tuple[Any, bool]:
        """Add item with all advanced options."""
        if isinstance(wishlist, Wishlist):
            item, created = WishlistItem.objects.get_or_create(
                wishlist=wishlist,
                product=product,
                variant=variant,
                defaults={
                    'notes': notes,
                    'price_at_add': product.current_price,
                    'priority': priority,
                    'desired_quantity': desired_quantity,
                    'target_price': target_price,
                    'notify_on_sale': notify_on_sale,
                    'notify_on_restock': notify_on_restock,
                    'notify_on_price_drop': notify_on_price_drop,
                }
            )
            
            if not created:
                # Update preferences if item exists
                item.notes = notes or item.notes
                item.priority = priority
                item.desired_quantity = desired_quantity
                item.target_price = target_price
                item.notify_on_sale = notify_on_sale
                item.notify_on_restock = notify_on_restock
                item.notify_on_price_drop = notify_on_price_drop
                item.save()
            
            # Add to collection
            if collection_id:
                try:
                    collection = wishlist.collections.get(pk=collection_id)
                    WishlistCollectionItem.objects.get_or_create(
                        collection=collection,
                        wishlist_item=item
                    )
                except WishlistCollection.DoesNotExist:
                    pass
            
            return item, created
        
        else:  # SessionWishlist
            item, created = SessionWishlistItem.objects.get_or_create(
                wishlist=wishlist,
                product=product,
                variant=variant,
                defaults={
                    'notes': notes,
                    'price_at_add': product.current_price,
                    'priority': priority,
                    'desired_quantity': desired_quantity,
                    'target_price': target_price,
                    'notify_on_sale': notify_on_sale,
                    'notify_on_restock': notify_on_restock,
                    'notify_on_price_drop': notify_on_price_drop,
                }
            )
            
            return item, created
    
    @classmethod
    def set_item_priority(cls, wishlist, item_id, priority: int) -> bool:
        """Set priority for a wishlist item."""
        try:
            item = wishlist.items.get(pk=item_id)
            item.priority = priority
            item.save(update_fields=['priority'])
            return True
        except (WishlistItem.DoesNotExist, SessionWishlistItem.DoesNotExist):
            return False
    
    @classmethod
    def set_reminder(cls, wishlist_item, reminder_date, message: str = '') -> bool:
        """Set reminder for wishlist item."""
        if isinstance(wishlist_item, WishlistItem):
            wishlist_item.set_reminder(reminder_date, message)
            return True
        return False
    
    @classmethod
    def get_due_reminders(cls, user) -> List[WishlistItem]:
        """Get wishlist items with due reminders."""
        today = timezone.now().date()
        return list(WishlistItem.objects.filter(
            wishlist__user=user,
            reminder_date__lte=today,
            reminder_sent=False,
        ).select_related('product', 'variant'))
    
    @classmethod
    def send_reminder_notifications(cls, user) -> int:
        """Send reminder notifications and mark as sent."""
        items = cls.get_due_reminders(user)
        count = 0
        
        for item in items:
            # Here you would trigger actual notification
            # For now, just mark as sent
            item.reminder_sent = True
            item.save(update_fields=['reminder_sent'])
            count += 1
        
        return count
    
    @classmethod
    def get_price_drop_items(cls, wishlist) -> List[Any]:
        """Get items with price drops."""
        return EnhancedWishlistService.get_items_with_price_drops(wishlist)
    
    @classmethod
    def get_back_in_stock_items(cls, wishlist) -> List[Any]:
        """Get items that are back in stock."""
        return EnhancedWishlistService.get_items_back_in_stock(wishlist)
    
    @classmethod
    def get_items_at_target_price(cls, wishlist) -> List[Any]:
        """Get items at or below target price."""
        if isinstance(wishlist, Wishlist):
            items = wishlist.items.filter(target_price__isnull=False).select_related('product', 'variant')
        else:
            items = wishlist.items.filter(target_price__isnull=False).select_related('product', 'variant')
        
        return [item for item in items if item.is_at_target_price] if isinstance(wishlist, Wishlist) else [
            item for item in items if item.current_price <= item.target_price
        ]
    
    @classmethod
    def move_all_to_cart_with_quantities(cls, wishlist, cart: Cart) -> Dict[str, Any]:
        """Move all wishlist items to cart using desired quantities."""
        added = []
        errors = []
        
        for item in list(wishlist.items.all()):
            try:
                quantity = item.desired_quantity if hasattr(item, 'desired_quantity') else 1
                
                cart_item = EnhancedCartService.add_item(
                    cart=cart,
                    product=item.product,
                    variant=item.variant,
                    quantity=quantity,
                )
                
                added.append({
                    'product_id': str(item.product.id),
                    'product_name': item.product.name,
                    'quantity': quantity,
                })
                
                item.delete()
                
            except Exception as e:
                errors.append({
                    'product_id': str(item.product.id),
                    'error': str(e),
                })
        
        return {
            'added': added,
            'errors': errors,
            'cart_summary': EnhancedCartService.get_cart_summary(cart),
        }
    
    @classmethod
    def get_comprehensive_summary(cls, wishlist) -> Dict[str, Any]:
        """Get comprehensive wishlist summary."""
        summary = EnhancedWishlistService.get_wishlist_summary(wishlist)
        
        # Add analysis
        if isinstance(wishlist, Wishlist):
            price_drops = cls.get_price_drop_items(wishlist)
            back_in_stock = cls.get_back_in_stock_items(wishlist)
            at_target = cls.get_items_at_target_price(wishlist)
            due_reminders = WishlistItem.objects.filter(
                wishlist=wishlist,
                reminder_date__lte=timezone.now().date(),
                reminder_sent=False,
            ).count()
            
            summary['analysis'] = {
                'price_drops_count': len(price_drops),
                'back_in_stock_count': len(back_in_stock),
                'at_target_price_count': len(at_target),
                'due_reminders_count': due_reminders,
            }
            
            # Calculate potential savings
            total_potential_savings = sum(
                item.savings_from_highest for item in wishlist.items.all()
            )
            summary['potential_savings'] = str(total_potential_savings)
            summary['formatted_potential_savings'] = f'৳{total_potential_savings:,.0f}'
        
        return summary
    
    @classmethod
    def export_wishlist(cls, wishlist) -> Dict[str, Any]:
        """Export wishlist data."""
        items = []
        
        for item in wishlist.items.select_related('product', 'variant'):
            item_data = {
                'product_id': str(item.product.id),
                'product_sku': item.product.sku,
                'product_name': item.product.name,
                'variant_id': str(item.variant.id) if item.variant else None,
                'notes': item.notes,
                'priority': item.priority if hasattr(item, 'priority') else 2,
                'desired_quantity': item.desired_quantity if hasattr(item, 'desired_quantity') else 1,
            }
            items.append(item_data)
        
        return {
            'export_date': timezone.now().isoformat(),
            'item_count': wishlist.item_count,
            'items': items,
        }
    
    @classmethod
    def create_public_share(
        cls,
        wishlist: Wishlist,
        name: str = '',
        expires_days: int = 30,
        allow_purchase: bool = True,
    ) -> WishlistShare:
        """Create a public shareable link for wishlist."""
        return WishlistService.create_share_link(
            wishlist=wishlist,
            expires_days=expires_days,
            allow_purchase=allow_purchase,
        )
