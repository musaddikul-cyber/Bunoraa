"""
Commerce views - Web views for cart, checkout, and wishlist
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.generic import TemplateView, ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.http import JsonResponse, HttpResponseBadRequest, Http404
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils import timezone
from decimal import Decimal

from .models import (
    Cart, CartItem, Wishlist, WishlistItem, WishlistShare,
    CheckoutSession, CartSettings
)
from .services import CartService, WishlistService, CheckoutService, EnhancedCartService
from apps.i18n.api.serializers import convert_currency_fields
from apps.i18n.services import CurrencyService, CurrencyConversionService, GeoService
from apps.accounts.services import AddressService
from apps.shipping.services import ShippingRateService, ShippingZoneService
from apps.shipping.models import ShippingRate
from apps.payments.models import PaymentGateway


def normalize_checkout_country(checkout_session):
    """Return normalized ISO country code without mutating stored country name."""
    if not checkout_session:
        return None

    normalized = CheckoutService.normalize_country_code(checkout_session.shipping_country)
    return normalized or checkout_session.shipping_country


def build_checkout_cart_summary(request, cart, checkout_session):
    """Build cart summary with currency conversion and formatted values."""
    if not cart:
        return {
            'items': [],
            'item_count': 0,
            'subtotal': '0',
            'discount_amount': '0',
            'total': '0',
            'formatted_subtotal': '',
            'formatted_discount': '',
            'formatted_shipping': '',
            'formatted_tax': '',
            'formatted_total': '',
            'currency': None,
        }

    summary = CartService.get_cart_summary(cart)
    from_code = getattr(cart, 'currency', None) or 'BDT'

    user_currency = CurrencyService.get_user_currency(request=request)

    if isinstance(summary.get('items'), list):
        for item in summary['items']:
            convert_currency_fields(
                item, ['unit_price', 'total', 'price_at_add'], from_code, request
            )

    session_currency_code = getattr(checkout_session, 'currency', None) if checkout_session else None
    session_currency = (
        CurrencyService.get_currency_by_code(session_currency_code)
        if session_currency_code else None
    )
    currency = (
        user_currency
        or session_currency
        or CurrencyService.get_currency_by_code(from_code)
        or CurrencyService.get_default_currency()
    )
    
    if currency:
        summary['currency'] = currency
        summary['currency_code'] = currency.code
        summary['currency_symbol'] = getattr(currency, 'native_symbol', None) or currency.symbol

        base_subtotal = Decimal(str(cart.subtotal or 0))
        base_discount = Decimal(str(cart.discount_amount or 0))
        base_shipping = Decimal('0')
        base_gift_wrap = Decimal('0')
        shipping_estimate = False
        shipping_selected = False

        if checkout_session:
            try:
                if checkout_session.shipping_method == CheckoutSession.SHIPPING_PICKUP:
                    shipping_selected = True
                elif getattr(checkout_session, 'shipping_rate', None):
                    shipping_selected = True
            except Exception:
                shipping_selected = False
            if (
                checkout_session.shipping_method == CheckoutSession.SHIPPING_PICKUP
                and getattr(checkout_session, 'pickup_location', None)
            ):
                summary['pickup_location_id'] = str(checkout_session.pickup_location.id)
                summary['pickup_location_name'] = checkout_session.pickup_location.name
                summary['shipping_method_code'] = CheckoutSession.SHIPPING_PICKUP
                summary['shipping_method_name'] = 'Store pickup'
            if getattr(checkout_session, 'shipping_rate', None):
                try:
                    summary['shipping_rate_id'] = str(checkout_session.shipping_rate.id)
                    method = checkout_session.shipping_rate.method
                    if method:
                        summary['shipping_method_name'] = method.name
                        summary['shipping_method_code'] = method.code
                except Exception:
                    pass

        # Prefer computing base shipping from rate/pickup instead of session snapshot
        if checkout_session:
            try:
                if (
                    checkout_session.shipping_method == CheckoutSession.SHIPPING_PICKUP
                    and getattr(checkout_session, 'pickup_location', None)
                ):
                    base_shipping = Decimal(str(checkout_session.pickup_location.pickup_fee or 0))
                elif getattr(checkout_session, 'shipping_rate', None):
                    shipping_rate = checkout_session.shipping_rate
                    try:
                        item_count = cart.item_count if cart else 1
                        base_shipping = Decimal(
                            str(
                                shipping_rate.calculate_rate(
                                    subtotal=base_subtotal,
                                    weight=Decimal('0'),
                                    item_count=item_count,
                                )
                            )
                        )
                    except Exception:
                        base_shipping = Decimal(str(getattr(shipping_rate, 'base_rate', 0) or 0))
                else:
                    base_shipping = Decimal(
                        str(
                            CheckoutService.DEFAULT_SHIPPING_COSTS.get(
                                checkout_session.shipping_method, Decimal('0.00')
                            )
                        )
                    )
            except Exception:
                base_shipping = Decimal('0')

        if checkout_session and not shipping_selected:
            try:
                country_value = (checkout_session.shipping_country or '').strip() or None
                country_code = normalize_checkout_country(checkout_session) or country_value
                state = (checkout_session.shipping_state or '').strip() or None
                postal_code = (checkout_session.shipping_postal_code or '').strip() or None
                city = (checkout_session.shipping_city or '').strip() or None
                if city:
                    estimate_label = city
                elif state:
                    estimate_label = state
                else:
                    estimate_label = country_value or country_code or ''

                if country_code:
                    try:
                        product_ids = list(
                            cart.items.values_list('product_id', flat=True)
                        )
                    except Exception:
                        product_ids = None

                    methods = ShippingRateService.get_available_methods(
                        country=country_code,
                        state=state,
                        postal_code=postal_code,
                        city=city,
                        subtotal=base_subtotal,
                        item_count=cart.item_count,
                        product_ids=product_ids,
                        currency_code=from_code,
                    )
                    if methods:
                        selected = methods[0]
                        base_shipping = Decimal(str(selected.get('rate') or 0))
                        shipping_estimate = True
                        summary['shipping_method_name'] = selected.get('name')
                        summary['shipping_method_code'] = selected.get('code')
                        summary['shipping_rate_id'] = selected.get('rate_id')
                        zone = ShippingZoneService.find_zone_for_location(
                            country_code, state, postal_code, city=city
                        )
                        if zone:
                            summary['shipping_zone'] = zone.name
                        if estimate_label:
                            summary['shipping_estimate_label'] = estimate_label
                if not shipping_estimate:
                    from .services import EnhancedCartService
                    division = (
                        (checkout_session.shipping_city or '').strip()
                        or (checkout_session.shipping_state or '').strip()
                        or (checkout_session.shipping_country or '').strip()
                        or 'Dhaka'
                    )
                    estimate = EnhancedCartService.calculate_shipping(cart, division=division)
                    base_shipping = Decimal(str(estimate.get('shipping_cost') or 0))
                    shipping_estimate = True
                    if estimate_label:
                        summary['shipping_estimate_label'] = estimate_label
            except Exception:
                shipping_estimate = False

        gift_wrap_amount = Decimal('0')
        gift_wrap_from_code = from_code
        gift_wrap_enabled = False
        gift_wrap_label = 'Gift Wrap'
        try:
            settings = CartSettings.get_settings()
            gift_wrap_enabled = bool(settings.gift_wrap_enabled)
            gift_wrap_label = settings.gift_wrap_label or gift_wrap_label
            gift_wrap_amount = Decimal(str(settings.gift_wrap_amount or 0))
            try:
                base_currency = CurrencyService.get_default_currency()
                if base_currency and base_currency.code:
                    gift_wrap_from_code = base_currency.code
            except Exception:
                gift_wrap_from_code = from_code
        except Exception:
            gift_wrap_enabled = False

        try:
            from apps.pages.models import SiteSettings
            tax_rate = Decimal(str(SiteSettings.get_settings().tax_rate or 0))
        except Exception:
            tax_rate = Decimal('0')

        taxable_amount = base_subtotal - base_discount
        if taxable_amount < 0:
            taxable_amount = Decimal('0')

        base_tax = (taxable_amount * tax_rate / Decimal('100')) if tax_rate else Decimal('0')
        gift_wrap_amount_base = gift_wrap_amount
        if gift_wrap_from_code and gift_wrap_from_code != from_code:
            try:
                gift_wrap_amount_base = CurrencyConversionService.convert_by_code(
                    gift_wrap_amount, gift_wrap_from_code, from_code, round_result=True
                )
            except Exception:
                gift_wrap_amount_base = gift_wrap_amount

        gift_wrap_amount_display = gift_wrap_amount
        if currency and gift_wrap_from_code and gift_wrap_from_code != currency.code:
            try:
                gift_wrap_amount_display = CurrencyConversionService.convert_by_code(
                    gift_wrap_amount, gift_wrap_from_code, currency.code, round_result=True
                )
            except Exception:
                gift_wrap_amount_display = gift_wrap_amount
        elif currency and gift_wrap_from_code == currency.code:
            gift_wrap_amount_display = gift_wrap_amount

        if getattr(checkout_session, 'gift_wrap', False):
            base_gift_wrap = Decimal(str(gift_wrap_amount_base))
        base_total = taxable_amount + base_shipping + base_gift_wrap + base_tax

        session_currency_code = getattr(checkout_session, 'currency', None) if checkout_session else None
        use_session_snapshot = bool(session_currency_code and currency and session_currency_code == currency.code)

        def convert_amount(amount: Decimal) -> Decimal:
            if currency and currency.code != from_code:
                return CurrencyConversionService.convert_by_code(
                    amount, from_code, currency.code, round_result=True
                )
            return amount

        def to_decimal(value):
            try:
                return Decimal(str(value))
            except Exception:
                return None

        def is_close(left, right, tolerance=Decimal('0.01')):
            if left is None or right is None:
                return False
            return abs(left - right) <= tolerance

        converted_subtotal = convert_amount(base_subtotal)
        converted_discount = convert_amount(base_discount)
        converted_shipping = convert_amount(base_shipping)
        converted_gift_wrap_amount = Decimal(str(gift_wrap_amount_display))
        converted_gift_wrap = (
            converted_gift_wrap_amount if getattr(checkout_session, 'gift_wrap', False) else Decimal('0')
        )
        converted_tax = convert_amount(base_tax)
        converted_total = convert_amount(base_total)

        if use_session_snapshot and currency and currency.code != from_code and checkout_session:
            decisions = []
            for snapshot_value, base_value, converted_value in [
                (getattr(checkout_session, 'subtotal', None), base_subtotal, converted_subtotal),
                (getattr(checkout_session, 'shipping_cost', None), base_shipping, converted_shipping),
                (getattr(checkout_session, 'total', None), base_total, converted_total),
            ]:
                snapshot_decimal = to_decimal(snapshot_value)
                if snapshot_decimal is None:
                    continue
                if is_close(snapshot_decimal, converted_value):
                    decisions.append(True)
                    continue
                if is_close(snapshot_decimal, base_value):
                    decisions.append(False)
                    continue
                decisions.append(abs(snapshot_decimal - converted_value) < abs(snapshot_decimal - base_value))
            if not decisions:
                use_session_snapshot = False
            else:
                use_session_snapshot = decisions.count(True) >= decisions.count(False)

        if use_session_snapshot:
            def session_or_convert(value, fallback):
                if value is None or value == '':
                    return fallback
                return Decimal(str(value))

            def session_or_convert_with_base(value, fallback, base_value):
                if value is None or value == '':
                    return fallback
                snapshot = Decimal(str(value))
                if currency and currency.code != from_code:
                    if is_close(snapshot, base_value) and not is_close(snapshot, fallback):
                        return fallback
                return snapshot

            display_subtotal = session_or_convert(
                getattr(checkout_session, 'subtotal', None), converted_subtotal
            )
            display_discount = session_or_convert(
                getattr(checkout_session, 'discount_amount', None), converted_discount
            )
            display_shipping = session_or_convert(
                getattr(checkout_session, 'shipping_cost', None), converted_shipping
            )
            display_gift_wrap = session_or_convert_with_base(
                getattr(checkout_session, 'gift_wrap_cost', None),
                converted_gift_wrap,
                base_gift_wrap,
            )
            display_gift_wrap_amount = converted_gift_wrap_amount
            display_tax = session_or_convert(
                getattr(checkout_session, 'tax_amount', None), converted_tax
            )
            if getattr(checkout_session, 'total', None) is not None:
                display_total = Decimal(str(getattr(checkout_session, 'total')))
            else:
                display_total = display_subtotal - display_discount + display_shipping + display_gift_wrap + display_tax
        else:
            display_subtotal = converted_subtotal
            display_discount = converted_discount
            display_shipping = converted_shipping
            display_gift_wrap = converted_gift_wrap
            display_gift_wrap_amount = converted_gift_wrap_amount
            display_tax = converted_tax
            display_total = converted_total

        summary['subtotal'] = str(display_subtotal)
        summary['discount_amount'] = str(display_discount)
        summary['discount'] = summary['discount_amount']
        summary['shipping_cost'] = str(display_shipping)
        summary['shipping_estimate'] = shipping_estimate
        summary['shipping_selected'] = shipping_selected
        summary['gift_wrap_cost'] = str(display_gift_wrap)
        summary['gift_wrap_amount'] = str(display_gift_wrap_amount)
        summary['tax_amount'] = str(display_tax)
        summary['total'] = str(display_total)
        summary['tax_rate'] = str(tax_rate)

        summary['formatted_subtotal'] = currency.format_amount(display_subtotal)
        summary['formatted_discount'] = currency.format_amount(display_discount)
        summary['formatted_shipping'] = currency.format_amount(display_shipping)
        summary['formatted_gift_wrap'] = currency.format_amount(display_gift_wrap)
        summary['formatted_gift_wrap_amount'] = currency.format_amount(display_gift_wrap_amount)
        summary['formatted_tax'] = currency.format_amount(display_tax)
        summary['formatted_total'] = currency.format_amount(display_total)
        summary['gift_wrap_label'] = gift_wrap_label
        summary['gift_wrap_enabled'] = gift_wrap_enabled

        for item in summary.get('items', []):
            try:
                item_unit = Decimal(str(item.get('unit_price') or '0'))
                item_total = Decimal(str(item.get('total') or '0'))
                item['formatted_unit_price'] = currency.format_amount(item_unit)
                item['formatted_total'] = currency.format_amount(item_total)
            except Exception:
                pass

    return summary


def sync_checkout_snapshot(request, cart, checkout_session):
    """Persist currency/tax snapshot from the checkout summary onto the session."""
    if not cart or not checkout_session:
        return None

    summary = build_checkout_cart_summary(request, cart, checkout_session)

    try:
        checkout_session.tax_rate = Decimal(str(summary.get('tax_rate') or 0))
        checkout_session.tax_amount = Decimal(str(summary.get('tax_amount') or 0))
    except Exception:
        checkout_session.tax_rate = Decimal('0')
        checkout_session.tax_amount = Decimal('0')

    try:
        checkout_session.subtotal = Decimal(str(summary.get('subtotal') or 0))
        checkout_session.discount_amount = Decimal(str(summary.get('discount_amount') or 0))
        checkout_session.shipping_cost = Decimal(str(summary.get('shipping_cost') or 0))
        checkout_session.total = Decimal(str(summary.get('total') or 0))
    except Exception:
        checkout_session.subtotal = Decimal('0')
        checkout_session.discount_amount = Decimal('0')
        checkout_session.shipping_cost = Decimal('0')
        checkout_session.total = Decimal('0')

    currency_code = summary.get('currency_code')
    if currency_code:
        checkout_session.currency = currency_code

        try:
            base_currency = CurrencyService.get_default_currency()
            if base_currency and base_currency.code != currency_code:
                checkout_session.exchange_rate = CurrencyConversionService.convert_by_code(
                    Decimal('1'), base_currency.code, currency_code, round_result=False
                )
            else:
                checkout_session.exchange_rate = Decimal('1')
        except Exception:
            checkout_session.exchange_rate = Decimal('1')

    # Payment fee snapshot (use selected gateway)
    payment_fee_amount = Decimal('0')
    payment_fee_label = ''
    if getattr(checkout_session, 'payment_method', None):
        gateway = PaymentGateway.objects.filter(
            code=checkout_session.payment_method,
            is_active=True
        ).first()
        if gateway:
            try:
                base_total = get_checkout_base_total(cart, checkout_session)
                base_fee = Decimal(str(gateway.calculate_fee(base_total) or 0))
            except Exception:
                base_fee = Decimal('0')

            payment_fee_label = gateway.name or gateway.code
            payment_fee_amount = base_fee

    checkout_session.payment_fee_amount = payment_fee_amount
    checkout_session.payment_fee_label = payment_fee_label

    checkout_session.save(update_fields=[
        'tax_rate',
        'tax_amount',
        'subtotal',
        'discount_amount',
        'shipping_cost',
        'total',
        'currency',
        'exchange_rate',
        'payment_fee_amount',
        'payment_fee_label',
    ])
    return summary


def get_checkout_base_total(cart, checkout_session) -> Decimal:
    """Calculate checkout total in base currency (before conversion)."""
    if not cart:
        return Decimal('0')

    base_subtotal = Decimal(str(cart.subtotal or 0))
    base_discount = Decimal(str(cart.discount_amount or 0))
    base_shipping = Decimal(str(getattr(checkout_session, 'shipping_cost', 0) or 0))

    try:
        from apps.pages.models import SiteSettings
        tax_rate = Decimal(str(SiteSettings.get_settings().tax_rate or 0))
    except Exception:
        tax_rate = Decimal('0')

    taxable_amount = base_subtotal - base_discount
    if taxable_amount < 0:
        taxable_amount = Decimal('0')

    base_tax = (taxable_amount * tax_rate / Decimal('100')) if tax_rate else Decimal('0')
    return taxable_amount + base_shipping + base_tax


def get_payment_fee_context(cart, checkout_session, request=None, payment_gateways=None):
    """Return payment fee info for the selected gateway (base currency)."""
    gateway = None
    fee_amount = Decimal('0')

    if checkout_session and getattr(checkout_session, 'payment_method', None):
        gateway = PaymentGateway.objects.filter(
            code=checkout_session.payment_method,
            is_active=True
        ).first()

    if not gateway and payment_gateways:
        try:
            gateway = payment_gateways[0] if len(payment_gateways) > 0 else None
        except Exception:
            gateway = None

    if gateway:
        try:
            base_total = get_checkout_base_total(cart, checkout_session)
            fee_amount = Decimal(str(gateway.calculate_fee(base_total) or 0))
        except Exception:
            fee_amount = Decimal('0')

    # Convert to user's currency for data attributes/JS usage
    fee_amount_converted = fee_amount
    try:
        target_currency = CurrencyService.get_user_currency(
            user=request.user if request and getattr(request, 'user', None) and request.user.is_authenticated else None,
            request=request
        )
        base_currency = CurrencyService.get_default_currency()
        if target_currency and base_currency and base_currency.code != target_currency.code:
            fee_amount_converted = CurrencyConversionService.convert_by_code(
                fee_amount, base_currency.code, target_currency.code, round_result=True
            )
    except Exception:
        fee_amount_converted = fee_amount

    return {
        'payment_fee_amount': fee_amount,
        'payment_fee_amount_converted': fee_amount_converted,
        'payment_fee_gateway': gateway,
    }


# =============================================================================
# Cart Views
# =============================================================================

class CartView(TemplateView):
    """Cart page view."""
    template_name = 'cart/cart.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        user = self.request.user if self.request.user.is_authenticated else None
        session_key = self.request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        context['cart'] = cart
        context['cart_items'] = cart.items.select_related('product', 'variant').all() if cart else []
        context['cart_summary'] = CartService.get_cart_summary(cart) if cart else None

        gift_state = {
            'is_gift': False,
            'gift_message': '',
            'gift_wrap': False,
            'gift_wrap_cost': '0',
            'formatted_gift_wrap_cost': '',
            'gift_wrap_amount': '0',
        }
        gift_wrap_enabled = False
        gift_wrap_label = 'Gift Wrap'
        gift_wrap_amount = Decimal('0')
        formatted_gift_wrap = ''

        if cart:
            try:
                settings = CartSettings.get_settings()
                gift_wrap_enabled = bool(settings.gift_wrap_enabled)
                gift_wrap_label = settings.gift_wrap_label or gift_wrap_label
                gift_wrap_amount = Decimal(str(settings.gift_wrap_amount or 0))
            except Exception:
                gift_wrap_enabled = False

            checkout_session = CheckoutService.get_active_session(
                cart=cart,
                user=user,
                session_key=session_key
            )

            if checkout_session:
                gift_state['is_gift'] = bool(checkout_session.is_gift)
                gift_state['gift_message'] = checkout_session.gift_message or ''
                gift_state['gift_wrap'] = bool(checkout_session.gift_wrap) if gift_wrap_enabled else False

            from_code = getattr(cart, 'currency', None) or 'BDT'
            target_currency = CurrencyService.get_user_currency(request=self.request) or CurrencyService.get_default_currency()

            display_gift_wrap_amount = gift_wrap_amount
            if target_currency and target_currency.code != from_code:
                try:
                    display_gift_wrap_amount = CurrencyConversionService.convert_by_code(
                        gift_wrap_amount, from_code, target_currency.code, round_result=True
                    )
                except Exception:
                    display_gift_wrap_amount = gift_wrap_amount

            formatted_gift_wrap = (
                target_currency.format_amount(display_gift_wrap_amount) if target_currency else str(display_gift_wrap_amount)
            )

            gift_wrap_cost = display_gift_wrap_amount if (gift_state['gift_wrap'] and gift_wrap_enabled) else Decimal('0')
            formatted_gift_wrap_cost = (
                target_currency.format_amount(gift_wrap_cost) if target_currency else str(gift_wrap_cost)
            )

            gift_state['gift_wrap_cost'] = str(gift_wrap_cost)
            gift_state['formatted_gift_wrap_cost'] = formatted_gift_wrap_cost
            gift_state['gift_wrap_amount'] = str(display_gift_wrap_amount)

        context['gift_state'] = gift_state
        context['gift_wrap_enabled'] = gift_wrap_enabled
        context['gift_wrap_label'] = gift_wrap_label
        context['gift_wrap_amount'] = gift_state.get('gift_wrap_amount', '0')
        context['formatted_gift_wrap'] = formatted_gift_wrap
        
        return context


class SharedCartView(TemplateView):
    """Shared cart view (read-only)."""
    template_name = 'cart/shared_cart.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        token = kwargs.get('token')
        password = self.request.GET.get('password')
        cart = EnhancedCartService.get_shared_cart(token, password=password)

        if not cart:
            raise Http404("Shared cart not found or expired")

        context['cart'] = cart
        context['cart_items'] = cart.items.select_related('product', 'variant').all()
        context['cart_summary'] = CartService.get_cart_summary(cart)
        return context


class CartAddView(View):
    """Add item to cart."""
    
    def post(self, request):
        from apps.catalog.models import Product, ProductVariant
        
        product_id = request.POST.get('product_id')
        variant_id = request.POST.get('variant_id')
        quantity = int(request.POST.get('quantity', 1))
        
        try:
            product = Product.objects.get(id=product_id, is_active=True, is_deleted=False)
            variant = ProductVariant.objects.get(id=variant_id) if variant_id else None
            
            user = request.user if request.user.is_authenticated else None
            if not request.session.session_key:
                request.session.create()
            session_key = request.session.session_key
            
            cart = CartService.get_or_create_cart(user=user, session_key=session_key)
            item = CartService.add_item(cart, product, quantity, variant)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': f'{product.name} added to cart',
                    'cart_count': cart.item_count,
                    'cart_total': str(cart.total),
                })
            
            messages.success(request, f'{product.name} added to cart')
            return redirect('commerce:cart')
            
        except Product.DoesNotExist:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Product not found'}, status=404)
            messages.error(request, 'Product not found')
            return redirect('catalog:product_list')
            
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': str(e)}, status=400)
            messages.error(request, str(e))
            return redirect(request.META.get('HTTP_REFERER', 'commerce:cart'))


class CartUpdateView(View):
    """Update cart item quantity."""
    
    def post(self, request):
        item_id = request.POST.get('item_id')
        quantity = int(request.POST.get('quantity', 1))
        
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Cart not found'}, status=404)
            messages.error(request, 'Cart not found')
            return redirect('commerce:cart')
        
        try:
            CartService.update_item_quantity(cart, item_id, quantity)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': 'Cart updated',
                    'cart_count': cart.item_count,
                    'cart_total': str(cart.total),
                    'summary': CartService.get_cart_summary(cart),
                })
            
            messages.success(request, 'Cart updated')
            
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': str(e)}, status=400)
            messages.error(request, str(e))
        
        return redirect('commerce:cart')


class CartRemoveView(View):
    """Remove item from cart."""
    
    def post(self, request):
        item_id = request.POST.get('item_id')
        
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if cart:
            CartService.remove_item(cart, item_id)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': 'Item removed',
                    'cart_count': cart.item_count,
                    'cart_total': str(cart.total),
                })
            
            messages.success(request, 'Item removed from cart')
        
        return redirect('commerce:cart')


class CartClearView(View):
    """Clear all items from cart."""
    
    def post(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if cart:
            CartService.clear_cart(cart)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'message': 'Cart cleared'})
            
            messages.success(request, 'Cart cleared')
        
        return redirect('commerce:cart')


class CartApplyCouponView(View):
    """Apply coupon to cart."""
    
    def post(self, request):
        coupon_code = request.POST.get('coupon_code', '').strip()
        
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Cart not found'}, status=404)
            messages.error(request, 'Cart not found')
            return redirect('commerce:cart')
        
        try:
            CartService.apply_coupon(cart, coupon_code)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': 'Coupon applied',
                    'summary': CartService.get_cart_summary(cart),
                })
            
            messages.success(request, 'Coupon applied successfully')
            
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': str(e)}, status=400)
            messages.error(request, str(e))
        
        return redirect('commerce:cart')


class CartRemoveCouponView(View):
    """Remove coupon from cart."""
    
    def post(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if cart:
            CartService.remove_coupon(cart)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': 'Coupon removed',
                    'summary': CartService.get_cart_summary(cart),
                })
            
            messages.success(request, 'Coupon removed')
        
        return redirect('commerce:cart')


# =============================================================================
# Wishlist Views
# =============================================================================

class WishlistView(LoginRequiredMixin, TemplateView):
    """Wishlist page view."""
    template_name = 'wishlist/list.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        wishlist = WishlistService.get_or_create_wishlist(self.request.user)
        items = wishlist.items.select_related('product', 'variant').prefetch_related('product__images').all()
        
        context['wishlist'] = wishlist
        context['wishlist_items'] = items
        context['price_drop_items'] = WishlistService.get_items_with_price_drops(wishlist)
        
        return context


class WishlistAddView(LoginRequiredMixin, View):
    """Add item to wishlist."""
    
    def post(self, request):
        from apps.catalog.models import Product, ProductVariant
        
        product_id = request.POST.get('product_id')
        variant_id = request.POST.get('variant_id')
        
        try:
            product = Product.objects.get(id=product_id, is_active=True, is_deleted=False)
            variant = ProductVariant.objects.get(id=variant_id) if variant_id else None
            
            wishlist = WishlistService.get_or_create_wishlist(request.user)
            WishlistService.add_item(wishlist, product, variant)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': f'{product.name} added to wishlist',
                    'wishlist_count': wishlist.item_count,
                })
            
            messages.success(request, f'{product.name} added to wishlist')
            
        except Product.DoesNotExist:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Product not found'}, status=404)
            messages.error(request, 'Product not found')
            
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': str(e)}, status=400)
            messages.error(request, str(e))
        
        return redirect(request.META.get('HTTP_REFERER', 'commerce:wishlist'))


class WishlistRemoveView(LoginRequiredMixin, View):
    """Remove item from wishlist."""
    
    def post(self, request):
        item_id = request.POST.get('item_id')
        
        wishlist = WishlistService.get_or_create_wishlist(request.user)
        WishlistService.remove_item(wishlist, item_id)
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': 'Item removed from wishlist',
                'wishlist_count': wishlist.item_count,
            })
        
        messages.success(request, 'Item removed from wishlist')
        return redirect('commerce:wishlist')


class WishlistMoveToCartView(LoginRequiredMixin, View):
    """Move wishlist item to cart."""
    
    def post(self, request):
        item_id = request.POST.get('item_id')
        
        try:
            wishlist = WishlistService.get_or_create_wishlist(request.user)
            item = wishlist.items.get(id=item_id)
            
            if not request.session.session_key:
                request.session.create()
            
            cart = CartService.get_or_create_cart(
                user=request.user,
                session_key=request.session.session_key
            )
            
            WishlistService.move_to_cart(item, cart)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': 'Item moved to cart',
                    'wishlist_count': wishlist.item_count,
                    'cart_count': cart.item_count,
                })
            
            messages.success(request, 'Item moved to cart')
            
        except WishlistItem.DoesNotExist:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Item not found'}, status=404)
            messages.error(request, 'Item not found')
            
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': str(e)}, status=400)
            messages.error(request, str(e))
        
        return redirect('commerce:wishlist')


class WishlistShareView(LoginRequiredMixin, View):
    """Create shareable wishlist link."""
    
    def post(self, request):
        wishlist = WishlistService.get_or_create_wishlist(request.user)
        
        share = WishlistService.create_share_link(
            wishlist,
            expires_days=request.POST.get('expires_days', 30),
            allow_purchase=request.POST.get('allow_purchase') == 'true'
        )
        
        share_url = request.build_absolute_uri(
            reverse('commerce:shared_wishlist', kwargs={'token': share.share_token})
        )
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'share_url': share_url,
                'token': share.share_token,
            })
        
        messages.success(request, f'Share link created: {share_url}')
        return redirect('commerce:wishlist')


class SharedWishlistView(TemplateView):
    """View a shared wishlist."""
    template_name = 'wishlist/shared.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        token = kwargs.get('token')
        wishlist = WishlistService.get_shared_wishlist(token)
        
        if not wishlist:
            raise Http404("Wishlist not found or expired")
        
        context['wishlist'] = wishlist
        context['wishlist_items'] = wishlist.items.select_related('product').prefetch_related('product__images').all()
        context['share'] = WishlistShare.objects.get(share_token=token)
        
        return context


# =============================================================================
# Checkout Views
# =============================================================================

class CheckoutView(TemplateView):
    """Main checkout page."""
    template_name = 'checkout/information.html'
    
    def get(self, request, *args, **kwargs):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart or not cart.items.exists():
            messages.warning(request, 'Your cart is empty')
            return redirect('commerce:cart')
        
        return super().get(request, *args, **kwargs)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        user = self.request.user if self.request.user.is_authenticated else None
        session_key = self.request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key,
            request=self.request
        )
        normalize_checkout_country(checkout_session)
        
        context['cart'] = cart
        context['cart_items'] = cart.items.select_related('product', 'variant').all()
        context['cart_summary'] = build_checkout_cart_summary(self.request, cart, checkout_session)
        context['checkout_session'] = checkout_session
        context['checkout'] = checkout_session
        context['shipping_methods'] = self._get_shipping_methods()
        context['payment_methods'] = self._get_payment_methods()
        if user:
            context['saved_addresses'] = AddressService.get_user_addresses(user)
        else:
            context['saved_addresses'] = []
        context['countries'] = GeoService.get_all_countries()
        
        return context
    
    def _get_shipping_methods(self):
        """Get available shipping methods."""
        return [
            {'value': 'standard', 'label': 'Standard Shipping (5-7 days)', 'price': '60.00'},
            {'value': 'express', 'label': 'Express Shipping (2-3 days)', 'price': '120.00'},
            {'value': 'overnight', 'label': 'Overnight Shipping', 'price': '200.00'},
            {'value': 'pickup', 'label': 'Store Pickup', 'price': '0.00'},
        ]
    
    def _get_payment_methods(self):
        """Get available payment methods."""
        return [
            {'value': 'cod', 'label': 'Cash on Delivery', 'description': '+৳20 COD fee'},
            {'value': 'bkash', 'label': 'bKash', 'description': 'Pay with bKash'},
            {'value': 'nagad', 'label': 'Nagad', 'description': 'Pay with Nagad'},
            {'value': 'card', 'label': 'Credit/Debit Card', 'description': 'Visa, Mastercard'},
        ]


class CheckoutUpdateInfoView(View):
    """Update checkout shipping information."""
    
    def post(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Cart not found'}, status=404)
            messages.error(request, 'Cart not found')
            return redirect('commerce:cart')
        
        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key
        )
        
        full_name = (request.POST.get('full_name') or '').strip()
        first_name = (request.POST.get('first_name') or '').strip()
        last_name = (request.POST.get('last_name') or '').strip()

        if full_name and (not first_name or not last_name):
            parts = [p for p in full_name.split(' ') if p]
            if not first_name and parts:
                first_name = parts[0]
            if not last_name and len(parts) > 1:
                last_name = ' '.join(parts[1:])

        if not first_name and user:
            first_name = user.first_name or ''
        if not last_name and user:
            last_name = user.last_name or ''

        country_value = (request.POST.get('country') or '').strip() or 'Bangladesh'

        data = {
            'email': (request.POST.get('email') or '').strip(),
            'shipping_first_name': first_name or '',
            'shipping_last_name': last_name or '',
            'shipping_email': (request.POST.get('email') or '').strip(),
            'shipping_phone': (request.POST.get('phone') or '').strip(),
            'shipping_address_line_1': request.POST.get('address_line_1') or request.POST.get('address_line1') or '',
            'shipping_address_line_2': request.POST.get('address_line_2') or request.POST.get('address_line2') or '',
            'shipping_city': (request.POST.get('city') or '').strip(),
            'shipping_state': (request.POST.get('state') or '').strip(),
            'shipping_postal_code': (request.POST.get('postal_code') or '').strip(),
            'shipping_country': country_value,
        }

        required_fields = [
            'email',
            'shipping_first_name',
            'shipping_address_line_1',
            'shipping_city',
            'shipping_postal_code',
            'shipping_country',
        ]
        missing = [field for field in required_fields if not data.get(field)]
        if missing:
            errors = {field: 'This field is required.' for field in missing}
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'message': 'Please fill in all required fields.',
                    'errors': errors,
                }, status=400)
            messages.error(request, 'Please fill in all required fields.')
            return redirect('commerce:checkout')

        # Gift options
        is_gift = str(request.POST.get('is_gift', '')).lower() in {'1', 'true', 'on', 'yes'}
        gift_message = (request.POST.get('gift_message') or '').strip() if is_gift else ''
        gift_wrap = str(request.POST.get('gift_wrap', '')).lower() in {'1', 'true', 'on', 'yes'}
        gift_wrap_cost = Decimal('0')
        try:
            from .models import CartSettings
            settings = CartSettings.get_settings()
            if gift_wrap and settings.gift_wrap_enabled:
                gift_wrap_cost = Decimal(str(settings.gift_wrap_amount or 0))
        except Exception:
            gift_wrap_cost = Decimal('0')

        try:
            CheckoutService.update_shipping_info(checkout_session, data)
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
            sync_checkout_snapshot(request, cart, checkout_session)
        except Exception as exc:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': str(exc)}, status=400)
            messages.error(request, str(exc))
            return redirect('commerce:checkout')

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': 'Information saved',
                'redirect_url': reverse('commerce:checkout_shipping')
            })

        return redirect('commerce:checkout_shipping')


class CheckoutSelectShippingView(View):
    """Select shipping method."""
    
    def get(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key

        cart = CartService.get_cart(user=user, session_key=session_key)
        if not cart or not cart.items.exists():
            messages.warning(request, 'Your cart is empty')
            return redirect('commerce:cart')

        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key,
            request=request
        )
        normalize_checkout_country(checkout_session)

        cart_summary = build_checkout_cart_summary(request, cart, checkout_session)

        country_code = normalize_checkout_country(checkout_session) or 'BD'
        state = (checkout_session.shipping_state or '').strip()
        city = (checkout_session.shipping_city or '').strip()
        postal_code = (checkout_session.shipping_postal_code or '').strip()

        shipping_methods = ShippingRateService.get_available_methods(
            country=country_code,
            state=state or None,
            city=city or None,
            postal_code=postal_code or None,
            subtotal=Decimal(str(cart.subtotal or 0)),
            item_count=cart.item_count,
            product_ids=[str(it.product_id) for it in cart.items.all()],
            currency_code=cart_summary.get('currency_code') or cart.currency
        )
        if shipping_methods:
            shipping_methods = shipping_methods[:5]

        shipping_options = []
        for method in shipping_methods:
            carrier_name = method.get('carrier', {}).get('name') if isinstance(method.get('carrier'), dict) else None
            shipping_options.append({
                'id': method.get('id'),
                'rate_id': method.get('rate_id'),
                'name': method.get('name'),
                'cost': method.get('rate'),
                'formatted_cost': method.get('rate_display'),
                'is_free': method.get('is_free'),
                'delivery_estimate': method.get('delivery_estimate'),
                'zone': None,
                'carrier': carrier_name,
            })

        country_obj = GeoService.get_country_by_code(country_code) if country_code else None

        context = {
            'cart': cart,
            'cart_items': cart.items.select_related('product', 'variant').all(),
            'cart_summary': cart_summary,
            'checkout_summary': cart_summary,
            'checkout_session': checkout_session,
            'checkout': checkout_session,
            'shipping_options': shipping_options,
            'shipping_country_name': country_obj.name if country_obj else None,
        }

        return render(request, 'checkout/shipping.html', context)

    def post(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Cart not found'}, status=404)
            messages.error(request, 'Cart not found')
            return redirect('commerce:cart')
        
        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key
        )

        # Preserve gift wrap settings from checkout form
        gift_wrap = str(request.POST.get('gift_wrap', '')).lower() in {'1', 'true', 'on', 'yes'}
        is_gift = str(request.POST.get('is_gift', '')).lower() in {'1', 'true', 'on', 'yes'}
        gift_message = (request.POST.get('gift_message') or '').strip() if is_gift else ''
        
        if gift_wrap or gift_message or is_gift:
            gift_wrap_cost = Decimal('0')
            try:
                from .models import CartSettings
                settings = CartSettings.get_settings()
                if gift_wrap and settings.gift_wrap_enabled:
                    gift_wrap_cost = Decimal(str(settings.gift_wrap_amount or 0))
            except Exception:
                gift_wrap_cost = Decimal('0')
            
            checkout_session.gift_wrap = gift_wrap
            checkout_session.gift_wrap_cost = gift_wrap_cost
            checkout_session.is_gift = is_gift
            checkout_session.gift_message = gift_message
            checkout_session.save(update_fields=['gift_wrap', 'gift_wrap_cost', 'is_gift', 'gift_message'])

        shipping_type = (request.POST.get('shipping_type') or 'delivery').strip()
        if shipping_type == 'pickup':
            pickup_location_id = request.POST.get('pickup_location')
            if pickup_location_id:
                try:
                    from apps.contacts.models import StoreLocation
                    location = StoreLocation.objects.filter(pk=pickup_location_id).first()
                    if location:
                        checkout_session.pickup_location = location
                except Exception:
                    pass

            checkout_session.shipping_rate = None
            CheckoutService.select_shipping_method(checkout_session, CheckoutSession.SHIPPING_PICKUP)
            sync_checkout_snapshot(request, cart, checkout_session)

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': 'Pickup selected',
                    'shipping_cost': str(checkout_session.shipping_cost),
                    'total': str(checkout_session.total),
                    'redirect_url': reverse('commerce:checkout_payment')
                })

            return redirect('commerce:checkout_payment')

        rate_id = request.POST.get('shipping_rate_id') or request.POST.get('shipping_method')
        if not rate_id:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Please select a shipping method.'}, status=400)
            messages.error(request, 'Please select a shipping method.')
            return redirect('commerce:checkout_shipping')

        rate = None
        try:
            rate = ShippingRate.objects.select_related('method').get(pk=rate_id)
        except (ShippingRate.DoesNotExist, ValueError, TypeError):
            rate = None

        country_code = normalize_checkout_country(checkout_session) or 'BD'
        state = (checkout_session.shipping_state or '').strip() or None
        city = (checkout_session.shipping_city or '').strip() or None
        postal_code = (checkout_session.shipping_postal_code or '').strip() or None

        if not rate:
            zone = ShippingZoneService.find_zone_for_location(country_code, state, postal_code, city=city)
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

        if not rate:
            message = 'Invalid shipping rate. Please update your address and try again.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': message}, status=400)
            messages.error(request, message)
            return redirect('commerce:checkout_shipping')

        method = rate.method.code or rate.method.name or checkout_session.shipping_method
        CheckoutService.select_shipping_method(checkout_session, method, shipping_rate=rate)
        sync_checkout_snapshot(request, cart, checkout_session)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': 'Shipping method selected',
                'shipping_cost': str(checkout_session.shipping_cost),
                'total': str(checkout_session.total),
                'redirect_url': reverse('commerce:checkout_payment')
            })

        return redirect('commerce:checkout_payment')


class CheckoutSelectPaymentView(View):
    """Select payment method."""
    
    def get(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key

        cart = CartService.get_cart(user=user, session_key=session_key)
        if not cart or not cart.items.exists():
            messages.warning(request, 'Your cart is empty')
            return redirect('commerce:cart')

        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key,
            request=request
        )
        normalize_checkout_country(checkout_session)

        cart_summary = build_checkout_cart_summary(request, cart, checkout_session)
        country_code = normalize_checkout_country(checkout_session)
        country_obj = GeoService.get_country_by_code(country_code) if country_code else None
        countries = GeoService.get_all_countries()
        country_names = {c.code: c.name for c in countries} if countries else {}

        currency_code = cart_summary.get('currency_code') or getattr(cart, 'currency', None)
        amount = None
        try:
            amount = Decimal(str(cart_summary.get('total') or cart.total or 0))
        except Exception:
            amount = None

        payment_gateways = PaymentGateway.get_active_gateways(
            currency=currency_code,
            country=country_code,
            amount=amount
        )
        base_currency = CurrencyService.get_default_currency()
        target_currency = CurrencyService.get_user_currency(request=request)
        base_total = get_checkout_base_total(cart, checkout_session)

        for gateway in payment_gateways:
            try:
                if gateway.code == PaymentGateway.CODE_COD:
                    base_fee = Decimal('0')
                    gateway.fee_text = 'No extra fee'
                else:
                    base_fee = Decimal(str(gateway.calculate_fee(base_total) or 0))
                if target_currency and base_currency and base_currency.code != target_currency.code:
                    gateway.fee_amount_converted = CurrencyConversionService.convert_by_code(
                        base_fee, base_currency.code, target_currency.code, round_result=True
                    )
                else:
                    gateway.fee_amount_converted = base_fee
            except Exception:
                gateway.fee_amount_converted = Decimal('0')

        payment_fee_ctx = get_payment_fee_context(
            cart,
            checkout_session,
            request=request,
            payment_gateways=payment_gateways
        )

        context = {
            'cart': cart,
            'cart_items': cart.items.select_related('product', 'variant').all(),
            'cart_summary': cart_summary,
            'checkout_summary': cart_summary,
            'checkout_session': checkout_session,
            'checkout': checkout_session,
            'shipping_country_name': country_obj.name if country_obj else None,
            'shipping_country_code': country_code,
            'stripe_publishable_key': getattr(settings, 'STRIPE_PUBLIC_KEY', ''),
            'countries': countries,
            'country_names': country_names,
            'payment_gateways': payment_gateways,
            **payment_fee_ctx,
        }

        return render(request, 'checkout/payment.html', context)

    def post(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Cart not found'}, status=404)
            messages.error(request, 'Cart not found')
            return redirect('commerce:cart')
        
        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key
        )
        
        method = request.POST.get('payment_method')
        if not method:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Please select a payment method.'}, status=400)
            messages.error(request, 'Please select a payment method.')
            return redirect('commerce:checkout_payment')

        CheckoutService.select_payment_method(checkout_session, method)
        sync_checkout_snapshot(request, cart, checkout_session)
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': 'Payment method selected',
                'redirect_url': reverse('commerce:checkout_review')
            })
        
        return redirect('commerce:checkout_review')


class CheckoutReviewView(TemplateView):
    """Review order page."""
    template_name = 'checkout/review.html'

    def get(self, request, *args, **kwargs):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key

        cart = CartService.get_cart(user=user, session_key=session_key)
        if not cart or not cart.items.exists():
            messages.warning(request, 'Your cart is empty')
            return redirect('commerce:cart')

        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user = self.request.user if self.request.user.is_authenticated else None
        session_key = self.request.session.session_key

        cart = CartService.get_cart(user=user, session_key=session_key)
        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key,
            request=self.request
        )
        normalize_checkout_country(checkout_session)

        cart_summary = build_checkout_cart_summary(self.request, cart, checkout_session)
        shipping_country_code = normalize_checkout_country(checkout_session)
        shipping_country = (
            GeoService.get_country_by_code(shipping_country_code)
            if shipping_country_code
            else None
        )
        billing_country_value = getattr(checkout_session, 'billing_country', None)
        billing_country_code = (
            CheckoutService.normalize_country_code(billing_country_value)
            if billing_country_value
            else None
        )
        billing_country = (
            GeoService.get_country_by_code(billing_country_code)
            if billing_country_code
            else None
        )
        payment_fee_ctx = get_payment_fee_context(cart, checkout_session, request=self.request)

        context.update({
            'cart': cart,
            'cart_items': cart.items.select_related('product', 'variant').all(),
            'cart_summary': cart_summary,
            'checkout_summary': cart_summary,
            'checkout_session': checkout_session,
            'checkout': checkout_session,
            'shipping_country_name': shipping_country.name if shipping_country else None,
            'billing_country_name': billing_country.name if billing_country else None,
            **payment_fee_ctx,
        })

        return context


class CheckoutCompleteView(View):
    """Complete checkout and place order."""
    
    def post(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        if not cart:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Cart not found'}, status=404)
            messages.error(request, 'Cart not found')
            return redirect('commerce:cart')
        
        checkout_session = CheckoutService.get_or_create_session(
            cart=cart,
            user=user,
            session_key=session_key
        )
        
        try:
            sync_checkout_snapshot(request, cart, checkout_session)
            order = CheckoutService.complete_checkout(checkout_session)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': 'Order placed successfully',
                    'order_number': order.order_number,
                    'redirect_url': reverse('commerce:order_confirmation', kwargs={'order_number': order.order_number})
                })

            messages.success(request, f'Order placed successfully! Order number: {order.order_number}')
            return redirect('commerce:order_confirmation', order_number=order.order_number)
            
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': str(e)}, status=400)
            messages.error(request, str(e))
            return redirect('commerce:checkout_review')


class OrderConfirmationView(TemplateView):
    """Order confirmation page."""
    template_name = 'checkout/confirmation.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        order_number = kwargs.get('order_number')
        user = self.request.user if self.request.user.is_authenticated else None
        
        # Get order from orders app
        from apps.orders.services import OrderService
        order = OrderService.get_order_by_number(order_number, user)
        
        if not order:
            raise Http404("Order not found")
        
        context['order'] = order
        order_items = order.items.all()
        context['order_items'] = order_items

        # Currency-aware totals for confirmation page (use order snapshot currency)
        try:
            target_currency = CurrencyService.get_currency_by_code(getattr(order, 'currency', None)) if getattr(order, 'currency', None) else CurrencyService.get_default_currency()
        except Exception:
            target_currency = CurrencyService.get_default_currency()

        def to_decimal(value):
            try:
                return Decimal(str(value or 0))
            except Exception:
                return Decimal('0')

        base_subtotal = to_decimal(order.subtotal)
        base_discount = to_decimal(order.discount)
        base_shipping = to_decimal(order.shipping_cost)
        base_tax = to_decimal(order.tax)
        base_gift_wrap = to_decimal(getattr(order, 'gift_wrap_cost', 0))
        base_payment_fee = to_decimal(getattr(order, 'payment_fee_amount', 0))

        base_total = base_subtotal - base_discount + base_shipping + base_tax + base_gift_wrap + base_payment_fee

        display_subtotal = base_subtotal
        display_discount = base_discount
        display_shipping = base_shipping
        display_gift_wrap = base_gift_wrap
        display_payment_fee = base_payment_fee
        display_tax = base_tax
        display_total = display_subtotal - display_discount + display_shipping + display_tax + display_gift_wrap + display_payment_fee

        if target_currency:
            context['formatted_subtotal'] = target_currency.format_amount(display_subtotal)
            context['formatted_discount'] = target_currency.format_amount(display_discount)
            context['formatted_shipping'] = target_currency.format_amount(display_shipping)
            context['formatted_gift_wrap'] = target_currency.format_amount(display_gift_wrap)
            context['formatted_payment_fee'] = target_currency.format_amount(display_payment_fee)
            context['formatted_tax'] = target_currency.format_amount(display_tax)
            context['formatted_total'] = target_currency.format_amount(display_total)
            context['currency_symbol_local'] = getattr(target_currency, 'symbol', None) or getattr(target_currency, 'native_symbol', None) or ''
            context['currency_code_local'] = target_currency.code
        else:
            context['formatted_subtotal'] = str(display_subtotal)
            context['formatted_discount'] = str(display_discount)
            context['formatted_shipping'] = str(display_shipping)
            context['formatted_gift_wrap'] = str(display_gift_wrap)
            context['formatted_payment_fee'] = str(display_payment_fee)
            context['formatted_tax'] = str(display_tax)
            context['formatted_total'] = str(display_total)
            context['currency_symbol_local'] = ''
            context['currency_code_local'] = ''

        context['display_shipping'] = display_shipping
        context['display_gift_wrap'] = display_gift_wrap
        context['display_payment_fee'] = display_payment_fee

        # Attach formatted item prices for templates
        for item in order_items:
            unit_price = to_decimal(item.unit_price)
            line_total = to_decimal(item.line_total)
            display_unit = unit_price
            display_line = line_total
            try:
                item.formatted_unit_price = target_currency.format_amount(display_unit) if target_currency else str(display_unit)
                item.formatted_line_total = target_currency.format_amount(display_line) if target_currency else str(display_line)
            except Exception:
                item.formatted_unit_price = str(display_unit)
                item.formatted_line_total = str(display_line)
        
        return context


# =============================================================================
# AJAX/API Helper Views
# =============================================================================

class CartCountView(View):
    """Get current cart item count (AJAX)."""
    
    def get(self, request):
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        cart = CartService.get_cart(user=user, session_key=session_key)
        
        return JsonResponse({
            'count': cart.item_count if cart else 0,
            'total': str(cart.total) if cart else '0',
        })


class WishlistCountView(LoginRequiredMixin, View):
    """Get current wishlist item count (AJAX)."""
    
    def get(self, request):
        try:
            wishlist = Wishlist.objects.get(user=request.user)
            count = wishlist.item_count
        except Wishlist.DoesNotExist:
            count = 0
        
        return JsonResponse({'count': count})


class CheckWishlistView(LoginRequiredMixin, View):
    """Check if product is in wishlist (AJAX)."""
    
    def get(self, request):
        product_id = request.GET.get('product_id')
        
        try:
            wishlist = Wishlist.objects.get(user=request.user)
            in_wishlist = wishlist.items.filter(product_id=product_id).exists()
        except Wishlist.DoesNotExist:
            in_wishlist = False
        
        return JsonResponse({'in_wishlist': in_wishlist})


class ToggleWishlistView(LoginRequiredMixin, View):
    """Toggle product in wishlist (AJAX)."""
    
    def post(self, request):
        from apps.catalog.models import Product
        
        product_id = request.POST.get('product_id')
        
        try:
            product = Product.objects.get(id=product_id)
            wishlist = WishlistService.get_or_create_wishlist(request.user)
            
            # Check if already in wishlist
            existing = wishlist.items.filter(product=product).first()
            
            if existing:
                WishlistService.remove_item(wishlist, existing.id)
                return JsonResponse({
                    'success': True,
                    'in_wishlist': False,
                    'message': 'Removed from wishlist',
                    'count': wishlist.item_count,
                })
            else:
                WishlistService.add_item(wishlist, product)
                return JsonResponse({
                    'success': True,
                    'in_wishlist': True,
                    'message': 'Added to wishlist',
                    'count': wishlist.item_count,
                })
                
        except Product.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Product not found'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)}, status=400)
