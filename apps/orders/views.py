"""
Orders views
"""
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages

from .models import Order
from .services import OrderService


class OrderListView(LoginRequiredMixin, ListView):
    """User order list."""
    model = Order
    template_name = 'accounts/orders.html'
    context_object_name = 'orders'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Order.objects.filter(
            user=self.request.user,
            is_deleted=False
        ).prefetch_related('items')
        
        # Filter by status (validate against allowed choices)
        status = self.request.GET.get('status')
        if status:
            status = str(status).lower()
            allowed = [s[0] for s in Order.STATUS_CHOICES]
            if status in allowed:
                queryset = queryset.filter(status=status)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'My Orders'
        context['status_choices'] = Order.STATUS_CHOICES        # Expose raw status param as `status` (used by template) and also `current_status` for clarity
        status_param = self.request.GET.get('status', '')
        context['status'] = status_param
        context['current_status'] = status_param

        # Compute formatted totals using the order's snapshot currency (no checkout session fallback)
        from apps.i18n.services import CurrencyService
        from decimal import Decimal

        try:
            orders_page = context.get('orders')
            orders_iter = orders_page.object_list if hasattr(orders_page, 'object_list') else list(orders_page)

            default_currency = CurrencyService.get_default_currency()

            for order in orders_iter:
                try:
                    # Determine currency to format with
                    cur = CurrencyService.get_currency_by_code(getattr(order, 'currency', None)) or default_currency
                    display_total = Decimal(str(order.total or 0))
                    order.formatted_total = cur.format_amount(display_total.quantize(Decimal('0.01')))
                except Exception:
                    # Fallback to formatting stored total with default currency
                    try:
                        order.formatted_total = default_currency.format_amount(order.total)
                    except Exception:
                        order.formatted_total = str(order.total)
        except Exception:
            # If anything fails, don't break the page
            pass

        return context


class OrderDetailView(LoginRequiredMixin, DetailView):
    """Order detail page."""
    model = Order
    template_name = 'accounts/order_detail.html'
    context_object_name = 'order'
    
    def get_queryset(self):
        return Order.objects.filter(
            user=self.request.user,
            is_deleted=False
        ).prefetch_related('items', 'status_history')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = f'Order {self.object.order_number}'

        from decimal import Decimal
        from apps.i18n.services import CurrencyService
        currency_symbol_local = None
        currency_code_local = None
        try:
            currency_obj = CurrencyService.get_currency_by_code(getattr(self.object, 'currency', None))
            if currency_obj:
                currency_symbol_local = getattr(currency_obj, 'symbol', None)
                currency_code_local = getattr(currency_obj, 'code', None)
        except Exception:
            currency_symbol_local = None
            currency_code_local = None

        # Determine checkout/display currency
        try:
            checkout_currency = CurrencyService.get_currency_by_code(currency_code_local) if currency_code_local else CurrencyService.get_default_currency()
        except Exception:
            checkout_currency = CurrencyService.get_default_currency()

        # Resolve human-readable country names for display
        try:
            from apps.i18n.models import Country
            shipping_country_obj = Country.objects.filter(code=self.object.shipping_country).first()
            billing_country_obj = Country.objects.filter(code=self.object.billing_country).first()
            context['shipping_country_name'] = shipping_country_obj.name if shipping_country_obj else self.object.shipping_country
            context['billing_country_name'] = billing_country_obj.name if billing_country_obj else self.object.billing_country
        except Exception:
            context['shipping_country_name'] = self.object.shipping_country
            context['billing_country_name'] = self.object.billing_country

        # Compute items subtotal (recompute from items for safety)
        try:
            items_total = sum((Decimal(str(it.unit_price)) * Decimal(it.quantity)) for it in self.object.items.all())
        except Exception:
            items_total = self.object.subtotal or Decimal('0')

        # Provide country display fallback to templates if not set earlier
        if 'shipping_country_name' not in context:
            context['shipping_country_name'] = getattr(self.object, 'shipping_country', '')
        if 'billing_country_name' not in context:
            context['billing_country_name'] = getattr(self.object, 'billing_country', '')

        # Determine display shipping/tax from order snapshot
        display_shipping = self.object.shipping_cost or Decimal('0')
        display_tax = self.object.tax or Decimal('0')

        # Compute display total
        display_total = (items_total or Decimal('0')) - (self.object.discount or Decimal('0')) + (display_shipping or Decimal('0')) + (display_tax or Decimal('0')) + (self.object.gift_wrap_cost or Decimal('0'))

        # Formatted strings
        formatted_subtotal = checkout_currency.format_amount(items_total.quantize(Decimal('0.01'))) if checkout_currency else f"{currency_symbol_local}{items_total}"
        formatted_shipping = 'Free' if Decimal(str(display_shipping)) == Decimal('0') else (checkout_currency.format_amount(display_shipping) if checkout_currency else f"{currency_symbol_local}{display_shipping}")
        formatted_tax = checkout_currency.format_amount(display_tax.quantize(Decimal('0.01'))) if checkout_currency else f"{currency_symbol_local}{display_tax}"
        formatted_total = checkout_currency.format_amount(display_total.quantize(Decimal('0.01'))) if checkout_currency else f"{currency_symbol_local}{display_total}"

        context['formatted_subtotal'] = formatted_subtotal
        context['formatted_shipping'] = formatted_shipping
        context['formatted_tax'] = formatted_tax
        context['formatted_total'] = formatted_total

        # Attach formatted item prices for templates
        for item in self.object.items.all():
            try:
                item.formatted_unit_price = checkout_currency.format_amount(Decimal(str(item.unit_price or 0)))
                item.formatted_line_total = checkout_currency.format_amount(Decimal(str(item.line_total or 0)))
            except Exception:
                item.formatted_unit_price = str(item.unit_price)
                item.formatted_line_total = str(item.line_total)

        return context
    
    def post(self, request, *args, **kwargs):
        """Handle order cancellation."""
        self.object = self.get_object()
        
        if 'cancel_order' in request.POST:
            reason = request.POST.get('cancel_reason', '')
            success, message = OrderService.cancel_order(
                self.object,
                reason=reason,
                cancelled_by=request.user
            )
            
            if success:
                messages.success(request, message)
            else:
                messages.error(request, message)
        
        return redirect('orders:detail', pk=self.object.pk)


class OrderTrackView(TemplateView):
    """Public order tracking page."""
    template_name = 'accounts/orders.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        order_number = self.kwargs.get('order_number')
        
        order = get_object_or_404(
            Order.objects.prefetch_related('status_history'),
            order_number=order_number,
            is_deleted=False
        )
        
        context['page_title'] = f'Track Order {order_number}'
        context['order'] = order
        
        return context


class OrderInvoiceView(LoginRequiredMixin, TemplateView):
    """Render a printable invoice for an order. Accessible by owner or staff."""
    template_name = 'orders/invoice.html'
    login_url = '/account/login/'

    def dispatch(self, request, *args, **kwargs):
        # Ensure the order exists and user is owner or staff
        order_number = self.kwargs.get('order_number')
        order = get_object_or_404(Order, order_number=order_number, is_deleted=False)
        if not (request.user.is_authenticated and (request.user.is_staff or order.user == request.user)):
            # For guests trying to access, show 404 to avoid leaking info
            from django.http import Http404
            raise Http404("Order not found")
        self.order = order
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = f'Invoice {self.order.order_number}'
        context['order'] = self.order

        # Compute formatted amounts using currencies service (reuse logic from success/order detail)
        try:
            from apps.i18n.services import CurrencyService
            from decimal import Decimal
            # Attempt to show amounts in order display currency if available
            checkout_session = None
            try:
                from apps.commerce.models import CheckoutSession
                checkout_session = CheckoutSession.objects.filter(order=self.order).select_related('shipping_rate').first()
            except Exception:
                checkout_session = None

            currency_obj = None
            if checkout_session and getattr(checkout_session, 'currency', None):
                try:
                    currency_obj = CurrencyService.get_currency_by_code(checkout_session.currency)
                except Exception:
                    currency_obj = CurrencyService.get_default_currency()
            else:
                currency_obj = CurrencyService.get_default_currency()

            # items total
            items_total = sum((Decimal(str(it.unit_price)) * Decimal(it.quantity)) for it in self.order.items.all())
            display_shipping = self.order.shipping_cost
            # convert shipping if rate currency differs
            try:
                rate_currency_obj = getattr(checkout_session.shipping_rate, 'currency', None) if checkout_session else None
                if rate_currency_obj and rate_currency_obj.code != currency_obj.code:
                    from apps.i18n.services import CurrencyConversionService
                    display_shipping = CurrencyConversionService.convert_by_code(self.order.shipping_cost, rate_currency_obj.code, currency_obj.code)
            except Exception:
                display_shipping = self.order.shipping_cost

            # tax snapshot
            display_tax = self.order.tax
            try:
                if checkout_session and getattr(checkout_session, 'tax_amount', None) is not None:
                    display_tax = Decimal(str(checkout_session.tax_amount))
            except Exception:
                display_tax = self.order.tax

            display_total = (items_total or Decimal('0')) - (self.order.discount or Decimal('0')) + (display_shipping or Decimal('0')) + (display_tax or Decimal('0')) + (self.order.gift_wrap_cost or Decimal('0'))

            context['currency'] = currency_obj
            context['items_total'] = items_total.quantize(Decimal('0.01'))
            context['display_shipping'] = Decimal(str(display_shipping)).quantize(Decimal('0.01'))
            context['display_tax'] = Decimal(str(display_tax)).quantize(Decimal('0.01'))
            context['display_total'] = display_total.quantize(Decimal('0.01'))
        except Exception:
            pass

        return context
