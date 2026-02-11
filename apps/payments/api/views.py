"""
Payments API views
"""
import stripe
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser

from ..models import Payment, PaymentMethod, Refund, PaymentGateway, PaymentLink, BNPLProvider, RecurringCharge, PaymentTransaction
from ..services import StripeService, PaymentService, PaymentMethodService, SSLCommerzService, BkashService, NagadService
from .serializers import (
    PaymentSerializer, PaymentMethodSerializer, RefundSerializer,
    CreatePaymentIntentSerializer, SavePaymentMethodSerializer,
    SetDefaultPaymentMethodSerializer, RefundCreateSerializer,
    PaymentGatewaySerializer, PaymentLinkSerializer, PaymentLinkCreateSerializer,
    BNPLProviderSerializer, RecurringChargeSerializer
)


class PaymentGatewayViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for payment gateways (public read-only).
    
    GET /api/v1/payments/gateways/ - List available payment gateways
    GET /api/v1/payments/gateways/{code}/ - Get gateway detail
    GET /api/v1/payments/gateways/available/ - Get gateways for current context
    """
    serializer_class = PaymentGatewaySerializer
    permission_classes = [AllowAny]
    lookup_field = 'code'
    
    def get_queryset(self):
        return PaymentGateway.objects.filter(is_active=True)
    
    def list(self, request, *args, **kwargs):
        """List all active payment gateways."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Payment gateways retrieved successfully',
            'data': serializer.data,
            'meta': {'count': queryset.count()}
        })
    
    @action(detail=False, methods=['get'])
    def available(self, request):
        """Get payment gateways available for current context."""
        currency = request.query_params.get('currency')
        country = request.query_params.get('country')
        amount = request.query_params.get('amount')

        try:
            from apps.commerce.services import CheckoutService

            raw_country = country
            normalized_country = CheckoutService.normalize_country_code(country)
            if raw_country and len(raw_country.strip()) in (2, 3) and normalized_country:
                country = normalized_country
            else:
                country = None
        except Exception:
            pass
        
        if amount:
            try:
                amount = float(amount)
            except ValueError:
                amount = None
        
        gateways = PaymentGateway.get_active_gateways(
            currency=currency,
            country=country,
            amount=amount
        )
        
        # If no gateways configured, return defaults
        if not gateways:
            gateways = self._get_default_gateways(currency)
        
        serializer = self.get_serializer(gateways, many=True)
        return Response({
            'success': True,
            'message': 'Available payment gateways retrieved',
            'data': serializer.data,
            'meta': {'count': len(gateways)}
        })
    
    def _get_default_gateways(self, currency=None):
        """Return an empty list when no gateways are configured.

        We intentionally remove placeholder defaults to ensure admin config is required.
        """
        return []


class PaymentViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for payments.
    
    GET /api/v1/payments/ - List user's payments
    GET /api/v1/payments/{id}/ - Get payment detail
    """
    serializer_class = PaymentSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Payment.objects.filter(user=self.request.user)
    
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            return Response({
                'success': True,
                'message': 'Payments retrieved successfully',
                'data': serializer.data,
                'meta': {
                    'count': self.paginator.page.paginator.count,
                    'next': response.data.get('next'),
                    'previous': response.data.get('previous')
                }
            })
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Payments retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })
    
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        
        # Include refunds
        refunds = RefundSerializer(instance.refunds.all(), many=True).data
        
        return Response({
            'success': True,
            'message': 'Payment retrieved successfully',
            'data': {
                **serializer.data,
                'refunds': refunds
            },
            'meta': {}
        })
    
    @action(detail=False, methods=['post'], url_path='create-intent')
    def create_intent(self, request):
        """Create a payment intent for an order."""
        serializer = CreatePaymentIntentSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        from apps.orders.models import Order
        order = Order.objects.filter(
            id=serializer.validated_data['order_id'],
            user=request.user
        ).first()
        
        if not order:
            return Response({
                'success': False,
                'message': 'Order not found',
                'data': {},
                'meta': {}
            }, status=status.HTTP_404_NOT_FOUND)
        
        try:
            # Get or create Stripe customer
            customer_id = None
            if hasattr(request.user, 'stripe_customer_id') and request.user.stripe_customer_id:
                customer_id = request.user.stripe_customer_id
            
            # Create payment intent
            amount_cents = int(order.total_amount * 100)
            pi = StripeService.create_payment_intent(
                amount=amount_cents,
                currency='usd',
                customer_id=customer_id,
                metadata={
                    'order_id': str(order.id),
                    'order_number': order.order_number,
                    'user_id': str(request.user.id)
                }
            )
            
            # Create payment record
            PaymentService.create_payment_for_order(order, pi.id)
            
            return Response({
                'success': True,
                'message': 'Payment intent created successfully',
                'data': {
                    'client_secret': pi.client_secret,
                    'payment_intent_id': pi.id
                },
                'meta': {}
            })
        
        except stripe.error.StripeError as e:
            return Response({
                'success': False,
                'message': str(e),
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)


class PaymentMethodViewSet(viewsets.ModelViewSet):
    """
    ViewSet for payment methods.
    
    GET /api/v1/payments/methods/ - List saved payment methods
    POST /api/v1/payments/methods/ - Save a payment method
    DELETE /api/v1/payments/methods/{id}/ - Remove payment method
    POST /api/v1/payments/methods/{id}/set-default/ - Set as default
    GET /api/v1/payments/methods/setup-intent/ - Get setup intent
    """
    serializer_class = PaymentMethodSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return PaymentMethod.objects.filter(user=self.request.user, is_active=True)
    
    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Payment methods retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })
    
    def create(self, request, *args, **kwargs):
        """Save a new payment method."""
        serializer = SavePaymentMethodSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            # Ensure user has Stripe customer
            customer_id = getattr(request.user, 'stripe_customer_id', None)
            if not customer_id:
                customer = StripeService.create_customer(
                    email=request.user.email,
                    name=request.user.get_full_name(),
                    metadata={'user_id': str(request.user.id)}
                )
                customer_id = customer.id
                request.user.stripe_customer_id = customer_id
                request.user.save(update_fields=['stripe_customer_id'])
            
            # Attach payment method to customer
            pm_id = serializer.validated_data['payment_method_id']
            StripeService.attach_payment_method(pm_id, customer_id)
            
            # Save to database
            payment_method = PaymentMethodService.save_payment_method(request.user, pm_id)
            
            return Response({
                'success': True,
                'message': 'Payment method saved successfully',
                'data': PaymentMethodSerializer(payment_method).data,
                'meta': {}
            }, status=status.HTTP_201_CREATED)
        
        except stripe.error.StripeError as e:
            return Response({
                'success': False,
                'message': str(e),
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)
    
    def destroy(self, request, *args, **kwargs):
        """Remove a payment method."""
        instance = self.get_object()
        PaymentMethodService.delete_payment_method(instance)
        return Response({
            'success': True,
            'message': 'Payment method removed successfully',
            'data': {},
            'meta': {}
        })
    
    @action(detail=True, methods=['post'], url_path='set-default')
    def set_default(self, request, pk=None):
        """Set payment method as default."""
        instance = self.get_object()
        PaymentMethodService.set_default_payment_method(request.user, instance.id)
        return Response({
            'success': True,
            'message': 'Default payment method updated',
            'data': {},
            'meta': {}
        })
    
    @action(detail=False, methods=['get'], url_path='setup-intent')
    def setup_intent(self, request):
        """Get a setup intent for saving payment methods."""
        try:
            # Ensure user has Stripe customer
            customer_id = getattr(request.user, 'stripe_customer_id', None)
            if not customer_id:
                customer = StripeService.create_customer(
                    email=request.user.email,
                    name=request.user.get_full_name(),
                    metadata={'user_id': str(request.user.id)}
                )
                customer_id = customer.id
                request.user.stripe_customer_id = customer_id
                request.user.save(update_fields=['stripe_customer_id'])
            
            setup_intent = StripeService.create_setup_intent(customer_id)
            
            return Response({
                'success': True,
                'message': 'Setup intent created successfully',
                'data': {
                    'client_secret': setup_intent.client_secret
                },
                'meta': {}
            })
        
        except stripe.error.StripeError as e:
            return Response({
                'success': False,
                'message': str(e),
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)


class RefundAdminViewSet(viewsets.ModelViewSet):
    """
    Admin ViewSet for refunds.
    """
    queryset = Refund.objects.all().order_by('-created_at')
    serializer_class = RefundSerializer
    permission_classes = [IsAdminUser]

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)

        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            return Response({
                'success': True,
                'message': 'Refunds retrieved successfully',
                'data': serializer.data,
                'meta': {
                    'count': self.paginator.page.paginator.count
                }
            })

        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Refunds retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })

    def create(self, request, *args, **kwargs):
        """Create a refund."""
        serializer = RefundCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        payment = Payment.objects.filter(
            id=serializer.validated_data['payment_id']
        ).first()

        if not payment:
            return Response({
                'success': False,
                'message': 'Payment not found',
                'data': {},
                'meta': {}
            }, status=status.HTTP_404_NOT_FOUND)

        try:
            refund = PaymentService.refund_payment(
                payment=payment,
                amount=serializer.validated_data.get('amount'),
                reason=serializer.validated_data['reason'],
                notes=serializer.validated_data.get('notes'),
                created_by=request.user
            )

            return Response({
                'success': True,
                'message': 'Refund processed successfully',
                'data': RefundSerializer(refund).data,
                'meta': {}
            }, status=status.HTTP_201_CREATED)

        except ValueError as e:
            return Response({
                'success': False,
                'message': str(e),
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)

        except stripe.error.StripeError as e:
            return Response({
                'success': False,
                'message': str(e),
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)


class PaymentLinkViewSet(viewsets.GenericViewSet):
    """Create and retrieve payment links."""
    serializer_class = PaymentLinkSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return PaymentLink.objects.filter(order__user=self.request.user)

    @action(detail=False, methods=['post'])
    def create_link(self, request):
        serializer = PaymentLinkCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        from apps.orders.models import Order
        order = Order.objects.filter(id=serializer.validated_data['order_id'], user=request.user).first()
        if not order:
            return Response({'success': False, 'message': 'Order not found'}, status=status.HTTP_404_NOT_FOUND)
        from django.utils.crypto import get_random_string
        code = get_random_string(32)
        gateway = None
        gw_code = serializer.validated_data.get('gateway_code')
        if gw_code:
            gateway = PaymentGateway.objects.filter(code=gw_code).first()
        amount = serializer.validated_data.get('amount') or order.total_amount
        currency = serializer.validated_data.get('currency') or 'BDT'
        expires_at = serializer.validated_data.get('expires_at')
        pl = PaymentLink.objects.create(order=order, gateway=gateway, code=code, amount=amount, currency=currency, expires_at=expires_at)
        return Response({'success': True, 'message': 'Payment link created', 'data': PaymentLinkSerializer(pl).data}, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['get'])
    def my_links(self, request):
        qs = self.get_queryset()
        serializer = PaymentLinkSerializer(qs, many=True)
        return Response({'success': True, 'data': serializer.data})


class BNPLViewSet(viewsets.ReadOnlyModelViewSet):
    """Public endpoints to list BNPL providers and check agreements."""
    serializer_class = BNPLProviderSerializer
    permission_classes = [IsAuthenticated]
    queryset = BNPLProvider.objects.all()

    def list(self, request):
        providers = BNPLProvider.objects.filter(is_active=True)
        data = [{'code': p.code, 'name': p.name, 'config': p.config} for p in providers]
        return Response({'success': True, 'data': data})

    @action(detail=False, methods=['post'])
    def create_agreement(self, request):
        code = request.data.get('provider_code')
        provider = BNPLProvider.objects.filter(code=code).first()
        if not provider:
            return Response({'success': False, 'message': 'BNPL provider not found'}, status=status.HTTP_404_NOT_FOUND)
        # TODO: initiate provider-specific flow and return URL or widget data
        return Response({'success': True, 'message': 'BNPL flow initiated (stub)'} )


class RecurringChargeAdminViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = RecurringCharge.objects.order_by('-created_at')
    serializer_class = RecurringChargeSerializer
    permission_classes = [IsAdminUser]

    def list(self, request, *args, **kwargs):
        qs = self.filter_queryset(self.get_queryset())
        serializer = RecurringChargeSerializer(qs, many=True)
        return Response({'success': True, 'data': serializer.data})


@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt
def stripe_webhook(request):
    """Handle Stripe webhooks."""
    payload = request.body
    sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
    
    try:
        event = StripeService.construct_webhook_event(payload, sig_header)
    except ValueError:
        return HttpResponse(status=400)
    except stripe.error.SignatureVerificationError:
        return HttpResponse(status=400)
    
    # Handle the event
    event_type = event['type']
    data = event['data']['object']
    
    if event_type == 'payment_intent.succeeded':
        PaymentService.process_payment_success(data['id'])
    
    elif event_type == 'payment_intent.payment_failed':
        failure_message = data.get('last_payment_error', {}).get('message')
        PaymentService.process_payment_failure(data['id'], failure_message)
    
    return HttpResponse(status=200)


@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt
def sslcommerz_ipn(request):
    """Handle SSLCommerz IPN notifications.

    Docs: https://developer.sslcommerz.com
    """
    payload = request.POST.dict() if request.method == 'POST' else {}
    gateway_code = request.GET.get('gateway')
    gateway = PaymentGateway.objects.filter(code=gateway_code).first() if gateway_code else None

    # Verify and normalize
    result = SSLCommerzService.verify_transaction(payload, gateway)
    if not result.get('success'):
        return HttpResponse(status=400)
    # Create transaction log
    PaymentTransaction.objects.create(
        gateway=gateway,
        reference=result.get('reference'),
        payload=payload,
        event_type='sslcommerz_ipn'
    )
    # TODO: tie to Payment and update order
    return HttpResponse('OK')


@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt
def bkash_webhook(request):
    payload = request.body
    gateway = PaymentGateway.objects.filter(code=PaymentGateway.CODE_BKASH).first()
    result = BkashService.verify_payment(payload, gateway)
    if not result.get('success'):
        return HttpResponse(status=400)
    PaymentTransaction.objects.create(
        gateway=gateway,
        reference=result.get('reference'),
        payload={'raw': payload.decode('utf-8') if isinstance(payload, (bytes,)) else str(payload)},
        event_type='bkash_webhook'
    )
    # TODO: process and link to Payment
    return HttpResponse('OK')


@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt
def nagad_webhook(request):
    payload = request.body
    gateway = PaymentGateway.objects.filter(code=PaymentGateway.CODE_NAGAD).first()
    result = NagadService.verify_callback(payload, gateway)
    if not result.get('success'):
        return HttpResponse(status=400)
    PaymentTransaction.objects.create(
        gateway=gateway,
        reference=result.get('reference'),
        payload={'raw': payload.decode('utf-8') if isinstance(payload, (bytes,)) else str(payload)},
        event_type='nagad_webhook'
    )
    # TODO: process and link to Payment
    return HttpResponse('OK')
