"""
Payments API serializers
"""
from rest_framework import serializers
from django.conf import settings
from decimal import Decimal

from ..models import Payment, PaymentMethod, Refund, PaymentGateway, PaymentLink, BNPLProvider, RecurringCharge
from apps.i18n.services import CurrencyService, CurrencyConversionService


class PaymentGatewaySerializer(serializers.ModelSerializer):
    """Serializer for payment gateways (public facing)."""
    icon_url = serializers.SerializerMethodField()
    public_key = serializers.SerializerMethodField()
    requires_client = serializers.SerializerMethodField()
    fee_amount_converted = serializers.SerializerMethodField()
    
    class Meta:
        model = PaymentGateway
        fields = [
            'code', 'name', 'description', 'icon_url', 'icon_class',
            'color', 'fee_type', 'fee_amount', 'fee_amount_converted', 'fee_text', 'instructions',
            'public_key', 'requires_client'
        ]

    def get_icon_url(self, obj):
        if obj.icon:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.icon.url)
            return obj.icon.url
        return None

    def get_public_key(self, obj):
        # Expose only publishable/public keys to frontend. For Stripe, 'api_key' is publishable in admin.
        if obj.code == PaymentGateway.CODE_STRIPE:
            return obj.api_key or getattr(settings, 'STRIPE_PUBLISHABLE_KEY', '')
        return None

    def get_requires_client(self, obj):
        # Indicate whether the gateway requires client-side JS (e.g., Stripe)
        return obj.code in (PaymentGateway.CODE_STRIPE,)

    def get_fee_amount_converted(self, obj):
        """Convert gateway fee to user's currency for frontend display."""
        try:
            request = self.context.get('request')
            user = request.user if request and hasattr(request, 'user') and request.user.is_authenticated else None
            target = CurrencyService.get_user_currency(user=user, request=request)
            base = CurrencyService.get_default_currency()
            amount = Decimal(str(obj.fee_amount or 0))
            if target and base and base.code != target.code:
                converted = CurrencyConversionService.convert_by_code(
                    amount, base.code, target.code, round_result=True
                )
                return float(converted)
            return float(amount)
        except Exception:
            try:
                return float(obj.fee_amount or 0)
            except Exception:
                return 0.0
    
    def get_icon_url(self, obj):
        if obj.icon:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.icon.url)
            return obj.icon.url
        return None


class PaymentMethodSerializer(serializers.ModelSerializer):
    """Serializer for payment methods."""
    display_name = serializers.CharField(source='__str__', read_only=True)
    type_display = serializers.CharField(source='get_type_display', read_only=True)
    
    class Meta:
        model = PaymentMethod
        fields = [
            'id', 'type', 'type_display', 'display_name',
            'card_brand', 'card_last_four', 'card_exp_month', 'card_exp_year',
            'paypal_email', 'is_default', 'created_at'
        ]
        read_only_fields = [
            'id', 'card_brand', 'card_last_four', 'card_exp_month',
            'card_exp_year', 'paypal_email', 'created_at'
        ]


class PaymentSerializer(serializers.ModelSerializer):
    """Serializer for payments."""
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    order_number = serializers.CharField(source='order.order_number', read_only=True)
    
    class Meta:
        model = Payment
        fields = [
            'id', 'order', 'order_number', 'amount', 'currency',
            'status', 'status_display', 'method_type',
            'refunded_amount', 'paid_at', 'created_at'
        ]
        read_only_fields = fields


class RefundSerializer(serializers.ModelSerializer):
    """Serializer for refunds."""
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    reason_display = serializers.CharField(source='get_reason_display', read_only=True)
    
    class Meta:
        model = Refund
        fields = [
            'id', 'amount', 'reason', 'reason_display', 'notes',
            'status', 'status_display', 'created_at', 'processed_at'
        ]
        read_only_fields = fields


class CreatePaymentIntentSerializer(serializers.Serializer):
    """Serializer for creating payment intent."""
    order_id = serializers.UUIDField()


class SavePaymentMethodSerializer(serializers.Serializer):
    """Serializer for saving payment method."""
    payment_method_id = serializers.CharField(max_length=100)


class SetDefaultPaymentMethodSerializer(serializers.Serializer):
    """Serializer for setting default payment method."""
    payment_method_id = serializers.UUIDField()


class RefundCreateSerializer(serializers.Serializer):
    """Serializer for creating refunds."""
    payment_id = serializers.UUIDField()
    amount = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    reason = serializers.ChoiceField(choices=Refund.REASON_CHOICES)
    notes = serializers.CharField(required=False, allow_blank=True)


class PaymentLinkSerializer(serializers.ModelSerializer):
    class Meta:
        model = PaymentLink
        fields = ['id', 'order', 'gateway', 'code', 'amount', 'currency', 'expires_at', 'is_active', 'created_at']
        read_only_fields = fields


class PaymentLinkCreateSerializer(serializers.Serializer):
    order_id = serializers.UUIDField()
    gateway_code = serializers.CharField(max_length=50, required=False)
    amount = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    currency = serializers.CharField(max_length=3, required=False)
    expires_at = serializers.DateTimeField(required=False)


class BNPLProviderSerializer(serializers.ModelSerializer):
    class Meta:
        model = BNPLProvider
        fields = ['code', 'name', 'is_active', 'config']


class RecurringChargeSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecurringCharge
        fields = ['id', 'subscription', 'payment', 'amount', 'currency', 'status', 'attempt_at', 'processed_at', 'created_at']
        read_only_fields = fields
