"""
Pre-orders API serializers
"""
from rest_framework import serializers
from django.utils import timezone

from ..models import (
    PreOrderCategory, PreOrderOption, PreOrderOptionChoice,
    PreOrder, PreOrderItem, PreOrderOptionValue, PreOrderDesign,
    PreOrderReference, PreOrderStatusHistory, PreOrderPayment,
    PreOrderMessage, PreOrderRevision, PreOrderQuote, PreOrderTemplate
)


class PreOrderOptionChoiceSerializer(serializers.ModelSerializer):
    """Serializer for pre-order option choices."""
    
    class Meta:
        model = PreOrderOptionChoice
        fields = [
            'id', 'value', 'display_name', 'price_modifier',
            'color_code', 'image', 'order', 'is_active'
        ]


class PreOrderOptionSerializer(serializers.ModelSerializer):
    """Serializer for pre-order options."""
    choices = PreOrderOptionChoiceSerializer(many=True, read_only=True)
    
    class Meta:
        model = PreOrderOption
        fields = [
            'id', 'name', 'description', 'option_type', 'is_required',
            'min_length', 'max_length', 'price_modifier', 'placeholder',
            'help_text', 'order', 'is_active', 'choices'
        ]


class PreOrderCategorySerializer(serializers.ModelSerializer):
    """Serializer for pre-order categories (list view)."""
    preorder_count = serializers.SerializerMethodField()
    
    class Meta:
        model = PreOrderCategory
        fields = [
            'id', 'name', 'slug', 'description', 'icon', 'image',
            'base_price', 'deposit_percentage', 'min_production_days',
            'max_production_days', 'requires_design', 'requires_approval',
            'allow_rush_order', 'rush_order_fee_percentage',
            'min_quantity', 'max_quantity', 'is_active', 'order',
            'preorder_count'
        ]
    
    def get_preorder_count(self, obj):
        return obj.preorders.count()


class PreOrderCategoryDetailSerializer(PreOrderCategorySerializer):
    """Serializer for pre-order category detail view."""
    options = PreOrderOptionSerializer(many=True, read_only=True)
    
    class Meta(PreOrderCategorySerializer.Meta):
        fields = PreOrderCategorySerializer.Meta.fields + ['options']


class PreOrderTemplateSerializer(serializers.ModelSerializer):
    """Serializer for pre-order templates."""
    category_name = serializers.CharField(source='category.name', read_only=True)
    
    class Meta:
        model = PreOrderTemplate
        fields = [
            'id', 'name', 'slug', 'description', 'category', 'category_name',
            'image', 'default_quantity', 'base_price', 'estimated_days',
            'default_options', 'is_active', 'is_featured', 'order', 'use_count'
        ]


class PreOrderItemSerializer(serializers.ModelSerializer):
    """Serializer for pre-order items."""
    
    class Meta:
        model = PreOrderItem
        fields = [
            'id', 'name', 'description', 'quantity', 'unit_price',
            'total_price', 'customization_details', 'status', 'notes',
            'created_at', 'updated_at'
        ]


class PreOrderOptionValueSerializer(serializers.ModelSerializer):
    """Serializer for pre-order option values."""
    option_name = serializers.CharField(source='option.name', read_only=True)
    option_type = serializers.CharField(source='option.option_type', read_only=True)
    display_value = serializers.SerializerMethodField()
    
    class Meta:
        model = PreOrderOptionValue
        fields = [
            'id', 'option', 'option_name', 'option_type',
            'text_value', 'number_value', 'choice_value',
            'boolean_value', 'date_value', 'file_value',
            'price_modifier_applied', 'display_value'
        ]
    
    def get_display_value(self, obj):
        return str(obj.get_value()) if obj.get_value() else None


class PreOrderDesignSerializer(serializers.ModelSerializer):
    """Serializer for pre-order designs."""
    uploaded_by_name = serializers.CharField(source='uploaded_by.get_full_name', read_only=True)
    approved_by_name = serializers.CharField(source='approved_by.get_full_name', read_only=True)
    
    class Meta:
        model = PreOrderDesign
        fields = [
            'id', 'file', 'original_filename', 'file_size', 'design_type',
            'is_approved', 'approved_by', 'approved_by_name', 'approved_at',
            'version', 'is_current', 'notes', 'uploaded_by', 'uploaded_by_name',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['file_size', 'approved_at', 'created_at', 'updated_at']


class PreOrderReferenceSerializer(serializers.ModelSerializer):
    """Serializer for pre-order references."""
    
    class Meta:
        model = PreOrderReference
        fields = ['id', 'file', 'original_filename', 'description', 'created_at']


class PreOrderStatusHistorySerializer(serializers.ModelSerializer):
    """Serializer for pre-order status history."""
    changed_by_name = serializers.CharField(source='changed_by.get_full_name', read_only=True)
    from_status_display = serializers.SerializerMethodField()
    to_status_display = serializers.SerializerMethodField()
    
    class Meta:
        model = PreOrderStatusHistory
        fields = [
            'id', 'from_status', 'from_status_display',
            'to_status', 'to_status_display',
            'changed_by', 'changed_by_name', 'notes',
            'is_system', 'notification_sent', 'created_at'
        ]
    
    def get_from_status_display(self, obj):
        return dict(PreOrder.STATUS_CHOICES).get(obj.from_status, obj.from_status)
    
    def get_to_status_display(self, obj):
        return dict(PreOrder.STATUS_CHOICES).get(obj.to_status, obj.to_status)


class PreOrderPaymentSerializer(serializers.ModelSerializer):
    """Serializer for pre-order payments."""
    recorded_by_name = serializers.CharField(source='recorded_by.get_full_name', read_only=True)
    
    class Meta:
        model = PreOrderPayment
        fields = [
            'id', 'payment_type', 'amount', 'currency', 'status',
            'payment_method', 'transaction_id', 'notes',
            'receipt_url', 'receipt_sent', 'recorded_by', 'recorded_by_name',
            'created_at', 'paid_at'
        ]


class PreOrderMessageSerializer(serializers.ModelSerializer):
    """Serializer for pre-order messages."""
    sender_name = serializers.SerializerMethodField()
    
    class Meta:
        model = PreOrderMessage
        fields = [
            'id', 'sender', 'sender_name', 'is_from_customer', 'is_from_system',
            'subject', 'message', 'attachment', 'is_read', 'read_at', 'created_at'
        ]
    
    def get_sender_name(self, obj):
        if obj.is_from_system:
            return 'System'
        if obj.sender:
            return obj.sender.get_full_name() or obj.sender.email
        return 'Unknown'


class PreOrderRevisionSerializer(serializers.ModelSerializer):
    """Serializer for pre-order revisions."""
    requested_by_name = serializers.CharField(source='requested_by.get_full_name', read_only=True)
    responded_by_name = serializers.CharField(source='responded_by.get_full_name', read_only=True)
    
    class Meta:
        model = PreOrderRevision
        fields = [
            'id', 'revision_number', 'description', 'status',
            'additional_cost', 'requested_by', 'requested_by_name',
            'admin_response', 'responded_by', 'responded_by_name',
            'created_at', 'updated_at', 'completed_at'
        ]


class PreOrderQuoteSerializer(serializers.ModelSerializer):
    """Serializer for pre-order quotes."""
    created_by_name = serializers.CharField(source='created_by.get_full_name', read_only=True)
    is_expired = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = PreOrderQuote
        fields = [
            'id', 'quote_number', 'base_price', 'customization_cost',
            'rush_fee', 'discount', 'shipping', 'tax', 'total',
            'valid_from', 'valid_until', 'is_expired', 'status',
            'terms', 'notes', 'estimated_production_days',
            'estimated_delivery_date', 'created_by', 'created_by_name',
            'customer_response_notes', 'created_at', 'sent_at', 'responded_at'
        ]


class PreOrderSerializer(serializers.ModelSerializer):
    """Serializer for pre-orders (list view)."""
    category_name = serializers.CharField(source='category.name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    priority_display = serializers.CharField(source='get_priority_display', read_only=True)
    is_fully_paid = serializers.BooleanField(read_only=True)
    deposit_is_paid = serializers.BooleanField(read_only=True)
    days_until_deadline = serializers.IntegerField(read_only=True)
    is_overdue = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = PreOrder
        fields = [
            'id', 'preorder_number', 'category', 'category_name',
            'title', 'description', 'quantity', 'status', 'status_display',
            'priority', 'priority_display', 'is_rush_order',
            'estimated_price', 'final_price', 'total_amount',
            'deposit_required', 'deposit_paid', 'amount_paid', 'amount_remaining',
            'is_fully_paid', 'deposit_is_paid', 'currency',
            'requested_delivery_date', 'estimated_completion_date',
            'days_until_deadline', 'is_overdue',
            'created_at', 'submitted_at'
        ]


class PreOrderDetailSerializer(PreOrderSerializer):
    """Serializer for pre-order detail view."""
    items = PreOrderItemSerializer(many=True, read_only=True)
    option_values = PreOrderOptionValueSerializer(many=True, read_only=True)
    designs = PreOrderDesignSerializer(many=True, read_only=True)
    references = PreOrderReferenceSerializer(many=True, read_only=True)
    payments = PreOrderPaymentSerializer(many=True, read_only=True)
    messages = PreOrderMessageSerializer(many=True, read_only=True)
    revisions = PreOrderRevisionSerializer(many=True, read_only=True)
    quotes = PreOrderQuoteSerializer(many=True, read_only=True)
    status_history = PreOrderStatusHistorySerializer(many=True, read_only=True)
    user_email = serializers.EmailField(source='user.email', read_only=True)
    assigned_to_name = serializers.CharField(source='assigned_to.get_full_name', read_only=True)
    unread_messages_count = serializers.SerializerMethodField()
    
    class Meta(PreOrderSerializer.Meta):
        fields = PreOrderSerializer.Meta.fields + [
            'user', 'user_email', 'full_name', 'email', 'phone',
            'base_product', 'special_instructions', 'is_gift', 'gift_wrap', 'gift_message',
            'rush_order_fee', 'discount_amount', 'tax_amount', 'shipping_cost',
            'shipping_first_name', 'shipping_last_name',
            'shipping_address_line_1', 'shipping_address_line_2',
            'shipping_city', 'shipping_state', 'shipping_postal_code', 'shipping_country',
            'shipping_method', 'tracking_number', 'tracking_url',
            'assigned_to', 'assigned_to_name',
            'customer_notes', 'admin_notes', 'production_notes',
            'quote_valid_until', 'quote_notes',
            'revision_count', 'max_revisions',
            'production_start_date', 'actual_completion_date',
            'submitted_at', 'quoted_at', 'approved_at',
            'production_started_at', 'completed_at', 'shipped_at', 'delivered_at',
            'items', 'option_values', 'designs', 'references',
            'payments', 'messages', 'revisions', 'quotes', 'status_history',
            'unread_messages_count', 'updated_at'
        ]
    
    def get_unread_messages_count(self, obj):
        # Count unread messages from admin (for customer view)
        return obj.messages.filter(is_from_customer=False, is_read=False).count()


class PreOrderTrackingSerializer(serializers.ModelSerializer):
    """Serializer for public pre-order tracking view."""
    category_name = serializers.CharField(source='category.name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    priority_display = serializers.CharField(source='get_priority_display', read_only=True)
    is_fully_paid = serializers.BooleanField(read_only=True)
    deposit_is_paid = serializers.BooleanField(read_only=True)
    days_until_deadline = serializers.IntegerField(read_only=True)
    is_overdue = serializers.BooleanField(read_only=True)
    status_history = PreOrderStatusHistorySerializer(many=True, read_only=True)
    option_values = PreOrderOptionValueSerializer(many=True, read_only=True)
    designs = PreOrderDesignSerializer(many=True, read_only=True)
    references = PreOrderReferenceSerializer(many=True, read_only=True)
    quotes = PreOrderQuoteSerializer(many=True, read_only=True)
    payments = PreOrderPaymentSerializer(many=True, read_only=True)

    class Meta:
        model = PreOrder
        fields = [
            'id', 'preorder_number', 'category', 'category_name',
            'title', 'description', 'quantity', 'status', 'status_display',
            'priority', 'priority_display', 'is_rush_order',
            'estimated_price', 'final_price', 'total_amount',
            'deposit_required', 'deposit_paid', 'amount_paid', 'amount_remaining',
            'is_fully_paid', 'deposit_is_paid', 'currency',
            'requested_delivery_date', 'estimated_completion_date',
            'days_until_deadline', 'is_overdue',
            'shipping_method', 'tracking_number', 'tracking_url',
            'shipping_first_name', 'shipping_last_name',
            'shipping_address_line_1', 'shipping_address_line_2',
            'shipping_city', 'shipping_state', 'shipping_postal_code', 'shipping_country',
            'submitted_at', 'quoted_at', 'approved_at',
            'production_started_at', 'completed_at', 'shipped_at', 'delivered_at',
            'created_at', 'updated_at',
            'status_history', 'option_values', 'designs', 'references', 'quotes', 'payments',
        ]

class PreOrderCreateSerializer(serializers.Serializer):
    """Serializer for creating pre-orders."""
    category = serializers.UUIDField()
    title = serializers.CharField(max_length=300)
    description = serializers.CharField()
    quantity = serializers.IntegerField(min_value=1)
    
    # Contact info
    full_name = serializers.CharField(max_length=200)
    email = serializers.EmailField()
    phone = serializers.CharField(max_length=20, required=False, allow_blank=True)
    
    # Options
    options = serializers.DictField(required=False)
    
    # Optional fields
    special_instructions = serializers.CharField(required=False, allow_blank=True)
    is_gift = serializers.BooleanField(required=False, default=False)
    gift_wrap = serializers.BooleanField(required=False, default=False)
    gift_message = serializers.CharField(required=False, allow_blank=True)
    is_rush_order = serializers.BooleanField(required=False, default=False)
    requested_delivery_date = serializers.DateField(required=False, allow_null=True)
    customer_notes = serializers.CharField(required=False, allow_blank=True)
    
    # Shipping address
    shipping_first_name = serializers.CharField(max_length=100, required=False, allow_blank=True)
    shipping_last_name = serializers.CharField(max_length=100, required=False, allow_blank=True)
    shipping_address_line_1 = serializers.CharField(max_length=255, required=False, allow_blank=True)
    shipping_address_line_2 = serializers.CharField(max_length=255, required=False, allow_blank=True)
    shipping_city = serializers.CharField(max_length=100, required=False, allow_blank=True)
    shipping_state = serializers.CharField(max_length=100, required=False, allow_blank=True)
    shipping_postal_code = serializers.CharField(max_length=20, required=False, allow_blank=True)
    shipping_country = serializers.CharField(max_length=100, required=False, default='Bangladesh')
    
    # Auto-submit
    submit = serializers.BooleanField(required=False, default=False)
    
    def validate_category(self, value):
        try:
            return PreOrderCategory.objects.get(id=value, is_active=True)
        except PreOrderCategory.DoesNotExist:
            raise serializers.ValidationError('Invalid category')
    
    def validate(self, data):
        category = data.get('category')
        
        if isinstance(category, PreOrderCategory):
            quantity = data.get('quantity', 1)
            if quantity < category.min_quantity:
                raise serializers.ValidationError({
                    'quantity': f'Minimum quantity is {category.min_quantity}'
                })
            if quantity > category.max_quantity:
                raise serializers.ValidationError({
                    'quantity': f'Maximum quantity is {category.max_quantity}'
                })
            
            if data.get('is_rush_order') and not category.allow_rush_order:
                raise serializers.ValidationError({
                    'is_rush_order': 'Rush orders not available for this category'
                })
        
        return data
