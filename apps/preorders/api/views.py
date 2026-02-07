"""
Pre-orders API - RESTful API endpoints for pre-order operations
"""
from decimal import Decimal
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db.models import Q

from ..models import (
    PreOrderCategory, PreOrderOption, PreOrderOptionChoice,
    PreOrder, PreOrderItem, PreOrderDesign, PreOrderReference,
    PreOrderMessage, PreOrderRevision, PreOrderQuote, PreOrderTemplate,
    PreOrderOptionValue
)
from .serializers import (
    PreOrderCategorySerializer, PreOrderCategoryDetailSerializer,
    PreOrderOptionSerializer, PreOrderSerializer, PreOrderDetailSerializer,
    PreOrderCreateSerializer, PreOrderDesignSerializer, PreOrderMessageSerializer,
    PreOrderRevisionSerializer, PreOrderQuoteSerializer, PreOrderTemplateSerializer,
    PreOrderPaymentSerializer, PreOrderStatusHistorySerializer,
    PreOrderReferenceSerializer, PreOrderOptionValueSerializer,
    PreOrderTrackingSerializer
)
from ..services import PreOrderService, PreOrderCategoryService, PreOrderTemplateService


class PreOrderCategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoints for pre-order categories."""
    queryset = PreOrderCategory.objects.filter(is_active=True)
    permission_classes = [AllowAny]
    lookup_field = 'slug'
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return PreOrderCategoryDetailSerializer
        return PreOrderCategorySerializer
    
    def get_queryset(self):
        return PreOrderCategory.objects.filter(
            is_active=True
        ).prefetch_related('options__choices').order_by('order', 'name')
    
    @action(detail=True, methods=['get'])
    def options(self, request, slug=None):
        """Get all options for a category."""
        category = self.get_object()
        options = category.options.filter(is_active=True).order_by('order')
        serializer = PreOrderOptionSerializer(options, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def templates(self, request, slug=None):
        """Get all templates for a category."""
        category = self.get_object()
        templates = category.templates.filter(is_active=True).order_by('order')
        serializer = PreOrderTemplateSerializer(templates, many=True)
        return Response(serializer.data)


class PreOrderViewSet(viewsets.ModelViewSet):
    """API endpoints for pre-orders."""
    permission_classes = [IsAuthenticated]
    lookup_field = 'preorder_number'

    def get_permissions(self):
        if self.action == 'create':
            return [AllowAny()]
        return [permission() for permission in self.permission_classes]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return PreOrderCreateSerializer
        if self.action in ['retrieve', 'update', 'partial_update']:
            return PreOrderDetailSerializer
        return PreOrderSerializer
    
    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            queryset = PreOrder.objects.filter(is_deleted=False)
        else:
            queryset = PreOrder.objects.filter(user=user, is_deleted=False)
        
        return queryset.select_related(
            'category', 'user'
        ).prefetch_related(
            'items', 'designs', 'option_values'
        ).order_by('-created_at')

    def create(self, request, *args, **kwargs):
        serializer = PreOrderCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        category = data['category']

        preorder = PreOrderService.create_preorder(
            user=request.user if request.user and request.user.is_authenticated else None,
            category=category,
            data=data,
            options_data=data.get('options', {})
        )

        if data.get('submit', False):
            PreOrderService.submit_preorder(
                preorder, request.user if request.user and request.user.is_authenticated else None
            )

        output = PreOrderDetailSerializer(preorder, context={'request': request}).data
        return Response(output, status=status.HTTP_201_CREATED)

    def perform_create(self, serializer):
        """Create a new pre-order."""
        data = serializer.validated_data
        category = data['category']
        
        preorder = PreOrderService.create_preorder(
            user=self.request.user,
            category=category,
            data=data,
            options_data=data.get('options', {})
        )
        
        # Auto-submit if requested
        if data.get('submit', False):
            PreOrderService.submit_preorder(preorder, self.request.user)
        
        serializer.instance = preorder
    
    @action(detail=True, methods=['post'])
    def submit(self, request, preorder_number=None):
        """Submit a draft pre-order."""
        preorder = self.get_object()
        
        if preorder.status != PreOrder.STATUS_DRAFT:
            return Response(
                {'error': 'Only draft pre-orders can be submitted'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        preorder = PreOrderService.submit_preorder(preorder, request.user)
        serializer = PreOrderDetailSerializer(preorder)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, preorder_number=None):
        """Send a message on the pre-order."""
        preorder = self.get_object()
        message_text = request.data.get('message', '')
        subject = request.data.get('subject', '')
        
        if not message_text:
            return Response(
                {'error': 'Message is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        message = PreOrderService.send_message(
            preorder=preorder,
            message=message_text,
            subject=subject,
            sender=request.user,
            is_from_customer=not request.user.is_staff
        )
        
        serializer = PreOrderMessageSerializer(message)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['get'])
    def messages(self, request, preorder_number=None):
        """Get all messages for a pre-order."""
        preorder = self.get_object()
        messages = preorder.messages.all().order_by('created_at')
        serializer = PreOrderMessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def request_revision(self, request, preorder_number=None):
        """Request a revision for the pre-order."""
        preorder = self.get_object()
        description = request.data.get('description', '')
        
        if not description:
            return Response(
                {'error': 'Description is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            revision = PreOrderService.request_revision(
                preorder=preorder,
                description=description,
                user=request.user
            )
            serializer = PreOrderRevisionSerializer(revision)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_design(self, request, preorder_number=None):
        """Upload a design file."""
        preorder = self.get_object()
        file = request.FILES.get('file')
        notes = request.data.get('notes', '')
        
        if not file:
            return Response(
                {'error': 'File is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        design = PreOrderDesign.objects.create(
            preorder=preorder,
            file=file,
            original_filename=file.name,
            design_type=PreOrderDesign.DESIGN_CUSTOMER if not request.user.is_staff else PreOrderDesign.DESIGN_ADMIN,
            notes=notes,
            uploaded_by=request.user
        )
        
        serializer = PreOrderDesignSerializer(design)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_reference(self, request, preorder_number=None):
        """Upload a reference file."""
        preorder = self.get_object()
        file = request.FILES.get('file')
        description = request.data.get('description', '')

        if not file:
            return Response(
                {'error': 'File is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        reference = PreOrderReference.objects.create(
            preorder=preorder,
            file=file,
            original_filename=file.name,
            description=description
        )

        serializer = PreOrderReferenceSerializer(reference)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser], url_path='upload-option-file')
    def upload_option_file(self, request, preorder_number=None):
        """Upload a file for a file-type option."""
        preorder = self.get_object()
        option_id = request.data.get('option_id')
        file = request.FILES.get('file')

        if not option_id or not file:
            return Response(
                {'error': 'Option id and file are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        option = get_object_or_404(
            PreOrderOption,
            id=option_id,
            category=preorder.category
        )

        if option.option_type != PreOrderOption.OPTION_FILE:
            return Response(
                {'error': 'Selected option does not accept files'},
                status=status.HTTP_400_BAD_REQUEST
            )

        option_value, created = PreOrderOptionValue.objects.get_or_create(
            preorder=preorder,
            option=option
        )
        option_value.file_value = file
        option_value.price_modifier_applied = option.price_modifier
        option_value.save()

        if created and option.price_modifier:
            try:
                preorder.estimated_price = (preorder.estimated_price or Decimal('0')) + (
                    option.price_modifier * preorder.quantity
                )
                preorder.save(update_fields=['estimated_price'])
            except Exception:
                pass

        serializer = PreOrderOptionValueSerializer(option_value)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['get'])
    def designs(self, request, preorder_number=None):
        """Get all designs for a pre-order."""
        preorder = self.get_object()
        designs = preorder.designs.all().order_by('-created_at')
        serializer = PreOrderDesignSerializer(designs, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def quotes(self, request, preorder_number=None):
        """Get all quotes for a pre-order."""
        preorder = self.get_object()
        quotes = preorder.quotes.all().order_by('-created_at')
        serializer = PreOrderQuoteSerializer(quotes, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def accept_quote(self, request, preorder_number=None):
        """Accept a quote."""
        preorder = self.get_object()
        quote_id = request.data.get('quote_id')
        
        quote = get_object_or_404(PreOrderQuote, id=quote_id, preorder=preorder)
        
        if quote.is_expired:
            return Response(
                {'error': 'This quote has expired'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        preorder = PreOrderService.accept_quote(preorder, quote, request.user)
        serializer = PreOrderDetailSerializer(preorder)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def reject_quote(self, request, preorder_number=None):
        """Reject a quote."""
        preorder = self.get_object()
        quote_id = request.data.get('quote_id')
        reason = request.data.get('reason', '')
        
        quote = get_object_or_404(PreOrderQuote, id=quote_id, preorder=preorder)
        
        preorder = PreOrderService.reject_quote(preorder, quote, request.user, reason)
        serializer = PreOrderDetailSerializer(preorder)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def approve_work(self, request, preorder_number=None):
        """Approve completed work."""
        preorder = self.get_object()
        
        if preorder.status != PreOrder.STATUS_AWAITING_APPROVAL:
            return Response(
                {'error': 'Work can only be approved when awaiting approval'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        new_status = PreOrder.STATUS_READY_TO_SHIP if preorder.is_fully_paid else PreOrder.STATUS_FINAL_PAYMENT_PENDING
        preorder = PreOrderService.update_status(
            preorder,
            new_status,
            request.user,
            'Customer approved the work'
        )
        
        serializer = PreOrderDetailSerializer(preorder)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def status_history(self, request, preorder_number=None):
        """Get status history for a pre-order."""
        preorder = self.get_object()
        history = preorder.status_history.all().order_by('-created_at')
        serializer = PreOrderStatusHistorySerializer(history, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def summary(self, request, preorder_number=None):
        """Get pre-order summary."""
        preorder = self.get_object()
        return Response(PreOrderService.get_preorder_summary(preorder))


class PreOrderTemplateViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoints for pre-order templates."""
    queryset = PreOrderTemplate.objects.filter(is_active=True)
    serializer_class = PreOrderTemplateSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        queryset = PreOrderTemplate.objects.filter(is_active=True)
        
        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(category__slug=category)
        
        featured = self.request.query_params.get('featured')
        if featured:
            queryset = queryset.filter(is_featured=True)
        
        return queryset.select_related('category').order_by('order', 'name')
    
    @action(detail=True, methods=['post'])
    def use(self, request, pk=None):
        """Use this template to start a pre-order."""
        template = self.get_object()
        data = PreOrderTemplateService.use_template(template)
        return Response({
            'success': True,
            'template_data': data
        })


class PreOrderPriceCalculatorAPIView(APIView):
    """Calculate estimated price for a pre-order configuration."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        category_id = request.data.get('category_id')
        quantity = int(request.data.get('quantity', 1))
        options = request.data.get('options', {})
        is_rush = request.data.get('is_rush_order', False)
        
        try:
            category = PreOrderCategory.objects.get(id=category_id, is_active=True)
        except PreOrderCategory.DoesNotExist:
            return Response(
                {'error': 'Category not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        base_price = category.base_price * quantity
        options_price = Decimal('0')
        
        for option_id, value in options.items():
            try:
                option = PreOrderOption.objects.get(id=option_id)
                options_price += option.price_modifier * quantity
                
                if option.option_type == PreOrderOption.OPTION_SELECT and value:
                    choice = PreOrderOptionChoice.objects.get(id=value)
                    options_price += choice.price_modifier * quantity
                elif option.option_type == PreOrderOption.OPTION_MULTISELECT and value:
                    choices = PreOrderOptionChoice.objects.filter(id__in=value)
                    for choice in choices:
                        options_price += choice.price_modifier * quantity
            except:
                continue
        
        subtotal = base_price + options_price
        rush_fee = Decimal('0')
        
        if is_rush and category.allow_rush_order:
            rush_fee = (subtotal * category.rush_order_fee_percentage) / 100
        
        total = subtotal + rush_fee
        deposit = (total * category.deposit_percentage) / 100
        
        return Response({
            'base_price': str(base_price),
            'options_price': str(options_price),
            'rush_fee': str(rush_fee),
            'subtotal': str(subtotal),
            'total': str(total),
            'deposit_required': str(deposit),
            'deposit_percentage': category.deposit_percentage,
            'currency': 'BDT'
        })


class PreOrderStatisticsAPIView(APIView):
    """Get pre-order statistics for the current user."""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user if not request.user.is_staff else None
        stats = PreOrderService.get_preorder_statistics(user)
        return Response(stats)


class PreOrderTrackAPIView(APIView):
    """Public endpoint to track a pre-order by number and email."""
    permission_classes = [AllowAny]

    def post(self, request):
        preorder_number = (request.data.get('preorder_number') or '').strip()
        email = (request.data.get('email') or '').strip()

        if not preorder_number or not email:
            return Response(
                {'error': 'Order number and email are required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        preorder = PreOrder.objects.filter(
            preorder_number=preorder_number,
            email__iexact=email,
            is_deleted=False
        ).first()

        if not preorder:
            return Response(
                {'error': 'Pre-order not found. Please check your details.'},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = PreOrderTrackingSerializer(preorder, context={'request': request})
        return Response(serializer.data)


# Admin-only endpoints
class AdminPreOrderViewSet(viewsets.ModelViewSet):
    """Admin endpoints for pre-order management."""
    permission_classes = [IsAdminUser]
    queryset = PreOrder.objects.filter(is_deleted=False)
    lookup_field = 'preorder_number'
    
    def get_serializer_class(self):
        return PreOrderDetailSerializer
    
    def get_queryset(self):
        queryset = PreOrder.objects.filter(is_deleted=False)
        
        # Apply filters
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(category__slug=category)
        
        priority = self.request.query_params.get('priority')
        if priority:
            queryset = queryset.filter(priority=priority)
        
        assigned_to = self.request.query_params.get('assigned_to')
        if assigned_to:
            queryset = queryset.filter(assigned_to_id=assigned_to)
        
        return queryset.select_related(
            'category', 'user', 'assigned_to'
        ).order_by('-created_at')
    
    @action(detail=True, methods=['post'])
    def update_status(self, request, preorder_number=None):
        """Update pre-order status."""
        preorder = self.get_object()
        new_status = request.data.get('status')
        notes = request.data.get('notes', '')
        
        if new_status not in dict(PreOrder.STATUS_CHOICES):
            return Response(
                {'error': 'Invalid status'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        preorder = PreOrderService.update_status(
            preorder, new_status, request.user, notes
        )
        
        serializer = PreOrderDetailSerializer(preorder)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def assign(self, request, preorder_number=None):
        """Assign pre-order to staff member."""
        preorder = self.get_object()
        staff_id = request.data.get('staff_id')
        
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        try:
            staff = User.objects.get(id=staff_id, is_staff=True)
            preorder.assigned_to = staff
            preorder.save()
            
            serializer = PreOrderDetailSerializer(preorder)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response(
                {'error': 'Staff member not found'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=True, methods=['post'])
    def create_quote(self, request, preorder_number=None):
        """Create a quote for the pre-order."""
        preorder = self.get_object()
        
        quote = PreOrderService.create_quote(
            preorder=preorder,
            quote_data=request.data,
            user=request.user
        )
        
        serializer = PreOrderQuoteSerializer(quote)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def record_payment(self, request, preorder_number=None):
        """Record a payment for the pre-order."""
        preorder = self.get_object()
        
        amount = Decimal(request.data.get('amount', '0'))
        payment_method = request.data.get('payment_method', 'cash')
        payment_type = request.data.get('payment_type', 'deposit')
        transaction_id = request.data.get('transaction_id', '')
        notes = request.data.get('notes', '')
        
        payment = PreOrderService.process_payment(
            preorder=preorder,
            amount=amount,
            payment_method=payment_method,
            payment_type=payment_type,
            transaction_id=transaction_id,
            user=request.user
        )
        
        serializer = PreOrderPaymentSerializer(payment)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
