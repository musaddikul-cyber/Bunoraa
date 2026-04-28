"""
Orders API views
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle, ScopedRateThrottle
from django.http import Http404
from django.db.models import Q
from urllib.parse import quote

from ..models import Order
from ..services import OrderAccessService, OrderService
from .serializers import (
    OrderSerializer,
    OrderDetailSerializer,
    OrderListSerializer,
    CancelOrderSerializer,
    UpdateOrderStatusSerializer,
    AddTrackingSerializer,
    GuestOrderLookupSerializer,
)


class OrderViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for user order operations.
    
    Endpoints:
    - GET /api/v1/orders/ - List user orders
    - GET /api/v1/orders/{id}/ - Get order detail
    - POST /api/v1/orders/{id}/cancel/ - Cancel order
    - GET /api/v1/orders/{id}/track/ - Get tracking info
    """
    throttle_classes = [AnonRateThrottle, UserRateThrottle, ScopedRateThrottle]
    throttle_scope = 'orders'
    permission_classes = [IsAuthenticated]

    def _guest_access_token(self, request) -> str:
        return (request.query_params.get('access_token') or '').strip()

    def _is_guest_access_request(self, request) -> bool:
        return bool(
            self.action in {'retrieve', 'track'} and self._guest_access_token(request)
        )

    def get_permissions(self):
        if self.action == 'lookup' or self._is_guest_access_request(self.request):
            return [AllowAny()]
        return [permission() for permission in self.permission_classes]

    def get_throttles(self):
        if self.action == 'lookup' or self._is_guest_access_request(self.request):
            self.throttle_scope = 'guest-order-access'
        else:
            self.throttle_scope = 'orders'
        return super().get_throttles()
    
    def get_queryset(self):
        base_queryset = Order.objects.filter(is_deleted=False).prefetch_related('items', 'status_history')
        if self._is_guest_access_request(self.request):
            return base_queryset
        return base_queryset.filter(user=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return OrderDetailSerializer
        if self.action == 'list':
            return OrderListSerializer
        return OrderSerializer
    
    def list(self, request):
        """List user orders with optional filtering."""
        queryset = self.get_queryset()
        
        # Filter by status
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Date range filtering
        date_from = request.query_params.get('date_from')
        date_to = request.query_params.get('date_to')
        if date_from:
            queryset = queryset.filter(created_at__date__gte=date_from)
        if date_to:
            queryset = queryset.filter(created_at__date__lte=date_to)

        query = (request.query_params.get('q') or '').strip()
        if query:
            queryset = queryset.filter(
                Q(order_number__icontains=query)
                | Q(status__icontains=query)
                | Q(email__icontains=query)
                | Q(tracking_number__icontains=query)
            ).distinct()

        ordering_param = (
            request.query_params.get('ordering')
            or request.query_params.get('sort')
            or ''
        ).strip()
        ordering_map = {
            'newest': ('-created_at',),
            'oldest': ('created_at',),
            'total_high': ('-total', '-created_at'),
            'total_low': ('total', '-created_at'),
            'status': ('status', '-created_at'),
            'created_at': ('created_at',),
            '-created_at': ('-created_at',),
            'total': ('total',),
            '-total': ('-total',),
            'order_number': ('order_number',),
            '-order_number': ('-order_number',),
        }
        if ordering_param in ordering_map:
            queryset = queryset.order_by(*ordering_map[ordering_param])

        # Pagination
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = self.get_serializer(queryset, many=True)
        
        return Response({
            'success': True,
            'message': 'Orders retrieved',
            'data': serializer.data
        })
    
    def retrieve(self, request, pk=None):
        """Get order detail."""
        order = self._get_order_for_read_access()
        serializer = self.get_serializer(order)
        
        return Response({
            'success': True,
            'message': 'Order retrieved',
            'data': serializer.data
        })
    
    @action(detail=True, methods=['post'], url_path='cancel')
    def cancel(self, request, pk=None):
        """Cancel order."""
        order = self.get_object()
        
        serializer = CancelOrderSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'data': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        reason = serializer.validated_data.get('reason', '')
        success, message = OrderService.cancel_order(
            order,
            reason=reason,
            cancelled_by=request.user
        )
        
        if not success:
            return Response({
                'success': False,
                'message': message,
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)
        
        order.refresh_from_db()
        
        return Response({
            'success': True,
            'message': message,
            'data': OrderSerializer(order).data
        })
    
    @action(detail=True, methods=['get'], url_path='track')
    def track(self, request, pk=None):
        """Get tracking information."""
        order = self._get_order_for_read_access()
        
        return Response({
            'success': True,
            'message': 'Tracking info retrieved',
            'data': {
                'order_number': order.order_number,
                'status': order.status,
                'status_display': order.get_status_display(),
                'tracking_number': order.tracking_number,
                'tracking_url': order.tracking_url,
                'shipped_at': order.shipped_at,
                'delivered_at': order.delivered_at,
            }
        })

    @action(detail=False, methods=['post'], url_path='lookup')
    def lookup(self, request):
        """Lookup a guest order by order number and email."""
        serializer = GuestOrderLookupSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        order_number = serializer.validated_data['order_number'].strip()
        email = serializer.validated_data['email'].strip().lower()

        order = (
            Order.objects.filter(
                order_number__iexact=order_number,
                email__iexact=email,
                is_deleted=False,
                user__isnull=True,
            )
            .prefetch_related('items', 'status_history')
            .first()
        )

        if not order:
            return Response(
                {
                    'success': False,
                    'message': "We couldn't find an order matching those details.",
                    'data': None,
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        access_token = OrderAccessService.build_guest_access_token(order)
        encoded_access_token = quote(access_token, safe="")
        detail_url = f"/orders/{order.id}/?access_token={encoded_access_token}"
        track_url = f"/orders/{order.id}/track/?access_token={encoded_access_token}"

        return Response(
            {
                'success': True,
                'message': 'Order retrieved',
                'data': {
                    'order_id': str(order.id),
                    'order_number': order.order_number,
                    'access_token': access_token,
                    'status': order.status,
                    'status_display': order.get_status_display(),
                    'tracking_number': order.tracking_number,
                    'tracking_url': order.tracking_url,
                    'detail_url': detail_url,
                    'track_url': track_url,
                },
            }
        )

    def _get_order_for_read_access(self):
        order = self.get_object()
        request = self.request
        if request.user.is_authenticated and order.user_id == request.user.id:
            return order

        token = self._guest_access_token(request)
        if order.user_id is None and token and OrderAccessService.verify_guest_access_token(order, token):
            return order

        raise Http404("Order not found")


class OrderAdminViewSet(viewsets.ModelViewSet):
    """
    ViewSet for admin order management.

    Endpoints:
    - GET /api/v1/admin/orders/ - List all orders
    - GET /api/v1/admin/orders/{id}/ - Get order detail
    - PATCH /api/v1/admin/orders/{id}/status/ - Update status
    - POST /api/v1/admin/orders/{id}/tracking/ - Add tracking
    - GET /api/v1/admin/orders/statistics/ - Get statistics
    """
    throttle_classes = [UserRateThrottle]
    throttle_scope = 'admin-orders'
    permission_classes = [IsAdminUser]
    queryset = Order.objects.filter(is_deleted=False).prefetch_related('items', 'status_history')
    
    def get_serializer_class(self):
        if self.action in ['retrieve', 'update', 'partial_update']:
            return OrderDetailSerializer
        if self.action == 'list':
            return OrderListSerializer
        return OrderSerializer
    
    def list(self, request):
        """List all orders with filtering."""
        queryset = self.get_queryset()
        
        # Filters
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        user_filter = request.query_params.get('user')
        if user_filter:
            queryset = queryset.filter(user_id=user_filter)
        
        email_filter = request.query_params.get('email')
        if email_filter:
            queryset = queryset.filter(email__icontains=email_filter)
        
        # Date range
        date_from = request.query_params.get('date_from')
        date_to = request.query_params.get('date_to')
        if date_from:
            queryset = queryset.filter(created_at__date__gte=date_from)
        if date_to:
            queryset = queryset.filter(created_at__date__lte=date_to)

        query = (request.query_params.get('q') or '').strip()
        if query:
            queryset = queryset.filter(
                Q(order_number__icontains=query)
                | Q(email__icontains=query)
                | Q(status__icontains=query)
                | Q(tracking_number__icontains=query)
                | Q(shipping_first_name__icontains=query)
                | Q(shipping_last_name__icontains=query)
            ).distinct()

        ordering_param = (
            request.query_params.get('ordering')
            or request.query_params.get('sort')
            or ''
        ).strip()
        ordering_map = {
            'newest': ('-created_at',),
            'oldest': ('created_at',),
            'total_high': ('-total', '-created_at'),
            'total_low': ('total', '-created_at'),
            'status': ('status', '-created_at'),
            'created_at': ('created_at',),
            '-created_at': ('-created_at',),
            'total': ('total',),
            '-total': ('-total',),
            'order_number': ('order_number',),
            '-order_number': ('-order_number',),
        }
        if ordering_param in ordering_map:
            queryset = queryset.order_by(*ordering_map[ordering_param])
        
        # Pagination
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = self.get_serializer(queryset, many=True)
        
        return Response({
            'success': True,
            'message': 'Orders retrieved',
            'data': serializer.data
        })
    
    @action(detail=True, methods=['patch'], url_path='status')
    def update_status(self, request, pk=None):
        """Update order status."""
        order = self.get_object()
        
        serializer = UpdateOrderStatusSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'data': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        new_status = serializer.validated_data['status']
        notes = serializer.validated_data.get('notes', '')
        
        try:
            order = OrderService.update_order_status(
                order,
                new_status,
                changed_by=request.user,
                notes=notes
            )
        except ValueError as e:
            return Response({
                'success': False,
                'message': str(e),
                'data': None
            }, status=status.HTTP_400_BAD_REQUEST)
        
        return Response({
            'success': True,
            'message': f'Order status updated to {order.get_status_display()}',
            'data': OrderDetailSerializer(order).data
        })
    
    @action(detail=True, methods=['post'], url_path='tracking')
    def add_tracking(self, request, pk=None):
        """Add tracking information."""
        order = self.get_object()
        
        serializer = AddTrackingSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                'success': False,
                'message': 'Invalid data',
                'data': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        tracking_number = serializer.validated_data['tracking_number']
        tracking_url = serializer.validated_data.get('tracking_url', '')
        
        order = OrderService.add_tracking(order, tracking_number, tracking_url)
        
        return Response({
            'success': True,
            'message': 'Tracking added',
            'data': OrderDetailSerializer(order).data
        })
    
    @action(detail=True, methods=['post'], url_path='ship')
    def mark_shipped(self, request, pk=None):
        """Mark order as shipped."""
        order = self.get_object()
        
        tracking_number = request.data.get('tracking_number', '')
        tracking_url = request.data.get('tracking_url', '')
        
        order = OrderService.mark_shipped(
            order,
            tracking_number=tracking_number,
            tracking_url=tracking_url,
            shipped_by=request.user
        )
        
        return Response({
            'success': True,
            'message': 'Order marked as shipped',
            'data': OrderDetailSerializer(order).data
        })
    
    @action(detail=False, methods=['get'], url_path='statistics')
    def statistics(self, request):
        """Get order statistics."""
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        
        stats = OrderService.get_order_statistics(
            start_date=start_date,
            end_date=end_date
        )
        
        return Response({
            'success': True,
            'message': 'Statistics retrieved',
            'data': stats
        })
