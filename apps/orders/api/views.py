"""
Orders API views
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle

from ..models import Order
from ..services import OrderService
from .serializers import (
    OrderSerializer,
    OrderDetailSerializer,
    OrderListSerializer,
    CancelOrderSerializer,
    UpdateOrderStatusSerializer,
    AddTrackingSerializer,
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
    throttle_classes = [UserRateThrottle]
    throttle_scope = 'orders'
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return Order.objects.filter(
            user=self.request.user,
            is_deleted=False
        ).prefetch_related('items', 'status_history')
    
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
        order = self.get_object()
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
        order = self.get_object()
        
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
