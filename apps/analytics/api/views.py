"""
Analytics API views
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAdminUser
from django.contrib.auth import get_user_model 
from core.authentication import OptionalJWTAuthentication, CsrfExemptSessionAuthentication

User = get_user_model()

from ..models import DailyStat
from ..services import DashboardService, AnalyticsService
from .serializers import (
    DailyStatSerializer, OverviewStatSerializer, RevenueChartSerializer,
    TopProductSerializer, PopularSearchSerializer, DeviceBreakdownSerializer,
    CartAnalyticsSerializer, TrackEventSerializer
)


class DashboardViewSet(viewsets.ViewSet):
    """
    ViewSet for dashboard analytics (admin only).
    
    GET /api/v1/analytics/dashboard/ - Get overview stats
    GET /api/v1/analytics/dashboard/revenue/ - Get revenue chart data
    GET /api/v1/analytics/dashboard/top-products/ - Get top products
    GET /api/v1/analytics/dashboard/searches/ - Get popular searches
    GET /api/v1/analytics/dashboard/devices/ - Get device breakdown
    GET /api/v1/analytics/dashboard/cart/ - Get cart analytics
    """
    permission_classes = [IsAdminUser]
    
    def list(self, request):
        """Get overview statistics."""
        days = int(request.query_params.get('days', 30))
        stats = DashboardService.get_overview_stats(days)
        serializer = OverviewStatSerializer(stats)
        
        return Response({
            'success': True,
            'message': 'Overview statistics retrieved successfully',
            'data': serializer.data,
            'meta': {}
        })
    
    @action(detail=False, methods=['get'])
    def revenue(self, request):
        """Get revenue chart data."""
        days = int(request.query_params.get('days', 30))
        data = DashboardService.get_revenue_chart_data(days)
        
        return Response({
            'success': True,
            'message': 'Revenue data retrieved successfully',
            'data': data,
            'meta': {'days': days}
        })
    
    @action(detail=False, methods=['get'], url_path='top-products')
    def top_products(self, request):
        """Get top performing products."""
        days = int(request.query_params.get('days', 30))
        limit = int(request.query_params.get('limit', 10))
        data = DashboardService.get_top_products(days, limit)
        
        return Response({
            'success': True,
            'message': 'Top products retrieved successfully',
            'data': data,
            'meta': {'days': days, 'limit': limit}
        })
    
    @action(detail=False, methods=['get'])
    def searches(self, request):
        """Get popular search queries."""
        days = int(request.query_params.get('days', 7))
        limit = int(request.query_params.get('limit', 20))
        data = DashboardService.get_popular_searches(days, limit)
        
        return Response({
            'success': True,
            'message': 'Popular searches retrieved successfully',
            'data': data,
            'meta': {'days': days, 'limit': limit}
        })
    
    @action(detail=False, methods=['get'])
    def devices(self, request):
        """Get device type breakdown."""
        days = int(request.query_params.get('days', 30))
        data = DashboardService.get_device_breakdown(days)
        
        return Response({
            'success': True,
            'message': 'Device breakdown retrieved successfully',
            'data': data,
            'meta': {'days': days}
        })
    
    @action(detail=False, methods=['get'])
    def cart(self, request):
        """Get cart analytics."""
        days = int(request.query_params.get('days', 30))
        data = DashboardService.get_cart_analytics(days)
        serializer = CartAnalyticsSerializer(data)
        
        return Response({
            'success': True,
            'message': 'Cart analytics retrieved successfully',
            'data': serializer.data,
            'meta': {'days': days}
        })

    @action(detail=False, methods=['get'], url_path='new-users')
    def new_users(self, request):
        """Get new user registrations over time."""
        days = int(request.query_params.get('days', 30))
        data = DashboardService.get_new_users_data(days)
        return Response({
            'success': True,
            'message': 'New users data retrieved successfully',
            'data': data,
            'meta': {'days': days}
        })

    @action(detail=False, methods=['get'], url_path='product-views-over-time')
    def product_views_over_time(self, request):
        """Get product views over time."""
        days = int(request.query_params.get('days', 30))
        data = DashboardService.get_product_views_data(days)
        return Response({
            'success': True,
            'message': 'Product views data retrieved successfully',
            'data': data,
            'meta': {'days': days}
        })


class DailyStatViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for daily statistics (admin only).
    
    GET /api/v1/analytics/daily/ - List daily stats
    GET /api/v1/analytics/daily/{date}/ - Get specific date stats
    """
    queryset = DailyStat.objects.all()
    serializer_class = DailyStatSerializer
    permission_classes = [IsAdminUser]
    lookup_field = 'date'
    
    def list(self, request, *args, **kwargs):
        days = int(request.query_params.get('days', 30))
        queryset = self.get_queryset()[:days]
        serializer = self.get_serializer(queryset, many=True)
        
        return Response({
            'success': True,
            'message': 'Daily statistics retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })
    
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        
        return Response({
            'success': True,
            'message': 'Daily statistics retrieved successfully',
            'data': serializer.data,
            'meta': {}
        })


class TrackingViewSet(viewsets.ViewSet):
    """
    ViewSet for tracking events (public).
    
    POST /api/v1/analytics/track/ - Track an event
    """
    permission_classes = [AllowAny]
    authentication_classes = [OptionalJWTAuthentication, CsrfExemptSessionAuthentication]
    
    def create(self, request):
        """Track an event."""
        serializer = TrackEventSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        event_type = serializer.validated_data['event_type']
        
        try:
            if event_type == 'page_view':
                metadata = serializer.validated_data.get('metadata', {})
                AnalyticsService.track_page_view(
                    request,
                    time_on_page=metadata.get('time_on_page', 0),
                    page_path=metadata.get('page_path'),
                    query_string=metadata.get('query_string'),
                    referrer=metadata.get('referrer'),
                )
            
            elif event_type == 'product_view':
                product_id = serializer.validated_data.get('product_id')
                if product_id:
                    from apps.catalog.models import Product
                    product = Product.objects.filter(id=product_id).first()
                    if product:
                        source = serializer.validated_data.get('source')
                        AnalyticsService.track_product_view(product, request, source)
            
            elif event_type == 'search':
                query = serializer.validated_data.get('query')
                if query:
                    metadata = serializer.validated_data.get('metadata', {})
                    results_count = metadata.get('results_count', 0)
                    AnalyticsService.track_search(query, results_count, request)
            
            elif event_type == 'cart_add':
                product_id = serializer.validated_data.get('product_id')
                metadata = serializer.validated_data.get('metadata', {})
                from apps.catalog.models import Product
                product = Product.objects.filter(id=product_id).first() if product_id else None
                from ..models import CartEvent
                AnalyticsService.track_cart_event(
                    CartEvent.EVENT_ADD,
                    request,
                    product=product,
                    quantity=metadata.get('quantity', 1),
                    cart_value=metadata.get('cart_value', 0)
                )
            
            elif event_type == 'cart_remove':
                product_id = serializer.validated_data.get('product_id')
                from apps.catalog.models import Product
                product = Product.objects.filter(id=product_id).first() if product_id else None
                from ..models import CartEvent
                AnalyticsService.track_cart_event(CartEvent.EVENT_REMOVE, request, product=product)
            
            elif event_type == 'checkout_start':
                metadata = serializer.validated_data.get('metadata', {})
                from ..models import CartEvent
                AnalyticsService.track_cart_event(
                    CartEvent.EVENT_CHECKOUT_START,
                    request,
                    cart_value=metadata.get('cart_value', 0)
                )

            elif event_type == 'checkout_complete':
                metadata = serializer.validated_data.get('metadata', {})
                from ..models import CartEvent
                AnalyticsService.track_cart_event(
                    CartEvent.EVENT_CHECKOUT_COMPLETE,
                    request,
                    cart_value=metadata.get('cart_value', 0)
                )
            
            return Response({
                'success': True,
                'message': 'Event tracked successfully',
                'data': {},
                'meta': {}
            })
        
        except Exception as e:
            return Response({
                'success': False,
                'message': 'Failed to track event',
                'data': {'error': str(e)},
                'meta': {}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PublicAnalyticsViewSet(viewsets.ViewSet):
    """
    ViewSet for public analytics endpoints (no authentication required).
    
    GET /api/v1/analytics/active-visitors/ - Get active visitor count
    GET /api/v1/analytics/recent-purchases/ - Get recent purchase notifications
    """
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['get'])
    def active_visitors(self, request):
        """
        Get count of active visitors in the last 5 minutes.
        
        Returns:
            {
                'active_visitors': 28,
                'count': 28
            }
        """
        from django.utils import timezone
        from datetime import timedelta
        from ..models import PageView
        
        try:
            # Get active sessions from last 5 minutes
            five_minutes_ago = timezone.now() - timedelta(minutes=5)
            active_sessions = PageView.objects.filter(
                created_at__gte=five_minutes_ago
            ).values('session_key').distinct().count()
            
            return Response({
                'success': True,
                'message': 'Active visitors retrieved successfully',
                'data': {
                    'active_visitors': active_sessions,
                    'count': active_sessions
                },
                'meta': {}
            })
        except Exception as e:
            return Response({
                'success': False,
                'message': 'Failed to retrieve active visitors',
                'data': {'error': str(e)},
                'meta': {}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'], url_path='active-visitors')
    def get_active_visitors(self, request):
        """Alias for active_visitors endpoint"""
        return self.active_visitors(request)
    
    @action(detail=False, methods=['get'])
    def recent_purchases(self, request):
        """
        Get recent purchase notifications for social proof popups.
        
        Returns:
            {
                'purchases': [
                    {
                        'message': 'Someone in Dhaka purchased Handcrafted Leather Bag',
                        'time_ago': '2 min ago'
                    },
                    ...
                ]
            }
        """
        from django.utils import timezone
        from datetime import timedelta, datetime
        from apps.orders.models import Order
        from apps.catalog.models import Product
        
        try:
            # Get recent completed orders from last 2 hours
            two_hours_ago = timezone.now() - timedelta(hours=2)
            
            recent_orders = Order.objects.filter(
                created_at__gte=two_hours_ago,
                status__in=['completed', 'shipped', 'delivered']
            ).select_related('user').prefetch_related('items__product').order_by('-created_at')[:10]
            
            purchases = []
            for order in recent_orders:
                try:
                    # Get first product name from order
                    product_name = "an item"
                    if order.items.exists():
                        product_name = order.items.first().product.name
                    
                    # Get user's city if available
                    city = "a location"
                    if order.user and order.user.profile:
                        if hasattr(order.user.profile, 'city') and order.user.profile.city:
                            city = order.user.profile.city
                    elif order.shipping_address and isinstance(order.shipping_address, dict):
                        city = order.shipping_address.get('city', city)
                    
                    # Calculate time ago
                    time_diff = timezone.now() - order.created_at
                    if time_diff.total_seconds() < 60:
                        time_ago = "just now"
                    elif time_diff.total_seconds() < 3600:
                        minutes = int(time_diff.total_seconds() / 60)
                        time_ago = f"{minutes} min ago" if minutes > 1 else "1 min ago"
                    else:
                        hours = int(time_diff.total_seconds() / 3600)
                        time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
                    
                    purchases.append({
                        'message': f"Someone in {city} purchased {product_name}",
                        'time_ago': time_ago
                    })
                except Exception as e:
                    continue
            
            return Response({
                'success': True,
                'message': 'Recent purchases retrieved successfully',
                'data': {
                    'purchases': purchases
                },
                'meta': {}
            })
        
        except Exception as e:
            return Response({
                'success': False,
                'message': 'Failed to retrieve recent purchases',
                'data': {'error': str(e)},
                'meta': {}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PerformanceAnalyticsViewSet(viewsets.ViewSet):
    """
    ViewSet for performance metrics tracking from frontend.
    
    POST /api/v1/analytics/performance/ - Submit web vitals and performance metrics
    GET /api/v1/analytics/performance/stats/ - Get aggregated performance stats (admin)
    """
    permission_classes = [AllowAny]
    authentication_classes = [OptionalJWTAuthentication, CsrfExemptSessionAuthentication]
    
    def create(self, request):
        """
        Receive performance metrics from frontend web-vitals library.
        
        Expected payload:
        {
            'metrics': [
                {'name': 'LCP', 'value': 1200, 'rating': 'good'},
                {'name': 'FID', 'value': 16, 'rating': 'good'},
                {'name': 'CLS', 'value': 0.1, 'rating': 'needs-improvement'},
                ...
            ],
            'page': '/products/123',
            'userAgent': '...',
            'url': 'https://...'
        }
        """
        try:
            metrics_data = request.data.get('metrics', [])
            page = request.data.get('page', 'unknown')
            user_agent = request.data.get('userAgent', '')
            
            if not metrics_data:
                return Response({
                    'success': False,
                    'message': 'No metrics provided',
                    'data': {},
                    'meta': {}
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Store metrics in Redis for aggregation
            from django.core.cache import cache
            from django.utils import timezone
            import json
            
            timestamp = timezone.now().isoformat()
            session_key = request.session.session_key or 'anonymous'
            
            # Create metric entry
            metric_entry = {
                'timestamp': timestamp,
                'page': page,
                'session_key': session_key,
                'user_agent': user_agent[:200] if user_agent else '',
                'metrics': metrics_data
            }
            
            # Store in Redis list (keep last 10000 entries)
            cache_key = 'performance_metrics'
            existing = cache.get(cache_key, [])
            if existing:
                existing = json.loads(existing) if isinstance(existing, str) else existing
            
            existing.append(metric_entry)
            # Keep only last 10000 entries
            existing = existing[-10000:]
            
            cache.set(cache_key, json.dumps(existing), timeout=86400 * 7)  # 7 days
            
            # Also update daily aggregates
            today = timezone.now().strftime('%Y-%m-%d')
            daily_key = f'performance_metrics:{today}'
            daily_data = cache.get(daily_key, {})
            if daily_data:
                daily_data = json.loads(daily_data) if isinstance(daily_data, str) else daily_data
            
            for metric in metrics_data:
                metric_name = metric.get('name')
                value = metric.get('value')
                rating = metric.get('rating')
                
                if metric_name and value is not None:
                    if metric_name not in daily_data:
                        daily_data[metric_name] = {'values': [], 'good': 0, 'poor': 0, 'needs_improvement': 0}
                    
                    daily_data[metric_name]['values'].append(float(value))
                    if rating:
                        daily_data[metric_name][rating.replace('-', '_')] = daily_data[metric_name].get(rating.replace('-', '_'), 0) + 1
            
            cache.set(daily_key, json.dumps(daily_data), timeout=86400 * 30)  # 30 days
            
            return Response({
                'success': True,
                'message': 'Performance metrics recorded',
                'data': {'metrics_received': len(metrics_data)},
                'meta': {
                    'page': page,
                    'timestamp': timestamp
                }
            })
        
        except Exception as e:
            logger.exception("Error saving performance metrics")
            return Response({
                'success': False,
                'message': 'Failed to record metrics',
                'data': {'error': str(e)},
                'meta': {}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'], permission_classes=[IsAdminUser])
    def stats(self, request):
        """
        Get aggregated performance statistics (admin only).
        
        GET /api/v1/analytics/performance/stats/
        """
        from django.core.cache import cache
        from django.utils import timezone
        import json
        import statistics
        
        try:
            days = int(request.query_params.get('days', 7))
            
            # Aggregate metrics from last N days
            all_metrics = {}
            
            for i in range(days):
                date = (timezone.now() - timezone.timedelta(days=i)).strftime('%Y-%m-%d')
                daily_key = f'performance_metrics:{date}'
                daily_data = cache.get(daily_key)
                
                if daily_data:
                    daily_data = json.loads(daily_data) if isinstance(daily_data, str) else daily_data
                    
                    for metric_name, data in daily_data.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = {'values': [], 'good': 0, 'poor': 0, 'needs_improvement': 0}
                        
                        all_metrics[metric_name]['values'].extend(data.get('values', []))
                        all_metrics[metric_name]['good'] += data.get('good', 0)
                        all_metrics[metric_name]['poor'] += data.get('poor', 0)
                        all_metrics[metric_name]['needs_improvement'] += data.get('needs_improvement', 0)
            
            # Calculate statistics
            result = {}
            for metric_name, data in all_metrics.items():
                values = data.get('values', [])
                if values:
                    result[metric_name] = {
                        'count': len(values),
                        'avg': round(statistics.mean(values), 2),
                        'median': round(statistics.median(values), 2),
                        'p75': round(sorted(values)[int(len(values) * 0.75)], 2) if len(values) > 1 else values[0],
                        'p95': round(sorted(values)[int(len(values) * 0.95)], 2) if len(values) > 1 else values[0],
                        'min': round(min(values), 2),
                        'max': round(max(values), 2),
                        'good': data.get('good', 0),
                        'needs_improvement': data.get('needs_improvement', 0),
                        'poor': data.get('poor', 0),
                    }
            
            return Response({
                'success': True,
                'message': 'Performance statistics retrieved',
                'data': result,
                'meta': {
                    'days': days,
                    'metrics_tracked': list(result.keys())
                }
            })
        
        except Exception as e:
            logger.exception("Error retrieving performance stats")
            return Response({
                'success': False,
                'message': 'Failed to retrieve stats',
                'data': {'error': str(e)},
                'meta': {}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'], url_path='current', permission_classes=[IsAdminUser])
    def current_metrics(self, request):
        """
        Get current/recent performance metrics (last hour).
        
        GET /api/v1/analytics/performance/current/
        """
        from django.core.cache import cache
        import json
        
        try:
            cache_key = 'performance_metrics'
            data = cache.get(cache_key)
            
            if not data:
                return Response({
                    'success': True,
                    'message': 'No recent metrics available',
                    'data': [],
                    'meta': {}
                })
            
            metrics = json.loads(data) if isinstance(data, str) else data
            
            # Return last 100 entries
            return Response({
                'success': True,
                'message': 'Recent metrics retrieved',
                'data': metrics[-100:] if len(metrics) > 100 else metrics,
                'meta': {'count': min(len(metrics), 100)}
            })
        
        except Exception as e:
            return Response({
                'success': False,
                'message': 'Failed to retrieve current metrics',
                'data': {'error': str(e)},
                'meta': {}
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
