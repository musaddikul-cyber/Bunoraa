"""
Analytics services
"""
from django.db.models import Sum, Count, Avg, F
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth import get_user_model 
from apps.catalog.models import Product 

from .models import PageView, ProductView, SearchQuery, CartEvent, DailyStat, ProductStat


class AnalyticsService:
    """Service for tracking and querying analytics."""
    
    @staticmethod
    def track_page_view(request, time_on_page=0):
        """Track a page view."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        # Get device info from user agent
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        device_info = AnalyticsService._parse_user_agent(user_agent)
        
        # Get IP address
        ip = AnalyticsService._get_client_ip(request)
        
        PageView.objects.create(
            user=user,
            session_key=session_key,
            path=request.path,
            query_string=request.META.get('QUERY_STRING', ''),
            referrer=request.META.get('HTTP_REFERER'),
            user_agent=user_agent,
            device_type=device_info.get('device_type'),
            browser=device_info.get('browser'),
            os=device_info.get('os'),
            ip_address=ip,
            time_on_page=time_on_page
        )
    
    @staticmethod
    def track_product_view(product, request, source=None):
        """Track a product view."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        ProductView.objects.create(
            product=product,
            user=user,
            session_key=session_key,
            source=source,
            referrer=request.META.get('HTTP_REFERER')
        )
        
        # Update product view count
        product.view_count = F('view_count') + 1
        product.save(update_fields=['view_count'])
    
    @staticmethod
    def track_search(query, results_count, request, clicked_product=None):
        """Track a search query."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        SearchQuery.objects.create(
            query=query,
            user=user,
            session_key=session_key,
            results_count=results_count,
            clicked_product=clicked_product
        )
    
    @staticmethod
    def track_cart_event(event_type, request, product=None, quantity=1, cart_value=0):
        """Track a cart event."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key
        
        CartEvent.objects.create(
            event_type=event_type,
            user=user,
            session_key=session_key,
            product=product,
            quantity=quantity,
            cart_value=cart_value
        )
    
    @staticmethod
    def _parse_user_agent(user_agent):
        """Parse user agent string for device info."""
        user_agent_lower = user_agent.lower()
        
        # Device type
        if 'mobile' in user_agent_lower or 'android' in user_agent_lower:
            device_type = 'mobile'
        elif 'tablet' in user_agent_lower or 'ipad' in user_agent_lower:
            device_type = 'tablet'
        else:
            device_type = 'desktop'
        
        # Browser
        if 'chrome' in user_agent_lower:
            browser = 'Chrome'
        elif 'firefox' in user_agent_lower:
            browser = 'Firefox'
        elif 'safari' in user_agent_lower:
            browser = 'Safari'
        elif 'edge' in user_agent_lower:
            browser = 'Edge'
        else:
            browser = 'Other'
        
        # OS
        if 'windows' in user_agent_lower:
            os = 'Windows'
        elif 'mac' in user_agent_lower:
            os = 'macOS'
        elif 'linux' in user_agent_lower:
            os = 'Linux'
        elif 'android' in user_agent_lower:
            os = 'Android'
        elif 'ios' in user_agent_lower or 'iphone' in user_agent_lower:
            os = 'iOS'
        else:
            os = 'Other'
        
        return {
            'device_type': device_type,
            'browser': browser,
            'os': os
        }
    
    @staticmethod
    def _get_client_ip(request):
        """Get client IP address from request."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0].strip()
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class DashboardService:
    """Service for dashboard analytics."""
    
    @staticmethod
    def get_overview_stats(days=30):
        """Get overview statistics for dashboard."""
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        stats = DailyStat.objects.filter(
            date__gte=start_date,
            date__lte=end_date
        ).aggregate(
            total_revenue=Sum('orders_revenue'),
            total_orders=Sum('orders_count'),
            total_visitors=Sum('unique_visitors'),
            avg_order_value=Avg('average_order_value'),
            avg_conversion_rate=Avg('conversion_rate')
        )
        
        return {
            'total_revenue': stats['total_revenue'] or 0,
            'total_orders': stats['total_orders'] or 0,
            'total_visitors': stats['total_visitors'] or 0,
            'avg_order_value': stats['avg_order_value'] or 0,
            'avg_conversion_rate': stats['avg_conversion_rate'] or 0,
            'period_days': days
        }
    
    @staticmethod
    def get_revenue_chart_data(days=30):
        """Get revenue data for chart."""
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        stats = DailyStat.objects.filter(
            date__gte=start_date,
            date__lte=end_date
        ).values('date').annotate(
            revenue=Sum('orders_revenue'),
            orders=Sum('orders_count')
        ).order_by('date')
        
        return list(stats)
    
    @staticmethod
    def get_top_products(days=30, limit=10):
        """Get top performing products."""
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        top_products = ProductStat.objects.filter(
            date__gte=start_date,
            date__lte=end_date
        ).values('product__id', 'product__name', 'product__slug').annotate(
            total_revenue=Sum('revenue'),
            total_orders=Sum('orders_count'),
            total_views=Sum('views')
        ).order_by('-total_revenue')[:limit]
        
        return list(top_products)
    
    @staticmethod
    def get_popular_searches(days=7, limit=20):
        """Get popular search queries."""
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)
        
        searches = SearchQuery.objects.filter(
            created_at__gte=start_date
        ).values('query').annotate(
            count=Count('id'),
            avg_results=Avg('results_count')
        ).order_by('-count')[:limit]
        
        return list(searches)
    
    @staticmethod
    def get_traffic_sources(days=30):
        """Get traffic sources breakdown."""
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)
        
        sources = PageView.objects.filter(
            created_at__gte=start_date
        ).values('referrer').annotate(
            count=Count('id')
        ).order_by('-count')[:10]
        
        return list(sources)
    
    @staticmethod
    def get_device_breakdown(days=30):
        """Get device type breakdown."""
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)
        
        devices = PageView.objects.filter(
            created_at__gte=start_date
        ).values('device_type').annotate(
            count=Count('id')
        ).order_by('-count')
        
        return list(devices)
    
    @staticmethod
    def get_cart_analytics(days=30):
        """Get cart analytics."""
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)
        
        events = CartEvent.objects.filter(
            created_at__gte=start_date
        ).values('event_type').annotate(
            count=Count('id')
        )
        
        cart_data = {e['event_type']: e['count'] for e in events}
        
        add_count = cart_data.get(CartEvent.EVENT_ADD, 0)
        checkout_start = cart_data.get(CartEvent.EVENT_CHECKOUT_START, 0)
        checkout_complete = cart_data.get(CartEvent.EVENT_CHECKOUT_COMPLETE, 0)
        
        abandonment_rate = 0
        if checkout_start > 0:
            abandonment_rate = ((checkout_start - checkout_complete) / checkout_start) * 100
        
        return {
            'add_to_cart': add_count,
            'checkout_started': checkout_start,
            'checkout_completed': checkout_complete,
            'abandonment_rate': round(abandonment_rate, 2)
        }

    @staticmethod
    def get_new_users_data(days=30):
        """Get new user registrations over time."""
        User = get_user_model()
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)

        new_users = User.objects.filter(
            date_joined__date__gte=start_date,
            date_joined__date__lte=end_date
        ).annotate(date=TruncDate('date_joined')).values('date').annotate(count=Count('id')).order_by('date')
        
        # Fill missing dates with 0
        date_range = [start_date + timedelta(n) for n in range(days + 1)]
        data_dict = {str(d): {'date': d, 'count': 0} for d in date_range}
        for entry in new_users:
            data_dict[str(entry['date'])] = {'date': entry['date'], 'count': entry['count']}
        
        return list(data_dict.values())

    @staticmethod
    def get_product_views_data(days=30):
        """Get product views over time."""
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)

        product_views = ProductView.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date
        ).annotate(date=TruncDate('created_at')).values('date').annotate(count=Count('id')).order_by('date')

        # Fill missing dates with 0
        date_range = [start_date + timedelta(n) for n in range(days + 1)]
        data_dict = {str(d): {'date': d, 'count': 0} for d in date_range}
        for entry in product_views:
            data_dict[str(entry['date'])] = {'date': entry['date'], 'count': entry['count']}
        
        return list(data_dict.values())


class ReportService:
    """Service for generating reports."""
    
    @staticmethod
    def generate_daily_stats(date=None):
        """Generate daily statistics for a specific date."""
        if date is None:
            date = timezone.now().date() - timedelta(days=1)
        
        start_of_day = timezone.make_aware(
            timezone.datetime.combine(date, timezone.datetime.min.time())
        )
        end_of_day = start_of_day + timedelta(days=1)
        
        # Page views
        page_views = PageView.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day
        ).count()
        
        # Unique visitors
        unique_visitors = PageView.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day
        ).values('session_key').distinct().count()
        
        # Product views
        product_views = ProductView.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day
        ).count()
        
        # Cart events
        cart_adds = CartEvent.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day,
            event_type=CartEvent.EVENT_ADD
        ).values('product').distinct().count()
        
        checkout_starts = CartEvent.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day,
            event_type=CartEvent.EVENT_CHECKOUT_START
        ).count()
        
        checkout_completions = CartEvent.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day,
            event_type=CartEvent.EVENT_CHECKOUT_COMPLETE
        ).count()
        
        # Orders (from Order model)
        from apps.orders.models import Order
        orders = Order.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day
        ).aggregate(
            count=Count('id'),
            revenue=Sum('total_amount')
        )
        
        orders_count = orders['count'] or 0
        orders_revenue = orders['revenue'] or 0
        avg_order_value = orders_revenue / orders_count if orders_count > 0 else 0
        
        # Conversion rate
        conversion_rate = 0
        if unique_visitors > 0:
            conversion_rate = (orders_count / unique_visitors) * 100
        
        # Cart abandonment
        cart_abandonment_rate = 0
        if checkout_starts > 0:
            cart_abandonment_rate = ((checkout_starts - checkout_completions) / checkout_starts) * 100
        
        # New registrations
        from django.contrib.auth import get_user_model
        User = get_user_model()
        new_registrations = User.objects.filter(
            date_joined__gte=start_of_day,
            date_joined__lt=end_of_day
        ).count()
        
        # Create or update daily stat
        stat, _ = DailyStat.objects.update_or_create(
            date=date,
            defaults={
                'page_views': page_views,
                'unique_visitors': unique_visitors,
                'product_views': product_views,
                'products_added_to_cart': cart_adds,
                'orders_count': orders_count,
                'orders_revenue': orders_revenue,
                'average_order_value': avg_order_value,
                'checkout_starts': checkout_starts,
                'checkout_completions': checkout_completions,
                'conversion_rate': round(conversion_rate, 2),
                'cart_abandonment_rate': round(cart_abandonment_rate, 2),
                'new_registrations': new_registrations
            }
        )
        
        return stat
