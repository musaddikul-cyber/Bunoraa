"""
Bunoraa Admin Dashboard
Comprehensive analytics and KPI dashboard for the admin panel.
"""
import json
from datetime import datetime, timedelta
from decimal import Decimal

from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Sum, Count, Avg, F, Q
from django.db.models.functions import TruncDate, TruncMonth
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.cache import cache_page
from django.views.decorators.http import require_GET
from apps.analytics.services import DashboardService 


def get_date_range(period='30d'):
    """Get date range based on period string."""
    now = timezone.now()
    if period == '7d':
        start = now - timedelta(days=7)
    elif period == '30d':
        start = now - timedelta(days=30)
    elif period == '90d':
        start = now - timedelta(days=90)
    elif period == '1y':
        start = now - timedelta(days=365)
    elif period == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start = now - timedelta(days=30)
    return start, now


def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


@staff_member_required
@require_GET
def admin_dashboard(request):
    """Main admin dashboard view."""
    period = request.GET.get('period', '30d')
    start_date, end_date = get_date_range(period)
    
    context = {
        'title': 'Dashboard',
        'period': period,
        'start_date': start_date.isoformat(), 
        'end_date': end_date.isoformat(),     
        'selected_period': period,
    }
    
    return render(request, 'admin/dashboard/index.html', context)


@staff_member_required
@require_GET
@cache_page(60 * 5)  # Cache for 5 minutes
def dashboard_stats_api(request):
    """API endpoint for dashboard statistics."""
    period = request.GET.get('period', '30d')
    days = int(period.replace('d', '').replace('y', '365') if period != 'today' else 0) if period else 30 
    if period == 'today':
        days = 1
    
    start_date, end_date = get_date_range(period)
    
    stats = get_summary_stats(start_date, end_date)
    
    # Add new users and product views data
    stats['new_users_over_time'] = DashboardService.get_new_users_data(days)
    stats['product_views_over_time'] = DashboardService.get_product_views_data(days)
    
    # Convert Decimals to floats for JSON serialization
    for key, value in stats.items():
        if isinstance(value, Decimal):
            stats[key] = float(value)
        elif isinstance(value, dict):
            stats[key] = {k: decimal_to_float(v) for k, v in value.items()}
        elif isinstance(value, list): # Handle lists (like chart data)
            stats[key] = [{k: decimal_to_float(v) for k, v in item.items()} if isinstance(item, dict) else item for item in value]
    
    return JsonResponse(stats)


def get_summary_stats(start_date, end_date):
    """Get summary statistics for the dashboard."""
    from apps.orders.models import Order
    from apps.accounts.models import User
    from apps.catalog.models import Product
    from apps.analytics.models import PageView, ProductView
    
    # Previous period for comparison
    period_length = (end_date - start_date).days
    prev_start = start_date - timedelta(days=period_length)
    prev_end = start_date
    
    # Order statistics
    orders = Order.objects.filter(
        created_at__gte=start_date,
        created_at__lte=end_date,
        is_deleted=False
    )
    prev_orders = Order.objects.filter(
        created_at__gte=prev_start,
        created_at__lte=prev_end,
        is_deleted=False
    )
    
    total_revenue = orders.aggregate(total=Sum('total'))['total'] or Decimal('0')
    prev_revenue = prev_orders.aggregate(total=Sum('total'))['total'] or Decimal('0')
    
    total_orders = orders.count()
    prev_total_orders = prev_orders.count()
    
    avg_order_value = orders.aggregate(avg=Avg('total'))['avg'] or Decimal('0')
    
    # User statistics
    new_users = User.objects.filter(
        created_at__gte=start_date,
        created_at__lte=end_date,
        is_deleted=False
    ).count()
    prev_new_users = User.objects.filter(
        created_at__gte=prev_start,
        created_at__lte=prev_end,
        is_deleted=False
    ).count()
    
    total_users = User.objects.filter(is_deleted=False).count()
    active_users = User.objects.filter(
        is_deleted=False,
        last_login__gte=start_date
    ).count()
    
    # Product statistics
    total_products = Product.objects.filter(is_deleted=False).count()
    active_products = Product.objects.filter(is_deleted=False, is_active=True).count()
    low_stock_products = Product.objects.filter(
        is_deleted=False,
        is_active=True,
        stock_quantity__lte=F('low_stock_threshold')
    ).count()
    out_of_stock = Product.objects.filter(
        is_deleted=False,
        is_active=True,
        stock_quantity=0
    ).count()
    
    # Page views
    try:
        page_views = PageView.objects.filter(
            created_at__gte=start_date,
            created_at__lte=end_date
        ).count()
        prev_page_views = PageView.objects.filter(
            created_at__gte=prev_start,
            created_at__lte=prev_end
        ).count()
    except Exception:
        page_views = 0
        prev_page_views = 0
    
    # Product views
    try:
        product_views = ProductView.objects.filter(
            created_at__gte=start_date,
            created_at__lte=end_date
        ).count()
    except Exception:
        product_views = 0
    
    # Order status breakdown
    order_status_breakdown = dict(
        orders.values('status').annotate(count=Count('id')).values_list('status', 'count')
    )
    
    # Calculate growth percentages
    def calc_growth(current, previous):
        if previous == 0:
            return 100 if current > 0 else 0
        return round(((current - previous) / previous) * 100, 1)
    
    revenue_growth = calc_growth(float(total_revenue), float(prev_revenue))
    orders_growth = calc_growth(total_orders, prev_total_orders)
    users_growth = calc_growth(new_users, prev_new_users)
    views_growth = calc_growth(page_views, prev_page_views)
    
    return {
        # Revenue
        'total_revenue': total_revenue,
        'prev_revenue': prev_revenue,
        'revenue_growth': revenue_growth,
        'avg_order_value': avg_order_value,
        
        # Orders
        'total_orders': total_orders,
        'prev_orders': prev_total_orders,
        'orders_growth': orders_growth,
        'order_status_breakdown': order_status_breakdown,
        
        # Users
        'total_users': total_users,
        'new_users': new_users,
        'prev_new_users': prev_new_users,
        'users_growth': users_growth,
        'active_users': active_users,
        
        # Products
        'total_products': total_products,
        'active_products': active_products,
        'low_stock_products': low_stock_products,
        'out_of_stock': out_of_stock,
        
        # Traffic
        'page_views': page_views,
        'prev_page_views': prev_page_views,
        'views_growth': views_growth,
        'product_views': product_views,
    }


@staff_member_required
@require_GET
@cache_page(60 * 5)
def revenue_chart_api(request):
    """API endpoint for revenue chart data."""
    from apps.orders.models import Order
    
    period = request.GET.get('period', '30d')
    start_date, end_date = get_date_range(period)
    
    orders = Order.objects.filter(
        created_at__gte=start_date,
        created_at__lte=end_date,
        is_deleted=False
    ).annotate(
        date=TruncDate('created_at')
    ).values('date').annotate(
        revenue=Sum('total'),
        orders=Count('id')
    ).order_by('date')
    
    data = {
        'labels': [o['date'].strftime('%Y-%m-%d') for o in orders],
        'revenue': [float(o['revenue'] or 0) for o in orders],
        'orders': [o['orders'] for o in orders],
    }
    
    return JsonResponse(data)


@staff_member_required
@require_GET
@cache_page(60 * 5)
def top_products_api(request):
    """API endpoint for top selling products."""
    from apps.orders.models import OrderItem
    
    period = request.GET.get('period', '30d')
    limit = int(request.GET.get('limit', 10))
    start_date, end_date = get_date_range(period)
    
    top_products = OrderItem.objects.filter(
        order__created_at__gte=start_date,
        order__created_at__lte=end_date,
        order__is_deleted=False
    ).values(
        'product_name'
    ).annotate(
        total_quantity=Sum('quantity'),
        total_revenue=Sum(F('quantity') * F('price'))
    ).order_by('-total_revenue')[:limit]
    
    data = {
        'products': [
            {
                'name': p['product_name'],
                'quantity': p['total_quantity'],
                'revenue': float(p['total_revenue'] or 0)
            }
            for p in top_products
        ]
    }
    
    return JsonResponse(data)


@staff_member_required
@require_GET
@cache_page(60 * 5)
def recent_orders_api(request):
    """API endpoint for recent orders."""
    from apps.orders.models import Order
    
    limit = int(request.GET.get('limit', 10))
    
    orders = Order.objects.filter(
        is_deleted=False
    ).select_related('user').order_by('-created_at')[:limit]
    
    data = {
        'orders': [
            {
                'id': str(o.id),
                'order_number': o.order_number,
                'customer': o.user.email if o.user else o.email,
                'total': float(o.total),
                'status': o.status,
                'created_at': o.created_at.isoformat(),
            }
            for o in orders
        ]
    }
    
    return JsonResponse(data)


@staff_member_required
@require_GET
@cache_page(60 * 5)
def low_stock_api(request):
    """API endpoint for low stock products."""
    from apps.catalog.models import Product
    
    limit = int(request.GET.get('limit', 10))
    
    products = Product.objects.filter(
        is_deleted=False,
        is_active=True,
        stock_quantity__lte=F('low_stock_threshold')
    ).order_by('stock_quantity')[:limit]
    
    data = {
        'products': [
            {
                'id': str(p.id),
                'name': p.name,
                'sku': p.sku,
                'stock': p.stock_quantity,
                'threshold': p.low_stock_threshold,
            }
            for p in products
        ]
    }
    
    return JsonResponse(data)


@staff_member_required
@require_GET
@cache_page(60 * 5)
def user_activity_api(request):
    """API endpoint for user activity chart."""
    from apps.accounts.models import User
    
    period = request.GET.get('period', '30d')
    start_date, end_date = get_date_range(period)
    
    # New user registrations by day
    new_users = User.objects.filter(
        created_at__gte=start_date,
        created_at__lte=end_date,
        is_deleted=False
    ).annotate(
        date=TruncDate('created_at')
    ).values('date').annotate(
        count=Count('id')
    ).order_by('date')
    
    data = {
        'labels': [u['date'].strftime('%Y-%m-%d') for u in new_users],
        'registrations': [u['count'] for u in new_users],
    }
    
    return JsonResponse(data)


@staff_member_required
@require_GET
def system_health_api(request):
    """API endpoint for system health status."""
    from django.core.cache import cache
    from django.db import connection
    
    health = {
        'database': False,
        'cache': False,
        'storage': False,
    }
    
    # Check database
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
        health['database'] = True
    except Exception:
        pass
    
    # Check cache
    try:
        cache.set('health_check', 'ok', 10)
        if cache.get('health_check') == 'ok':
            health['cache'] = True
    except Exception:
        pass
    
    # Check storage
    try:
        from django.core.files.storage import default_storage
        health['storage'] = default_storage.exists('') or True
    except Exception:
        pass
    
    return JsonResponse(health)
