"""
Bunoraa Admin Utilities
Provides utility views for admin panel operations.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from functools import wraps

from django.contrib import admin, messages
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.core.cache import cache
from django.db import connection
from django.conf import settings
from django.utils import timezone
from django.utils.translation import gettext as _

logger = logging.getLogger(__name__)


# =============================================================================
# DECORATORS
# =============================================================================

def superuser_required(view_func):
    """Decorator that requires superuser access."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_superuser:
            messages.error(request, _("Superuser access required."))
            return redirect('admin:index')
        return view_func(request, *args, **kwargs)
    return wrapper


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

@staff_member_required
def cache_management(request):
    """View for cache management operations."""
    context = {
        'title': _('Cache Management'),
        'has_permission': request.user.is_superuser,
    }
    
    # Get cache statistics if available
    try:
        from django_redis import get_redis_connection
        redis_conn = get_redis_connection("default")
        info = redis_conn.info()
        context['cache_stats'] = {
            'used_memory': info.get('used_memory_human', 'N/A'),
            'connected_clients': info.get('connected_clients', 'N/A'),
            'total_keys': redis_conn.dbsize(),
            'uptime_days': info.get('uptime_in_days', 'N/A'),
            'hit_rate': f"{info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1) * 100:.1f}%",
        }
    except Exception as e:
        context['cache_stats'] = {'error': str(e)}
    
    return render(request, 'admin/utilities/cache_management.html', context)


@staff_member_required
@superuser_required
@require_POST
def clear_all_cache(request):
    """Clear all cached data."""
    try:
        cache.clear()
        messages.success(request, _("All cache cleared successfully."))
        logger.info(f"Cache cleared by {request.user.email}")
    except Exception as e:
        messages.error(request, f"Error clearing cache: {str(e)}")
        logger.error(f"Cache clear error: {e}")
    
    return redirect('admin_cache_management')


@staff_member_required
@superuser_required
@require_POST
def clear_view_cache(request):
    """Clear view/template cache."""
    try:
        # Clear specific cache keys
        from django.core.cache import caches
        for cache_name in caches.all():
            try:
                cache_name.clear()
            except:
                pass
        messages.success(request, _("View cache cleared."))
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
    
    return redirect('admin_cache_management')


@staff_member_required
@superuser_required
@require_POST
def clear_session_cache(request):
    """Clear session cache (excluding current session)."""
    try:
        from django.contrib.sessions.models import Session
        current_session_key = request.session.session_key
        
        # Delete all sessions except current
        deleted = Session.objects.exclude(session_key=current_session_key).delete()
        messages.success(request, f"Cleared {deleted[0]} sessions.")
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
    
    return redirect('admin_cache_management')


# =============================================================================
# MAINTENANCE MODE
# =============================================================================

MAINTENANCE_MODE_CACHE_KEY = 'bunoraa_maintenance_mode'

@staff_member_required
def maintenance_mode_view(request):
    """View for maintenance mode settings."""
    context = {
        'title': _('Maintenance Mode'),
        'has_permission': request.user.is_superuser,
        'maintenance_enabled': cache.get(MAINTENANCE_MODE_CACHE_KEY, False),
        'maintenance_message': cache.get(f'{MAINTENANCE_MODE_CACHE_KEY}_message', ''),
        'maintenance_end': cache.get(f'{MAINTENANCE_MODE_CACHE_KEY}_end', None),
    }
    
    return render(request, 'admin/utilities/maintenance_mode.html', context)


@staff_member_required
@superuser_required
@require_POST
def enable_maintenance(request):
    """Enable maintenance mode."""
    message = request.POST.get('message', _('We are currently performing maintenance. Please check back soon.'))
    duration_hours = int(request.POST.get('duration', 1))
    end_time = timezone.now() + timedelta(hours=duration_hours)
    
    cache.set(MAINTENANCE_MODE_CACHE_KEY, True, timeout=duration_hours * 3600)
    cache.set(f'{MAINTENANCE_MODE_CACHE_KEY}_message', message, timeout=duration_hours * 3600)
    cache.set(f'{MAINTENANCE_MODE_CACHE_KEY}_end', end_time.isoformat(), timeout=duration_hours * 3600)
    
    messages.warning(request, f"Maintenance mode enabled for {duration_hours} hour(s).")
    logger.warning(f"Maintenance mode enabled by {request.user.email} for {duration_hours}h")
    
    return redirect('admin_maintenance_mode')


@staff_member_required
@superuser_required
@require_POST
def disable_maintenance(request):
    """Disable maintenance mode."""
    cache.delete(MAINTENANCE_MODE_CACHE_KEY)
    cache.delete(f'{MAINTENANCE_MODE_CACHE_KEY}_message')
    cache.delete(f'{MAINTENANCE_MODE_CACHE_KEY}_end')
    
    messages.success(request, _("Maintenance mode disabled."))
    logger.info(f"Maintenance mode disabled by {request.user.email}")
    
    return redirect('admin_maintenance_mode')


def is_maintenance_mode():
    """Check if maintenance mode is enabled."""
    return cache.get(MAINTENANCE_MODE_CACHE_KEY, False)


def get_maintenance_message():
    """Get maintenance mode message."""
    return cache.get(f'{MAINTENANCE_MODE_CACHE_KEY}_message', '')


# =============================================================================
# SYSTEM HEALTH
# =============================================================================

@staff_member_required
def system_health_view(request):
    """View for system health dashboard."""
    from core.admin_dashboard import check_system_health
    
    health = check_system_health()
    
    context = {
        'title': _('System Health'),
        'has_permission': request.user.is_superuser,
        'health': health,
        'overall_status': 'healthy' if all(s.get('status') == 'healthy' for s in health.values()) else 'degraded',
    }
    
    return render(request, 'admin/utilities/system_health.html', context)


@staff_member_required
@require_GET
def system_health_api(request):
    """API endpoint for system health check."""
    from core.admin_dashboard import check_system_health
    
    health = check_system_health()
    overall = 'healthy' if all(s.get('status') == 'healthy' for s in health.values()) else 'degraded'
    
    return JsonResponse({
        'status': overall,
        'services': health,
        'timestamp': timezone.now().isoformat(),
    })


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

@staff_member_required
def database_stats_view(request):
    """View for database statistics."""
    context = {
        'title': _('Database Statistics'),
        'has_permission': request.user.is_superuser,
    }
    
    try:
        with connection.cursor() as cursor:
            # Get table sizes (PostgreSQL specific)
            if 'postgresql' in settings.DATABASES['default']['ENGINE']:
                cursor.execute("""
                    SELECT 
                        relname as table_name,
                        pg_size_pretty(pg_total_relation_size(relid)) as total_size,
                        pg_size_pretty(pg_table_size(relid)) as data_size,
                        pg_size_pretty(pg_indexes_size(relid)) as index_size,
                        n_live_tup as row_count
                    FROM pg_stat_user_tables
                    ORDER BY pg_total_relation_size(relid) DESC
                    LIMIT 20;
                """)
                columns = [col[0] for col in cursor.description]
                context['table_stats'] = [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                context['table_stats'] = []
                context['db_note'] = 'Table size info only available for PostgreSQL'
                
    except Exception as e:
        context['table_stats'] = []
        context['db_error'] = str(e)
    
    return render(request, 'admin/utilities/database_stats.html', context)


@staff_member_required
@superuser_required
@require_POST
def vacuum_database(request):
    """Run VACUUM on PostgreSQL database."""
    try:
        if 'postgresql' in settings.DATABASES['default']['ENGINE']:
            with connection.cursor() as cursor:
                cursor.execute("VACUUM ANALYZE;")
            messages.success(request, _("Database VACUUM completed."))
            logger.info(f"Database VACUUM run by {request.user.email}")
        else:
            messages.info(request, _("VACUUM only available for PostgreSQL."))
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
    
    return redirect('admin_database_stats')


# =============================================================================
# BULK OPERATIONS
# =============================================================================

@staff_member_required
def bulk_operations_view(request):
    """View for bulk operations."""
    context = {
        'title': _('Bulk Operations'),
        'has_permission': request.user.is_superuser,
    }
    
    return render(request, 'admin/utilities/bulk_operations.html', context)


@staff_member_required
@superuser_required
@require_POST
def recalculate_all_product_stats(request):
    """Recalculate all product statistics."""
    from apps.catalog.models import Product
    from apps.analytics.models import ProductView
    from apps.orders.models import OrderItem
    from django.db.models import Sum, Count
    
    try:
        products = Product.objects.all()
        updated = 0
        
        for product in products:
            views = ProductView.objects.filter(product=product).count()
            sales_data = OrderItem.objects.filter(
                product=product,
                order__status='delivered'
            ).aggregate(
                total_qty=Sum('quantity'),
                total_orders=Count('id')
            )
            
            product.views_count = views
            product.sales_count = sales_data['total_qty'] or 0
            product.save(update_fields=['views_count', 'sales_count'])
            updated += 1
        
        messages.success(request, f"Updated statistics for {updated} products.")
        logger.info(f"Product stats recalculated by {request.user.email}")
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
    
    return redirect('admin_bulk_operations')


@staff_member_required
@superuser_required
@require_POST
def recalculate_category_counts(request):
    """Recalculate product counts for all categories."""
    from apps.catalog.models import Category
    
    try:
        categories = Category.objects.all()
        for category in categories:
            if hasattr(category, 'update_product_count'):
                category.update_product_count()
        
        messages.success(request, f"Updated product counts for {categories.count()} categories.")
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
    
    return redirect('admin_bulk_operations')


@staff_member_required
@superuser_required
@require_POST
def cleanup_orphaned_images(request):
    """Find and optionally delete orphaned product images."""
    from apps.catalog.models import ProductImage
    
    try:
        # Find images with no product
        orphaned = ProductImage.objects.filter(product__isnull=True)
        count = orphaned.count()
        
        if request.POST.get('delete') == 'true':
            orphaned.delete()
            messages.success(request, f"Deleted {count} orphaned images.")
            logger.info(f"Orphaned images cleanup by {request.user.email}: {count} deleted")
        else:
            messages.info(request, f"Found {count} orphaned images. Check 'delete' to remove them.")
    except Exception as e:
        messages.error(request, f"Error: {str(e)}")
    
    return redirect('admin_bulk_operations')


# =============================================================================
# AUDIT LOG VIEWER
# =============================================================================

@staff_member_required
def audit_log_view(request):
    """View for audit logs."""
    from django.contrib.admin.models import LogEntry
    from django.contrib.contenttypes.models import ContentType
    
    # Filters
    user_id = request.GET.get('user')
    action_flag = request.GET.get('action')
    content_type_id = request.GET.get('content_type')
    days = int(request.GET.get('days', 7))
    
    queryset = LogEntry.objects.select_related('user', 'content_type')
    
    if user_id:
        queryset = queryset.filter(user_id=user_id)
    if action_flag:
        queryset = queryset.filter(action_flag=action_flag)
    if content_type_id:
        queryset = queryset.filter(content_type_id=content_type_id)
    
    if days:
        cutoff = timezone.now() - timedelta(days=days)
        queryset = queryset.filter(action_time__gte=cutoff)
    
    queryset = queryset.order_by('-action_time')[:500]
    
    # Get filter options
    from apps.accounts.models import User
    staff_users = User.objects.filter(is_staff=True).values('id', 'email')
    content_types = ContentType.objects.filter(
        logentry__action_time__gte=timezone.now() - timedelta(days=30)
    ).distinct()
    
    context = {
        'title': _('Audit Log'),
        'has_permission': request.user.is_superuser,
        'logs': queryset,
        'staff_users': staff_users,
        'content_types': content_types,
        'selected_user': user_id,
        'selected_action': action_flag,
        'selected_content_type': content_type_id,
        'selected_days': days,
    }
    
    return render(request, 'admin/utilities/audit_log.html', context)


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

@staff_member_required
@superuser_required
def export_all_users_csv(request):
    """Export all users to CSV."""
    import csv
    from apps.accounts.models import User
    
    response = HttpResponse(content_type='text/csv')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = f'attachment; filename="users_export_{timestamp}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['ID', 'Email', 'First Name', 'Last Name', 'Phone', 'Verified', 
                     'Active', 'Staff', 'Newsletter', 'Created At', 'Last Login'])
    
    for user in User.objects.filter(is_deleted=False).order_by('-created_at'):
        writer.writerow([
            user.id,
            user.email,
            user.first_name,
            user.last_name,
            user.phone or '',
            'Yes' if user.is_verified else 'No',
            'Yes' if user.is_active else 'No',
            'Yes' if user.is_staff else 'No',
            'Yes' if user.newsletter_subscribed else 'No',
            user.created_at.isoformat() if user.created_at else '',
            user.last_login.isoformat() if user.last_login else '',
        ])
    
    logger.info(f"Users CSV export by {request.user.email}")
    return response


@staff_member_required
@superuser_required
def export_all_orders_csv(request):
    """Export all orders to CSV."""
    import csv
    from apps.orders.models import Order
    
    response = HttpResponse(content_type='text/csv')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = f'attachment; filename="orders_export_{timestamp}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Order Number', 'Customer Email', 'Status', 'Payment Status', 
                     'Subtotal', 'Discount', 'Tax', 'Total', 'Items', 'Created At'])
    
    for order in Order.objects.filter(is_deleted=False).order_by('-created_at'):
        writer.writerow([
            order.order_number,
            order.email,
            order.status,
            order.payment_status,
            order.subtotal,
            order.discount,
            order.tax,
            order.total,
            order.items.count(),
            order.created_at.isoformat() if order.created_at else '',
        ])
    
    logger.info(f"Orders CSV export by {request.user.email}")
    return response


@staff_member_required
@superuser_required
def export_all_products_csv(request):
    """Export all products to CSV."""
    import csv
    from apps.catalog.models import Product
    
    response = HttpResponse(content_type='text/csv')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = f'attachment; filename="products_export_{timestamp}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['SKU', 'Name', 'Category', 'Price', 'Sale Price', 'Stock', 
                     'Active', 'Featured', 'Views', 'Sales', 'Created At'])
    
    for product in Product.objects.select_related('primary_category').order_by('-created_at'):
        writer.writerow([
            product.sku,
            product.name,
            product.primary_category.name if product.primary_category else '',
            product.price,
            product.sale_price or '',
            product.stock_quantity,
            'Yes' if product.is_active else 'No',
            'Yes' if product.is_featured else 'No',
            product.views_count or 0,
            product.sales_count or 0,
            product.created_at.isoformat() if product.created_at else '',
        ])
    
    logger.info(f"Products CSV export by {request.user.email}")
    return response
