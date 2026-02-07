"""
Celery tasks for Analytics app.
Handles data aggregation and reporting.
"""
import logging
from celery import shared_task
from django.utils import timezone
from datetime import timedelta

logger = logging.getLogger('bunoraa.analytics')


@shared_task
def generate_daily_report():
    """
    Generate daily analytics report.
    """
    logger.info("Generating daily analytics report...")
    
    try:
        from apps.analytics.models import DailyStat
        from apps.orders.models import Order
        from apps.accounts.models import User
        
        yesterday = timezone.now().date() - timedelta(days=1)
        
        # Get or create daily stat
        stat, created = DailyStat.objects.get_or_create(date=yesterday)
        
        if created:
            logger.info(f"Created daily stat for {yesterday}")
        
        # Calculate additional metrics
        orders = Order.objects.filter(created_at__date=yesterday)
        new_users = User.objects.filter(created_at__date=yesterday)
        
        report = {
            'date': str(yesterday),
            'page_views': stat.page_views,
            'unique_visitors': stat.unique_visitors,
            'orders': stat.orders_count,
            'revenue': float(stat.orders_revenue or 0),
            'new_users': new_users.count(),
            'conversion_rate': float(stat.conversion_rate or 0),
        }
        
        logger.info(f"Daily report generated: {report}")
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        return {'error': str(e)}


@shared_task
def generate_weekly_report():
    """
    Generate weekly analytics report.
    """
    logger.info("Generating weekly analytics report...")
    
    try:
        from apps.analytics.models import DailyStat
        from django.db.models import Sum, Avg
        
        end_date = timezone.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        
        stats = DailyStat.objects.filter(
            date__gte=start_date,
            date__lte=end_date
        ).aggregate(
            total_page_views=Sum('page_views'),
            total_visitors=Sum('unique_visitors'),
            total_orders=Sum('orders_count'),
            total_revenue=Sum('orders_revenue'),
            avg_conversion=Avg('conversion_rate'),
        )
        
        report = {
            'week_start': str(start_date),
            'week_end': str(end_date),
            **stats
        }
        
        logger.info(f"Weekly report generated: {report}")
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate weekly report: {e}")
        return {'error': str(e)}


@shared_task
def cleanup_old_data(days: int = 365):
    """
    Clean up old analytics data.
    """
    logger.info(f"Cleaning up analytics data older than {days} days...")
    
    try:
        from apps.analytics.models import PageView, ProductView, SearchQuery, CartEvent
        
        cutoff = timezone.now() - timedelta(days=days)
        
        deleted_counts = {
            'page_views': PageView.objects.filter(created_at__lt=cutoff).delete()[0],
            'product_views': ProductView.objects.filter(created_at__lt=cutoff).delete()[0],
            'search_queries': SearchQuery.objects.filter(created_at__lt=cutoff).delete()[0],
            'cart_events': CartEvent.objects.filter(created_at__lt=cutoff).delete()[0],
        }
        
        total = sum(deleted_counts.values())
        logger.info(f"Deleted {total} old analytics records: {deleted_counts}")
        
        return deleted_counts
        
    except Exception as e:
        logger.error(f"Failed to cleanup analytics data: {e}")
        return {'error': str(e)}


@shared_task
def track_conversion_funnel():
    """
    Track daily conversion funnel metrics.
    """
    logger.info("Tracking conversion funnel...")
    
    try:
        from apps.analytics.models import CartEvent, PageView
        
        today = timezone.now().date()
        
        # Get funnel steps
        visitors = PageView.objects.filter(created_at__date=today).values('session_key').distinct().count()
        product_viewers = PageView.objects.filter(
            created_at__date=today,
            path__startswith='/products/'
        ).values('session_key').distinct().count()
        
        cart_adds = CartEvent.objects.filter(
            created_at__date=today,
            event_type='add'
        ).values('session_key').distinct().count()
        
        checkout_starts = CartEvent.objects.filter(
            created_at__date=today,
            event_type='checkout_start'
        ).values('session_key').distinct().count()
        
        purchases = CartEvent.objects.filter(
            created_at__date=today,
            event_type='checkout_complete'
        ).count()
        
        funnel = {
            'date': str(today),
            'visitors': visitors,
            'product_viewers': product_viewers,
            'cart_adds': cart_adds,
            'checkout_starts': checkout_starts,
            'purchases': purchases,
        }
        
        logger.info(f"Conversion funnel: {funnel}")
        return funnel
        
    except Exception as e:
        logger.error(f"Failed to track funnel: {e}")
        return {'error': str(e)}
