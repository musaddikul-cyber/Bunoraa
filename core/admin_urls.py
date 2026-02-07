"""
Admin URLs for Bunoraa Dashboard and Utilities
"""
from django.urls import path
from . import admin_dashboard
from . import admin_utilities

urlpatterns = [
    # Dashboard
    path('', admin_dashboard.admin_dashboard, name='admin_dashboard'),
    path('api/stats/', admin_dashboard.dashboard_stats_api, name='dashboard_stats_api'),
    path('api/revenue/', admin_dashboard.revenue_chart_api, name='dashboard_revenue_api'),
    path('api/top-products/', admin_dashboard.top_products_api, name='dashboard_top_products_api'),
    path('api/recent-orders/', admin_dashboard.recent_orders_api, name='dashboard_recent_orders_api'),
    path('api/low-stock/', admin_dashboard.low_stock_api, name='dashboard_low_stock_api'),
    path('api/user-activity/', admin_dashboard.user_activity_api, name='dashboard_user_activity_api'),
    path('api/system-health/', admin_dashboard.system_health_api, name='dashboard_system_health_api'),
    
    # Cache Management
    path('utilities/cache/', admin_utilities.cache_management, name='admin_cache_management'),
    path('utilities/cache/clear-all/', admin_utilities.clear_all_cache, name='admin_clear_all_cache'),
    path('utilities/cache/clear-views/', admin_utilities.clear_view_cache, name='admin_clear_view_cache'),
    path('utilities/cache/clear-sessions/', admin_utilities.clear_session_cache, name='admin_clear_session_cache'),
    
    # Maintenance Mode
    path('utilities/maintenance/', admin_utilities.maintenance_mode_view, name='admin_maintenance_mode'),
    path('utilities/maintenance/enable/', admin_utilities.enable_maintenance, name='admin_enable_maintenance'),
    path('utilities/maintenance/disable/', admin_utilities.disable_maintenance, name='admin_disable_maintenance'),
    
    # System Health
    path('utilities/health/', admin_utilities.system_health_view, name='admin_system_health'),
    path('utilities/health/api/', admin_utilities.system_health_api, name='admin_system_health_api'),
    
    # Database Stats
    path('utilities/database/', admin_utilities.database_stats_view, name='admin_database_stats'),
    path('utilities/database/vacuum/', admin_utilities.vacuum_database, name='admin_vacuum_database'),
    
    # Bulk Operations
    path('utilities/bulk/', admin_utilities.bulk_operations_view, name='admin_bulk_operations'),
    path('utilities/bulk/recalculate-stats/', admin_utilities.recalculate_all_product_stats, name='admin_recalculate_product_stats'),
    path('utilities/bulk/recalculate-categories/', admin_utilities.recalculate_category_counts, name='admin_recalculate_category_counts'),
    path('utilities/bulk/cleanup-images/', admin_utilities.cleanup_orphaned_images, name='admin_cleanup_orphaned_images'),
    
    # Audit Log
    path('utilities/audit-log/', admin_utilities.audit_log_view, name='admin_audit_log'),
    
    # Exports
    path('utilities/export/users/', admin_utilities.export_all_users_csv, name='admin_export_users_csv'),
    path('utilities/export/orders/', admin_utilities.export_all_orders_csv, name='admin_export_orders_csv'),
    path('utilities/export/products/', admin_utilities.export_all_products_csv, name='admin_export_products_csv'),
]
