"""
Analytics admin configuration
"""
from django.contrib import admin
from .models import PageView, ProductView, SearchQuery, CartEvent, DailyStat, ProductStat, CategoryStat


@admin.register(PageView)
class PageViewAdmin(admin.ModelAdmin):
    list_display = ['path', 'user', 'device_type', 'browser', 'created_at']
    list_filter = ['device_type', 'browser', 'os', 'created_at']
    search_fields = ['path', 'user__email']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(ProductView)
class ProductViewAdmin(admin.ModelAdmin):
    list_display = ['product', 'user', 'source', 'created_at']
    list_filter = ['source', 'created_at']
    search_fields = ['product__name', 'user__email']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ['query', 'results_count', 'clicked_product', 'user', 'created_at']
    list_filter = ['created_at']
    search_fields = ['query', 'user__email']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(CartEvent)
class CartEventAdmin(admin.ModelAdmin):
    list_display = ['event_type', 'product', 'quantity', 'cart_value', 'user', 'created_at']
    list_filter = ['event_type', 'created_at']
    search_fields = ['user__email', 'product__name']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(DailyStat)
class DailyStatAdmin(admin.ModelAdmin):
    list_display = [
        'date', 'page_views', 'unique_visitors', 'orders_count',
        'orders_revenue', 'conversion_rate'
    ]
    list_filter = ['date']
    readonly_fields = [
        'date', 'page_views', 'unique_visitors', 'new_visitors',
        'returning_visitors', 'product_views', 'products_added_to_cart',
        'orders_count', 'orders_revenue', 'average_order_value',
        'checkout_starts', 'checkout_completions', 'conversion_rate',
        'cart_abandonment_rate', 'new_registrations', 'created_at', 'updated_at'
    ]
    date_hierarchy = 'date'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(ProductStat)
class ProductStatAdmin(admin.ModelAdmin):
    list_display = ['product', 'date', 'views', 'orders_count', 'revenue', 'conversion_rate']
    list_filter = ['date']
    search_fields = ['product__name']
    readonly_fields = ['product', 'date', 'views', 'add_to_cart_count', 'orders_count', 'revenue', 'conversion_rate']
    date_hierarchy = 'date'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(CategoryStat)
class CategoryStatAdmin(admin.ModelAdmin):
    list_display = ['category', 'date', 'views', 'product_views', 'orders_count', 'revenue']
    list_filter = ['date']
    search_fields = ['category__name']
    readonly_fields = ['category', 'date', 'views', 'product_views', 'orders_count', 'revenue']
    date_hierarchy = 'date'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False
