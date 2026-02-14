"""
Analytics admin configuration
"""
from datetime import datetime, timedelta

from django.contrib import admin, messages
from django.core.cache import cache
from django.db.models import Max, Min, Sum
from django.contrib.auth import get_user_model
from django.utils import timezone

from .models import PageView, ProductView, SearchQuery, CartEvent, DailyStat, ProductStat, CategoryStat
from core.admin_mixins import ImportExportEnhancedModelAdmin


class AnalyticsStatsRefreshMixin:
    """
    Keep analytics aggregate tables reasonably fresh for admin pages.
    This avoids stale/empty DailyStat/ProductStat/CategoryStat screens when
    celery workers are down or lagging.
    """

    auto_refresh_ttl_seconds = 300
    reconcile_ttl_seconds = 900
    max_auto_backfill_days = 90

    def _refresh_cache_key(self):
        return "admin_analytics_auto_refresh_lock_v1"

    def _reconcile_cache_key(self):
        return "admin_analytics_recent_reconcile_lock_v1"

    def _iter_dates(self, start_date, end_date):
        current = start_date
        while current <= end_date:
            yield current
            current += timedelta(days=1)

    def _first_tracked_date(self):
        from apps.orders.models import Order

        earliest = None
        models = [PageView, ProductView, SearchQuery, CartEvent]
        for model in models:
            value = model.objects.aggregate(first=Min("created_at")).get("first")
            if value:
                day = timezone.localtime(value).date()
                earliest = day if earliest is None else min(earliest, day)

        first_order = (
            Order.objects.filter(is_deleted=False)
            .exclude(status__in=[Order.STATUS_CANCELLED, Order.STATUS_REFUNDED])
            .aggregate(first=Min("created_at"))
            .get("first")
        )
        if first_order:
            day = timezone.localtime(first_order).date()
            earliest = day if earliest is None else min(earliest, day)

        return earliest

    def _latest_source_timestamp(self):
        from apps.orders.models import Order

        sources = [
            PageView.objects.aggregate(last=Max("created_at")).get("last"),
            ProductView.objects.aggregate(last=Max("created_at")).get("last"),
            SearchQuery.objects.aggregate(last=Max("created_at")).get("last"),
            CartEvent.objects.aggregate(last=Max("created_at")).get("last"),
            Order.objects.filter(is_deleted=False).aggregate(last=Max("created_at")).get("last"),
        ]
        return max((ts for ts in sources if ts is not None), default=None)

    def _rebuild_dates(self, dates):
        from .services import ReportService

        unique_dates = sorted(set(date for date in dates if date))
        rebuilt = 0
        errors = []
        for day in unique_dates:
            try:
                ReportService.generate_daily_stats(day)
                rebuilt += 1
            except Exception as exc:  # pragma: no cover - defensive admin path
                errors.append(f"{day}: {exc}")
        cache.delete(self._refresh_cache_key())
        cache.delete(self._reconcile_cache_key())
        return rebuilt, errors

    def _needs_recent_reconcile(self, days=30):
        """
        Detect stale aggregate tables when source activity exists but recent
        DailyStat totals remain zero/mismatched.
        """
        from apps.orders.models import Order

        days = max(1, int(days or 30))
        end_date = timezone.localdate()
        start_date = end_date - timedelta(days=days - 1)
        start_dt = timezone.make_aware(datetime.combine(start_date, datetime.min.time()))
        end_dt = timezone.make_aware(datetime.combine(end_date + timedelta(days=1), datetime.min.time()))

        excluded_statuses = [Order.STATUS_CANCELLED, Order.STATUS_REFUNDED]
        User = get_user_model()

        source_page_views = PageView.objects.filter(
            created_at__gte=start_dt,
            created_at__lt=end_dt,
        ).count()
        source_product_views = ProductView.objects.filter(
            created_at__gte=start_dt,
            created_at__lt=end_dt,
        ).count()
        source_orders_qs = Order.objects.filter(
            created_at__gte=start_dt,
            created_at__lt=end_dt,
            is_deleted=False,
        ).exclude(status__in=excluded_statuses)
        source_orders_count = source_orders_qs.count()
        source_orders_revenue = source_orders_qs.aggregate(total=Sum("total")).get("total") or 0
        source_new_users = User.objects.filter(
            created_at__gte=start_dt,
            created_at__lt=end_dt,
            is_deleted=False,
        ).count()

        has_source_activity = any(
            (
                source_page_views > 0,
                source_product_views > 0,
                source_orders_count > 0,
                source_orders_revenue > 0,
                source_new_users > 0,
            )
        )
        if not has_source_activity:
            return False

        daily_rows = DailyStat.objects.filter(date__gte=start_date, date__lte=end_date)
        if not daily_rows.exists():
            return True

        totals = daily_rows.aggregate(
            page_views=Sum("page_views"),
            product_views=Sum("product_views"),
            orders_count=Sum("orders_count"),
            orders_revenue=Sum("orders_revenue"),
            new_registrations=Sum("new_registrations"),
        )
        daily_page_views = totals.get("page_views") or 0
        daily_product_views = totals.get("product_views") or 0
        daily_orders_count = totals.get("orders_count") or 0
        daily_orders_revenue = totals.get("orders_revenue") or 0
        daily_new_users = totals.get("new_registrations") or 0

        return any(
            (
                daily_page_views != source_page_views,
                daily_product_views != source_product_views,
                daily_orders_count != source_orders_count,
                abs(float(daily_orders_revenue) - float(source_orders_revenue)) > 0.01,
                daily_new_users != source_new_users,
            )
        )

    def _auto_refresh_recent_stats(self):
        """
        Auto-refresh recent aggregates with a throttle.
        - Bootstrap up to last N days when no daily rows exist.
        - Fill missing days up to today.
        - Refresh today/yesterday when today's row is stale.
        """
        try:
            from .services import AnalyticsService

            AnalyticsService.sync_product_views_from_impressions(
                days=self.max_auto_backfill_days
            )
            AnalyticsService.sync_cart_events_from_checkout_events(
                days=self.max_auto_backfill_days
            )
        except Exception:
            pass

        cache_key = self._refresh_cache_key()
        if cache.get(cache_key):
            return

        today = timezone.localdate()
        latest = DailyStat.objects.order_by("-date").first()
        dates_to_rebuild = []

        if latest is None:
            earliest = self._first_tracked_date()
            if earliest is None:
                cache.set(cache_key, True, self.auto_refresh_ttl_seconds)
                return
            floor = today - timedelta(days=self.max_auto_backfill_days - 1)
            start = max(earliest, floor)
            dates_to_rebuild.extend(self._iter_dates(start, today))
        else:
            if latest.date < today:
                gap_start = latest.date + timedelta(days=1)
                floor = today - timedelta(days=self.max_auto_backfill_days - 1)
                gap_start = max(gap_start, floor)
                dates_to_rebuild.extend(self._iter_dates(gap_start, today))

            stale_cutoff = timezone.now() - timedelta(minutes=30)
            if latest.date >= today and latest.updated_at < stale_cutoff:
                dates_to_rebuild.extend([today, today - timedelta(days=1)])

            latest_source_ts = self._latest_source_timestamp()
            if latest.date >= today and latest_source_ts and latest_source_ts > latest.updated_at:
                dates_to_rebuild.extend([today])

        if dates_to_rebuild:
            self._rebuild_dates(dates_to_rebuild)

        # Reconcile recent window when source activity and aggregate totals drift.
        reconcile_key = self._reconcile_cache_key()
        if not cache.get(reconcile_key):
            if self._needs_recent_reconcile(days=min(self.max_auto_backfill_days, 30)):
                self._rebuild_recent_days(request=None, days=min(self.max_auto_backfill_days, 30), as_action=False)
            cache.set(reconcile_key, True, self.reconcile_ttl_seconds)

        cache.set(cache_key, True, self.auto_refresh_ttl_seconds)

    def _rebuild_recent_days(self, request, days, as_action=True):
        today = timezone.localdate()
        start = today - timedelta(days=days - 1)
        rebuilt, errors = self._rebuild_dates(self._iter_dates(start, today))
        if not as_action:
            return rebuilt, errors
        if errors:
            self.message_user(
                request,
                f"Rebuilt {rebuilt} day(s) with {len(errors)} error(s). First error: {errors[0]}",
                level=messages.WARNING,
            )
        else:
            self.message_user(request, f"Successfully rebuilt last {rebuilt} day(s).")

    @admin.action(description="Rebuild Selected Day Stats")
    def rebuild_selected_days(self, request, queryset):
        dates = queryset.values_list("date", flat=True)
        rebuilt, errors = self._rebuild_dates(dates)
        if errors:
            self.message_user(
                request,
                f"Rebuilt {rebuilt} selected day(s) with {len(errors)} error(s). First error: {errors[0]}",
                level=messages.WARNING,
            )
        elif rebuilt:
            self.message_user(request, f"Successfully rebuilt {rebuilt} selected day(s).")
        else:
            self.message_user(request, "No valid dates selected.", level=messages.WARNING)

    @admin.action(description="Rebuild Last 30 Days")
    def rebuild_last_30_days(self, request, queryset):
        self._rebuild_recent_days(request, 30, as_action=True)

    @admin.action(description="Rebuild Last 90 Days")
    def rebuild_last_90_days(self, request, queryset):
        self._rebuild_recent_days(request, 90, as_action=True)

    @admin.action(description="Rebuild Last 365 Days")
    def rebuild_last_365_days(self, request, queryset):
        self._rebuild_recent_days(request, 365, as_action=True)


@admin.register(PageView)
class PageViewAdmin(ImportExportEnhancedModelAdmin):
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
class ProductViewAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['product', 'user', 'source', 'created_at']
    list_filter = ['source', 'created_at']
    search_fields = ['product__name', 'user__email']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False

    def get_queryset(self, request):
        try:
            from .services import AnalyticsService

            AnalyticsService.sync_product_views_from_impressions(days=90)
        except Exception:
            pass
        return super().get_queryset(request)


@admin.register(SearchQuery)
class SearchQueryAdmin(ImportExportEnhancedModelAdmin):
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
class CartEventAdmin(ImportExportEnhancedModelAdmin):
    list_display = ['event_type', 'product', 'quantity', 'cart_value', 'user', 'created_at']
    list_filter = ['event_type', 'created_at']
    search_fields = ['user__email', 'product__name']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False

    def get_queryset(self, request):
        try:
            from .services import AnalyticsService

            AnalyticsService.sync_cart_events_from_checkout_events(days=90)
        except Exception:
            pass
        return super().get_queryset(request)


@admin.register(DailyStat)
class DailyStatAdmin(AnalyticsStatsRefreshMixin, ImportExportEnhancedModelAdmin):
    list_display = [
        'date', 'page_views', 'unique_visitors', 'orders_count',
        'orders_revenue', 'average_order_value', 'conversion_rate', 'updated_at'
    ]
    list_filter = ['date']
    ordering = ['-date']
    actions = [
        'rebuild_selected_days',
        'rebuild_last_30_days',
        'rebuild_last_90_days',
        'rebuild_last_365_days',
    ]
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
        if obj is None:
            # Keep changelist + admin actions available while preventing edits.
            return True
        return False

    def get_queryset(self, request):
        self._auto_refresh_recent_stats()
        return super().get_queryset(request)


@admin.register(ProductStat)
class ProductStatAdmin(AnalyticsStatsRefreshMixin, ImportExportEnhancedModelAdmin):
    list_display = ['product', 'date', 'views', 'orders_count', 'revenue', 'conversion_rate']
    list_filter = ['date', 'product__primary_category']
    search_fields = ['product__name']
    ordering = ['-date', '-revenue']
    list_select_related = ['product', 'product__primary_category']
    actions = [
        'rebuild_selected_days',
        'rebuild_last_30_days',
        'rebuild_last_90_days',
    ]
    readonly_fields = ['product', 'date', 'views', 'add_to_cart_count', 'orders_count', 'revenue', 'conversion_rate']
    date_hierarchy = 'date'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        if obj is None:
            return True
        return False

    def get_queryset(self, request):
        self._auto_refresh_recent_stats()
        return super().get_queryset(request)


@admin.register(CategoryStat)
class CategoryStatAdmin(AnalyticsStatsRefreshMixin, ImportExportEnhancedModelAdmin):
    list_display = ['category', 'date', 'views', 'product_views', 'orders_count', 'revenue']
    list_filter = ['date']
    search_fields = ['category__name']
    ordering = ['-date', '-revenue']
    list_select_related = ['category']
    actions = [
        'rebuild_selected_days',
        'rebuild_last_30_days',
        'rebuild_last_90_days',
    ]
    readonly_fields = ['category', 'date', 'views', 'product_views', 'orders_count', 'revenue']
    date_hierarchy = 'date'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        if obj is None:
            return True
        return False

    def get_queryset(self, request):
        self._auto_refresh_recent_stats()
        return super().get_queryset(request)
