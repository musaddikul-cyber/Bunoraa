"""
Bunoraa Admin Dashboard
Robust analytics and KPI APIs for the admin panel.
"""
from datetime import date, datetime, timedelta
from decimal import Decimal
import time

from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import (
    Avg,
    Case,
    CharField,
    Count,
    DecimalField,
    ExpressionWrapper,
    F,
    Q,
    Sum,
    Value,
    When,
)
from django.db.models.functions import Cast, Concat, TruncDate
from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils import timezone
from django.views.decorators.cache import cache_page
from django.views.decorators.http import require_GET


PERIOD_DAY_MAP = {
    "today": 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "1y": 365,
}


def _safe_int(value, default, min_value=None, max_value=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _parse_iso_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def get_date_range(period="30d", start_param=None, end_param=None):
    """
    Resolve a period to a timezone-aware [start, end) window.
    Supports predefined periods and custom start/end query dates.
    """
    now = timezone.now()
    period_key = (period or "30d").lower().strip()

    if period_key == "custom":
        start_date = _parse_iso_date(start_param)
        end_date = _parse_iso_date(end_param)
        if start_date and end_date and start_date <= end_date:
            start = timezone.make_aware(datetime.combine(start_date, datetime.min.time()))
            end_exclusive = timezone.make_aware(
                datetime.combine(end_date + timedelta(days=1), datetime.min.time())
            )
            # Do not project into the future for live analytics.
            end_exclusive = min(end_exclusive, now + timedelta(seconds=1))
            if end_exclusive > start:
                return start, end_exclusive, "custom"

    if period_key == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now + timedelta(seconds=1), "today"

    day_count = PERIOD_DAY_MAP.get(period_key, PERIOD_DAY_MAP["30d"])
    start = now - timedelta(days=day_count)
    return start, now + timedelta(seconds=1), period_key


def _period_days(start_date, end_date):
    delta_seconds = max((end_date - start_date).total_seconds(), 1)
    return max(1, int(delta_seconds // 86400))


def _growth(current, previous):
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    return round(((current - previous) / previous) * 100, 2)


def _visitor_key_expression():
    return Case(
        When(
            user_id__isnull=False,
            then=Concat(Value("u:"), Cast("user_id", output_field=CharField())),
        ),
        When(
            Q(session_key__isnull=False) & ~Q(session_key=""),
            then=Concat(Value("s:"), F("session_key")),
        ),
        When(
            ip_address__isnull=False,
            then=Concat(Value("ip:"), Cast("ip_address", output_field=CharField())),
        ),
        default=Concat(Value("pv:"), Cast("id", output_field=CharField())),
        output_field=CharField(),
    )


def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


def _json_safe(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@staff_member_required
@require_GET
def admin_dashboard(request):
    """Redirect to analytics model pages (frontend handled by React app)."""
    return redirect("/admin/analytics/dailystat/")


@staff_member_required
@require_GET
def dashboard_stats_api(request):
    """API endpoint for admin dashboard statistics."""
    start_date, end_date, period = get_date_range(
        period=request.GET.get("period", "30d"),
        start_param=request.GET.get("start"),
        end_param=request.GET.get("end"),
    )
    stats = get_summary_stats(start_date, end_date)
    stats["period"] = period
    stats["start_date"] = start_date
    stats["end_date"] = end_date
    stats["generated_at"] = timezone.now()
    return JsonResponse(_json_safe(stats))


def get_summary_stats(start_date, end_date):
    """Get robust summary statistics for the admin dashboard."""
    from apps.accounts.models import User
    from apps.analytics.models import CartEvent, PageView, ProductView
    from apps.catalog.models import Product
    from apps.orders.models import Order, OrderItem

    # Current and previous windows use [start, end) for exact non-overlapping comparisons.
    window = end_date - start_date
    if window.total_seconds() <= 0:
        window = timedelta(days=1)
        end_date = start_date + window
    prev_start = start_date - window
    prev_end = start_date

    excluded_statuses = [Order.STATUS_CANCELLED, Order.STATUS_REFUNDED]
    line_total_expr = ExpressionWrapper(
        F("quantity") * F("unit_price"),
        output_field=DecimalField(max_digits=14, decimal_places=2),
    )
    visitor_key_expr = _visitor_key_expression()

    current_orders = Order.objects.filter(
        created_at__gte=start_date,
        created_at__lt=end_date,
        is_deleted=False,
    ).exclude(status__in=excluded_statuses)
    previous_orders = Order.objects.filter(
        created_at__gte=prev_start,
        created_at__lt=prev_end,
        is_deleted=False,
    ).exclude(status__in=excluded_statuses)

    total_revenue = current_orders.aggregate(total=Sum("total"))["total"] or Decimal("0")
    prev_revenue = previous_orders.aggregate(total=Sum("total"))["total"] or Decimal("0")
    total_orders = current_orders.count()
    prev_total_orders = previous_orders.count()
    avg_order_value = (
        (total_revenue / total_orders) if total_orders else Decimal("0")
    )

    current_page_views_qs = PageView.objects.filter(
        created_at__gte=start_date,
        created_at__lt=end_date,
    )
    previous_page_views_qs = PageView.objects.filter(
        created_at__gte=prev_start,
        created_at__lt=prev_end,
    )
    page_views = current_page_views_qs.count()
    prev_page_views = previous_page_views_qs.count()
    unique_visitors = current_page_views_qs.annotate(
        visitor_key=visitor_key_expr
    ).values("visitor_key").distinct().count()
    prev_unique_visitors = previous_page_views_qs.annotate(
        visitor_key=visitor_key_expr
    ).values("visitor_key").distinct().count()

    product_views = ProductView.objects.filter(
        created_at__gte=start_date,
        created_at__lt=end_date,
    ).count()
    prev_product_views = ProductView.objects.filter(
        created_at__gte=prev_start,
        created_at__lt=prev_end,
    ).count()

    new_users = User.objects.filter(
        created_at__gte=start_date,
        created_at__lt=end_date,
        is_deleted=False,
    ).count()
    prev_new_users = User.objects.filter(
        created_at__gte=prev_start,
        created_at__lt=prev_end,
        is_deleted=False,
    ).count()
    total_users = User.objects.filter(is_deleted=False).count()
    active_users = User.objects.filter(
        is_deleted=False,
        last_login__gte=start_date,
    ).count()

    total_products = Product.objects.filter(is_deleted=False).count()
    active_products = Product.objects.filter(is_deleted=False, is_active=True).count()
    low_stock_products = Product.objects.filter(
        is_deleted=False,
        is_active=True,
        stock_quantity__lte=F("low_stock_threshold"),
    ).count()
    out_of_stock = Product.objects.filter(
        is_deleted=False,
        is_active=True,
        stock_quantity=0,
    ).count()

    cart_events_qs = CartEvent.objects.filter(
        created_at__gte=start_date,
        created_at__lt=end_date,
    )
    products_added_to_cart = cart_events_qs.filter(
        event_type=CartEvent.EVENT_ADD
    ).count()
    checkout_starts = cart_events_qs.filter(
        event_type=CartEvent.EVENT_CHECKOUT_START
    ).count()
    checkout_completions = cart_events_qs.filter(
        event_type=CartEvent.EVENT_CHECKOUT_COMPLETE
    ).count()
    cart_abandonment_rate = (
        ((checkout_starts - checkout_completions) / checkout_starts) * 100
        if checkout_starts
        else 0
    )
    conversion_rate = (
        (total_orders / unique_visitors) * 100
        if unique_visitors
        else 0
    )

    active_visitors_5m = PageView.objects.filter(
        created_at__gte=timezone.now() - timedelta(minutes=5)
    ).annotate(
        visitor_key=visitor_key_expr
    ).values(
        "visitor_key"
    ).distinct().count()

    order_status_breakdown = {
        row["status"]: row["count"]
        for row in current_orders.values("status").annotate(count=Count("id")).order_by("-count")
    }

    device_breakdown = list(
        current_page_views_qs.values("device_type")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    traffic_sources = list(
        current_page_views_qs.annotate(
            source_type=Case(
                When(Q(referrer__isnull=True) | Q(referrer=""), then=Value("direct")),
                When(referrer__icontains="bunoraa.com", then=Value("internal")),
                default=Value("external"),
                output_field=CharField(),
            )
        )
        .values("source_type")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    top_products_rows = (
        OrderItem.objects.filter(order__in=current_orders)
        .values("product_name")
        .annotate(total_quantity=Sum("quantity"), total_revenue=Sum(line_total_expr))
        .order_by("-total_revenue")[:10]
    )
    top_products = [
        {
            "name": row["product_name"] or "Unnamed product",
            "quantity": row["total_quantity"] or 0,
            "revenue": row["total_revenue"] or Decimal("0"),
        }
        for row in top_products_rows
    ]

    top_categories_rows = (
        OrderItem.objects.filter(
            order__in=current_orders,
            product__primary_category__isnull=False,
        )
        .values("product__primary_category__name")
        .annotate(total_quantity=Sum("quantity"), total_revenue=Sum(line_total_expr))
        .order_by("-total_revenue")[:10]
    )
    top_categories = [
        {
            "name": row["product__primary_category__name"] or "Uncategorized",
            "quantity": row["total_quantity"] or 0,
            "revenue": row["total_revenue"] or Decimal("0"),
        }
        for row in top_categories_rows
    ]

    start_day = start_date.date()
    end_day = (end_date - timedelta(seconds=1)).date()
    day_count = max((end_day - start_day).days + 1, 1)
    timeline = {}
    for offset in range(day_count):
        day = start_day + timedelta(days=offset)
        timeline[day] = {
            "date": day.isoformat(),
            "revenue": Decimal("0"),
            "orders": 0,
            "page_views": 0,
            "unique_visitors": 0,
            "product_views": 0,
            "new_users": 0,
            "conversion_rate": 0.0,
        }

    order_series = (
        current_orders.annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(revenue=Sum("total"), orders=Count("id"))
        .order_by("day")
    )
    for row in order_series:
        day = row["day"]
        if day in timeline:
            timeline[day]["revenue"] = row["revenue"] or Decimal("0")
            timeline[day]["orders"] = row["orders"] or 0

    page_series = (
        current_page_views_qs.annotate(day=TruncDate("created_at"), visitor_key=visitor_key_expr)
        .values("day")
        .annotate(page_views=Count("id"), unique_visitors=Count("visitor_key", distinct=True))
        .order_by("day")
    )
    for row in page_series:
        day = row["day"]
        if day in timeline:
            timeline[day]["page_views"] = row["page_views"] or 0
            timeline[day]["unique_visitors"] = row["unique_visitors"] or 0

    product_view_series = (
        ProductView.objects.filter(created_at__gte=start_date, created_at__lt=end_date)
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )
    for row in product_view_series:
        day = row["day"]
        if day in timeline:
            timeline[day]["product_views"] = row["count"] or 0

    new_user_series = (
        User.objects.filter(
            created_at__gte=start_date,
            created_at__lt=end_date,
            is_deleted=False,
        )
        .annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )
    for row in new_user_series:
        day = row["day"]
        if day in timeline:
            timeline[day]["new_users"] = row["count"] or 0

    timeseries = []
    for day in sorted(timeline.keys()):
        entry = timeline[day]
        if entry["unique_visitors"] > 0:
            entry["conversion_rate"] = round(
                (entry["orders"] / entry["unique_visitors"]) * 100,
                2,
            )
        timeseries.append(entry)

    revenue_growth = _growth(float(total_revenue), float(prev_revenue))
    orders_growth = _growth(total_orders, prev_total_orders)
    users_growth = _growth(new_users, prev_new_users)
    views_growth = _growth(page_views, prev_page_views)
    visitors_growth = _growth(unique_visitors, prev_unique_visitors)
    product_views_growth = _growth(product_views, prev_product_views)

    return {
        "total_revenue": total_revenue,
        "prev_revenue": prev_revenue,
        "revenue_growth": revenue_growth,
        "avg_order_value": avg_order_value,
        "total_orders": total_orders,
        "prev_orders": prev_total_orders,
        "orders_growth": orders_growth,
        "order_status_breakdown": order_status_breakdown,
        "total_users": total_users,
        "new_users": new_users,
        "prev_new_users": prev_new_users,
        "users_growth": users_growth,
        "active_users": active_users,
        "total_products": total_products,
        "active_products": active_products,
        "low_stock_products": low_stock_products,
        "out_of_stock": out_of_stock,
        "page_views": page_views,
        "prev_page_views": prev_page_views,
        "views_growth": views_growth,
        "product_views": product_views,
        "products_added_to_cart": products_added_to_cart,
        "checkout_starts": checkout_starts,
        "checkout_completions": checkout_completions,
        "cart_abandonment_rate": round(cart_abandonment_rate, 2),
        "unique_visitors": unique_visitors,
        "conversion_rate": round(conversion_rate, 2),
        "active_visitors_5m": active_visitors_5m,
        "visitors_growth": visitors_growth,
        "product_views_growth": product_views_growth,
        "top_products": top_products,
        "top_categories": top_categories,
        "device_breakdown": device_breakdown,
        "traffic_sources": traffic_sources,
        "timeseries": timeseries,
        "new_users_over_time": [
            {"date": point["date"], "count": point["new_users"]}
            for point in timeseries
        ],
        "product_views_over_time": [
            {"date": point["date"], "count": point["product_views"]}
            for point in timeseries
        ],
        "kpis": {
            "revenue": total_revenue,
            "orders": total_orders,
            "visitors": unique_visitors,
            "conversion_rate": round(conversion_rate, 2),
            "avg_order_value": avg_order_value,
            "active_visitors_5m": active_visitors_5m,
        },
        "growth": {
            "revenue": revenue_growth,
            "orders": orders_growth,
            "users": users_growth,
            "views": views_growth,
            "visitors": visitors_growth,
            "product_views": product_views_growth,
        },
    }


@staff_member_required
@require_GET
def revenue_chart_api(request):
    """API endpoint for revenue chart data."""
    start_date, end_date, _ = get_date_range(
        period=request.GET.get("period", "30d"),
        start_param=request.GET.get("start"),
        end_param=request.GET.get("end"),
    )
    summary = get_summary_stats(start_date, end_date)
    series = summary.get("timeseries", [])
    return JsonResponse(
        {
            "labels": [point["date"] for point in series],
            "revenue": [decimal_to_float(point["revenue"]) for point in series],
            "orders": [point["orders"] for point in series],
            "visitors": [point["unique_visitors"] for point in series],
            "conversion_rate": [point["conversion_rate"] for point in series],
        }
    )


@staff_member_required
@require_GET
@cache_page(60 * 5)
def top_products_api(request):
    """API endpoint for top selling products."""
    limit = _safe_int(request.GET.get("limit", 10), default=10, min_value=1, max_value=50)
    start_date, end_date, _ = get_date_range(
        period=request.GET.get("period", "30d"),
        start_param=request.GET.get("start"),
        end_param=request.GET.get("end"),
    )
    summary = get_summary_stats(start_date, end_date)
    products = summary.get("top_products", [])[:limit]
    return JsonResponse({"products": _json_safe(products)})


@staff_member_required
@require_GET
@cache_page(60 * 5)
def recent_orders_api(request):
    """API endpoint for recent orders."""
    from apps.orders.models import Order

    limit = _safe_int(request.GET.get("limit", 10), default=10, min_value=1, max_value=100)
    orders = Order.objects.filter(is_deleted=False).select_related("user").order_by("-created_at")[:limit]
    data = {
        "orders": [
            {
                "id": str(order.id),
                "order_number": order.order_number,
                "customer": order.user.email if order.user else order.email,
                "total": float(order.total),
                "status": order.status,
                "created_at": order.created_at.isoformat(),
            }
            for order in orders
        ]
    }
    return JsonResponse(data)


@staff_member_required
@require_GET
@cache_page(60 * 5)
def low_stock_api(request):
    """API endpoint for low stock products."""
    from apps.catalog.models import Product

    limit = _safe_int(request.GET.get("limit", 10), default=10, min_value=1, max_value=100)
    products = Product.objects.filter(
        is_deleted=False,
        is_active=True,
        stock_quantity__lte=F("low_stock_threshold"),
    ).order_by("stock_quantity")[:limit]
    data = {
        "products": [
            {
                "id": str(product.id),
                "name": product.name,
                "sku": product.sku,
                "stock": product.stock_quantity,
                "threshold": product.low_stock_threshold,
            }
            for product in products
        ]
    }
    return JsonResponse(data)


@staff_member_required
@require_GET
@cache_page(60 * 5)
def user_activity_api(request):
    """API endpoint for user activity chart."""
    from apps.accounts.models import User

    start_date, end_date, _ = get_date_range(
        period=request.GET.get("period", "30d"),
        start_param=request.GET.get("start"),
        end_param=request.GET.get("end"),
    )
    new_users = (
        User.objects.filter(
            created_at__gte=start_date,
            created_at__lt=end_date,
            is_deleted=False,
        )
        .annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )
    data = {
        "labels": [user["date"].strftime("%Y-%m-%d") for user in new_users],
        "registrations": [user["count"] for user in new_users],
    }
    return JsonResponse(data)


@staff_member_required
@require_GET
def system_health_api(request):
    """API endpoint for system health status."""
    health = check_system_health()
    overall = "healthy" if all(service.get("status") == "healthy" for service in health.values()) else "degraded"
    return JsonResponse(
        {
            "status": overall,
            "services": health,
            "timestamp": timezone.now().isoformat(),
        }
    )


def check_system_health():
    """Check core services health for admin utilities and APIs."""
    from django.core.cache import cache
    from django.core.files.storage import default_storage
    from django.db import connection

    health = {}

    try:
        start = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        health["database"] = {"status": "healthy", "latency_ms": elapsed_ms}
    except Exception as exc:
        health["database"] = {"status": "down", "error": str(exc)}

    try:
        start = time.perf_counter()
        cache.set("health_check", "ok", 10)
        cache_ok = cache.get("health_check") == "ok"
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        health["cache"] = {
            "status": "healthy" if cache_ok else "down",
            "latency_ms": elapsed_ms,
        }
    except Exception as exc:
        health["cache"] = {"status": "down", "error": str(exc)}

    try:
        start = time.perf_counter()
        _ = default_storage.exists("") or True
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        health["storage"] = {"status": "healthy", "latency_ms": elapsed_ms}
    except Exception as exc:
        health["storage"] = {"status": "down", "error": str(exc)}

    return health
