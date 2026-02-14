"""
Analytics services
"""
from datetime import datetime, timedelta

from django.db.models import (
    Avg,
    Case,
    CharField,
    Count,
    DecimalField,
    ExpressionWrapper,
    F,
    Max,
    Q,
    Sum,
    Value,
    When,
)
from django.db.models.functions import Cast, Concat, TruncDate
from django.utils import timezone
from django.contrib.auth import get_user_model
from apps.catalog.models import Product

from .models import (
    PageView,
    ProductView,
    SearchQuery,
    CartEvent,
    DailyStat,
    ProductStat,
    CategoryStat,
)


class AnalyticsService:
    """Service for tracking and querying analytics."""

    @staticmethod
    def sync_product_views_from_impressions(days=90):
        """
        One-time backfill for ProductView rows from legacy ProductImpression rows.
        Runs only while ProductView is empty to avoid duplicate analytics.
        """
        if ProductView.objects.exists():
            return 0

        try:
            from apps.catalog.models import ProductImpression
        except Exception:
            return 0

        days = max(1, int(days or 90))
        start = timezone.now() - timedelta(days=days)

        impressions = ProductImpression.objects.filter(
            occurred_at__gte=start
        ).values(
            'product_id', 'user_id', 'session_key', 'occurred_at'
        )

        to_create = []
        for row in impressions.iterator():
            raw_session = row.get('session_key') or ''
            session_key = raw_session[:40] if raw_session else None
            to_create.append(
                ProductView(
                    product_id=row.get('product_id'),
                    user_id=row.get('user_id'),
                    session_key=session_key,
                    source='catalog_impression_backfill',
                    referrer='',
                    created_at=row.get('occurred_at'),
                )
            )

        if not to_create:
            return 0

        ProductView.objects.bulk_create(to_create, batch_size=1000)
        return len(to_create)

    @staticmethod
    def sync_cart_events_from_checkout_events(days=90):
        """
        Backfill checkout_start/checkout_complete analytics from commerce checkout
        events when CartEvent data is missing.
        """
        days = max(1, int(days or 90))
        start = timezone.now() - timedelta(days=days)

        has_checkout_events = CartEvent.objects.filter(
            event_type__in=[CartEvent.EVENT_CHECKOUT_START, CartEvent.EVENT_CHECKOUT_COMPLETE],
            created_at__gte=start,
        ).exists()
        if has_checkout_events:
            return 0

        try:
            from apps.commerce.models import CheckoutEvent
        except Exception:
            return 0

        source_events = (
            CheckoutEvent.objects.filter(
                created_at__gte=start,
                event_type__in=[
                    CheckoutEvent.EVENT_STARTED,
                    CheckoutEvent.EVENT_ORDER_CREATED,
                    CheckoutEvent.EVENT_PAYMENT_COMPLETED,
                ],
            )
            .select_related('checkout_session')
            .order_by('created_at')
        )

        mapping = {
            CheckoutEvent.EVENT_STARTED: CartEvent.EVENT_CHECKOUT_START,
            CheckoutEvent.EVENT_ORDER_CREATED: CartEvent.EVENT_CHECKOUT_COMPLETE,
            CheckoutEvent.EVENT_PAYMENT_COMPLETED: CartEvent.EVENT_CHECKOUT_COMPLETE,
        }

        to_create = []
        for event in source_events.iterator():
            checkout_session = getattr(event, 'checkout_session', None)
            if not checkout_session:
                continue
            mapped_type = mapping.get(event.event_type)
            if not mapped_type:
                continue

            to_create.append(
                CartEvent(
                    event_type=mapped_type,
                    user_id=checkout_session.user_id,
                    session_key=checkout_session.session_key,
                    cart_value=checkout_session.total or 0,
                    created_at=event.created_at,
                )
            )

        if not to_create:
            return 0

        CartEvent.objects.bulk_create(to_create, batch_size=1000)
        return len(to_create)
    
    @staticmethod
    def track_page_view(
        request,
        time_on_page=0,
        page_path=None,
        query_string=None,
        referrer=None,
    ):
        """Track a page view."""
        user = request.user if request.user.is_authenticated else None
        session_key = request.session.session_key

        path_value = str(page_path or request.path or "/")[:500]
        query_value = str(
            request.META.get('QUERY_STRING', '')
            if query_string is None
            else query_string
        )[:1000]
        referrer_value = (
            request.META.get('HTTP_REFERER')
            if referrer is None
            else referrer
        )
        if referrer_value:
            referrer_value = str(referrer_value)[:1000]
        
        # Get device info from user agent
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        device_info = AnalyticsService._parse_user_agent(user_agent)
        
        # Get IP address
        ip = AnalyticsService._get_client_ip(request)
        
        PageView.objects.create(
            user=user,
            session_key=session_key,
            path=path_value,
            query_string=query_value,
            referrer=referrer_value,
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
    def _iter_dates(start_date, end_date):
        current = start_date
        while current <= end_date:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _ensure_daily_stats(days=30):
        """
        Ensure DailyStat/ProductStat/CategoryStat rows exist for the requested range.
        This keeps admin analytics usable even if scheduled jobs were skipped.
        """
        days = max(1, int(days or 30))
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)

        existing_dates = set(
            DailyStat.objects.filter(date__gte=start_date, date__lte=end_date).values_list("date", flat=True)
        )
        missing_dates = [d for d in DashboardService._iter_dates(start_date, end_date) if d not in existing_dates]

        if missing_dates:
            # Guardrail: avoid expensive full-history rebuild on request path.
            if len(missing_dates) > 400:
                missing_dates = missing_dates[-400:]

            for day in missing_dates:
                ReportService.generate_daily_stats(day)

        # Keep today's aggregate fresh when new raw events/orders arrived.
        today_stat = DailyStat.objects.filter(date=end_date).first()
        if not today_stat:
            return

        from apps.orders.models import Order

        latest_sources = [
            PageView.objects.aggregate(last=Max("created_at")).get("last"),
            ProductView.objects.aggregate(last=Max("created_at")).get("last"),
            SearchQuery.objects.aggregate(last=Max("created_at")).get("last"),
            CartEvent.objects.aggregate(last=Max("created_at")).get("last"),
            Order.objects.filter(is_deleted=False).aggregate(last=Max("created_at")).get("last"),
        ]
        latest_source_ts = max((ts for ts in latest_sources if ts is not None), default=None)
        if latest_source_ts and latest_source_ts > today_stat.updated_at:
            ReportService.generate_daily_stats(end_date)
    
    @staticmethod
    def get_overview_stats(days=30):
        """Get overview statistics for dashboard."""
        DashboardService._ensure_daily_stats(days)
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
        DashboardService._ensure_daily_stats(days)
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
        DashboardService._ensure_daily_stats(days)
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

    @staticmethod
    def generate_daily_stats(date=None):
        """Generate daily statistics for a specific date and related aggregates."""
        from apps.orders.models import Order, OrderItem

        if date is None:
            date = timezone.now().date() - timedelta(days=1)

        start_of_day = timezone.make_aware(datetime.combine(date, datetime.min.time()))
        end_of_day = start_of_day + timedelta(days=1)

        visitor_key_expr = ReportService._visitor_key_expression()

        page_views_qs = PageView.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day,
        )
        page_views = page_views_qs.count()
        unique_visitors = page_views_qs.annotate(
            visitor_key=visitor_key_expr
        ).values("visitor_key").distinct().count()

        # Returning visitors = visitors seen before this date.
        day_visitors = page_views_qs.annotate(visitor_key=visitor_key_expr).values(
            "visitor_key"
        ).distinct()
        historical_visitor_keys = PageView.objects.filter(
            created_at__lt=start_of_day
        ).annotate(
            visitor_key=visitor_key_expr
        ).values(
            "visitor_key"
        )
        returning_visitors = day_visitors.filter(
            visitor_key__in=historical_visitor_keys
        ).count()
        new_visitors = max(unique_visitors - returning_visitors, 0)

        product_views_rows = (
            ProductView.objects.filter(
                created_at__gte=start_of_day,
                created_at__lt=end_of_day,
            )
            .values("product")
            .annotate(views=Count("id"))
        )

        # Backward compatibility for deployments that recorded ProductImpression
        # before dedicated ProductView tracking was added.
        impression_rows = []
        try:
            from apps.catalog.models import ProductImpression

            impression_rows = list(
                ProductImpression.objects.filter(
                    occurred_at__gte=start_of_day,
                    occurred_at__lt=end_of_day,
                )
                .values("product")
                .annotate(views=Count("id"))
            )
        except Exception:
            impression_rows = []

        views_map = {row["product"]: row["views"] for row in product_views_rows}
        for row in impression_rows:
            product_id = row.get("product")
            if not product_id:
                continue
            # Use impressions as fallback only when ProductView rows are missing.
            if product_id not in views_map:
                views_map[product_id] = row.get("views", 0) or 0

        product_views = sum(views_map.values())

        cart_events_qs = CartEvent.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day,
        )
        cart_adds = cart_events_qs.filter(event_type=CartEvent.EVENT_ADD).count()
        checkout_starts = cart_events_qs.filter(
            event_type=CartEvent.EVENT_CHECKOUT_START
        ).count()
        checkout_completions = cart_events_qs.filter(
            event_type=CartEvent.EVENT_CHECKOUT_COMPLETE
        ).count()

        excluded_statuses = [Order.STATUS_CANCELLED, Order.STATUS_REFUNDED]
        orders_qs = Order.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day,
            is_deleted=False,
        ).exclude(status__in=excluded_statuses)
        orders_agg = orders_qs.aggregate(count=Count("id"), revenue=Sum("total"))
        orders_count = orders_agg["count"] or 0
        orders_revenue = orders_agg["revenue"] or 0
        avg_order_value = orders_revenue / orders_count if orders_count > 0 else 0

        conversion_rate = (orders_count / unique_visitors) * 100 if unique_visitors > 0 else 0
        cart_abandonment_rate = (
            ((checkout_starts - checkout_completions) / checkout_starts) * 100
            if checkout_starts > 0
            else 0
        )

        User = get_user_model()
        new_registrations = User.objects.filter(
            created_at__gte=start_of_day,
            created_at__lt=end_of_day,
            is_deleted=False,
        ).count()

        stat, _ = DailyStat.objects.update_or_create(
            date=date,
            defaults={
                "page_views": page_views,
                "unique_visitors": unique_visitors,
                "new_visitors": new_visitors,
                "returning_visitors": returning_visitors,
                "product_views": product_views,
                "products_added_to_cart": cart_adds,
                "orders_count": orders_count,
                "orders_revenue": orders_revenue,
                "average_order_value": avg_order_value,
                "checkout_starts": checkout_starts,
                "checkout_completions": checkout_completions,
                "conversion_rate": round(conversion_rate, 2),
                "cart_abandonment_rate": round(cart_abandonment_rate, 2),
                "new_registrations": new_registrations,
            },
        )

        line_total_expr = ExpressionWrapper(
            F("quantity") * F("unit_price"),
            output_field=DecimalField(max_digits=14, decimal_places=2),
        )

        # Product-level aggregates.
        product_rows = (
            OrderItem.objects.filter(order__in=orders_qs, product__isnull=False)
            .values("product")
            .annotate(
                orders_count=Count("order", distinct=True),
                revenue=Sum(line_total_expr),
            )
        )
        cart_add_rows = (
            cart_events_qs.filter(
                event_type=CartEvent.EVENT_ADD,
                product__isnull=False,
            )
            .values("product")
            .annotate(add_to_cart_count=Count("id"))
        )

        add_map = {row["product"]: row["add_to_cart_count"] for row in cart_add_rows}
        revenue_map = {
            row["product"]: {
                "orders_count": row["orders_count"] or 0,
                "revenue": row["revenue"] or 0,
            }
            for row in product_rows
        }

        product_ids = set(views_map.keys()) | set(add_map.keys()) | set(revenue_map.keys())
        for product_id in product_ids:
            views = views_map.get(product_id, 0)
            add_to_cart_count = add_map.get(product_id, 0)
            order_info = revenue_map.get(product_id, {})
            product_orders_count = order_info.get("orders_count", 0)
            product_revenue = order_info.get("revenue", 0)
            conversion = (product_orders_count / views) * 100 if views > 0 else 0
            ProductStat.objects.update_or_create(
                product_id=product_id,
                date=date,
                defaults={
                    "views": views,
                    "add_to_cart_count": add_to_cart_count,
                    "orders_count": product_orders_count,
                    "revenue": product_revenue,
                    "conversion_rate": round(conversion, 2),
                },
            )

        # Category-level aggregates (prefer primary_category, fallback to M2M categories).
        category_rows_primary = (
            OrderItem.objects.filter(
                order__in=orders_qs,
                product__primary_category__isnull=False,
            )
            .values("product__primary_category")
            .annotate(
                orders_count=Count("order", distinct=True),
                revenue=Sum(line_total_expr),
            )
        )
        category_rows_m2m = (
            OrderItem.objects.filter(
                order__in=orders_qs,
                product__primary_category__isnull=True,
                product__categories__isnull=False,
            )
            .values("product__categories")
            .annotate(
                orders_count=Count("order", distinct=True),
                revenue=Sum(line_total_expr),
            )
        )
        category_page_views_rows = (
            PageView.objects.filter(
                created_at__gte=start_of_day,
                created_at__lt=end_of_day,
                path__startswith="/categories/",
            )
            .values("path")
            .annotate(views=Count("id"))
        )

        # Map category page views by slug fragment for best-effort attribution.
        page_view_by_slug = {}
        for row in category_page_views_rows:
            path = row["path"] or ""
            slug = path.rstrip("/").split("/")[-1] if path else ""
            if slug:
                page_view_by_slug[slug] = page_view_by_slug.get(slug, 0) + (row["views"] or 0)

        category_order_map = {
            row["product__primary_category"]: {
                "orders_count": row["orders_count"] or 0,
                "revenue": row["revenue"] or 0,
            }
            for row in category_rows_primary
        }
        for row in category_rows_m2m:
            category_id = row.get("product__categories")
            if not category_id:
                continue
            slot = category_order_map.setdefault(category_id, {"orders_count": 0, "revenue": 0})
            slot["orders_count"] += row.get("orders_count") or 0
            slot["revenue"] += row.get("revenue") or 0

        category_product_views_map = {}
        if views_map:
            from apps.catalog.models import Product

            products = Product.objects.filter(id__in=list(views_map.keys())).prefetch_related("categories")
            for product in products:
                view_count = views_map.get(product.id, 0) or 0
                if view_count <= 0:
                    continue
                if product.primary_category_id:
                    category_product_views_map[product.primary_category_id] = (
                        category_product_views_map.get(product.primary_category_id, 0) + view_count
                    )
                    continue

                for category_id in product.categories.values_list("id", flat=True):
                    category_product_views_map[category_id] = (
                        category_product_views_map.get(category_id, 0) + view_count
                    )

        category_ids = set(category_order_map.keys()) | set(category_product_views_map.keys())

        from apps.catalog.models import Category
        slug_map = dict(Category.objects.filter(id__in=category_ids).values_list("id", "slug"))

        for category_id in category_ids:
            order_info = category_order_map.get(category_id, {})
            product_view_count = category_product_views_map.get(category_id, 0)
            slug = slug_map.get(category_id)
            category_page_views = page_view_by_slug.get(slug, 0) if slug else 0
            CategoryStat.objects.update_or_create(
                category_id=category_id,
                date=date,
                defaults={
                    "views": category_page_views,
                    "product_views": product_view_count,
                    "orders_count": order_info.get("orders_count", 0),
                    "revenue": order_info.get("revenue", 0),
                },
            )

        return stat
