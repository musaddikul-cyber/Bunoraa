from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.db.models import Sum
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken
from rest_framework_simplejwt.token_blacklist.models import OutstandingToken

from social_core.exceptions import AuthException
from social_django.utils import load_backend, load_strategy

from apps.accounts.services import MfaService, AuthSessionService
from apps.admin_api.models import AdminAuditLog
from apps.admin_api.permissions import IsStaffWithMfa
from core.views import check_cache, check_database, check_redis, check_storage

from apps.orders.api.views import OrderAdminViewSet

from .serializers import (
    AdminAuditLogSerializer,
    AdminGroupSerializer,
    AdminPermissionSerializer,
    AdminUserSerializer,
)

User = get_user_model()


class AdminHealthView(APIView):
    permission_classes = [IsStaffWithMfa]

    def get(self, request):
        checks = {
            "database": check_database(),
            "cache": check_cache(),
            "redis": check_redis(),
            "storage": check_storage(),
        }
        all_ok = all(c.get("status") == "ok" for c in checks.values())
        return Response(
            {
                "success": True,
                "message": "Health check complete.",
                "data": {
                    "status": "ok" if all_ok else "degraded",
                    "service": "bunoraa-admin",
                    "version": getattr(settings, "VERSION", "1.0.0"),
                    "environment": "production" if not settings.DEBUG else "development",
                    "checks": checks,
                    "timestamp": timezone.now().isoformat(),
                },
                "meta": None,
            }
        )


class AdminDashboardView(APIView):
    permission_classes = [IsStaffWithMfa]

    def get(self, request):
        from apps.catalog.models import Product
        from apps.orders.models import Order

        now = timezone.now()
        window_start = now - timedelta(days=30)

        total_users = User.objects.count()
        total_products = Product.objects.filter(is_deleted=False).count()
        total_orders = Order.objects.filter(is_deleted=False).count()
        pending_orders = Order.objects.filter(
            is_deleted=False,
            status=Order.STATUS_PENDING,
        ).count()
        revenue_30d = (
            Order.objects.filter(
                is_deleted=False,
                payment_status=Order.PAYMENT_SUCCEEDED,
                created_at__gte=window_start,
            )
            .aggregate(total=Sum("total"))
            .get("total")
            or 0
        )

        return Response(
            {
                "success": True,
                "message": "Dashboard metrics loaded.",
                "data": {
                    "generated_at": now.isoformat(),
                    "window_days": 30,
                    "totals": {
                        "users": total_users,
                        "products": total_products,
                        "orders": total_orders,
                        "orders_pending": pending_orders,
                        "revenue_30d": str(revenue_30d),
                    },
                },
                "meta": None,
            }
        )


class AdminSocialLoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        provider = request.data.get("provider")
        access_token = request.data.get("access_token")

        if not provider or not access_token:
            return Response(
                {
                    "success": False,
                    "message": "provider and access_token are required.",
                    "data": None,
                    "meta": None,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            strategy = load_strategy(request)
            backend = load_backend(strategy, provider, redirect_uri=None)
            user = backend.do_auth(access_token)
        except AuthException as exc:
            return Response(
                {
                    "success": False,
                    "message": f"Social authentication failed: {exc}",
                    "data": None,
                    "meta": None,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            return Response(
                {
                    "success": False,
                    "message": f"Social authentication error: {exc}",
                    "data": None,
                    "meta": None,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not user or not getattr(user, "is_active", False):
            return Response(
                {
                    "success": False,
                    "message": "User not active or not found.",
                    "data": None,
                    "meta": None,
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        if not getattr(user, "is_staff", False):
            return Response(
                {
                    "success": False,
                    "message": "Admin access required.",
                    "data": None,
                    "meta": None,
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        if getattr(settings, "ADMIN_MFA_REQUIRED", True):
            if not MfaService.is_mfa_enabled(user):
                return Response(
                    {
                        "success": False,
                        "message": "MFA is required for admin accounts.",
                        "data": None,
                        "meta": None,
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

            return Response(
                {
                    "success": True,
                    "message": "MFA required.",
                    "data": {
                        "mfa_required": True,
                        "mfa_token": MfaService.create_mfa_token(user),
                        "methods": list(MfaService.available_methods(user)),
                    },
                    "meta": None,
                },
                status=status.HTTP_200_OK,
            )

        refresh = RefreshToken.for_user(user)
        access = refresh.access_token
        expires_at = datetime.fromtimestamp(refresh["exp"], tz=dt_timezone.utc)
        OutstandingToken.objects.get_or_create(
            jti=str(refresh["jti"]),
            user=user,
            token=str(refresh),
            expires_at=expires_at,
        )
        AuthSessionService.create_session(
            user,
            request,
            str(access["jti"]),
            str(refresh["jti"]),
        )

        return Response(
            {
                "success": True,
                "message": "Admin login successful.",
                "data": {
                    "access": str(access),
                    "refresh": str(refresh),
                    "mfa_required": False,
                },
                "meta": None,
            }
        )


class AdminAuditLogViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = [IsStaffWithMfa]
    serializer_class = AdminAuditLogSerializer
    queryset = AdminAuditLog.objects.select_related("actor")
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["action", "path", "actor__email", "resource_type", "resource_id"]
    ordering_fields = ["created_at", "status_code"]
    ordering = ["-created_at"]


class AdminUserViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = [IsStaffWithMfa]
    serializer_class = AdminUserSerializer
    queryset = User.objects.all().order_by("-date_joined")
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["email", "first_name", "last_name"]
    ordering_fields = ["date_joined", "last_login", "email"]
    ordering = ["-date_joined"]


class AdminGroupViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = [IsStaffWithMfa]
    serializer_class = AdminGroupSerializer
    queryset = Group.objects.prefetch_related("permissions").all().order_by("name")
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["name"]
    ordering_fields = ["name"]
    ordering = ["name"]


class AdminPermissionViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = [IsStaffWithMfa]
    serializer_class = AdminPermissionSerializer
    queryset = Permission.objects.select_related("content_type").all().order_by(
        "content_type__app_label",
        "codename",
    )
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["codename", "name", "content_type__app_label"]
    ordering_fields = ["codename", "name"]
    ordering = ["content_type__app_label", "codename"]


class AdminOrderViewSet(OrderAdminViewSet):
    permission_classes = [IsStaffWithMfa]
