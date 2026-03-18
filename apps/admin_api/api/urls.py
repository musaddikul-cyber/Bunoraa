from django.urls import include, path

from core.api.routers import DefaultRouter

from .views import (
    AdminAuditLogViewSet,
    AdminDashboardView,
    AdminGroupViewSet,
    AdminHealthView,
    AdminOrderViewSet,
    AdminPermissionViewSet,
    AdminSocialLoginView,
    AdminUserViewSet,
)

router = DefaultRouter()
router.register(r"audit-logs", AdminAuditLogViewSet, basename="admin-audit-logs")
router.register(r"users", AdminUserViewSet, basename="admin-users")
router.register(r"groups", AdminGroupViewSet, basename="admin-groups")
router.register(r"permissions", AdminPermissionViewSet, basename="admin-permissions")
router.register(r"orders", AdminOrderViewSet, basename="admin-orders")

urlpatterns = [
    path("auth/social/", AdminSocialLoginView.as_view(), name="admin-social-login"),
    path("dashboard/", AdminDashboardView.as_view(), name="admin-dashboard"),
    path("health/", AdminHealthView.as_view(), name="admin-health"),
    path("", include(router.urls)),
]
