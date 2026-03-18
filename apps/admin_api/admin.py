from django.contrib import admin

from .models import AdminAuditLog


@admin.register(AdminAuditLog)
class AdminAuditLogAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "actor",
        "action",
        "method",
        "path",
        "status_code",
    )
    list_filter = ("action", "method", "status_code")
    search_fields = ("path", "resource_type", "resource_id", "actor__email")
    ordering = ("-created_at",)
    readonly_fields = (
        "actor",
        "action",
        "resource_type",
        "resource_id",
        "path",
        "method",
        "status_code",
        "ip_address",
        "user_agent",
        "metadata",
        "created_at",
    )
