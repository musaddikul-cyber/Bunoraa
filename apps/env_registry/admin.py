from django.contrib import admin
from django.conf import settings

from core.admin_mixins import EnhancedModelAdmin

from .forms import EnvValueForm
from .models import EnvCategory, EnvValue, EnvVariable
from .runtime import apply_runtime_overrides
from .schema_loader import sync_from_schema


@admin.register(EnvCategory)
class EnvCategoryAdmin(EnhancedModelAdmin):
    list_display = ("name", "slug", "order", "is_active")
    list_filter = ("is_active",)
    search_fields = ("name", "slug")
    ordering = ("order", "name")


@admin.register(EnvVariable)
class EnvVariableAdmin(EnhancedModelAdmin):
    list_display = (
        "key",
        "category",
        "is_secret",
        "required",
        "restart_required",
        "runtime_apply",
        "is_active",
    )
    list_filter = ("category", "is_secret", "required", "restart_required", "runtime_apply", "is_active")
    search_fields = ("key", "description")
    ordering = ("key",)
    actions = ["sync_from_schema_action", "apply_runtime_action"]

    def sync_from_schema_action(self, request, queryset):
        sync_from_schema(None, env=settings.ENVIRONMENT, force=False, prune=False)
        self.message_user(request, "Env registry synced from schema.")

    sync_from_schema_action.short_description = "Sync env registry from schema"

    def apply_runtime_action(self, request, queryset):
        updated = apply_runtime_overrides(settings.ENVIRONMENT)
        self.message_user(request, f"Applied {updated} runtime overrides.")

    apply_runtime_action.short_description = "Apply runtime overrides"


@admin.register(EnvValue)
class EnvValueAdmin(EnhancedModelAdmin):
    form = EnvValueForm
    list_display = ("variable", "environment", "masked_value", "updated_at", "updated_by")
    list_filter = ("environment", "variable__category", "variable__is_secret", "variable__is_active")
    search_fields = ("variable__key",)
    ordering = ("variable__key", "environment")
    actions = ["apply_runtime_action"]

    def masked_value(self, obj):
        return obj.masked_value()

    masked_value.short_description = "Value"

    def save_model(self, request, obj, form, change):
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)

    def apply_runtime_action(self, request, queryset):
        updated = apply_runtime_overrides(settings.ENVIRONMENT)
        self.message_user(request, f"Applied {updated} runtime overrides.")

    apply_runtime_action.short_description = "Apply runtime overrides"
