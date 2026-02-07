from django.contrib import admin
from django.utils import timezone
from django.urls import reverse
from django.utils.html import format_html
from .models import Plan, Subscription
from django.contrib.admin import SimpleListFilter

# Import payments models lazily
from apps.payments.models import RecurringCharge
from .services import SubscriptionService


class RecurringChargeInline(admin.TabularInline):
    model = RecurringCharge
    extra = 0
    readonly_fields = ("amount", "currency", "status", "attempt_at", "processed_at", "stripe_subscription_id")
    can_delete = False


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ("name", "interval", "price_amount", "currency", "active")
    list_filter = ("interval", "active")
    search_fields = ("name", "stripe_price_id")
    readonly_fields = ("id", "created_at", "updated_at")


class IsDeletedFilter(SimpleListFilter):
    title = 'deleted'
    parameter_name = 'is_deleted'

    def lookups(self, request, model_admin):
        return (
            ('deleted', 'Deleted'),
            ('active', 'Not deleted'),
        )

    def queryset(self, request, queryset):
        if self.value() == 'deleted':
            return queryset.filter(is_deleted=True)
        if self.value() == 'active':
            return queryset.filter(is_deleted=False)
        return queryset


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user_link",
        "plan",
        "status",
        "stripe_subscription_id",
        "current_period_end",
        "next_billing_at",
        "is_deleted",
    )
    list_filter = ("status", "plan", IsDeletedFilter)
    search_fields = ("user__email", "stripe_subscription_id", "plan__name")
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = (RecurringChargeInline,)
    actions = ("action_cancel_subscriptions", "action_resume_subscriptions", "action_sync_with_stripe")

    def user_link(self, obj):
        try:
            url = reverse("admin:auth_user_change", args=(obj.user.id,))
            return format_html('<a href="{}">{}</a>', url, obj.user.email or obj.user.get_username())
        except Exception:
            return str(obj.user)

    user_link.short_description = "User"

    def action_cancel_subscriptions(self, request, queryset):
        for sub in queryset:
            SubscriptionService.cancel(sub, at=timezone.now(), cancel_at_period_end=False)
        self.message_user(request, "Selected subscriptions canceled.")

    action_cancel_subscriptions.short_description = "Cancel selected subscriptions"

    def action_resume_subscriptions(self, request, queryset):
        for sub in queryset:
            try:
                SubscriptionService.resume(sub)
            except Exception:
                pass
        self.message_user(request, "Resumed selected subscriptions where possible.")

    action_resume_subscriptions.short_description = "Resume selected subscriptions"

    def action_sync_with_stripe(self, request, queryset):
        for sub in queryset:
            try:
                SubscriptionService.sync_with_stripe(sub)
            except Exception:
                pass
        self.message_user(request, "Synced selected subscriptions with Stripe.")

    action_sync_with_stripe.short_description = "Sync selected subscriptions with Stripe"