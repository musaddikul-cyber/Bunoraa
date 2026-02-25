"""
Reviews admin configuration.

Reviews in production are stored in catalog review models. The admin in this app
registers those canonical models so customer reviews are visible in one place.
"""
from __future__ import annotations

from django.contrib import admin
from django.utils import timezone
from django.utils.html import format_html

from apps.reviews.models import Review, ReviewImage, ReviewReport, ReviewVote
from core.admin_mixins import ImportExportEnhancedModelAdmin


class ReviewImageInline(admin.TabularInline):
    model = ReviewImage
    extra = 0
    readonly_fields = ["preview", "caption", "sort_order", "created_at"]
    fields = ["image", "preview", "caption", "sort_order", "created_at"]

    def preview(self, obj):
        if obj and obj.image:
            return format_html(
                '<img src="{}" style="max-width:120px; max-height:80px; object-fit:cover; border-radius:4px;" />',
                obj.image.url,
            )
        return "-"


class ReviewVoteInline(admin.TabularInline):
    model = ReviewVote
    extra = 0
    can_delete = False
    readonly_fields = ["user", "is_helpful", "created_at", "updated_at"]
    fields = ["user", "is_helpful", "created_at", "updated_at"]


class ReviewReportInline(admin.TabularInline):
    model = ReviewReport
    extra = 0
    can_delete = False
    readonly_fields = [
        "reporter",
        "reason",
        "details",
        "status",
        "resolved_by",
        "resolved_at",
        "created_at",
        "updated_at",
    ]
    fields = [
        "reporter",
        "reason",
        "details",
        "status",
        "resolved_by",
        "resolved_at",
        "created_at",
        "updated_at",
    ]


@admin.register(Review)
class ReviewAdmin(ImportExportEnhancedModelAdmin):
    list_display = [
        "product",
        "user",
        "rating",
        "moderation_status",
        "verified_purchase",
        "is_featured",
        "helpful_votes",
        "report_count",
        "created_at",
    ]
    list_filter = [
        "moderation_status",
        "rating",
        "verified_purchase",
        "is_featured",
        "created_at",
    ]
    list_editable = ["moderation_status", "is_featured"]
    search_fields = ["product__name", "user__email", "title", "body", "pros", "cons"]
    readonly_fields = [
        "id",
        "product",
        "user",
        "created_at",
        "updated_at",
        "published_at",
        "edited_at",
        "helpful_votes",
        "not_helpful_votes",
        "report_count",
        "verified_purchase",
    ]
    actions = [
        "approve_reviews",
        "reject_reviews",
        "mark_featured",
        "clear_featured",
        "refresh_engagement",
    ]
    inlines = [ReviewImageInline, ReviewVoteInline, ReviewReportInline]

    fieldsets = (
        ("Review Info", {"fields": ("id", "product", "user", "verified_purchase")}),
        ("Content", {"fields": ("rating", "title", "body", "pros", "cons", "would_recommend", "is_anonymous")}),
        ("Moderation", {"fields": ("moderation_status", "moderation_notes", "is_featured", "published_at", "edited_at")}),
        ("Engagement", {"fields": ("helpful_votes", "not_helpful_votes", "report_count")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    def save_model(self, request, obj, form, change):
        now = timezone.now()
        if "moderation_status" in form.changed_data:
            obj.moderated_by = request.user
            obj.moderated_at = now
            if obj.moderation_status == Review.MODERATION_APPROVED and not obj.published_at:
                obj.published_at = now
        if obj.moderation_status != Review.MODERATION_APPROVED:
            obj.is_featured = False
        super().save_model(request, obj, form, change)

    def approve_reviews(self, request, queryset):
        updated = 0
        for review in queryset:
            review.mark_approved(moderator=request.user)
            updated += 1
        self.message_user(request, f"{updated} reviews approved.")

    approve_reviews.short_description = "Approve selected reviews"

    def reject_reviews(self, request, queryset):
        updated = 0
        for review in queryset:
            review.mark_rejected(moderator=request.user)
            updated += 1
        self.message_user(request, f"{updated} reviews rejected.")

    reject_reviews.short_description = "Reject selected reviews"

    def mark_featured(self, request, queryset):
        updated = queryset.filter(moderation_status=Review.MODERATION_APPROVED).update(
            is_featured=True,
            updated_at=timezone.now(),
        )
        self.message_user(request, f"{updated} approved reviews marked as featured.")

    mark_featured.short_description = "Mark selected as featured"

    def clear_featured(self, request, queryset):
        updated = queryset.update(is_featured=False, updated_at=timezone.now())
        self.message_user(request, f"{updated} reviews unfeatured.")

    clear_featured.short_description = "Clear featured flag"

    def refresh_engagement(self, request, queryset):
        for review in queryset:
            review.refresh_engagement_counters()
        self.message_user(request, f"Refreshed engagement counters for {queryset.count()} reviews.")

    refresh_engagement.short_description = "Refresh helpful/report counters"


@admin.register(ReviewImage)
class ReviewImageAdmin(ImportExportEnhancedModelAdmin):
    list_display = ["review", "caption", "sort_order", "created_at"]
    list_filter = ["created_at"]
    search_fields = ["review__product__name", "caption"]


@admin.register(ReviewVote)
class ReviewVoteAdmin(ImportExportEnhancedModelAdmin):
    list_display = ["review", "user", "is_helpful", "created_at"]
    list_filter = ["is_helpful", "created_at"]
    search_fields = ["review__product__name", "user__email"]
    readonly_fields = ["review", "user", "is_helpful", "created_at", "updated_at"]


@admin.register(ReviewReport)
class ReviewReportAdmin(ImportExportEnhancedModelAdmin):
    list_display = ["review", "reporter", "reason", "status", "created_at"]
    list_filter = ["reason", "status", "created_at"]
    search_fields = ["review__product__name", "reporter__email", "details"]
    readonly_fields = ["review", "reporter", "reason", "details", "created_at", "updated_at"]
