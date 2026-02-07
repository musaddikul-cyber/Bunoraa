"""
Reviews admin configuration
"""
from django.contrib import admin
from .models import Review, ReviewImage, ReviewVote, ReviewReply


class ReviewImageInline(admin.TabularInline):
    model = ReviewImage
    extra = 0
    readonly_fields = ['image', 'caption']


class ReviewReplyInline(admin.StackedInline):
    model = ReviewReply
    extra = 0
    readonly_fields = ['replied_by', 'created_at', 'updated_at']


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = [
        'product', 'user', 'rating', 'status',
        'is_verified_purchase', 'helpful_count', 'created_at'
    ]
    list_filter = ['status', 'rating', 'is_verified_purchase', 'created_at']
    search_fields = ['product__name', 'user__email', 'title', 'content']
    readonly_fields = [
        'id', 'product', 'user', 'is_verified_purchase',
        'helpful_count', 'not_helpful_count', 'created_at', 'updated_at'
    ]
    
    inlines = [ReviewImageInline, ReviewReplyInline]
    
    actions = ['approve_reviews', 'reject_reviews']
    
    fieldsets = (
        ('Review Info', {
            'fields': ('id', 'product', 'user', 'is_verified_purchase')
        }),
        ('Content', {
            'fields': ('rating', 'title', 'content')
        }),
        ('Moderation', {
            'fields': ('status', 'moderation_notes')
        }),
        ('Engagement', {
            'fields': ('helpful_count', 'not_helpful_count')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def approve_reviews(self, request, queryset):
        updated = queryset.filter(status=Review.STATUS_PENDING).update(
            status=Review.STATUS_APPROVED
        )
        self.message_user(request, f'{updated} reviews approved.')
    approve_reviews.short_description = 'Approve selected reviews'
    
    def reject_reviews(self, request, queryset):
        updated = queryset.filter(status=Review.STATUS_PENDING).update(
            status=Review.STATUS_REJECTED
        )
        self.message_user(request, f'{updated} reviews rejected.')
    reject_reviews.short_description = 'Reject selected reviews'


@admin.register(ReviewImage)
class ReviewImageAdmin(admin.ModelAdmin):
    list_display = ['review', 'image', 'caption', 'sort_order']
    list_filter = ['created_at']
    search_fields = ['review__product__name', 'caption']


@admin.register(ReviewVote)
class ReviewVoteAdmin(admin.ModelAdmin):
    list_display = ['review', 'user', 'is_helpful', 'created_at']
    list_filter = ['is_helpful', 'created_at']
    search_fields = ['review__product__name', 'user__email']
    readonly_fields = ['review', 'user', 'is_helpful', 'created_at']


@admin.register(ReviewReply)
class ReviewReplyAdmin(admin.ModelAdmin):
    list_display = ['review', 'replied_by', 'created_at']
    list_filter = ['created_at']
    search_fields = ['review__product__name', 'content']
    readonly_fields = ['review', 'replied_by', 'created_at', 'updated_at']
