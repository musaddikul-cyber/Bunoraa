"""
Contacts Admin Configuration
"""
from django.contrib import admin
from django.utils.html import format_html

from .models import (
    ContactCategory, ContactInquiry, ContactResponse,
    ContactAttachment, StoreLocation, ContactSettings,
    CustomizationRequest
)


class ContactResponseInline(admin.TabularInline):
    """Inline for contact responses."""
    
    model = ContactResponse
    extra = 0
    readonly_fields = ['sent_by', 'sent_at', 'delivered', 'opened', 'created_at']
    fields = ['subject', 'message', 'sent_by', 'sent_at', 'delivered']


class ContactAttachmentInline(admin.TabularInline):
    """Inline for contact attachments."""
    
    model = ContactAttachment
    extra = 0
    readonly_fields = ['filename', 'file_size', 'content_type', 'created_at']
    fields = ['file', 'filename', 'file_size', 'content_type']


@admin.register(ContactCategory)
class ContactCategoryAdmin(admin.ModelAdmin):
    """Admin for ContactCategory model."""
    
    list_display = ['name', 'slug', 'order', 'inquiries_count', 'is_active']
    list_filter = ['is_active']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    ordering = ['order', 'name']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'slug', 'description')
        }),
        ('Email Routing', {
            'fields': ('email_recipients',)
        }),
        ('Auto-Response', {
            'fields': ('auto_response_subject', 'auto_response_message'),
            'classes': ('collapse',)
        }),
        ('Display', {
            'fields': ('order', 'is_active')
        }),
    )
    
    def inquiries_count(self, obj):
        return obj.inquiries.count()
    inquiries_count.short_description = 'Inquiries'


@admin.register(ContactInquiry)
class ContactInquiryAdmin(admin.ModelAdmin):
    """Admin for ContactInquiry model."""
    
    list_display = [
        'subject', 'name', 'email', 'category', 'status_badge',
        'priority', 'created_at', 'responded_at'
    ]
    list_filter = ['status', 'priority', 'category', 'created_at']
    search_fields = ['name', 'email', 'subject', 'message', 'order_number']
    readonly_fields = [
        'user', 'ip_address', 'user_agent', 'source_page',
        'auto_response_sent', 'created_at', 'updated_at'
    ]
    ordering = ['-created_at']
    inlines = [ContactAttachmentInline, ContactResponseInline]
    
    fieldsets = (
        (None, {
            'fields': ('name', 'email', 'phone', 'company')
        }),
        ('Inquiry', {
            'fields': ('category', 'subject', 'message', 'order_number')
        }),
        ('Status', {
            'fields': ('status', 'priority', 'responded_by', 'responded_at')
        }),
        ('Internal', {
            'fields': ('internal_notes',)
        }),
        ('Tracking', {
            'fields': ('user', 'ip_address', 'user_agent', 'source_page', 'auto_response_sent'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['mark_responded', 'mark_closed', 'mark_spam']
    
    def status_badge(self, obj):
        colors = {
            'new': 'blue',
            'in_progress': 'orange',
            'responded': 'green',
            'closed': 'gray',
            'spam': 'red',
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; '
            'border-radius: 3px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def mark_responded(self, request, queryset):
        from django.utils import timezone
        count = queryset.update(
            status='responded',
            responded_by=request.user,
            responded_at=timezone.now()
        )
        self.message_user(request, f'{count} inquiries marked as responded.')
    mark_responded.short_description = 'Mark as responded'
    
    def mark_closed(self, request, queryset):
        count = queryset.update(status='closed')
        self.message_user(request, f'{count} inquiries closed.')
    mark_closed.short_description = 'Mark as closed'
    
    def mark_spam(self, request, queryset):
        count = queryset.update(status='spam')
        self.message_user(request, f'{count} inquiries marked as spam.')
    mark_spam.short_description = 'Mark as spam'


@admin.register(CustomizationRequest)
class CustomizationRequestAdmin(admin.ModelAdmin):
    """Admin for CustomizationRequest model."""
    
    list_display = ['product', 'name', 'email', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['name', 'email', 'message', 'product__name']
    readonly_fields = ['product', 'user', 'ip_address', 'user_agent', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    fieldsets = (
        (None, {
            'fields': ('product', 'user', 'name', 'email', 'phone')
        }),
        ('Request Details', {
            'fields': ('message',)
        }),
        ('Status', {
            'fields': ('status',)
        }),
        ('Internal Notes', {
            'fields': ('internal_notes',)
        }),
        ('Tracking', {
            'fields': ('ip_address', 'user_agent', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['mark_in_progress', 'mark_quoted', 'mark_ordered', 'mark_closed']
    
    def mark_in_progress(self, request, queryset):
        updated = queryset.update(status='in_progress')
        self.message_user(request, f'{updated} requests marked as in progress.')
    mark_in_progress.short_description = 'Mark as In Progress'

    def mark_quoted(self, request, queryset):
        updated = queryset.update(status='quoted')
        self.message_user(request, f'{updated} requests marked as quoted.')
    mark_quoted.short_description = 'Mark as Quoted'

    def mark_ordered(self, request, queryset):
        updated = queryset.update(status='ordered')
        self.message_user(request, f'{updated} requests marked as ordered.')
    mark_ordered.short_description = 'Mark as Ordered'

    def mark_closed(self, request, queryset):
        updated = queryset.update(status='closed')
        self.message_user(request, f'{updated} requests marked as closed.')
    mark_closed.short_description = 'Mark as Closed'


@admin.register(ContactResponse)
class ContactResponseAdmin(admin.ModelAdmin):
    """Admin for ContactResponse model."""
    
    list_display = ['inquiry', 'subject', 'sent_by', 'sent_at', 'delivered', 'opened']
    list_filter = ['delivered', 'opened', 'created_at']
    search_fields = ['inquiry__subject', 'subject', 'message']
    readonly_fields = ['inquiry', 'sent_by', 'sent_at', 'delivered', 'opened', 'opened_at', 'created_at']
    ordering = ['-created_at']


@admin.register(StoreLocation)
class StoreLocationAdmin(admin.ModelAdmin):
    """Admin for StoreLocation model."""
    
    list_display = [
        'name', 'city', 'country', 'is_pickup_location',
        'is_returns_location', 'is_main', 'is_active', 'order'
    ]
    list_filter = ['is_active', 'is_main', 'is_pickup_location', 'is_returns_location', 'country']
    search_fields = ['name', 'address_line1', 'city', 'country']
    prepopulated_fields = {'slug': ('name',)}
    ordering = ['order', 'name']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'slug', 'description', 'image')
        }),
        ('Address', {
            'fields': (
                'address_line1', 'address_line2', 'city', 'state',
                'postal_code', 'country'
            )
        }),
        ('Location', {
            'fields': ('latitude', 'longitude')
        }),
        ('Contact', {
            'fields': ('phone', 'email', 'fax')
        }),
        ('Hours', {
            'fields': (
                'monday_hours', 'tuesday_hours', 'wednesday_hours',
                'thursday_hours', 'friday_hours', 'saturday_hours', 'sunday_hours'
            ),
            'classes': ('collapse',)
        }),
        ('Features', {
            'fields': ('is_pickup_location', 'is_returns_location')
        }),
        ('Display', {
            'fields': ('order', 'is_active', 'is_main')
        }),
    )


@admin.register(ContactSettings)
class ContactSettingsAdmin(admin.ModelAdmin):
    """Admin for ContactSettings model."""
    
    list_display = ['general_email', 'phone', 'allow_attachments', 'enable_auto_response']
    
    fieldsets = (
        ('Contact Information', {
            'fields': ('general_email', 'support_email', 'sales_email', 'phone')
        }),
        ('Social Media', {
            'fields': (
                'facebook_url', 'twitter_url', 'instagram_url',
                'linkedin_url', 'youtube_url', 'pinterest_url', 'tiktok_url'
            ),
            'classes': ('collapse',)
        }),
        ('Form Settings', {
            'fields': ('allow_attachments', 'max_attachment_size_mb', 'allowed_file_types')
        }),
        ('Notifications', {
            'fields': ('notify_on_new_inquiry', 'notification_emails')
        }),
        ('Auto-Response', {
            'fields': (
                'enable_auto_response',
                'default_auto_response_subject',
                'default_auto_response_message'
            )
        }),
        ('Business Hours', {
            'fields': ('business_hours_note',),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        if ContactSettings.objects.exists():
            return False
        return super().has_add_permission(request)
    
    def has_delete_permission(self, request, obj=None):
        return False
