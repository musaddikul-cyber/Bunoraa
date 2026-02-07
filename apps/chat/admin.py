"""
Django Admin Configuration for Bunoraa Chat System
"""
from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from django.urls import reverse

from .models import (
    ChatAgent, Conversation, Message, MessageAttachment,
    CannedResponse, TypingIndicator, ChatSettings, ChatAnalytics
)


@admin.register(ChatAgent)
class ChatAgentAdmin(admin.ModelAdmin):
    """Admin for Chat Agents."""
    
    list_display = [
        'user', 'is_online', 'is_accepting_chats',
        'current_chat_count', 'max_concurrent_chats', 'avg_rating'
    ]
    list_filter = ['is_online', 'is_accepting_chats', 'created_at']
    search_fields = ['user__email', 'user__first_name', 'user__last_name']
    readonly_fields = [
        'id', 'created_at', 'updated_at', 'current_chat_count',
        'total_chats_handled', 'avg_rating', 'avg_response_time_seconds', 'last_active_at'
    ]
    
    fieldsets = (
        ('Agent Info', {
            'fields': ('user',)
        }),
        ('Status', {
            'fields': ('is_online', 'is_accepting_chats', 'auto_accept')
        }),
        ('Capacity', {
            'fields': ('max_concurrent_chats', 'current_chat_count')
        }),
        ('Skills', {
            'fields': ('languages', 'categories')
        }),
        ('Performance', {
            'fields': ('total_chats_handled', 'avg_rating', 'avg_response_time_seconds')
        }),
        ('Notifications', {
            'fields': ('notification_sound', 'desktop_notifications')
        }),
        ('Activity', {
            'fields': ('last_active_at', 'created_at', 'updated_at')
        }),
        ('Metadata', {
            'fields': ('id',),
            'classes': ('collapse',)
        }),
    )


class MessageInline(admin.TabularInline):
    """Inline messages in conversation admin."""
    model = Message
    extra = 0
    readonly_fields = ['created_at', 'is_read', 'read_at']
    fields = ['sender', 'is_from_customer', 'is_from_bot', 'content', 'is_read', 'created_at']
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    """Admin for Conversations."""
    
    list_display = [
        'id_short', 'customer_display', 'category', 'status', 'agent',
        'is_bot_handling', 'rating', 'created_at'
    ]
    list_filter = ['status', 'category', 'is_bot_handling', 'source', 'created_at']
    search_fields = ['customer__email', 'customer_name', 'subject', 'id']
    readonly_fields = [
        'id', 'created_at', 'updated_at', 'started_at', 'first_response_at',
        'resolved_at', 'closed_at', 'last_message_at'
    ]
    autocomplete_fields = ['customer', 'agent']
    inlines = [MessageInline]
    
    fieldsets = (
        ('Customer Info', {
            'fields': ('customer', 'customer_name', 'customer_email', 'customer_phone')
        }),
        ('Conversation', {
            'fields': ('subject', 'category', 'initial_message', 'source', 'order_reference')
        }),
        ('Status', {
            'fields': ('status', 'priority', 'is_bot_handling', 'bot_handoff_requested')
        }),
        ('Agent', {
            'fields': ('agent',)
        }),
        ('Resolution', {
            'fields': ('rating', 'rating_comment', 'internal_notes')
        }),
        ('Tags', {
            'fields': ('tags',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'started_at', 'first_response_at', 'resolved_at', 'closed_at', 'last_message_at')
        }),
        ('Metadata', {
            'fields': ('id',),
            'classes': ('collapse',)
        }),
    )
    
    def id_short(self, obj):
        """Display shortened ID."""
        return str(obj.id)[:8] + '...'
    id_short.short_description = 'ID'
    
    def customer_display(self, obj):
        """Display customer name or email."""
        return obj.customer_name or obj.customer.email
    customer_display.short_description = 'Customer'


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Admin for Messages."""
    
    list_display = [
        'id_short', 'conversation_link', 'sender_display', 'message_type',
        'content_preview', 'is_read', 'created_at'
    ]
    list_filter = ['message_type', 'is_from_customer', 'is_from_bot', 'is_read', 'created_at']
    search_fields = ['conversation__id', 'content', 'sender__email']
    readonly_fields = ['id', 'created_at', 'read_at', 'edited_at', 'delivered_at', 'deleted_at']
    autocomplete_fields = ['conversation', 'sender', 'reply_to']
    
    fieldsets = (
        ('Conversation', {
            'fields': ('conversation', 'reply_to')
        }),
        ('Sender', {
            'fields': ('sender', 'is_from_customer', 'is_from_bot')
        }),
        ('Content', {
            'fields': ('message_type', 'content')
        }),
        ('Status', {
            'fields': ('is_read', 'read_at', 'is_delivered', 'delivered_at', 'is_edited', 'edited_at', 'is_deleted', 'deleted_at')
        }),
        ('Reactions', {
            'fields': ('reactions',)
        }),
        ('Metadata', {
            'fields': ('id', 'created_at', 'metadata'),
            'classes': ('collapse',)
        }),
    )
    
    def id_short(self, obj):
        return str(obj.id)[:8] + '...'
    id_short.short_description = 'ID'
    
    def conversation_link(self, obj):
        url = reverse('admin:chat_conversation_change', args=[obj.conversation.id])
        return format_html('<a href="{}">{}</a>', url, str(obj.conversation.id)[:8])
    conversation_link.short_description = 'Conversation'
    
    def sender_display(self, obj):
        if obj.is_from_bot:
            return '🤖 Bot'
        if obj.sender:
            return obj.sender.email
        return 'System'
    sender_display.short_description = 'From'
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content'


@admin.register(MessageAttachment)
class MessageAttachmentAdmin(admin.ModelAdmin):
    """Admin for Message Attachments."""
    
    list_display = ['id_short', 'message', 'file_name', 'file_type', 'file_size', 'created_at']
    list_filter = ['file_type', 'created_at']
    search_fields = ['file_name', 'message__conversation__id']
    readonly_fields = ['id', 'created_at', 'file_size']
    
    def id_short(self, obj):
        return str(obj.id)[:8] + '...'
    id_short.short_description = 'ID'


@admin.register(CannedResponse)
class CannedResponseAdmin(admin.ModelAdmin):
    """Admin for Canned Responses."""
    
    list_display = ['title', 'shortcut', 'category', 'agent', 'usage_count', 'is_active']
    list_filter = ['category', 'is_active', 'created_at']
    search_fields = ['title', 'shortcut', 'content', 'agent__user__email']
    readonly_fields = ['id', 'created_at', 'updated_at', 'usage_count']
    autocomplete_fields = ['agent']
    
    fieldsets = (
        ('Response', {
            'fields': ('title', 'shortcut', 'content')
        }),
        ('Classification', {
            'fields': ('category',)
        }),
        ('Ownership', {
            'fields': ('agent',)
        }),
        ('Status', {
            'fields': ('is_active', 'usage_count')
        }),
        ('Metadata', {
            'fields': ('id', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ChatSettings)
class ChatSettingsAdmin(admin.ModelAdmin):
    """Admin for Chat Settings (singleton)."""
    
    list_display = ['__str__', 'is_chat_enabled', 'ai_enabled']
    readonly_fields = ['id', 'updated_at']
    
    fieldsets = (
        ('General', {
            'fields': ('is_chat_enabled',)
        }),
        ('Messages', {
            'fields': ('welcome_message', 'offline_message', 'rating_prompt')
        }),
        ('Business Hours', {
            'fields': ('business_hours_enabled', 'business_hours', 'timezone')
        }),
        ('Bot Settings', {
            'fields': ('bot_name', 'bot_avatar')
        }),
        ('AI Settings', {
            'fields': (
                'ai_enabled', 'ai_model', 'ai_temperature', 
                'ai_max_tokens', 'max_ai_responses_before_handoff', 'auto_reply_delay_seconds'
            )
        }),
        ('AI Prompt', {
            'fields': ('ai_system_prompt',)
        }),
        ('Notifications', {
            'fields': ('notify_on_new_chat', 'notify_on_customer_waiting', 'customer_waiting_threshold_minutes')
        }),
        ('Attachments', {
            'fields': ('allowed_file_types', 'max_file_size_mb', 'max_message_length')
        }),
        ('Rating', {
            'fields': ('request_rating',)
        }),
        ('Metadata', {
            'fields': ('id', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        # Only allow one settings object
        return not ChatSettings.objects.exists()
    
    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(ChatAnalytics)
class ChatAnalyticsAdmin(admin.ModelAdmin):
    """Admin for Chat Analytics."""
    
    list_display = [
        'date', 'total_conversations', 'resolved_conversations',
        'total_messages', 'avg_rating_display', 'avg_response_time'
    ]
    list_filter = ['date']
    readonly_fields = [
        'id', 'date', 'total_conversations', 'new_conversations',
        'resolved_conversations', 'total_messages', 'customer_messages',
        'agent_messages', 'bot_messages', 'avg_first_response_seconds',
        'avg_resolution_time_seconds', 'avg_rating', 'category_breakdown',
        'agent_performance', 'created_at', 'updated_at'
    ]
    ordering = ['-date']
    
    def avg_rating_display(self, obj):
        if obj.avg_rating:
            stars = '⭐' * int(obj.avg_rating)
            return f'{obj.avg_rating:.1f} {stars}'
        return '-'
    avg_rating_display.short_description = 'Avg Rating'
    
    def avg_response_time(self, obj):
        if obj.avg_first_response_seconds:
            minutes = obj.avg_first_response_seconds / 60
            return f'{minutes:.1f} min'
        return '-'
    avg_response_time.short_description = 'Avg Response'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


# Register typing indicators only in DEBUG mode
from django.conf import settings as django_settings

if getattr(django_settings, 'DEBUG', False):
    @admin.register(TypingIndicator)
    class TypingIndicatorAdmin(admin.ModelAdmin):
        list_display = ['conversation', 'user', 'started_at']
        readonly_fields = ['conversation', 'user', 'started_at']
