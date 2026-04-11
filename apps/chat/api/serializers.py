"""
DRF Serializers for Bunoraa Chat System API
"""
from urllib.parse import urlparse

from rest_framework import serializers
from django.contrib.auth import get_user_model

from apps.chat.models import (
    ChatAgent, Conversation, Message, MessageAttachment,
    CannedResponse, ChatSettings, ChatAnalytics, ConversationStatus
)
from apps.chat.services import user_can_access_conversation, get_agent_for_user

User = get_user_model()


def normalize_public_asset_url(value: str | None, request=None) -> str:
    if not value:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        return raw
    if raw.startswith("//"):
        return f"https:{raw}"
    normalized = raw if raw.startswith("/") else f"/{raw.lstrip('/')}"
    if request:
        try:
            return request.build_absolute_uri(normalized)
        except Exception:
            return normalized
    return normalized


def get_default_agent_avatar_url(request=None) -> str:
    from django.conf import settings

    return normalize_public_asset_url(
        getattr(settings, "DEFAULT_AGENT_AVATAR_URL", "/static/images/assets/favicon.ico"),
        request=request,
    )


class SiteLogoMixin:
    """Provide a cached site logo URL for support avatars."""

    def _get_site_logo_url(self):
        if hasattr(self, '_site_logo_url_cache'):
            return self._site_logo_url_cache
        request = self.context.get('request') if hasattr(self, 'context') else None
        logo_url = None
        try:
            from apps.pages.models import SiteSettings
            site_settings = SiteSettings.get_settings()
            logo_field = site_settings.logo or site_settings.logo_dark
            if logo_field:
                logo_url = normalize_public_asset_url(logo_field.url, request=request)
        except Exception:
            logo_url = None
        self._site_logo_url_cache = logo_url
        return logo_url


class UserMinimalSerializer(SiteLogoMixin, serializers.ModelSerializer):
    """Minimal user serializer for chat display."""
    
    full_name = serializers.SerializerMethodField()
    avatar_url = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'email', 'full_name', 'avatar_url']
        read_only_fields = fields
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.email

    def get_avatar_url(self, obj):
        request = self.context.get('request')
        if obj.is_staff or obj.is_superuser:
            site_logo_url = self._get_site_logo_url()
            if site_logo_url:
                return site_logo_url
        if obj.avatar:
            return normalize_public_asset_url(obj.avatar.url, request=request)
        return get_default_agent_avatar_url(request=request)


class ChatAgentSerializer(SiteLogoMixin, serializers.ModelSerializer):
    """Serializer for Chat Agents."""
    
    user = UserMinimalSerializer(read_only=True)
    user_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(),
        source='user',
        write_only=True
    )
    display_name = serializers.SerializerMethodField()
    avatar_url = serializers.SerializerMethodField()
    role = serializers.SerializerMethodField()
    skills = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    total_ratings = serializers.SerializerMethodField()
    last_activity = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatAgent
        fields = [
            'id', 'user', 'user_id', 'display_name', 'avatar_url', 'role', 'bio',
            'is_online', 'is_accepting_chats', 'is_active',
            'max_concurrent_chats', 'current_chat_count',
            'languages', 'categories', 'skills',
            'total_chats_handled', 'avg_rating', 'total_ratings',
            'last_activity', 'created_at'
        ]
        read_only_fields = [
            'id', 'current_chat_count', 'total_chats_handled',
            'avg_rating', 'total_ratings', 'last_activity', 'created_at'
        ]

    def get_display_name(self, obj):
        """Derive display name from the associated user."""
        return obj.user.get_full_name() or obj.user.email

    def get_avatar_url(self, obj):
        """Return the agent's avatar URL or a default if not set."""
        request = self.context.get('request')
        if obj.user.is_staff or obj.user.is_superuser:
            site_logo_url = self._get_site_logo_url()
            if site_logo_url:
                return site_logo_url
        if obj.avatar:
            return normalize_public_asset_url(obj.avatar.url, request=request)
        return get_default_agent_avatar_url(request=request)

    def get_role(self, obj):
        if obj.user.is_superuser:
            return "admin"
        if obj.user.is_staff:
            return "staff"
        return "agent"

    def get_total_ratings(self, obj):
        return obj.conversations.filter(rating__isnull=False).count()

    def get_last_activity(self, obj):
        return obj.last_active_at


class ChatAgentPublicSerializer(SiteLogoMixin, serializers.ModelSerializer):
    """Public agent info (for customers)."""
    
    display_name = serializers.SerializerMethodField()
    avatar_url = serializers.SerializerMethodField()
    role = serializers.SerializerMethodField()

    class Meta:
        model = ChatAgent
        fields = ['id', 'display_name', 'avatar_url', 'role']
        read_only_fields = fields

    def get_avatar_url(self, obj):
        """Return the agent's avatar URL or a default if not set."""
        request = self.context.get('request')
        if obj.user.is_staff or obj.user.is_superuser:
            site_logo_url = self._get_site_logo_url()
            if site_logo_url:
                return site_logo_url
        if obj.avatar:
            return normalize_public_asset_url(obj.avatar.url, request=request)
        return get_default_agent_avatar_url(request=request)

    def get_display_name(self, obj):
        return obj.user.get_full_name() or obj.user.email

    def get_role(self, obj):
        if obj.user.is_superuser:
            return "admin"
        if obj.user.is_staff:
            return "staff"
        return "agent"


class MessageAttachmentSerializer(serializers.ModelSerializer):
    """Serializer for Message Attachments."""
    
    download_url = serializers.SerializerMethodField()
    
    class Meta:
        model = MessageAttachment
        fields = [
            'id', 'file', 'file_name', 'file_type', 'file_size',
            'thumbnail', 'download_url', 'created_at'
        ]
        read_only_fields = ['id', 'file_name', 'file_type', 'file_size', 'created_at']
    
    def get_download_url(self, obj):
        request = self.context.get('request')
        if obj.file and request:
            return request.build_absolute_uri(obj.file.url)
        return None


class MessageSerializer(SiteLogoMixin, serializers.ModelSerializer):
    """Serializer for Chat Messages."""
    
    sender = UserMinimalSerializer(read_only=True)
    attachments = MessageAttachmentSerializer(many=True, read_only=True)
    reply_to_preview = serializers.SerializerMethodField()
    sender_display_name = serializers.SerializerMethodField()
    sender_avatar_url = serializers.SerializerMethodField()
    sender_role = serializers.SerializerMethodField()
    
    class Meta:
        model = Message
        fields = [
            'id', 'conversation', 'sender', 'is_from_customer', 'is_from_bot',
            'sender_display_name', 'sender_avatar_url', 'sender_role',
            'message_type', 'content', 'attachments',
            'is_read', 'read_at', 'is_edited', 'edited_at', 'is_deleted',
            'reactions', 'reply_to', 'reply_to_preview',
            'created_at'
        ]
        read_only_fields = [
            'id', 'sender', 'is_from_customer', 'is_from_bot',
            'is_read', 'read_at', 'is_edited', 'edited_at',
            'reactions', 'created_at'
        ]
    
    def get_reply_to_preview(self, obj):
        if obj.reply_to and not obj.reply_to.is_deleted:
            return {
                'id': str(obj.reply_to.id),
                'content': obj.reply_to.content[:100],
                'sender_name': obj.reply_to.sender.get_full_name() if obj.reply_to.sender else 'Bot'
            }
        return None

    def _get_chat_settings(self):
        if not hasattr(self, '_chat_settings_cache'):
            self._chat_settings_cache = ChatSettings.get_settings()
        return self._chat_settings_cache

    def get_sender_display_name(self, obj):
        request = self.context.get('request')
        if (
            request
            and getattr(request, 'user', None)
            and request.user.is_authenticated
            and obj.sender_id == request.user.id
            and obj.is_from_customer
        ):
            return ''
        if obj.is_from_bot:
            return self._get_chat_settings().bot_name
        if obj.sender:
            return obj.sender.get_full_name() or obj.sender.email
        return 'Support'

    def get_sender_avatar_url(self, obj):
        request = self.context.get('request')
        settings_obj = self._get_chat_settings()

        def _abs(url):
            return normalize_public_asset_url(url, request=request)

        if obj.is_from_bot:
            if settings_obj.bot_avatar:
                return _abs(settings_obj.bot_avatar.url)
            return get_default_agent_avatar_url(request=request)

        sender = obj.sender
        if not sender:
            return get_default_agent_avatar_url(request=request)

        if sender.is_staff or sender.is_superuser:
            site_logo_url = self._get_site_logo_url()
            if site_logo_url:
                return site_logo_url

        # Prefer explicit agent avatar for support users.
        agent_profile = getattr(sender, 'chat_agent_profile', None)
        if agent_profile and getattr(agent_profile, 'avatar', None):
            return _abs(agent_profile.avatar.url)
        if sender.avatar:
            return _abs(sender.avatar.url)

        return get_default_agent_avatar_url(request=request)

    def get_sender_role(self, obj):
        if obj.is_from_bot:
            return 'bot'
        if not obj.sender:
            return 'agent'
        if obj.sender.is_superuser:
            return 'admin'
        if obj.sender.is_staff:
            return 'staff'
        return 'customer'


class MessageCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating messages."""
    
    attachments = serializers.ListField(
        child=serializers.FileField(),
        required=False,
        write_only=True
    )
    
    class Meta:
        model = Message
        fields = ['conversation', 'content', 'message_type', 'reply_to', 'attachments']

    def validate(self, attrs):
        request = self.context.get('request')
        conversation = attrs.get('conversation')

        if request and conversation:
            if not user_can_access_conversation(request.user, conversation):
                raise serializers.ValidationError('Not authorized to post to this conversation.')
            agent = get_agent_for_user(request.user)
            if (
                agent
                and conversation.agent_id is None
                and conversation.status == ConversationStatus.WAITING
                and not request.user.is_staff
            ):
                raise serializers.ValidationError('Assign the conversation before sending a message.')

        settings_obj = ChatSettings.get_settings()
        content = attrs.get('content') or ''
        if settings_obj.max_message_length and len(content) > settings_obj.max_message_length:
            raise serializers.ValidationError(f'Message exceeds max length of {settings_obj.max_message_length} characters.')

        attachments = attrs.get('attachments') or []
        allowed_types = settings_obj.allowed_file_types or []
        if isinstance(allowed_types, str):
            allowed_types = [t.strip() for t in allowed_types.split(',') if t.strip()]
        allowed_types = [str(t).lower() for t in allowed_types if t]
        max_size_bytes = int(settings_obj.max_file_size_mb) * 1024 * 1024
        for file in attachments:
            if max_size_bytes and file.size > max_size_bytes:
                raise serializers.ValidationError(f'Attachment exceeds max size of {settings_obj.max_file_size_mb} MB.')
            if allowed_types:
                import os
                content_type = (file.content_type or '').lower()
                file_ext = os.path.splitext(file.name or '')[1].lstrip('.').lower()
                allowed_mimes = [t for t in allowed_types if '/' in t]
                allowed_exts = [t.lstrip('.').lower() for t in allowed_types if '/' not in t]
                allowed = False
                if allowed_mimes and content_type in allowed_mimes:
                    allowed = True
                if allowed_exts and file_ext in allowed_exts:
                    allowed = True
                if not allowed:
                    raise serializers.ValidationError('Attachment type not allowed.')

        return attrs

    def create(self, validated_data):
        attachments_data = validated_data.pop('attachments', [])
        request = self.context.get('request')
        
        # Set sender from request
        validated_data['sender'] = request.user
        
        # Determine sender side from conversation ownership, not role only.
        conversation = validated_data['conversation']
        validated_data['is_from_customer'] = conversation.customer_id == request.user.id
        
        message = Message.objects.create(**validated_data)
        
        # Handle attachments
        for file in attachments_data:
            file_name = file.name
            if file_name:
                import os
                file_name = os.path.basename(file_name)
            MessageAttachment.objects.create(
                message=message,
                file=file,
                file_name=file_name or file.name,
                file_type=file.content_type,
                file_size=file.size
            )
        
        return message


class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for Conversations."""

    customer = UserMinimalSerializer(read_only=True)
    agent = ChatAgentPublicSerializer(read_only=True)
    last_message = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()
    message_count = serializers.SerializerMethodField()
    feedback = serializers.CharField(source='rating_comment', read_only=True)
    internal_notes = serializers.SerializerMethodField()
    customer_avatar_url = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'customer', 'customer_name', 'customer_email', 'customer_phone',
            'customer_avatar_url',
            'agent', 'category', 'subject', 'status', 'priority',
            'is_bot_handling', 'source', 'order_reference',
            'rating', 'feedback', 'message_count', 'last_message', 'unread_count',
            'internal_notes',
            'created_at', 'started_at', 'first_response_at', 'resolved_at', 'last_message_at'
        ]
        read_only_fields = [
            'id', 'customer', 'agent', 'message_count',
            'created_at', 'started_at', 'first_response_at', 'resolved_at', 'last_message_at'
        ]
    
    def get_message_count(self, obj):
        return obj.messages.filter(is_deleted=False).count()
    
    def get_last_message(self, obj):
        last = obj.messages.order_by('-created_at').first()
        if last:
            return {
                'id': str(last.id),
                'content': last.content[:100] if not last.is_deleted else '[Deleted]',
                'is_from_customer': last.is_from_customer,
                'created_at': last.created_at.isoformat()
            }
        return None
    
    def get_unread_count(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            # For agents, count unread customer messages
            # For customers, count unread agent/bot messages
            is_agent = ChatAgent.objects.filter(user=request.user, is_active=True).exists()
            if is_agent:
                return obj.messages.filter(is_from_customer=True, is_read=False).count()
            else:
                return obj.messages.filter(is_from_customer=False, is_read=False).count()
        return 0

    def get_internal_notes(self, obj):
        request = self.context.get('request')
        if not request or not request.user.is_authenticated:
            return ''
        if request.user.is_staff:
            return obj.internal_notes
        agent = get_agent_for_user(request.user)
        if agent and obj.agent_id == agent.id:
            return obj.internal_notes
        return ''

    def get_customer_avatar_url(self, obj):
        request = self.context.get('request')
        if obj.customer and obj.customer.avatar:
            return normalize_public_asset_url(obj.customer.avatar.url, request=request)
        return get_default_agent_avatar_url(request=request)


class ConversationDetailSerializer(ConversationSerializer):
    """Detailed conversation serializer with messages."""
    
    messages = serializers.SerializerMethodField()
    
    class Meta(ConversationSerializer.Meta):
        # Only include fields that exist on the model
        fields = ConversationSerializer.Meta.fields + ['messages', 'initial_message']
    
    def get_messages(self, obj):
        """Get last 50 messages (oldest first for display)."""
        messages = obj.messages.filter(is_deleted=False).select_related('sender').order_by('-created_at')[:50]
        # Reverse to get chronological order (oldest first)
        messages_list = list(messages)
        messages_list.reverse()
        return MessageSerializer(messages_list, many=True, context=self.context).data


class ConversationCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating conversations."""
    
    initial_message = serializers.CharField(required=False, allow_blank=True)
    
    class Meta:
        model = Conversation
        fields = ['category', 'subject', 'initial_message', 'source', 'order_reference']
    
    def create(self, validated_data):
        from apps.chat.services import ChatService
        
        request = self.context.get('request')
        customer = request.user
        
        conversation, error = ChatService.start_conversation(
            customer=customer,
            category=validated_data.get('category', 'general'),
            subject=validated_data.get('subject', ''),
            initial_message=validated_data.get('initial_message', ''),
            source=validated_data.get('source', 'website'),
            order_reference=validated_data.get('order_reference')
        )
        
        if error:
            raise serializers.ValidationError(error)
        
        return conversation


class ConversationRatingSerializer(serializers.Serializer):
    """Serializer for rating a conversation."""
    
    rating = serializers.IntegerField(min_value=1, max_value=5)
    feedback = serializers.CharField(required=False, allow_blank=True, max_length=1000)
    
    def update(self, instance, validated_data):
        instance.rating = validated_data['rating']
        instance.rating_comment = validated_data.get('feedback', '')
        instance.save(update_fields=['rating', 'rating_comment'])
        return instance


class CannedResponseSerializer(serializers.ModelSerializer):
    """Serializer for Canned Responses."""

    tags = serializers.ListField(child=serializers.CharField(), required=False, default=list)

    class Meta:
        model = CannedResponse
        fields = [
            'id', 'title', 'shortcut', 'content', 'category', 'tags',
            'agent', 'is_global', 'is_active', 'use_count', 'last_used_at', 'created_at'
        ]
        read_only_fields = ['id', 'use_count', 'last_used_at', 'created_at']


class ChatSettingsSerializer(serializers.ModelSerializer):
    """Serializer for Chat Settings."""
    
    class Meta:
        model = ChatSettings
        fields = [
            'id', 'is_chat_enabled', 'welcome_message', 'offline_message',
            'wait_message', 'ai_enabled', 'ai_model', 'ai_temperature',
            'ai_max_tokens', 'ai_system_prompt',
            'max_ai_responses_before_handoff', 'auto_reply_delay_seconds',
            'max_concurrent_chats',
            'business_hours_enabled', 'business_hours',
            'allowed_file_types', 'max_file_size_mb',
            'support_inbox', 'email_reply_from'
        ]
        read_only_fields = ['id']

    def validate_ai_model(self, value):
        model_id = (value or "").strip()
        if not model_id:
            raise serializers.ValidationError("AI model cannot be empty.")
        if any(ch in model_id for ch in ("\n", "\r", "\t")):
            raise serializers.ValidationError("AI model contains invalid characters.")
        return model_id


class ChatAnalyticsSerializer(serializers.ModelSerializer):
    """Serializer for Chat Analytics."""

    resolution_rate = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatAnalytics
        fields = [
            'id', 'date', 'total_conversations', 'new_conversations',
            'resolved_conversations', 'resolution_rate',
            'total_messages', 'customer_messages', 'agent_messages', 'bot_messages',
            'avg_first_response_seconds', 'avg_resolution_time_seconds', 'avg_rating',
            'category_breakdown', 'channel_breakdown', 'hourly_breakdown', 'agent_performance'
        ]
        read_only_fields = fields
    
    def get_resolution_rate(self, obj):
        if obj.total_conversations > 0:
            return round(obj.resolved_conversations / obj.total_conversations * 100, 1)
        return 0


class ChatQueueSerializer(serializers.Serializer):
    """Serializer for chat queue display."""
    
    waiting_count = serializers.IntegerField()
    active_count = serializers.IntegerField()
    available_agents = serializers.IntegerField()
    estimated_wait = serializers.IntegerField()  # minutes
    conversations = ConversationSerializer(many=True)


class AgentDashboardSerializer(serializers.Serializer):
    """Serializer for agent dashboard data."""
    
    agent = ChatAgentSerializer()
    active_chats = ConversationSerializer(many=True)
    queue = ChatQueueSerializer()
    today_stats = serializers.DictField()
