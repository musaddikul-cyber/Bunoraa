"""
DRF Serializers for Bunoraa Chat System API
"""
from rest_framework import serializers
from django.contrib.auth import get_user_model

from apps.chat.models import (
    ChatAgent, Conversation, Message, MessageAttachment,
    CannedResponse, ChatSettings, ChatAnalytics, ConversationStatus
)

User = get_user_model()


class UserMinimalSerializer(serializers.ModelSerializer):
    """Minimal user serializer for chat display."""
    
    full_name = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'email', 'full_name']
        read_only_fields = fields
    
    def get_full_name(self, obj):
        return obj.get_full_name() or obj.email


class ChatAgentSerializer(serializers.ModelSerializer):
    """Serializer for Chat Agents."""
    
    user = UserMinimalSerializer(read_only=True)
    user_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(),
        source='user',
        write_only=True
    )
    
    class Meta:
        model = ChatAgent
        fields = [
            'id', 'user', 'user_id', 'display_name', 'avatar', 'role', 'bio',
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


class ChatAgentPublicSerializer(serializers.ModelSerializer):
    """Public agent info (for customers)."""
    
    class Meta:
        model = ChatAgent
        fields = ['id', 'display_name', 'avatar', 'role']
        read_only_fields = fields


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


class MessageSerializer(serializers.ModelSerializer):
    """Serializer for Chat Messages."""
    
    sender = UserMinimalSerializer(read_only=True)
    attachments = MessageAttachmentSerializer(many=True, read_only=True)
    reply_to_preview = serializers.SerializerMethodField()
    
    class Meta:
        model = Message
        fields = [
            'id', 'conversation', 'sender', 'is_from_customer', 'is_from_bot',
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
    
    def create(self, validated_data):
        attachments_data = validated_data.pop('attachments', [])
        request = self.context.get('request')
        
        # Set sender from request
        validated_data['sender'] = request.user
        
        # Determine if from customer or agent
        is_agent = ChatAgent.objects.filter(user=request.user, is_active=True).exists()
        validated_data['is_from_customer'] = not is_agent
        
        message = Message.objects.create(**validated_data)
        
        # Handle attachments
        for file in attachments_data:
            MessageAttachment.objects.create(
                message=message,
                file=file,
                file_name=file.name,
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
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'customer', 'customer_name', 'customer_email', 'customer_phone',
            'agent', 'category', 'subject', 'status', 'priority',
            'is_bot_handling', 'source', 'order_reference',
            'rating', 'feedback', 'message_count', 'last_message', 'unread_count',
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


class ConversationDetailSerializer(ConversationSerializer):
    """Detailed conversation serializer with messages."""
    
    messages = serializers.SerializerMethodField()
    
    class Meta(ConversationSerializer.Meta):
        # Only include fields that exist on the model
        fields = ConversationSerializer.Meta.fields + ['messages', 'initial_message']
    
    def get_messages(self, obj):
        """Get last 50 messages (oldest first for display)."""
        messages = obj.messages.filter(is_deleted=False).order_by('-created_at')[:50]
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
    
    class Meta:
        model = CannedResponse
        fields = [
            'id', 'title', 'shortcut', 'content', 'category', 'tags',
            'agent', 'is_global', 'is_active', 'use_count', 'created_at'
        ]
        read_only_fields = ['id', 'use_count', 'created_at']


class ChatSettingsSerializer(serializers.ModelSerializer):
    """Serializer for Chat Settings."""
    
    class Meta:
        model = ChatSettings
        fields = [
            'id', 'is_chat_enabled', 'welcome_message', 'offline_message',
            'wait_message', 'ai_enabled', 'max_concurrent_chats',
            'business_hours_enabled', 'business_hours',
            'allowed_file_types', 'max_file_size_mb'
        ]
        read_only_fields = ['id']


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
            'category_breakdown', 'hourly_breakdown'
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
