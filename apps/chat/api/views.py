"""
DRF ViewSets for Bunoraa Chat System API
"""
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.views import APIView
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.exceptions import PermissionDenied
from rest_framework.pagination import PageNumberPagination
from django.utils import timezone
from django.db.models import Q, F
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.contrib.auth import get_user_model
import logging
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
import uuid

from apps.chat.models import (
    ChatAgent, Conversation, Message, MessageAttachment,
    CannedResponse, ChatSettings, ChatAnalytics, MessageType,
    ConversationStatus, ConversationCategory
)
from apps.chat.services import ChatService, ChatAnalyticsService
from apps.chat.tasks import generate_ai_response, send_chat_rating_request
from apps.chat.utils import redact_payload

from .serializers import (
    ChatAgentSerializer, ChatAgentPublicSerializer,
    ConversationSerializer, ConversationDetailSerializer,
    ConversationCreateSerializer, ConversationRatingSerializer,
    MessageSerializer, MessageCreateSerializer, MessageAttachmentSerializer,
    CannedResponseSerializer, ChatSettingsSerializer, ChatAnalyticsSerializer,
    AgentDashboardSerializer, ChatQueueSerializer
)


class IsAgentOrAdmin(permissions.BasePermission):
    """Permission for chat agents or admins."""
    
    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        if request.user.is_staff:
            return True
        return ChatAgent.objects.filter(user=request.user, is_active=True).exists()


class MessagePagination(PageNumberPagination):
    """Pagination for messages."""
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 100


class ChatAgentViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Chat Agents.
    
    Endpoints:
    - GET /api/chat/agents/ - List agents (admin only)
    - POST /api/chat/agents/ - Create agent (admin only)
    - GET /api/chat/agents/{id}/ - Get agent details
    - PUT /api/chat/agents/{id}/ - Update agent
    - DELETE /api/chat/agents/{id}/ - Delete agent
    - POST /api/chat/agents/me/status/ - Update own status
    - GET /api/chat/agents/me/dashboard/ - Get agent dashboard
    """
    serializer_class = ChatAgentSerializer
    permission_classes = [IsAgentOrAdmin]
    lookup_field = 'id'
    lookup_value_regex = r"[0-9a-f-]{36}"
    throttle_scope = 'chat_agents'
    
    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return ChatAgent.objects.none()
        user = self.request.user
        if user.is_staff:
            return ChatAgent.objects.all()
        return ChatAgent.objects.filter(user=user)
    
    @action(detail=False, methods=['get'])
    def me(self, request):
        """Get current user's agent profile."""
        agent = get_object_or_404(ChatAgent, user=request.user)
        serializer = self.get_serializer(agent)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'], url_path='me/status')
    def update_status(self, request):
        """Update agent's online/accepting status."""
        agent = get_object_or_404(ChatAgent, user=request.user)
        
        is_online = request.data.get('is_online')
        is_accepting = request.data.get('is_accepting_chats')
        
        if is_online is not None:
            agent.is_online = is_online
        if is_accepting is not None:
            agent.is_accepting_chats = is_accepting
        
        agent.last_active_at = timezone.now()
        agent.save(update_fields=['is_online', 'is_accepting_chats', 'last_active_at'])
        
        return Response(self.get_serializer(agent).data)
    
    @action(detail=False, methods=['get'], url_path='me/dashboard')
    def dashboard(self, request):
        """Get agent dashboard data."""
        agent = get_object_or_404(ChatAgent, user=request.user)
        
        # Get active chats for this agent
        active_chats = Conversation.objects.filter(
            agent=agent,
            status__in=[ConversationStatus.OPEN, ConversationStatus.ACTIVE, ConversationStatus.WAITING]
        ).order_by('-last_message_at')
        
        # Get queue stats
        waiting = Conversation.objects.filter(
            status=ConversationStatus.WAITING,
            agent__isnull=True
        ).count()
        
        active_total = Conversation.objects.filter(
            status__in=[ConversationStatus.OPEN, ConversationStatus.ACTIVE]
        ).count()
        
        available_agents = ChatAgent.objects.filter(
            is_online=True,
            is_accepting_chats=True,
            is_active=True
        ).filter(
            current_chat_count__lt=F('max_concurrent_chats')
        ).count()
        
        # Today's stats
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_stats = {
            'handled': Conversation.objects.filter(
                agent=agent, created_at__gte=today_start
            ).count(),
            'resolved': Conversation.objects.filter(
                agent=agent, resolved_at__gte=today_start
            ).count(),
            'avg_rating': agent.avg_rating,
            'messages_sent': Message.objects.filter(
                sender=request.user,
                is_from_customer=False,
                created_at__gte=today_start
            ).count()
        }
        
        return Response({
            'agent': ChatAgentSerializer(agent).data,
            'active_chats': ConversationSerializer(active_chats, many=True, context={'request': request}).data,
            'queue': {
                'waiting_count': waiting,
                'active_count': active_total,
                'available_agents': available_agents,
                'estimated_wait': waiting * 2 if available_agents > 0 else waiting * 5  # minutes estimate
            },
            'today_stats': today_stats
        })
    
    @action(detail=False, methods=['get'])
    def available(self, request):
        """List available agents (public)."""
        agents = ChatAgent.objects.filter(
            is_online=True,
            is_accepting_chats=True,
            is_active=True
        )
        serializer = ChatAgentPublicSerializer(agents, many=True)
        return Response(serializer.data)


class ConversationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Conversations.
    
    Endpoints:
    - GET /api/chat/conversations/ - List user's conversations
    - POST /api/chat/conversations/ - Start new conversation
    - GET /api/chat/conversations/{id}/ - Get conversation with messages
    - POST /api/chat/conversations/{id}/rate/ - Rate conversation
    - POST /api/chat/conversations/{id}/close/ - Close conversation
    - POST /api/chat/conversations/{id}/request_agent/ - Request human agent
    - POST /api/chat/conversations/{id}/assign/ - Assign agent (agents only)
    - POST /api/chat/conversations/{id}/transfer/ - Transfer to another agent
    """
    permission_classes = [permissions.IsAuthenticated]
    lookup_field = 'id'
    lookup_value_regex = r"[0-9a-f-]{36}"
    throttle_scope = 'chat'
    
    def get_serializer_class(self):
        if self.action == 'create':
            return ConversationCreateSerializer
        if self.action in ['retrieve', 'messages']:
            return ConversationDetailSerializer
        return ConversationSerializer
    
    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Conversation.objects.none()
        user = self.request.user
        
        # Agents can see all conversations
        is_agent = ChatAgent.objects.filter(user=user, is_active=True).exists()
        if is_agent or user.is_staff:
            queryset = Conversation.objects.all()
        else:
            queryset = Conversation.objects.filter(customer=user)
        
        # Filter by status
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by category
        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(category=category)
        
        return queryset.order_by('-last_message_at', '-created_at')

    @action(detail=True, methods=['post'], url_path='email', permission_classes=[IsAgentOrAdmin])
    def send_email(self, request, id=None):
        """Send an email reply tied to a conversation."""
        conversation = self.get_object()
        settings_obj = ChatSettings.get_settings()
        logger = logging.getLogger('bunoraa.chat')

        to_email = request.data.get('to_email') or conversation.customer_email
        subject = request.data.get('subject') or conversation.subject or 'Support'
        html_body = request.data.get('html_body')
        text_body = request.data.get('text_body') or request.data.get('body')

        if not to_email:
            return Response({'detail': 'Recipient email required'}, status=status.HTTP_400_BAD_REQUEST)
        if not (html_body or text_body):
            return Response({'detail': 'Email body required'}, status=status.HTTP_400_BAD_REQUEST)

        from_email = settings_obj.email_reply_from or getattr(settings, 'DEFAULT_FROM_EMAIL', '')
        if not from_email:
            return Response({'detail': 'Email sender not configured'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            from apps.email_service.models import EmailMessage, APIKey
            from apps.email_service.engine import QueueManager
        except Exception:
            return Response({'detail': 'Email service not available'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        api_key = APIKey.objects.filter(permission='send').first()
        if not api_key:
            return Response({'detail': 'Email API key missing'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if html_body and not text_body:
            from django.utils.html import strip_tags
            text_body = strip_tags(html_body)

        if not conversation.customer_email:
            conversation.customer_email = to_email
            conversation.save(update_fields=['customer_email'])

        message_id = f"chat_{conversation.id}_{uuid.uuid4().hex}@bunoraa.com"

        email_message = EmailMessage.objects.create(
            message_id=message_id,
            api_key=api_key,
            user=request.user,
            to_email=to_email,
            to_name=conversation.customer_name or '',
            from_email=from_email,
            from_name='Bunoraa Support',
            reply_to=settings_obj.support_inbox or '',
            subject=subject,
            html_body=html_body or '',
            text_body=text_body or '',
            status=EmailMessage.Status.QUEUED,
            metadata={
                'conversation_id': str(conversation.id),
                'channel': 'chat',
            }
        )

        QueueManager.enqueue(email_message)
        logger.info("[Chat] outbound email queued: %s", redact_payload({
            'conversation_id': str(conversation.id),
            'to_email': to_email,
            'message_id': message_id,
        }))

        # Store as chat message
        message = Message.objects.create(
            conversation=conversation,
            sender=request.user,
            is_from_customer=False,
            is_from_bot=False,
            message_type=MessageType.EMAIL,
            content=text_body or html_body or '',
            metadata={
                'email_message_id': message_id,
                'email_subject': subject,
                'email_to': to_email,
                'email_from': from_email,
            }
        )
        conversation.last_message_at = timezone.now()
        conversation.save(update_fields=['last_message_at'])

        return Response(MessageSerializer(message, context={'request': request}).data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get user's active conversation."""
        import logging
        logger = logging.getLogger('bunoraa.chat')
        
        try:
            user = request.user
            logger.debug(f"[Chat] active() called for user: {user}, is_authenticated: {user.is_authenticated}")
            
            if not user.is_authenticated:
                return Response({'detail': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
            
            conversation = Conversation.objects.filter(
                customer=user,
                status__in=[ConversationStatus.OPEN, ConversationStatus.ACTIVE, ConversationStatus.WAITING]
            ).order_by('-created_at').first()
            
            if conversation:
                logger.debug(f"[Chat] Found active conversation: {conversation.id}")
                serializer = ConversationDetailSerializer(conversation, context={'request': request})
                return Response(serializer.data)
            
            logger.debug(f"[Chat] No active conversation found for user {user.id}")
            # No active conversation is a valid state, so return 200 instead of 404.
            return Response(
                {
                    'conversation': None,
                    'detail': 'No active conversation',
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            logger.exception(f"[Chat] Error in active(): {e}")
            return Response(
                {'detail': f'Error retrieving active conversation: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def rate(self, request, id=None):
        """Rate a conversation."""
        conversation = self.get_object()
        
        # Only customers can rate
        if conversation.customer != request.user:
            return Response(
                {'detail': 'Only customers can rate conversations'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Only rate resolved conversations
        if conversation.status != ConversationStatus.RESOLVED:
            return Response(
                {'detail': 'Can only rate resolved conversations'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = ConversationRatingSerializer(conversation, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        return Response(ConversationSerializer(conversation, context={'request': request}).data)
    
    @action(detail=True, methods=['post'])
    def close(self, request, id=None):
        """Close a conversation."""
        conversation = self.get_object()
        
        # Verify permission
        is_agent = ChatAgent.objects.filter(user=request.user, is_active=True).exists()
        if not is_agent and conversation.customer != request.user:
            return Response(
                {'detail': 'Not authorized'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        conversation.status = ConversationStatus.CLOSED
        conversation.resolved_at = timezone.now()
        # Store resolution note in internal_notes
        resolution_note = request.data.get('resolution', 'Closed by user')
        if resolution_note:
            conversation.internal_notes = f"{conversation.internal_notes}\n[Closed] {resolution_note}".strip()
        conversation.save(update_fields=['status', 'resolved_at', 'internal_notes'])
        
        # Send rating request
        send_chat_rating_request.delay(str(conversation.id))
        
        return Response(ConversationSerializer(conversation, context={'request': request}).data)
    
    @action(detail=True, methods=['get'])
    def messages(self, request, id=None):
        """Get messages for a conversation."""
        conversation = self.get_object()
        
        # Get messages with pagination
        messages = conversation.messages.filter(is_deleted=False).order_by('created_at')
        
        # Apply pagination
        paginator = MessagePagination()
        page = paginator.paginate_queryset(messages, request)
        
        if page is not None:
            serializer = MessageSerializer(page, many=True, context={'request': request})
            return paginator.get_paginated_response(serializer.data)
        
        serializer = MessageSerializer(messages, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def attachments(self, request, id=None):
        """
        Upload attachment to a conversation.
        
        POST /api/v1/chat/conversations/{id}/attachments/
        
        Accepts multipart/form-data with 'file' field.
        Returns the created attachment details.
        """
        import logging
        logger = logging.getLogger('bunoraa.chat')
        
        try:
            conversation = self.get_object()
            
            # Verify permission - must be customer or assigned agent
            is_agent = ChatAgent.objects.filter(user=request.user, is_active=True).exists()
            if not is_agent and conversation.customer != request.user:
                return Response(
                    {'detail': 'Not authorized'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            file = request.FILES.get('file')
            if not file:
                return Response(
                    {'detail': 'No file provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check file size and type against settings
            chat_settings = ChatSettings.get_settings()
            max_size = int(chat_settings.max_file_size_mb or 0) * 1024 * 1024  # bytes

            if max_size and file.size > max_size:
                return Response(
                    {'detail': f'File too large. Maximum size is {chat_settings.max_file_size_mb}MB'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            import os
            file_name = os.path.basename(file.name or '')
            file_ext = os.path.splitext(file_name)[1].lstrip('.').lower()
            content_type = (file.content_type or '').lower()

            allowed_types = chat_settings.allowed_file_types or []
            if isinstance(allowed_types, str):
                allowed_types = [t.strip() for t in allowed_types.split(',') if t.strip()]
            allowed_types = [str(t).lower() for t in allowed_types if t]

            if allowed_types:
                allowed_mimes = [t for t in allowed_types if '/' in t]
                allowed_exts = [t.lstrip('.').lower() for t in allowed_types if '/' not in t]
                allowed = False
                if allowed_mimes and content_type in allowed_mimes:
                    allowed = True
                if allowed_exts and file_ext in allowed_exts:
                    allowed = True
                if not allowed:
                    return Response(
                        {'detail': 'File type not allowed.'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Create a message with the attachment
            message_type = MessageType.IMAGE if content_type.startswith('image/') else MessageType.FILE

            message = Message.objects.create(
                conversation=conversation,
                sender=request.user,
                content=f"[Attachment: {file_name or file.name}]",
                is_from_customer=(conversation.customer == request.user),
                message_type=message_type
            )
            
            # Create attachment
            attachment = MessageAttachment.objects.create(
                message=message,
                file=file,
                file_name=file_name or file.name,
                file_size=file.size,
                file_type=content_type or 'application/octet-stream'
            )
            
            # Update conversation last activity
            conversation.last_message_at = timezone.now()
            conversation.save(update_fields=['last_message_at'])
            
            return Response({
                'id': str(attachment.id),
                'message_id': str(message.id),
                'file_name': attachment.file_name,
                'file_size': attachment.file_size,
                'file_type': attachment.file_type,
                'file_url': attachment.file.url if attachment.file else None,
                'created_at': message.created_at.isoformat()
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.exception(f"[Chat] Error uploading attachment: {e}")
            return Response(
                {'detail': f'Failed to upload attachment: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def request_agent(self, request, id=None):
        """Request a human agent for bot-handled conversation."""
        conversation = self.get_object()
        
        if conversation.customer != request.user:
            return Response(
                {'detail': 'Not authorized'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        conversation.request_human_agent()

        settings_obj = ChatSettings.get_settings()
        available_agent = ChatService.find_available_agent(conversation.category)
        if not available_agent:
            Message.objects.create(
                conversation=conversation,
                sender=None,
                is_from_customer=False,
                is_from_bot=True,
                message_type='system',
                content=settings_obj.wait_message
            )

        return Response(ConversationSerializer(conversation, context={'request': request}).data)

    @action(detail=True, methods=['post'], url_path='internal-notes', permission_classes=[IsAgentOrAdmin])
    def internal_notes(self, request, id=None):
        """Update internal notes for a conversation."""
        conversation = self.get_object()
        notes = request.data.get('internal_notes', '')
        conversation.internal_notes = notes
        conversation.save(update_fields=['internal_notes'])
        return Response(ConversationSerializer(conversation, context={'request': request}).data)
    
    @action(detail=True, methods=['post'])
    def assign(self, request, id=None):
        """Assign agent to conversation (for agents)."""
        conversation = self.get_object()
        
        # Verify agent
        agent = ChatAgent.objects.filter(user=request.user, is_active=True).first()
        if not agent:
            return Response(
                {'detail': 'Not an agent'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Check capacity
        if agent.current_chat_count >= agent.max_concurrent_chats:
            return Response(
                {'detail': 'Maximum chat capacity reached'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        assigned_agent, error = ChatService.assign_agent(str(conversation.id), str(agent.id))
        
        if error:
            return Response({'detail': error}, status=status.HTTP_400_BAD_REQUEST)

        conversation.refresh_from_db()
        return Response(ConversationSerializer(conversation, context={'request': request}).data)
    
    @action(detail=True, methods=['post'])
    def transfer(self, request, id=None):
        """Transfer conversation to another agent."""
        conversation = self.get_object()
        
        # Verify current agent
        current_agent = ChatAgent.objects.filter(user=request.user, is_active=True).first()
        if not current_agent and not request.user.is_staff:
            return Response(
                {'detail': 'Not authorized'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        target_agent_id = request.data.get('agent_id')
        if not target_agent_id:
            return Response(
                {'detail': 'Target agent ID required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        target_agent = get_object_or_404(ChatAgent, id=target_agent_id)
        
        # Check target capacity
        if target_agent.current_chat_count >= target_agent.max_concurrent_chats:
            return Response(
                {'detail': 'Target agent at capacity'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Transfer
        old_agent = conversation.agent
        conversation.agent = target_agent
        if old_agent:
            note = f"Transferred from {old_agent.user.get_full_name() or old_agent.user.email}"
            conversation.internal_notes = f"{conversation.internal_notes}\n{note}".strip()
        conversation.save(update_fields=['agent', 'internal_notes'])
        
        # Update counts
        if old_agent:
            old_agent.current_chat_count = max(0, old_agent.current_chat_count - 1)
            old_agent.save()
        target_agent.current_chat_count += 1
        target_agent.save()
        
        return Response(ConversationSerializer(conversation, context={'request': request}).data)
    
    @action(detail=False, methods=['get'])
    def queue(self, request):
        """Get waiting conversations (for agents)."""
        if not ChatAgent.objects.filter(user=request.user, is_active=True).exists():
            return Response({'detail': 'Not an agent'}, status=status.HTTP_403_FORBIDDEN)
        
        waiting = Conversation.objects.filter(
            status=ConversationStatus.WAITING,
            agent__isnull=True
        ).order_by('created_at')
        
        return Response(ConversationSerializer(waiting, many=True, context={'request': request}).data)


class MessageViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Messages.
    
    Endpoints:
    - GET /api/chat/messages/?conversation={id} - List messages
    - POST /api/chat/messages/ - Send message
    - PUT /api/chat/messages/{id}/ - Edit message
    - DELETE /api/chat/messages/{id}/ - Delete message
    - POST /api/chat/messages/{id}/react/ - Add reaction
    """
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = MessagePagination
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    lookup_field = 'id'
    lookup_value_regex = r"[0-9a-f-]{36}"
    throttle_scope = 'chat_messages'

    def _user_can_access_conversation(self, conversation):
        if self.request.user.is_staff:
            return True
        if conversation.customer_id == self.request.user.id:
            return True
        return ChatAgent.objects.filter(user=self.request.user, is_active=True).exists()

    def _ensure_message_access(self, message):
        if not self._user_can_access_conversation(message.conversation):
            raise PermissionDenied('Not authorized to access this conversation')
    
    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return Message.objects.none()
        conversation_id = self.request.query_params.get('conversation')
        if conversation_id:
            try:
                conversation = Conversation.objects.get(id=conversation_id)
            except Conversation.DoesNotExist:
                return Message.objects.none()

            if not self._user_can_access_conversation(conversation):
                return Message.objects.none()

            return Message.objects.filter(conversation_id=conversation_id).order_by('created_at')

        if self.request.user.is_staff or ChatAgent.objects.filter(user=self.request.user, is_active=True).exists():
            return Message.objects.all()

        return Message.objects.filter(conversation__customer=self.request.user)
    
    def get_serializer_class(self):
        if self.action == 'create':
            return MessageCreateSerializer
        return MessageSerializer
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        message = serializer.save()
        
        conversation = message.conversation
        conversation.last_message_at = timezone.now()
        conversation.save(update_fields=['last_message_at'])
        
        # Trigger AI response if bot handling
        if conversation.is_bot_handling and message.is_from_customer:
            generate_ai_response.delay(str(conversation.id), message.content)
        
        return Response(
            MessageSerializer(message, context={'request': request}).data,
            status=status.HTTP_201_CREATED
        )

    def update(self, request, *args, **kwargs):
        message = self.get_object()
        self._ensure_message_access(message)
        return super().update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        message = self.get_object()
        self._ensure_message_access(message)
        return super().destroy(request, *args, **kwargs)
    
    @action(detail=True, methods=['post'])
    def react(self, request, id=None):
        """Add reaction to message."""
        message = self.get_object()
        self._ensure_message_access(message)
        
        emoji = request.data.get('emoji')
        if not emoji:
            return Response(
                {'detail': 'Emoji required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user_id = str(request.user.id)
        reactions = message.reactions or {}
        
        # Toggle reaction
        if emoji in reactions and user_id in reactions[emoji]:
            reactions[emoji].remove(user_id)
            if not reactions[emoji]:
                del reactions[emoji]
        else:
            if emoji not in reactions:
                reactions[emoji] = []
            reactions[emoji].append(user_id)
        
        message.reactions = reactions
        message.save()
        
        return Response(MessageSerializer(message, context={'request': request}).data)
    
    @action(detail=True, methods=['post'])
    def mark_read(self, request, id=None):
        """Mark message as read."""
        message = self.get_object()
        self._ensure_message_access(message)
        
        if not message.is_read:
            message.is_read = True
            message.read_at = timezone.now()
            message.save()
        
        return Response({'status': 'marked_read'})


class CannedResponseViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Canned Responses.
    
    Endpoints:
    - GET /api/chat/canned-responses/ - List responses (own + global)
    - POST /api/chat/canned-responses/ - Create response
    - PUT /api/chat/canned-responses/{id}/ - Update response
    - DELETE /api/chat/canned-responses/{id}/ - Delete response
    - POST /api/chat/canned-responses/{id}/use/ - Increment use count
    """
    serializer_class = CannedResponseSerializer
    permission_classes = [IsAgentOrAdmin]
    lookup_field = 'id'
    lookup_value_regex = r"[0-9a-f-]{36}"
    throttle_scope = 'chat_canned'

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return CannedResponse.objects.none()
        agent = ChatAgent.objects.filter(user=self.request.user).first()
        
        if self.request.user.is_staff:
            return CannedResponse.objects.filter(is_active=True)
        
        return CannedResponse.objects.filter(
            Q(is_global=True) | Q(agent=agent),
            is_active=True
        )
    
    def perform_create(self, serializer):
        agent = ChatAgent.objects.filter(user=self.request.user).first()
        is_global = serializer.validated_data.get('is_global', False)
        if self.request.user.is_staff and is_global:
            serializer.save(agent=None, is_global=True)
        else:
            serializer.save(agent=agent, is_global=False)
    
    @action(detail=True, methods=['post'])
    def use(self, request, id=None):
        """Record use of canned response."""
        response = self.get_object()
        response.use_count += 1
        response.last_used_at = timezone.now()
        response.save(update_fields=['use_count', 'last_used_at'])
        return Response({'use_count': response.use_count})
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Search canned responses by shortcut or content."""
        query = request.query_params.get('q', '')
        
        if not query:
            return Response([])
        
        responses = self.get_queryset().filter(
            Q(shortcut__icontains=query) |
            Q(title__icontains=query) |
            Q(content__icontains=query)
        )[:10]
        
        return Response(CannedResponseSerializer(responses, many=True).data)


class ChatEmailInboundView(APIView):
    """Inbound email webhook to create or append to conversations."""

    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'chat_email_inbound'

    def _parse_raw_email(self, request):
        if request.content_type and 'application/json' in request.content_type:
            raw = request.data.get('raw') if isinstance(request.data, dict) else None
            if isinstance(raw, str):
                return raw.encode('utf-8')
            if isinstance(raw, bytes):
                return raw
        return request.body or b''

    def _extract_body(self, message):
        text_body = ''
        html_body = ''
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                disposition = part.get_content_disposition()
                if disposition == 'attachment':
                    continue
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                charset = part.get_content_charset() or 'utf-8'
                try:
                    decoded = payload.decode(charset, errors='replace')
                except Exception:
                    decoded = payload.decode('utf-8', errors='replace')
                if content_type == 'text/plain' and not text_body:
                    text_body = decoded.strip()
                if content_type == 'text/html' and not html_body:
                    html_body = decoded.strip()
        else:
            payload = message.get_payload(decode=True) or b''
            charset = message.get_content_charset() or 'utf-8'
            try:
                text_body = payload.decode(charset, errors='replace').strip()
            except Exception:
                text_body = payload.decode('utf-8', errors='replace').strip()
        return text_body, html_body

    def post(self, request):
        logger = logging.getLogger('bunoraa.chat')
        secret = getattr(settings, 'CHAT_EMAIL_WEBHOOK_SECRET', '')
        provided = request.headers.get('X-Chat-Email-Secret') or request.query_params.get('secret')
        if secret and provided != secret:
            return Response({'detail': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)

        raw_bytes = self._parse_raw_email(request)
        if not raw_bytes:
            return Response({'detail': 'Missing raw email payload'}, status=status.HTTP_400_BAD_REQUEST)

        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        from_name, from_email = parseaddr(message.get('From', ''))
        to_value = message.get('To', '')
        subject = message.get('Subject', '').strip()
        message_id = (message.get('Message-ID', '') or '').strip('<>')
        in_reply_to = (message.get('In-Reply-To', '') or '').strip('<>')
        references = message.get('References', '') or ''
        reference_ids = [ref.strip('<>') for ref in references.split() if ref]

        if not from_email:
            return Response({'detail': 'Missing From address'}, status=status.HTTP_400_BAD_REQUEST)

        settings_obj = ChatSettings.get_settings()
        if settings_obj.support_inbox:
            if settings_obj.support_inbox.lower() not in (to_value or '').lower():
                return Response({'detail': 'Email not addressed to support inbox'}, status=status.HTTP_400_BAD_REQUEST)

        text_body, html_body = self._extract_body(message)
        if not text_body and html_body:
            from django.utils.html import strip_tags
            text_body = strip_tags(html_body).strip()

        if not text_body:
            text_body = '[No content]'

        thread_ids = [id for id in [in_reply_to] + reference_ids if id]

        User = get_user_model()
        user = User.objects.filter(email=from_email).first()
        if not user:
            user = User.objects.create_user(email=from_email, password=None, is_active=False)
            user.set_unusable_password()
            user.save(update_fields=['password'])

        conversation = None
        if thread_ids:
            prior_message = Message.objects.filter(
                metadata__email_message_id__in=thread_ids
            ).order_by('-created_at').first()
            if prior_message:
                conversation = prior_message.conversation
                if not conversation.customer_email:
                    conversation.customer_email = from_email
                    conversation.save(update_fields=['customer_email'])

        if not conversation:
            status_value = ConversationStatus.OPEN
            if settings_obj.business_hours_enabled and not settings_obj.is_within_business_hours():
                status_value = ConversationStatus.WAITING
            conversation = Conversation.objects.create(
                customer=user,
                category=ConversationCategory.GENERAL,
                status=status_value,
                subject=subject or 'Email Support',
                source='email',
                is_bot_handling=False,
                customer_email=from_email,
                customer_name=from_name or from_email,
                initial_message=text_body
            )
            ChatService.notify_agents_new_chat(conversation)

        chat_message = Message.objects.create(
            conversation=conversation,
            sender=user,
            is_from_customer=True,
            is_from_bot=False,
            message_type=MessageType.EMAIL,
            content=text_body,
            metadata={
                'email_message_id': message_id,
                'email_in_reply_to': in_reply_to,
                'email_references': reference_ids,
                'email_subject': subject,
                'email_from': from_email,
                'email_to': to_value,
            }
        )

        conversation.last_message_at = timezone.now()
        conversation.save(update_fields=['last_message_at'])

        logger.info("[Chat] inbound email processed: %s", redact_payload({
            'conversation_id': str(conversation.id),
            'from_email': from_email,
            'message_id': message_id,
        }))

        return Response(
            {'conversation_id': str(conversation.id), 'message_id': str(chat_message.id)},
            status=status.HTTP_201_CREATED
        )


class ChatSettingsViewSet(viewsets.ViewSet):
    """
    ViewSet for Chat Settings.
    
    Endpoints:
    - GET /api/chat/settings/ - Get settings
    - PUT /api/chat/settings/ - Update settings (admin only)
    - GET /api/chat/settings/public/ - Get public settings
    """
    permission_classes = [IsAgentOrAdmin]
    throttle_scope = 'chat_settings'
    
    def list(self, request):
        """Get chat settings."""
        settings = ChatSettings.get_settings()
        serializer = ChatSettingsSerializer(settings)
        return Response(serializer.data)
    
    def update(self, request, pk=None):
        """Update chat settings (admin only)."""
        if not request.user.is_staff:
            return Response(
                {'detail': 'Admin only'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        settings = ChatSettings.get_settings()
        serializer = ChatSettingsSerializer(settings, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'], permission_classes=[permissions.AllowAny])
    def public(self, request):
        """Get public chat settings (no auth required)."""
        settings = ChatSettings.get_settings()

        return Response({
            'is_chat_enabled': settings.is_chat_enabled,
            'welcome_message': settings.welcome_message,
            'wait_message': settings.wait_message,
            'offline_message': settings.offline_message,
            'ai_enabled': settings.ai_enabled,
            'allowed_file_types': settings.allowed_file_types,
            'max_file_size_mb': settings.max_file_size_mb,
            'business_hours_enabled': settings.business_hours_enabled,
            'is_within_business_hours': settings.is_within_business_hours()
        })


class ChatAnalyticsViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for Chat Analytics.
    
    Endpoints:
    - GET /api/chat/analytics/ - List analytics
    - GET /api/chat/analytics/today/ - Today's analytics
    - GET /api/chat/analytics/summary/ - Period summary
    """
    serializer_class = ChatAnalyticsSerializer
    permission_classes = [IsAgentOrAdmin]
    lookup_field = 'date'
    lookup_value_regex = r"\d{4}-\d{2}-\d{2}"
    throttle_scope = 'chat_analytics'

    def get_queryset(self):
        if getattr(self, 'swagger_fake_view', False):
            return ChatAnalytics.objects.none()
        return ChatAnalytics.objects.all().order_by('-date')
    
    @action(detail=False, methods=['get'])
    def today(self, request):
        """Get today's analytics."""
        today = timezone.now().date()
        
        # Update analytics first
        analytics = ChatAnalyticsService.update_daily_analytics(today)
        
        return Response(ChatAnalyticsSerializer(analytics).data)
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get summary for period."""
        days = int(request.query_params.get('days', 7))
        
        start_date = timezone.now().date() - timezone.timedelta(days=days)
        
        analytics = ChatAnalytics.objects.filter(date__gte=start_date)
        
        # Aggregate
        from django.db.models import Sum, Avg
        
        summary = analytics.aggregate(
            total_conversations=Sum('total_conversations'),
            resolved_conversations=Sum('resolved_conversations'),
            total_messages=Sum('total_messages'),
            avg_rating=Avg('avg_rating'),
            avg_response_time=Avg('avg_first_response_seconds')
        )
        
        return Response({
            'period_days': days,
            'start_date': start_date.isoformat(),
            'end_date': timezone.now().date().isoformat(),
            **summary,
            'daily': ChatAnalyticsSerializer(analytics, many=True).data
        })
