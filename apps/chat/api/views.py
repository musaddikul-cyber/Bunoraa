"""
DRF ViewSets for Bunoraa Chat System API
"""
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.pagination import PageNumberPagination
from django.utils import timezone
from django.db.models import Q, Count
from django.shortcuts import get_object_or_404

from apps.chat.models import (
    ChatAgent, Conversation, Message, MessageAttachment,
    CannedResponse, ChatSettings, ChatAnalytics,
    ConversationStatus, ConversationCategory
)
from apps.chat.services import ChatService, ChatAnalyticsService
from apps.chat.tasks import generate_ai_response, send_chat_rating_request

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
    
    def get_queryset(self):
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
        
        agent.last_activity = timezone.now()
        agent.save()
        
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
            is_accepting_chats=True
        ).annotate(
            available=Count('id') - Count('conversations_as_agent')
        ).filter(available__gt=0).count()
        
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
    
    def get_serializer_class(self):
        if self.action == 'create':
            return ConversationCreateSerializer
        if self.action in ['retrieve', 'messages']:
            return ConversationDetailSerializer
        return ConversationSerializer
    
    def get_queryset(self):
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
            return Response({'detail': 'No active conversation'}, status=status.HTTP_404_NOT_FOUND)
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
            max_size = chat_settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
            
            if file.size > max_size:
                return Response(
                    {'detail': f'File too large. Maximum size is {chat_settings.max_file_size_mb}MB'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check file type
            file_ext = file.name.split('.')[-1].lower() if '.' in file.name else ''
            allowed_types = [t.strip().lower().lstrip('.') for t in (chat_settings.allowed_file_types or 'jpg,jpeg,png,gif,pdf,doc,docx').split(',')]
            
            if file_ext and file_ext not in allowed_types:
                return Response(
                    {'detail': f'File type not allowed. Allowed types: {chat_settings.allowed_file_types}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create a message with the attachment
            message = Message.objects.create(
                conversation=conversation,
                sender=request.user,
                content=f"[Attachment: {file.name}]",
                is_from_customer=(conversation.customer == request.user)
            )
            
            # Create attachment
            attachment = MessageAttachment.objects.create(
                message=message,
                file=file,
                file_name=file.name,
                file_size=file.size,
                file_type=file.content_type or 'application/octet-stream'
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
        conversation.transferred_from = old_agent
        conversation.save()
        
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
    
    def get_queryset(self):
        conversation_id = self.request.query_params.get('conversation')
        if conversation_id:
            return Message.objects.filter(
                conversation_id=conversation_id
            ).order_by('created_at')
        return Message.objects.none()
    
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
    
    @action(detail=True, methods=['post'])
    def react(self, request, id=None):
        """Add reaction to message."""
        message = self.get_object()
        
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
    
    def get_queryset(self):
        agent = ChatAgent.objects.filter(user=self.request.user).first()
        
        if self.request.user.is_staff:
            return CannedResponse.objects.filter(is_active=True)
        
        return CannedResponse.objects.filter(
            Q(is_global=True) | Q(agent=agent),
            is_active=True
        )
    
    def perform_create(self, serializer):
        agent = ChatAgent.objects.filter(user=self.request.user).first()
        serializer.save(agent=agent)
    
    @action(detail=True, methods=['post'])
    def use(self, request, id=None):
        """Record use of canned response."""
        response = self.get_object()
        response.use_count += 1
        response.last_used_at = timezone.now()
        response.save()
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


class ChatSettingsViewSet(viewsets.ViewSet):
    """
    ViewSet for Chat Settings.
    
    Endpoints:
    - GET /api/chat/settings/ - Get settings
    - PUT /api/chat/settings/ - Update settings (admin only)
    - GET /api/chat/settings/public/ - Get public settings
    """
    permission_classes = [IsAgentOrAdmin]
    
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
    
    @action(detail=False, methods=['get'])
    def public(self, request):
        """Get public chat settings (no auth required)."""
        settings = ChatSettings.get_settings()
        
        return Response({
            'is_chat_enabled': settings.is_chat_enabled,
            'welcome_message': settings.welcome_message,
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
    
    def get_queryset(self):
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
