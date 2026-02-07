"""
Chat Services for Bunoraa Live Chat System

Provides business logic for:
- Conversation management
- Agent routing
- AI chatbot integration
- Analytics
"""
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from django.conf import settings
from django.utils import timezone
from django.db.models import Count, Avg, Q, F
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

if TYPE_CHECKING:
    from apps.chat.models import ChatAgent

logger = logging.getLogger('bunoraa.chat')


class ChatService:
    """Main chat service for managing conversations."""

    @staticmethod
    def start_conversation(
        customer,
        category: str = 'general',
        subject: str = '',
        initial_message: str = '',
        source: str = 'website',
        order_reference: str = None
    ):
        """Start a new chat conversation."""
        from apps.chat.models import Conversation, Message, ChatSettings, ConversationStatus
        
        chat_settings = ChatSettings.get_settings()
        
        # Check if chat is enabled
        if not chat_settings.is_chat_enabled:
            return None, 'Chat is currently unavailable.'
        
        # Create conversation
        conversation = Conversation.objects.create(
            customer=customer,
            category=category,
            subject=subject,
            initial_message=initial_message,
            source=source,
            order_reference=order_reference,
            customer_email=customer.email,
            customer_name=customer.get_full_name() or customer.email,
            is_bot_handling=chat_settings.ai_enabled,
            status=ConversationStatus.OPEN
        )
        
        # Create initial customer message if provided
        if initial_message:
            Message.objects.create(
                conversation=conversation,
                sender=customer,
                is_from_customer=True,
                content=initial_message
            )
            conversation.last_message_at = timezone.now()
            conversation.save(update_fields=['last_message_at'])
        
        # Send welcome message from bot
        if chat_settings.ai_enabled:
            Message.objects.create(
                conversation=conversation,
                sender=None,
                is_from_customer=False,
                is_from_bot=True,
                content=chat_settings.welcome_message
            )
        
        # Notify agents if configured
        if chat_settings.notify_on_new_chat:
            ChatService.notify_agents_new_chat(conversation)
        
        return conversation, None

    @staticmethod
    def get_or_create_active_conversation(customer):
        """Get customer's active conversation or create new one."""
        from apps.chat.models import Conversation, ConversationStatus
        
        # Look for open/active conversation
        active_statuses = [
            ConversationStatus.OPEN,
            ConversationStatus.WAITING,
            ConversationStatus.ACTIVE
        ]
        
        conversation = Conversation.objects.filter(
            customer=customer,
            status__in=active_statuses
        ).order_by('-created_at').first()
        
        if conversation:
            return conversation, False
        
        # Create new conversation
        conversation, error = ChatService.start_conversation(customer)
        return conversation, True

    @staticmethod
    def assign_agent(conversation_id: str, agent_id: str = None):
        """Assign an agent to a conversation."""
        from apps.chat.models import Conversation, ChatAgent, ConversationStatus
        
        conversation = Conversation.objects.get(id=conversation_id)
        
        if agent_id:
            agent = ChatAgent.objects.get(id=agent_id)
        else:
            # Auto-assign to available agent
            agent = ChatService.find_available_agent(
                category=conversation.category
            )
        
        if not agent:
            return None, 'No agents available'
        
        # Assign
        conversation.agent = agent
        conversation.status = ConversationStatus.ACTIVE
        conversation.is_bot_handling = False
        conversation.save()
        
        # Update agent's chat count
        agent.current_chat_count += 1
        agent.save(update_fields=['current_chat_count'])
        
        # Notify via WebSocket
        ChatService.notify_agent_assigned(conversation, agent)
        
        return agent, None

    @staticmethod
    def find_available_agent(category: str = None) -> Optional["ChatAgent"]:
        """Find the best available agent."""
        from apps.chat.models import ChatAgent
        
        agents = ChatAgent.objects.filter(
            is_online=True,
            is_accepting_chats=True
        ).annotate(
            available_slots=F('max_concurrent_chats') - F('current_chat_count')
        ).filter(available_slots__gt=0)
        
        # Filter by category if specified
        if category:
            agents = agents.filter(
                Q(categories=[]) | Q(categories__contains=[category])
            )
        
        # Order by: least loaded, then highest rated
        agents = agents.order_by('-available_slots', '-avg_rating')
        
        return agents.first()

    @staticmethod
    def notify_agents_new_chat(conversation):
        """Notify available agents of new chat."""
        channel_layer = get_channel_layer()
        
        # Skip if channel layer is not configured (running without Daphne/Channels)
        if channel_layer is None:
            logger.debug("Channel layer not configured, skipping notify_agents_new_chat")
            return
        
        try:
            async_to_sync(channel_layer.group_send)(
                'chat_agents_dashboard',
                {
                    'type': 'new_chat_notification',
                    'conversation': {
                        'id': str(conversation.id),
                        'category': conversation.category,
                        'customer_name': conversation.customer_name,
                        'subject': conversation.subject,
                        'initial_message': conversation.initial_message[:100] if conversation.initial_message else '',
                        'created_at': conversation.created_at.isoformat()
                    }
                }
            )
        except Exception as e:
            logger.warning(f"Failed to notify agents of new chat: {e}")

    @staticmethod
    def notify_agent_assigned(conversation, agent):
        """Notify WebSocket of agent assignment."""
        channel_layer = get_channel_layer()
        
        # Skip if channel layer is not configured
        if channel_layer is None:
            logger.debug("Channel layer not configured, skipping notify_agent_assigned")
            return
        
        try:
            # Notify the conversation room
            async_to_sync(channel_layer.group_send)(
                f'chat_{conversation.id}',
                {
                    'type': 'agent_assigned',
                    'agent_id': str(agent.id),
                    'agent_name': agent.user.get_full_name() or agent.user.email,
                    'timestamp': timezone.now().isoformat()
                }
            )
            
            # Notify agents dashboard
            async_to_sync(channel_layer.group_send)(
                'chat_agents_dashboard',
                {
                    'type': 'chat_assigned',
                    'conversation_id': str(conversation.id),
                    'agent_id': str(agent.id)
                }
            )
        except Exception as e:
            logger.warning(f"Failed to notify agent assigned: {e}")


class AIService:
    """AI chatbot service for automated responses."""

    @staticmethod
    def generate_response(conversation_id: str, customer_message: str) -> Optional[str]:
        """Generate AI response to customer message."""
        from apps.chat.models import Conversation, Message, ChatSettings
        import openai
        
        settings_obj = ChatSettings.get_settings()
        
        if not settings_obj.ai_enabled:
            return None
        
        conversation = Conversation.objects.get(id=conversation_id)
        
        # Check if we've exceeded max AI responses
        ai_message_count = Message.objects.filter(
            conversation=conversation,
            is_from_bot=True
        ).count()
        
        if ai_message_count >= settings_obj.max_ai_responses_before_handoff:
            # Request human handoff
            conversation.request_human_agent()
            return "I've reached the limit of what I can help with. Let me connect you with a human agent."
        
        try:
            # Build conversation history for context
            messages = AIService._build_message_history(conversation)
            messages.append({
                'role': 'user',
                'content': customer_message
            })
            
            # Call OpenAI
            client = openai.OpenAI(api_key=getattr(settings, 'OPENAI_API_KEY', ''))
            
            response = client.chat.completions.create(
                model=settings_obj.ai_model,
                messages=[
                    {'role': 'system', 'content': settings_obj.ai_system_prompt}
                ] + messages,
                temperature=settings_obj.ai_temperature,
                max_tokens=settings_obj.ai_max_tokens
            )
            
            ai_response = response.choices[0].message.content
            
            # Save AI response as message
            Message.objects.create(
                conversation=conversation,
                sender=None,
                is_from_customer=False,
                is_from_bot=True,
                content=ai_response
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return None

    @staticmethod
    def _build_message_history(conversation, limit: int = 10) -> List[Dict[str, str]]:
        """Build message history for AI context."""
        from apps.chat.models import Message
        
        messages = Message.objects.filter(
            conversation=conversation,
            is_deleted=False
        ).order_by('-created_at')[:limit]
        
        history = []
        for msg in reversed(messages):
            role = 'user' if msg.is_from_customer else 'assistant'
            history.append({
                'role': role,
                'content': msg.content
            })
        
        return history

    @staticmethod
    def categorize_conversation(message: str) -> str:
        """Use AI to categorize the conversation."""
        from apps.chat.models import ConversationCategory
        import openai
        
        try:
            client = openai.OpenAI(api_key=getattr(settings, 'OPENAI_API_KEY', ''))
            
            categories = [c[0] for c in ConversationCategory.choices]
            
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {
                        'role': 'system',
                        'content': f'Categorize the following customer message into one of these categories: {", ".join(categories)}. Respond with just the category code.'
                    },
                    {'role': 'user', 'content': message}
                ],
                temperature=0
            )
            
            category = response.choices[0].message.content.strip().lower()
            
            if category in categories:
                return category
            return ConversationCategory.GENERAL
            
        except Exception as e:
            logger.error(f"AI categorization failed: {e}")
            return ConversationCategory.GENERAL


class ChatAnalyticsService:
    """Service for chat analytics and reporting."""

    @staticmethod
    def update_daily_analytics(date=None):
        """Update or create daily analytics record."""
        from apps.chat.models import (
            ChatAnalytics, Conversation, Message, 
            ConversationStatus, ChatAgent
        )
        from django.db.models import Avg, Count
        
        if date is None:
            date = timezone.now().date()
        
        # Calculate metrics
        day_start = timezone.make_aware(
            timezone.datetime.combine(date, timezone.datetime.min.time())
        )
        day_end = timezone.make_aware(
            timezone.datetime.combine(date, timezone.datetime.max.time())
        )
        
        conversations = Conversation.objects.filter(
            created_at__range=(day_start, day_end)
        )
        
        messages = Message.objects.filter(
            created_at__range=(day_start, day_end)
        )
        
        # Volume metrics
        total_conversations = conversations.count()
        new_conversations = conversations.count()
        resolved_conversations = conversations.filter(
            status=ConversationStatus.RESOLVED
        ).count()
        
        # Message metrics
        total_messages = messages.count()
        customer_messages = messages.filter(is_from_customer=True).count()
        bot_messages = messages.filter(is_from_bot=True).count()
        agent_messages = total_messages - customer_messages - bot_messages
        
        # Performance metrics
        resolved = conversations.filter(
            first_response_at__isnull=False,
            resolved_at__isnull=False
        )
        
        avg_first_response = 0
        avg_resolution = 0
        avg_rating = 0
        
        if resolved.exists():
            # Calculate averages
            response_times = []
            resolution_times = []
            
            for conv in resolved:
                if conv.first_response_at:
                    rt = (conv.first_response_at - conv.started_at).total_seconds()
                    response_times.append(rt)
                if conv.resolved_at:
                    rest = (conv.resolved_at - conv.started_at).total_seconds()
                    resolution_times.append(rest)
            
            if response_times:
                avg_first_response = sum(response_times) / len(response_times)
            if resolution_times:
                avg_resolution = sum(resolution_times) / len(resolution_times)
        
        rated = conversations.filter(rating__isnull=False)
        if rated.exists():
            avg_rating = rated.aggregate(avg=Avg('rating'))['avg'] or 0
        
        # Category breakdown
        category_breakdown = dict(
            conversations.values('category').annotate(
                count=Count('id')
            ).values_list('category', 'count')
        )
        
        # Create or update analytics record
        analytics, _ = ChatAnalytics.objects.update_or_create(
            date=date,
            defaults={
                'total_conversations': total_conversations,
                'new_conversations': new_conversations,
                'resolved_conversations': resolved_conversations,
                'total_messages': total_messages,
                'customer_messages': customer_messages,
                'agent_messages': agent_messages,
                'bot_messages': bot_messages,
                'avg_first_response_seconds': avg_first_response,
                'avg_resolution_time_seconds': avg_resolution,
                'avg_rating': avg_rating,
                'category_breakdown': category_breakdown,
            }
        )
        
        return analytics

    @staticmethod
    def get_agent_performance(agent_id: str, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for an agent."""
        from apps.chat.models import Conversation, Message, ChatAgent
        from django.db.models import Avg, Count
        
        start_date = timezone.now() - timezone.timedelta(days=days)
        
        agent = ChatAgent.objects.get(id=agent_id)
        
        conversations = Conversation.objects.filter(
            agent=agent,
            created_at__gte=start_date
        )
        
        total = conversations.count()
        resolved = conversations.filter(status='resolved').count()
        
        # Average response time
        with_response = conversations.filter(first_response_at__isnull=False)
        response_times = []
        for conv in with_response:
            rt = (conv.first_response_at - conv.started_at).total_seconds()
            response_times.append(rt)
        
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        
        # Average rating
        rated = conversations.filter(rating__isnull=False)
        avg_rating = rated.aggregate(avg=Avg('rating'))['avg'] or 0
        
        # Message count
        messages = Message.objects.filter(
            conversation__agent=agent,
            is_from_customer=False,
            is_from_bot=False,
            created_at__gte=start_date
        ).count()
        
        return {
            'agent_id': str(agent_id),
            'agent_name': agent.user.get_full_name() or agent.user.email,
            'total_conversations': total,
            'resolved_conversations': resolved,
            'resolution_rate': (resolved / total * 100) if total > 0 else 0,
            'avg_response_time_seconds': avg_response,
            'avg_rating': float(avg_rating),
            'total_messages': messages,
            'period_days': days
        }
