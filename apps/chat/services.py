"""
Chat Services for Bunoraa Live Chat System

Provides business logic for:
- Conversation management
- Agent routing
- AI chatbot integration
- Analytics
"""
import logging
import re
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.db.models import Count, Avg, Q, F
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.cache import cache

if TYPE_CHECKING:
    from apps.chat.models import ChatAgent

logger = logging.getLogger('bunoraa.chat')

EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"\b(\+?\d[\d\s\-().]{6,}\d)\b")


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
        
        # Always send a welcome message from the support team
        welcome_text = (
            chat_settings.welcome_message
            if chat_settings.welcome_message
            else "Hello! How can we help you today?"
        )
        Message.objects.create(
            conversation=conversation,
            sender=None,
            is_from_customer=False,
            is_from_bot=True,
            content=welcome_text,
        )
        conversation.last_message_at = timezone.now()
        conversation.save(update_fields=['last_message_at'])

        # Create initial customer message if provided
        if initial_message:
            Message.objects.create(
                conversation=conversation,
                sender=customer,
                is_from_customer=True,
                content=initial_message,
            )
            conversation.last_message_at = timezone.now()
            conversation.save(update_fields=['last_message_at'])
        
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

        with transaction.atomic():
            conversation = Conversation.objects.select_for_update().get(id=conversation_id)

            if conversation.agent_id:
                existing_agent = conversation.agent
                return existing_agent, None

            if agent_id:
                agent = ChatAgent.objects.select_for_update().get(id=agent_id, is_active=True)
            else:
                # Auto-assign to best available agent with row-level locking.
                available = ChatAgent.objects.select_for_update().filter(
                    is_active=True,
                    is_online=True,
                    is_accepting_chats=True,
                ).annotate(
                    available_slots=F('max_concurrent_chats') - F('current_chat_count')
                ).filter(available_slots__gt=0)

                if conversation.category:
                    available = available.filter(
                        Q(categories=[]) | Q(categories__contains=[conversation.category])
                    )

                agent = available.order_by('-available_slots', '-avg_rating').first()

            if not agent:
                return None, 'No agents available'

            if agent.current_chat_count >= agent.max_concurrent_chats:
                return None, 'Agent at capacity'

            conversation.agent = agent
            conversation.status = ConversationStatus.ACTIVE
            conversation.is_bot_handling = False
            conversation.save(update_fields=['agent', 'status', 'is_bot_handling', 'updated_at'])

            ChatAgent.objects.filter(id=agent.id).update(
                current_chat_count=F('current_chat_count') + 1
            )

        # Notify outside transaction to avoid holding locks during network IO.
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

        settings_obj = ChatSettings.get_settings()
        
        if not settings_obj.ai_enabled:
            return None
        
        conversation = Conversation.objects.get(id=conversation_id)

        # Rate limit AI responses per conversation
        rate_limit = getattr(settings, 'CHAT_AI_RATE_LIMIT_PER_MINUTE', 10)
        if rate_limit:
            key = f"chat_ai:{conversation_id}"
            count = cache.get(key, 0)
            if count >= rate_limit:
                conversation.request_human_agent()
                return "I'm connecting you with a human agent for further assistance."
            cache.set(key, count + 1, timeout=60)
        
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
            history = AIService._build_message_history(
                conversation,
                limit=max(2, int(getattr(settings, 'CHAT_AI_CONTEXT_HISTORY_LIMIT', 10) or 10)),
            )
            history.append({'role': 'user', 'content': customer_message})

            ai_response = AIService._generate_local_model_response(
                conversation=conversation,
                settings_obj=settings_obj,
                history=history,
                customer_message=customer_message,
            )
            if not ai_response and getattr(settings, 'CHAT_AI_FALLBACK_TO_RULES', True):
                ai_response = AIService._rule_based_response(customer_message)

            if not ai_response:
                if getattr(settings, 'CHAT_AI_AUTO_HANDOFF_ON_FAILURE', False):
                    conversation.request_human_agent()
                    return "I'm connecting you with a human agent for more detailed support."
                return None

            Message.objects.create(
                conversation=conversation,
                sender=None,
                is_from_customer=False,
                is_from_bot=True,
                content=ai_response,
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
        """Categorize conversation using deterministic keyword mapping."""
        from apps.chat.models import ConversationCategory

        text = (message or '').strip().lower()
        if not text:
            return ConversationCategory.GENERAL

        keyword_map = {
            ConversationCategory.ORDER_INQUIRY: (
                'order', 'tracking', 'track', 'invoice', 'order number', 'purchase',
            ),
            ConversationCategory.SHIPPING: (
                'shipping', 'delivery', 'courier', 'dispatch', 'arrive', 'eta',
            ),
            ConversationCategory.RETURNS: (
                'return', 'refund', 'exchange', 'cancel', 'replacement',
            ),
            ConversationCategory.PAYMENT: (
                'payment', 'paid', 'transaction', 'card', 'bkash', 'nagad', 'sslcommerz',
            ),
            ConversationCategory.PRODUCT_QUESTION: (
                'product', 'size', 'material', 'color', 'stock', 'available', 'details',
            ),
            ConversationCategory.TECHNICAL: (
                'bug', 'error', 'issue', 'problem', 'login', 'cannot', 'failed',
            ),
            ConversationCategory.COMPLAINT: (
                'complaint', 'bad', 'worst', 'angry', 'disappointed',
            ),
            ConversationCategory.FEEDBACK: (
                'feedback', 'suggestion', 'feature request', 'improve',
            ),
        }

        for category, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                return category
        return ConversationCategory.GENERAL

    @staticmethod
    def _generate_local_model_response(conversation, settings_obj, history, customer_message: str = "") -> Optional[str]:
        if not (
            getattr(settings, 'ML_ENABLED', False)
            and getattr(settings, 'ML_CHAT_ASSISTANT_ENABLED', True)
            and getattr(settings, 'CHAT_AI_LOCAL_MODEL_ENABLED', True)
        ):
            return None

        try:
            from ml.services.chat_model_service import ChatModelService
        except Exception as exc:
            logger.warning("ML chat model service unavailable: %s", exc)
            return None

        personalization = AIService._build_personalization_context(
            conversation,
            latest_customer_message=customer_message,
        )
        model_override = AIService._resolve_model_override(settings_obj)

        return ChatModelService.generate_reply(
            system_prompt=f"{settings_obj.ai_system_prompt}\nFollow safety and privacy guidelines. Do not request sensitive credentials.",
            history=history,
            personalization=personalization,
            model_id_override=model_override,
            temperature=settings_obj.ai_temperature,
            max_new_tokens=settings_obj.ai_max_tokens,
        )

    @staticmethod
    def _resolve_model_override(settings_obj) -> Optional[str]:
        if not getattr(settings, 'CHAT_AI_USE_CHAT_SETTINGS_MODEL', False):
            return None
        configured = (getattr(settings_obj, 'ai_model', '') or '').strip()
        return configured or None

    @staticmethod
    def _build_personalization_context(conversation, latest_customer_message: str = "") -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        customer = getattr(conversation, 'customer', None)

        if getattr(settings, 'CHAT_AI_PERSONALIZATION_ENABLED', True):
            full_name = (conversation.customer_name or '').strip()
            if full_name:
                context['customer_name'] = full_name
                context['customer_first_name'] = full_name.split()[0]
            if conversation.category:
                context['conversation_category'] = conversation.category
            if conversation.order_reference:
                context['order_reference'] = conversation.order_reference

            if customer and getattr(settings, 'CHAT_AI_INCLUDE_USER_PREFERENCES', True):
                preferences = getattr(customer, 'preferences', None)
                if preferences:
                    context['language'] = getattr(preferences, 'language', None)
                    context['currency'] = getattr(preferences, 'currency', None)
                    context['timezone'] = getattr(preferences, 'timezone', None)

            if customer and getattr(settings, 'CHAT_AI_INCLUDE_BEHAVIOR_PROFILE', True):
                profile = getattr(customer, 'behavior_profile', None)
                if profile:
                    categories = (profile.category_preferences or {}) if hasattr(profile, 'category_preferences') else {}
                    top_categories = list(categories.keys())[:3]
                    context['engagement_score'] = float(profile.engagement_score or 0)
                    context['loyalty_score'] = float(profile.loyalty_score or 0)
                    context['total_orders'] = int(profile.total_orders or 0)
                    context['top_categories'] = ', '.join(top_categories) if top_categories else None

            if customer and getattr(settings, 'CHAT_AI_INCLUDE_ORDER_CONTEXT', True):
                AIService._augment_order_context(context, conversation, customer)

            if customer and getattr(settings, 'CHAT_AI_INCLUDE_ML_PROFILE', True):
                AIService._augment_ml_profile_context(context, customer)

            if latest_customer_message:
                context['detected_intent'] = AIService.categorize_conversation(latest_customer_message)
                if getattr(settings, 'CHAT_AI_INCLUDE_CANNED_KNOWLEDGE', True):
                    snippets = AIService._select_support_knowledge(
                        latest_customer_message=latest_customer_message,
                        category=context.get('conversation_category'),
                    )
                    if snippets:
                        context['support_knowledge'] = " | ".join(snippets)

        filtered = {key: value for key, value in context.items() if value not in (None, '', [], {}, ())}
        max_fields = max(4, int(getattr(settings, 'CHAT_AI_MAX_PERSONALIZATION_FIELDS', 16) or 16))
        if len(filtered) > max_fields:
            filtered = dict(list(filtered.items())[:max_fields])
        return filtered

    @staticmethod
    def _augment_order_context(context: Dict[str, Any], conversation, customer) -> None:
        try:
            from apps.orders.models import Order
        except Exception:
            return

        try:
            order_qs = Order.objects.filter(user_id=customer.id, is_deleted=False).order_by('-created_at')

            order_reference = (getattr(conversation, 'order_reference', None) or '').strip()
            if order_reference:
                referenced = order_qs.filter(order_number=order_reference).first()
                if referenced:
                    context['referenced_order_status'] = referenced.status
                    context['referenced_shipping_method'] = referenced.shipping_method
                    context['referenced_payment_status'] = referenced.payment_status
                    context['tracking_available'] = bool(referenced.tracking_number or referenced.tracking_url)

            recent_orders = list(order_qs.only('order_number', 'status')[:3])
            if recent_orders:
                summary = []
                for order in recent_orders:
                    masked = AIService._mask_order_reference(order.order_number)
                    summary.append(f"{masked}:{order.status}")
                context['recent_order_statuses'] = ', '.join(summary)
        except Exception as exc:
            logger.debug("Failed to load order context for conversation %s: %s", getattr(conversation, 'id', None), exc)

    @staticmethod
    def _augment_ml_profile_context(context: Dict[str, Any], customer) -> None:
        try:
            from ml.services.personalization_service import PersonalizationService
            profile = PersonalizationService().get_user_profile(customer.id)
        except Exception:
            return

        segments = profile.get('segments') or []
        if segments:
            context['customer_segments'] = ', '.join(str(item) for item in segments[:3])

        preferences = profile.get('preferences') or {}
        categories = preferences.get('categories') or []
        if categories:
            context['preferred_category_count'] = len(categories)

    @staticmethod
    def _select_support_knowledge(latest_customer_message: str, category: Optional[str] = None) -> List[str]:
        try:
            from apps.chat.models import CannedResponse
        except Exception:
            return []

        text = (latest_customer_message or '').strip().lower()
        if not text:
            return []

        keywords = {token for token in text.split() if len(token) >= 4}
        if not keywords:
            return []

        queryset = CannedResponse.objects.filter(
            is_active=True,
            is_global=True,
        )
        if category:
            queryset = queryset.filter(Q(category=category) | Q(category='general'))

        limit = max(1, int(getattr(settings, 'CHAT_AI_KNOWLEDGE_CANDIDATE_LIMIT', 20) or 20))
        candidates = list(queryset.values_list('content', flat=True)[:limit])

        scored: List[tuple[int, str]] = []
        for content in candidates:
            snippet = (content or '').strip()
            if not snippet:
                continue
            lower = snippet.lower()
            score = sum(1 for keyword in keywords if keyword in lower)
            if score > 0:
                scored.append((score, snippet))

        scored.sort(key=lambda item: item[0], reverse=True)
        max_snippets = max(1, int(getattr(settings, 'CHAT_AI_KNOWLEDGE_SNIPPET_LIMIT', 3) or 3))
        max_len = max(80, int(getattr(settings, 'CHAT_AI_KNOWLEDGE_SNIPPET_MAX_CHARS', 240) or 240))
        return [snippet[:max_len].strip() for _, snippet in scored[:max_snippets]]

    @staticmethod
    def _mask_order_reference(order_number: str) -> str:
        value = (order_number or '').strip()
        if not value:
            return ''
        if len(value) <= 6:
            return value
        return f"...{value[-6:]}"

    @staticmethod
    def _rule_based_response(customer_message: str) -> str:
        text = (customer_message or '').strip().lower()
        if not text:
            return "Thanks for reaching out. Could you share a few more details so I can help?"

        rules = (
            (
                ('refund', 'return', 'exchange', 'cancel'),
                "I can help with returns and refunds. Please share your order number and the reason, and I will guide you through the next steps.",
            ),
            (
                ('order', 'tracking', 'track', 'delivery', 'shipping'),
                "I can help with order and delivery updates. Please provide your order number so I can check the latest status.",
            ),
            (
                ('payment', 'failed', 'transaction', 'bkash', 'nagad', 'sslcommerz'),
                "I can help troubleshoot payment issues. Please share when the payment failed and which payment method you used.",
            ),
            (
                ('product', 'size', 'material', 'stock', 'available'),
                "I can help with product details. Please tell me the product name or link and what information you need.",
            ),
            (
                ('agent', 'human', 'person'),
                "No problem. I am connecting you with a human support agent now.",
            ),
        )

        for keywords, reply in rules:
            if any(keyword in text for keyword in keywords):
                return reply

        return "Thanks for your message. I can help with orders, delivery, returns, payments, and product questions. What would you like to do next?"


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
        from django.db.models.functions import TruncHour
        
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

        # Channel breakdown (by source)
        channel_breakdown = dict(
            conversations.values('source').annotate(
                count=Count('id')
            ).values_list('source', 'count')
        )

        # Hourly breakdown (messages per hour)
        hourly = messages.annotate(hour=TruncHour('created_at')).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')
        hourly_breakdown = {
            item['hour'].strftime('%H:00'): item['count']
            for item in hourly if item['hour']
        }

        # Agent performance summary
        agent_performance = {}
        for agent in ChatAgent.objects.filter(is_active=True):
            agent_conversations = conversations.filter(agent=agent)
            agent_messages = messages.filter(
                conversation__agent=agent,
                is_from_customer=False,
                is_from_bot=False
            )

            if not agent_conversations.exists() and not agent_messages.exists():
                continue

            resolved_count = agent_conversations.filter(status=ConversationStatus.RESOLVED).count()
            total_count = agent_conversations.count()

            response_times = []
            for conv in agent_conversations.filter(first_response_at__isnull=False):
                response_times.append((conv.first_response_at - conv.started_at).total_seconds())

            avg_first_response_agent = sum(response_times) / len(response_times) if response_times else 0
            avg_rating_agent = agent_conversations.filter(rating__isnull=False).aggregate(
                avg=Avg('rating')
            )['avg'] or 0

            agent_performance[str(agent.id)] = {
                'agent_name': agent.user.get_full_name() or agent.user.email,
                'total_conversations': total_count,
                'resolved_conversations': resolved_count,
                'resolution_rate': (resolved_count / total_count * 100) if total_count > 0 else 0,
                'avg_first_response_seconds': avg_first_response_agent,
                'avg_rating': float(avg_rating_agent),
                'messages_sent': agent_messages.count(),
            }
        
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
                'channel_breakdown': channel_breakdown,
                'hourly_breakdown': hourly_breakdown,
                'agent_performance': agent_performance,
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


def get_agent_for_user(user):
    """Return active chat agent profile for a user, if any."""
    from apps.chat.models import ChatAgent
    if not user or not user.is_authenticated:
        return None
    return ChatAgent.objects.filter(user=user, is_active=True).first()


def user_is_agent(user) -> bool:
    """True when user has an active chat agent profile."""
    return get_agent_for_user(user) is not None


def user_can_access_conversation(
    user,
    conversation,
    *,
    agent=None,
    allow_waiting_queue: bool = True,
) -> bool:
    """Evaluate whether a user can access a conversation."""
    from apps.chat.models import ConversationStatus

    if not user or not user.is_authenticated:
        return False

    if user.is_staff:
        return True

    if conversation.customer_id == user.id:
        return True

    if agent is None:
        agent = get_agent_for_user(user)
    if not agent:
        return False

    if conversation.agent_id == agent.id:
        return True

    if (
        allow_waiting_queue
        and conversation.agent_id is None
        and conversation.status == ConversationStatus.WAITING
    ):
        return True

    return False


def conversation_queryset_for_user(
    user,
    queryset=None,
    *,
    allow_waiting_queue: bool = True,
):
    """Return a conversation queryset scoped to the authenticated user."""
    from apps.chat.models import Conversation, ConversationStatus

    if queryset is None:
        queryset = Conversation.objects.all()

    if not user or not user.is_authenticated:
        return queryset.none()

    if user.is_staff:
        return queryset

    agent = get_agent_for_user(user)
    if agent:
        access_q = Q(agent=agent)
        if allow_waiting_queue:
            access_q |= Q(status=ConversationStatus.WAITING, agent__isnull=True)
        return queryset.filter(access_q)

    return queryset.filter(customer=user)


def redact_pii(text: str) -> str:
    if not text:
        return text
    masked = EMAIL_RE.sub(r"***@\2", text)
    masked = PHONE_RE.sub("***", masked)
    return masked


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: redact_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_payload(v) for v in value]
    if isinstance(value, str):
        return redact_pii(value)
    return value
