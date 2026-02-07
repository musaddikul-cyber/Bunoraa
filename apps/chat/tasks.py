"""
Celery Tasks for Bunoraa Chat System

Handles:
- AI response generation
- Chat analytics aggregation
- Cleanup tasks
- Notification tasks
"""
import logging
from celery import shared_task
from django.utils import timezone
from django.conf import settings

logger = logging.getLogger('bunoraa.chat.tasks')


@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def generate_ai_response(self, conversation_id: str, message_content: str):
    """Generate AI response for customer message."""
    try:
        from apps.chat.services import AIService
        from apps.chat.models import Conversation
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        
        conversation = Conversation.objects.get(id=conversation_id)
        
        # Check if still bot handling
        if not conversation.is_bot_handling:
            logger.info(f"Conversation {conversation_id} no longer bot handled, skipping AI response")
            return
        
        # Generate AI response
        response = AIService.generate_response(conversation_id, message_content)
        
        if response:
            # Send via WebSocket
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                f'chat_{conversation_id}',
                {
                    'type': 'ai_response',
                    'message': {
                        'content': response,
                        'is_from_bot': True,
                        'timestamp': timezone.now().isoformat()
                    }
                }
            )
            
            logger.info(f"AI response sent for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"AI response generation failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3)
def categorize_conversation_task(self, conversation_id: str, message: str):
    """Categorize conversation using AI."""
    try:
        from apps.chat.services import AIService
        from apps.chat.models import Conversation
        
        category = AIService.categorize_conversation(message)
        
        Conversation.objects.filter(id=conversation_id).update(category=category)
        
        logger.info(f"Conversation {conversation_id} categorized as {category}")
        
    except Exception as e:
        logger.error(f"Conversation categorization failed: {e}")
        raise self.retry(exc=e)


@shared_task
def update_daily_analytics():
    """Update chat analytics for today."""
    try:
        from apps.chat.services import ChatAnalyticsService
        
        analytics = ChatAnalyticsService.update_daily_analytics()
        
        logger.info(f"Updated chat analytics for {analytics.date}")
        
    except Exception as e:
        logger.error(f"Analytics update failed: {e}")


@shared_task
def update_historical_analytics(days: int = 7):
    """Update analytics for past N days (backfill)."""
    try:
        from apps.chat.services import ChatAnalyticsService
        
        today = timezone.now().date()
        
        for i in range(days):
            date = today - timezone.timedelta(days=i)
            ChatAnalyticsService.update_daily_analytics(date)
            logger.info(f"Updated analytics for {date}")
        
    except Exception as e:
        logger.error(f"Historical analytics update failed: {e}")


@shared_task
def cleanup_old_typing_indicators():
    """Clean up stale typing indicators."""
    try:
        from apps.chat.models import TypingIndicator
        
        threshold = timezone.now() - timezone.timedelta(seconds=30)
        
        deleted, _ = TypingIndicator.objects.filter(
            started_at__lt=threshold
        ).delete()
        
        logger.info(f"Cleaned up {deleted} stale typing indicators")
        
    except Exception as e:
        logger.error(f"Typing indicator cleanup failed: {e}")


@shared_task
def auto_resolve_inactive_conversations(hours: int = 24):
    """Auto-resolve conversations inactive for N hours."""
    try:
        from apps.chat.models import Conversation, ConversationStatus, Message
        
        threshold = timezone.now() - timezone.timedelta(hours=hours)
        
        inactive = Conversation.objects.filter(
            status__in=[ConversationStatus.OPEN, ConversationStatus.WAITING],
            last_message_at__lt=threshold
        )
        
        count = inactive.count()
        
        for conv in inactive:
            conv.status = ConversationStatus.CLOSED
            conv.resolved_at = timezone.now()
            conv.resolution = f'Auto-closed after {hours} hours of inactivity'
            conv.save()
            
            # Send closure message
            Message.objects.create(
                conversation=conv,
                sender=None,
                is_from_customer=False,
                is_from_bot=True,
                content=f"This conversation has been closed due to inactivity. Feel free to start a new chat if you need help!"
            )
        
        logger.info(f"Auto-resolved {count} inactive conversations")
        
    except Exception as e:
        logger.error(f"Auto-resolve failed: {e}")


@shared_task
def notify_waiting_customers():
    """Notify customers who have been waiting too long."""
    try:
        from apps.chat.models import Conversation, ConversationStatus, Message, ChatSettings
        
        settings_obj = ChatSettings.get_settings()
        threshold_minutes = 5  # Notify if waiting more than 5 minutes
        
        threshold = timezone.now() - timezone.timedelta(minutes=threshold_minutes)
        
        waiting = Conversation.objects.filter(
            status=ConversationStatus.WAITING,
            created_at__lt=threshold,
            agent__isnull=True
        )
        
        for conv in waiting:
            # Check if we already sent a wait message
            existing = Message.objects.filter(
                conversation=conv,
                is_from_bot=True,
                content__contains='still waiting'
            ).exists()
            
            if not existing:
                Message.objects.create(
                    conversation=conv,
                    sender=None,
                    is_from_customer=False,
                    is_from_bot=True,
                    content="Thanks for your patience! You're still waiting in queue. An agent will be with you shortly."
                )
        
        logger.info(f"Sent wait notifications to {waiting.count()} customers")
        
    except Exception as e:
        logger.error(f"Wait notification failed: {e}")


@shared_task
def update_agent_online_status():
    """Update agent online status based on activity."""
    try:
        from apps.chat.models import ChatAgent
        
        # Mark agents as offline if no activity for 15 minutes
        threshold = timezone.now() - timezone.timedelta(minutes=15)
        
        ChatAgent.objects.filter(
            is_online=True,
            last_activity__lt=threshold
        ).update(is_online=False, is_accepting_chats=False)
        
        logger.info("Updated agent online status")
        
    except Exception as e:
        logger.error(f"Agent status update failed: {e}")


@shared_task
def export_chat_transcript(conversation_id: str, email: str = None):
    """Export chat transcript as PDF/email."""
    try:
        from apps.chat.models import Conversation, Message
        from django.template.loader import render_to_string
        from django.core.mail import EmailMessage
        
        conversation = Conversation.objects.get(id=conversation_id)
        messages = Message.objects.filter(
            conversation=conversation
        ).order_by('created_at')
        
        # Render transcript
        transcript_html = render_to_string('chat/transcript.html', {
            'conversation': conversation,
            'messages': messages
        })
        
        if email:
            # Send via email
            email_msg = EmailMessage(
                subject=f'Chat Transcript - {conversation.id}',
                body=transcript_html,
                to=[email]
            )
            email_msg.content_subtype = 'html'
            email_msg.send()
            
            logger.info(f"Sent transcript to {email}")
        
        return transcript_html
        
    except Exception as e:
        logger.error(f"Transcript export failed: {e}")


@shared_task
def send_chat_rating_request(conversation_id: str):
    """Send rating request after chat resolution."""
    try:
        from apps.chat.models import Conversation, Message
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        
        conversation = Conversation.objects.get(id=conversation_id)
        
        # Only send if no rating yet
        if conversation.rating is not None:
            return
        
        # Send via WebSocket if customer still connected
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f'chat_{conversation_id}',
            {
                'type': 'rating_request',
                'conversation_id': str(conversation_id)
            }
        )
        
        logger.info(f"Sent rating request for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Rating request failed: {e}")


@shared_task
def sync_agent_metrics():
    """Sync calculated metrics to agent model."""
    try:
        from apps.chat.models import ChatAgent, Conversation
        from django.db.models import Avg, Count
        
        agents = ChatAgent.objects.all()
        
        for agent in agents:
            # Calculate current chat count
            current = Conversation.objects.filter(
                agent=agent,
                status__in=['open', 'active', 'waiting']
            ).count()
            
            # Calculate totals
            total = Conversation.objects.filter(agent=agent).count()
            
            # Calculate average rating
            rated = Conversation.objects.filter(
                agent=agent,
                rating__isnull=False
            ).aggregate(
                avg=Avg('rating'),
                count=Count('id')
            )
            
            agent.current_chat_count = current
            agent.total_chats_handled = total
            agent.avg_rating = rated['avg'] or 0
            agent.total_ratings = rated['count'] or 0
            agent.save()
        
        logger.info(f"Synced metrics for {agents.count()} agents")
        
    except Exception as e:
        logger.error(f"Agent metrics sync failed: {e}")
