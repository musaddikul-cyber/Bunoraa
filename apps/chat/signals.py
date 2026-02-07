"""
Django Signals for Bunoraa Chat System
"""
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.conf import settings
from django.contrib.auth import get_user_model
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import logging

logger = logging.getLogger('bunoraa.chat')
User = get_user_model()


def safe_channel_send(group_name, message):
    """Safely send to channel layer, handling cases where it's not configured."""
    channel_layer = get_channel_layer()
    if channel_layer is None:
        logger.debug(f"Channel layer not configured, skipping WebSocket notification to {group_name}")
        return False
    try:
        async_to_sync(channel_layer.group_send)(group_name, message)
        return True
    except Exception as e:
        logger.warning(f"Failed to send WebSocket message to {group_name}: {e}")
        return False


@receiver(post_save, sender='chat.Conversation')
def notify_conversation_update(sender, instance, created, **kwargs):
    """Notify WebSocket on conversation updates."""
    from apps.chat.models import ConversationStatus
    
    if created:
        # Notify agents of new conversation
        safe_channel_send(
            'chat_agents_dashboard',
            {
                'type': 'new_chat_notification',
                'conversation': {
                    'id': str(instance.id),
                    'category': instance.category,
                    'customer_name': instance.customer_name,
                    'subject': instance.subject,
                    'initial_message': instance.initial_message[:100] if instance.initial_message else '',
                    'created_at': instance.created_at.isoformat()
                }
            }
        )
    else:
        # Notify conversation room of status changes
        safe_channel_send(
            f'chat_{instance.id}',
            {
                'type': 'conversation_update',
                'conversation_id': str(instance.id),
                'status': instance.status,
                'agent_id': str(instance.agent_id) if instance.agent_id else None
            }
        )


@receiver(post_save, sender='chat.Message')
def notify_new_message(sender, instance, created, **kwargs):
    """Notify WebSocket of new messages."""
    if not created:
        return
    
    # Notify conversation room
    safe_channel_send(
        f'chat_{instance.conversation_id}',
        {
            'type': 'chat_message',
            'message': {
                'id': str(instance.id),
                'conversation_id': str(instance.conversation_id),
                'sender_id': str(instance.sender_id) if instance.sender_id else None,
                'sender_name': instance.sender.get_full_name() if instance.sender else 'Bot',
                'is_from_customer': instance.is_from_customer,
                'is_from_bot': instance.is_from_bot,
                'message_type': instance.message_type,
                'content': instance.content,
                'timestamp': instance.created_at.isoformat()
            }
        }
    )
    
    # Update conversation last_message_at
    from apps.chat.models import Conversation
    Conversation.objects.filter(id=instance.conversation_id).update(
        last_message_at=instance.created_at
    )


@receiver(post_save, sender='chat.ChatAgent')
def update_agent_status_broadcast(sender, instance, **kwargs):
    """Broadcast agent status changes."""
    safe_channel_send(
        'chat_agents_dashboard',
        {
            'type': 'agent_status_update',
            'agent_id': str(instance.id),
            'is_online': instance.is_online,
            'is_accepting_chats': instance.is_accepting_chats,
            'current_chat_count': instance.current_chat_count
        }
    )


# Auto-create agent profile signal (optional)
@receiver(post_save, sender=User)
def create_agent_profile(sender, instance, created, **kwargs):
    """Optionally create agent profile for staff users."""
    from apps.chat.models import ChatAgent
    
    # Only auto-create for staff users
    if instance.is_staff and not ChatAgent.objects.filter(user=instance).exists():
        ChatAgent.objects.create(
            user=instance,
            is_online=False,
            is_accepting_chats=True
        )
