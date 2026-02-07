"""
WebSocket Consumer for Real-time Chat

Handles:
- Real-time messaging
- Typing indicators
- Read receipts
- Agent presence
- Message reactions
"""
import json
import logging
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone
from asgiref.sync import sync_to_async

logger = logging.getLogger('bunoraa.chat')


class ChatConsumer(AsyncJsonWebsocketConsumer):
    """WebSocket consumer for live chat."""

    async def connect(self):
        """Handle WebSocket connection."""
        self.user = self.scope.get('user')
        
        if not self.user or not self.user.is_authenticated:
            await self.close()
            return

        self.conversation_id = self.scope['url_route']['kwargs'].get('conversation_id')
        self.room_group_name = f'chat_{self.conversation_id}'
        
        # Verify user has access to this conversation
        has_access = await self.check_conversation_access()
        if not has_access:
            await self.close()
            return

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        # Accept the connection
        await self.accept()

        # Send connection confirmation
        await self.send_json({
            'type': 'connection_established',
            'conversation_id': str(self.conversation_id),
            'user_id': str(self.user.id),
            'timestamp': timezone.now().isoformat()
        })

        # Mark user as present
        await self.mark_presence(True)

        # Send user joined notification
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'user_joined',
                'user_id': str(self.user.id),
                'user_name': self.user.get_full_name() or self.user.email,
                'timestamp': timezone.now().isoformat()
            }
        )

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        if hasattr(self, 'room_group_name'):
            # Clear typing indicator
            await self.clear_typing_indicator()
            
            # Mark user as not present
            await self.mark_presence(False)
            
            # Notify room
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'user_left',
                    'user_id': str(self.user.id),
                    'timestamp': timezone.now().isoformat()
                }
            )
            
            # Leave room group
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )

    async def receive_json(self, content):
        """Handle incoming WebSocket messages."""
        message_type = content.get('type')
        
        handlers = {
            'send_message': self.handle_send_message,
            'typing_start': self.handle_typing_start,
            'typing_stop': self.handle_typing_stop,
            'mark_read': self.handle_mark_read,
            'mark_all_read': self.handle_mark_all_read,
            'add_reaction': self.handle_add_reaction,
            'remove_reaction': self.handle_remove_reaction,
            'edit_message': self.handle_edit_message,
            'delete_message': self.handle_delete_message,
            'request_agent': self.handle_request_agent,
            'resolve_chat': self.handle_resolve_chat,
            'ping': self.handle_ping,
        }

        handler = handlers.get(message_type)
        if handler:
            try:
                await handler(content)
            except Exception as e:
                logger.error(f"Error handling {message_type}: {e}")
                await self.send_json({
                    'type': 'error',
                    'message': str(e)
                })
        else:
            await self.send_json({
                'type': 'error',
                'message': f'Unknown message type: {message_type}'
            })

    # Message handlers
    async def handle_send_message(self, content):
        """Handle sending a new message."""
        message_content = content.get('content', '').strip()
        message_type = content.get('message_type', 'text')
        reply_to_id = content.get('reply_to')
        metadata = content.get('metadata', {})

        if not message_content and message_type == 'text':
            return

        # Create message in database
        message = await self.create_message(
            content=message_content,
            message_type=message_type,
            reply_to_id=reply_to_id,
            metadata=metadata
        )

        # Broadcast to room
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message_id': str(message['id']),
                'sender_id': str(self.user.id),
                'sender_name': self.user.get_full_name() or self.user.email,
                'is_from_customer': message['is_from_customer'],
                'content': message_content,
                'message_type': message_type,
                'metadata': metadata,
                'reply_to': reply_to_id,
                'attachments': message.get('attachments', []),
                'timestamp': message['created_at']
            }
        )

        # Update conversation last_message_at
        await self.update_conversation_timestamp()

        # Trigger AI response if bot is handling and customer sent message
        if message['is_from_customer'] and await self.is_bot_handling():
            await self.trigger_ai_response(message_content)

    async def handle_typing_start(self, content):
        """Handle typing indicator start."""
        await self.set_typing_indicator(True)
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'typing_indicator',
                'user_id': str(self.user.id),
                'user_name': self.user.get_full_name() or self.user.email,
                'is_typing': True
            }
        )

    async def handle_typing_stop(self, content):
        """Handle typing indicator stop."""
        await self.clear_typing_indicator()
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'typing_indicator',
                'user_id': str(self.user.id),
                'is_typing': False
            }
        )

    async def handle_mark_read(self, content):
        """Mark specific message as read."""
        message_id = content.get('message_id')
        if message_id:
            await self.mark_message_read(message_id)
            
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'read_receipt',
                    'message_id': message_id,
                    'reader_id': str(self.user.id),
                    'read_at': timezone.now().isoformat()
                }
            )

    async def handle_mark_all_read(self, content):
        """Mark all messages as read."""
        count = await self.mark_all_messages_read()
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'all_read',
                'reader_id': str(self.user.id),
                'count': count,
                'read_at': timezone.now().isoformat()
            }
        )

    async def handle_add_reaction(self, content):
        """Add reaction to a message."""
        message_id = content.get('message_id')
        emoji = content.get('emoji')
        
        if message_id and emoji:
            await self.add_message_reaction(message_id, emoji)
            
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'message_reaction',
                    'message_id': message_id,
                    'user_id': str(self.user.id),
                    'emoji': emoji,
                    'action': 'add'
                }
            )

    async def handle_remove_reaction(self, content):
        """Remove reaction from a message."""
        message_id = content.get('message_id')
        emoji = content.get('emoji')
        
        if message_id and emoji:
            await self.remove_message_reaction(message_id, emoji)
            
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'message_reaction',
                    'message_id': message_id,
                    'user_id': str(self.user.id),
                    'emoji': emoji,
                    'action': 'remove'
                }
            )

    async def handle_edit_message(self, content):
        """Edit a message."""
        message_id = content.get('message_id')
        new_content = content.get('content')
        
        if message_id and new_content:
            success = await self.edit_message(message_id, new_content)
            if success:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'message_edited',
                        'message_id': message_id,
                        'content': new_content,
                        'edited_at': timezone.now().isoformat()
                    }
                )

    async def handle_delete_message(self, content):
        """Delete a message (soft delete)."""
        message_id = content.get('message_id')
        
        if message_id:
            success = await self.delete_message(message_id)
            if success:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'message_deleted',
                        'message_id': message_id,
                        'deleted_by': str(self.user.id)
                    }
                )

    async def handle_request_agent(self, content):
        """Request handoff to human agent."""
        await self.request_human_agent()
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'agent_requested',
                'requested_by': str(self.user.id),
                'timestamp': timezone.now().isoformat()
            }
        )

    async def handle_resolve_chat(self, content):
        """Resolve the chat conversation."""
        # Only agents can resolve
        is_agent = await self.is_agent()
        if not is_agent:
            return
            
        await self.resolve_conversation()
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_resolved',
                'resolved_by': str(self.user.id),
                'timestamp': timezone.now().isoformat()
            }
        )

    async def handle_ping(self, content):
        """Respond to ping with pong."""
        await self.send_json({
            'type': 'pong',
            'timestamp': timezone.now().isoformat()
        })

    # Event handlers (called when receiving from channel layer)
    async def chat_message(self, event):
        """Send chat message to WebSocket."""
        await self.send_json({
            'type': 'message',
            'message_id': event['message_id'],
            'sender_id': event['sender_id'],
            'sender_name': event['sender_name'],
            'is_from_customer': event['is_from_customer'],
            'content': event['content'],
            'message_type': event['message_type'],
            'metadata': event.get('metadata', {}),
            'reply_to': event.get('reply_to'),
            'attachments': event.get('attachments', []),
            'timestamp': event['timestamp']
        })

    async def typing_indicator(self, event):
        """Send typing indicator to WebSocket."""
        await self.send_json({
            'type': 'typing',
            'user_id': event['user_id'],
            'user_name': event.get('user_name'),
            'is_typing': event['is_typing']
        })

    async def read_receipt(self, event):
        """Send read receipt to WebSocket."""
        await self.send_json({
            'type': 'read',
            'message_id': event['message_id'],
            'reader_id': event['reader_id'],
            'read_at': event['read_at']
        })

    async def all_read(self, event):
        """Send all-read notification to WebSocket."""
        await self.send_json({
            'type': 'all_read',
            'reader_id': event['reader_id'],
            'count': event['count'],
            'read_at': event['read_at']
        })

    async def message_reaction(self, event):
        """Send reaction update to WebSocket."""
        await self.send_json({
            'type': 'reaction',
            'message_id': event['message_id'],
            'user_id': event['user_id'],
            'emoji': event['emoji'],
            'action': event['action']
        })

    async def message_edited(self, event):
        """Send message edit notification to WebSocket."""
        await self.send_json({
            'type': 'edited',
            'message_id': event['message_id'],
            'content': event['content'],
            'edited_at': event['edited_at']
        })

    async def message_deleted(self, event):
        """Send message delete notification to WebSocket."""
        await self.send_json({
            'type': 'deleted',
            'message_id': event['message_id'],
            'deleted_by': event['deleted_by']
        })

    async def user_joined(self, event):
        """Send user joined notification to WebSocket."""
        await self.send_json({
            'type': 'user_joined',
            'user_id': event['user_id'],
            'user_name': event.get('user_name'),
            'timestamp': event['timestamp']
        })

    async def user_left(self, event):
        """Send user left notification to WebSocket."""
        await self.send_json({
            'type': 'user_left',
            'user_id': event['user_id'],
            'timestamp': event['timestamp']
        })

    async def agent_assigned(self, event):
        """Send agent assigned notification to WebSocket."""
        await self.send_json({
            'type': 'agent_assigned',
            'agent_id': event['agent_id'],
            'agent_name': event['agent_name'],
            'timestamp': event['timestamp']
        })

    async def agent_requested(self, event):
        """Send agent requested notification to WebSocket."""
        await self.send_json({
            'type': 'agent_requested',
            'requested_by': event['requested_by'],
            'timestamp': event['timestamp']
        })

    async def chat_resolved(self, event):
        """Send chat resolved notification to WebSocket."""
        await self.send_json({
            'type': 'resolved',
            'resolved_by': event['resolved_by'],
            'timestamp': event['timestamp']
        })

    async def ai_response(self, event):
        """Send AI response to WebSocket."""
        await self.send_json({
            'type': 'ai_message',
            'message_id': event['message_id'],
            'content': event['content'],
            'metadata': event.get('metadata', {}),
            'timestamp': event['timestamp']
        })

    # Database operations
    @database_sync_to_async
    def check_conversation_access(self):
        """Check if user has access to the conversation."""
        from apps.chat.models import Conversation, ChatAgent
        
        try:
            conversation = Conversation.objects.get(id=self.conversation_id)
            
            # Customer can access their own conversations
            if conversation.customer_id == self.user.id:
                return True
            
            # Agents can access any conversation
            if ChatAgent.objects.filter(user=self.user).exists():
                return True
            
            # Staff can access any conversation
            if self.user.is_staff:
                return True
                
            return False
        except Conversation.DoesNotExist:
            return False

    @database_sync_to_async
    def create_message(self, content, message_type='text', reply_to_id=None, metadata=None):
        """Create a new message."""
        from apps.chat.models import Conversation, Message, ChatAgent
        
        conversation = Conversation.objects.get(id=self.conversation_id)
        
        # Determine if sender is customer or agent
        is_from_customer = conversation.customer_id == self.user.id
        
        message = Message.objects.create(
            conversation=conversation,
            sender=self.user,
            is_from_customer=is_from_customer,
            message_type=message_type,
            content=content,
            reply_to_id=reply_to_id,
            metadata=metadata or {}
        )
        
        # Update first response time if this is first agent response
        if not is_from_customer and not conversation.first_response_at:
            conversation.first_response_at = timezone.now()
            conversation.save(update_fields=['first_response_at'])
        
        return {
            'id': str(message.id),
            'is_from_customer': is_from_customer,
            'created_at': message.created_at.isoformat(),
            'attachments': []
        }

    @database_sync_to_async
    def update_conversation_timestamp(self):
        """Update conversation's last_message_at."""
        from apps.chat.models import Conversation
        Conversation.objects.filter(id=self.conversation_id).update(
            last_message_at=timezone.now()
        )

    @database_sync_to_async
    def set_typing_indicator(self, is_typing):
        """Set typing indicator."""
        from apps.chat.models import TypingIndicator, Conversation
        
        if is_typing:
            TypingIndicator.objects.update_or_create(
                conversation_id=self.conversation_id,
                user=self.user
            )
        else:
            TypingIndicator.objects.filter(
                conversation_id=self.conversation_id,
                user=self.user
            ).delete()

    @database_sync_to_async
    def clear_typing_indicator(self):
        """Clear typing indicator."""
        from apps.chat.models import TypingIndicator
        TypingIndicator.objects.filter(
            conversation_id=self.conversation_id,
            user=self.user
        ).delete()

    @database_sync_to_async
    def mark_message_read(self, message_id):
        """Mark a message as read."""
        from apps.chat.models import Message
        Message.objects.filter(
            id=message_id,
            conversation_id=self.conversation_id
        ).update(is_read=True, read_at=timezone.now())

    @database_sync_to_async
    def mark_all_messages_read(self):
        """Mark all unread messages as read."""
        from apps.chat.models import Message, Conversation
        
        conversation = Conversation.objects.get(id=self.conversation_id)
        is_customer = conversation.customer_id == self.user.id
        
        # Mark messages from the other party as read
        return Message.objects.filter(
            conversation_id=self.conversation_id,
            is_from_customer=not is_customer,
            is_read=False
        ).update(is_read=True, read_at=timezone.now())

    @database_sync_to_async
    def add_message_reaction(self, message_id, emoji):
        """Add reaction to message."""
        from apps.chat.models import Message
        try:
            message = Message.objects.get(id=message_id)
            message.add_reaction(str(self.user.id), emoji)
        except Message.DoesNotExist:
            pass

    @database_sync_to_async
    def remove_message_reaction(self, message_id, emoji):
        """Remove reaction from message."""
        from apps.chat.models import Message
        try:
            message = Message.objects.get(id=message_id)
            message.remove_reaction(str(self.user.id), emoji)
        except Message.DoesNotExist:
            pass

    @database_sync_to_async
    def edit_message(self, message_id, new_content):
        """Edit a message."""
        from apps.chat.models import Message
        return Message.objects.filter(
            id=message_id,
            sender=self.user
        ).update(
            content=new_content,
            is_edited=True,
            edited_at=timezone.now()
        ) > 0

    @database_sync_to_async
    def delete_message(self, message_id):
        """Soft delete a message."""
        from apps.chat.models import Message
        return Message.objects.filter(
            id=message_id,
            sender=self.user
        ).update(
            is_deleted=True,
            deleted_at=timezone.now()
        ) > 0

    @database_sync_to_async
    def is_bot_handling(self):
        """Check if conversation is being handled by bot."""
        from apps.chat.models import Conversation
        try:
            return Conversation.objects.get(id=self.conversation_id).is_bot_handling
        except Conversation.DoesNotExist:
            return False

    @database_sync_to_async
    def request_human_agent(self):
        """Request handoff to human agent."""
        from apps.chat.models import Conversation, ConversationStatus
        Conversation.objects.filter(id=self.conversation_id).update(
            bot_handoff_requested=True,
            status=ConversationStatus.WAITING
        )

    @database_sync_to_async
    def resolve_conversation(self):
        """Resolve the conversation."""
        from apps.chat.models import Conversation, ConversationStatus
        Conversation.objects.filter(id=self.conversation_id).update(
            status=ConversationStatus.RESOLVED,
            resolved_at=timezone.now()
        )

    @database_sync_to_async
    def is_agent(self):
        """Check if current user is an agent."""
        from apps.chat.models import ChatAgent
        return ChatAgent.objects.filter(user=self.user).exists() or self.user.is_staff

    @database_sync_to_async
    def mark_presence(self, is_present):
        """Mark user presence in conversation."""
        # Could be used for presence tracking
        pass

    async def trigger_ai_response(self, customer_message):
        """Trigger AI response to customer message."""
        from apps.chat.tasks import generate_ai_response
        # Trigger async task
        generate_ai_response.delay(str(self.conversation_id), customer_message)


class AgentDashboardConsumer(AsyncJsonWebsocketConsumer):
    """WebSocket consumer for agent dashboard - monitors all chats."""

    async def connect(self):
        """Handle connection."""
        self.user = self.scope.get('user')
        
        if not self.user or not self.user.is_authenticated:
            await self.close()
            return
        
        # Verify user is an agent or staff
        is_agent = await self.check_is_agent()
        if not is_agent:
            await self.close()
            return

        self.room_group_name = 'chat_agents_dashboard'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Mark agent as online
        await self.set_agent_online(True)
        
        # Send current queue status
        queue_status = await self.get_queue_status()
        await self.send_json({
            'type': 'queue_status',
            'data': queue_status
        })

    async def disconnect(self, close_code):
        """Handle disconnection."""
        if hasattr(self, 'room_group_name'):
            await self.set_agent_online(False)
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )

    async def receive_json(self, content):
        """Handle incoming messages."""
        message_type = content.get('type')
        
        if message_type == 'accept_chat':
            await self.handle_accept_chat(content)
        elif message_type == 'transfer_chat':
            await self.handle_transfer_chat(content)
        elif message_type == 'set_status':
            await self.handle_set_status(content)
        elif message_type == 'get_queue':
            queue = await self.get_queue_status()
            await self.send_json({'type': 'queue_status', 'data': queue})

    async def handle_accept_chat(self, content):
        """Accept a chat from the queue."""
        conversation_id = content.get('conversation_id')
        if conversation_id:
            result = await self.assign_chat_to_agent(conversation_id)
            await self.send_json({
                'type': 'chat_accepted',
                'conversation_id': conversation_id,
                'success': result
            })

    async def handle_transfer_chat(self, content):
        """Transfer chat to another agent."""
        conversation_id = content.get('conversation_id')
        target_agent_id = content.get('agent_id')
        if conversation_id and target_agent_id:
            result = await self.transfer_chat(conversation_id, target_agent_id)
            await self.send_json({
                'type': 'chat_transferred',
                'conversation_id': conversation_id,
                'success': result
            })

    async def handle_set_status(self, content):
        """Set agent accepting status."""
        is_accepting = content.get('is_accepting', True)
        await self.set_agent_accepting(is_accepting)

    # Event handlers
    async def new_chat_notification(self, event):
        """Notify of new chat in queue."""
        await self.send_json({
            'type': 'new_chat',
            'conversation': event['conversation']
        })

    async def chat_assigned(self, event):
        """Notify of chat assignment."""
        await self.send_json({
            'type': 'chat_assigned',
            'conversation_id': event['conversation_id'],
            'agent_id': event['agent_id']
        })

    async def queue_update(self, event):
        """Send queue update."""
        await self.send_json({
            'type': 'queue_update',
            'data': event['data']
        })

    # Database operations
    @database_sync_to_async
    def check_is_agent(self):
        """Check if user is an agent."""
        from apps.chat.models import ChatAgent
        return ChatAgent.objects.filter(user=self.user).exists() or self.user.is_staff

    @database_sync_to_async
    def set_agent_online(self, is_online):
        """Set agent online status."""
        from apps.chat.models import ChatAgent
        ChatAgent.objects.filter(user=self.user).update(
            is_online=is_online,
            last_active_at=timezone.now()
        )

    @database_sync_to_async
    def set_agent_accepting(self, is_accepting):
        """Set agent accepting status."""
        from apps.chat.models import ChatAgent
        ChatAgent.objects.filter(user=self.user).update(
            is_accepting_chats=is_accepting
        )

    @database_sync_to_async
    def get_queue_status(self):
        """Get current chat queue status."""
        from apps.chat.models import Conversation, ConversationStatus, ChatAgent
        
        waiting = Conversation.objects.filter(
            status=ConversationStatus.WAITING
        ).count()
        
        active = Conversation.objects.filter(
            status=ConversationStatus.ACTIVE
        ).count()
        
        online_agents = ChatAgent.objects.filter(is_online=True).count()
        
        return {
            'waiting': waiting,
            'active': active,
            'online_agents': online_agents
        }

    @database_sync_to_async
    def assign_chat_to_agent(self, conversation_id):
        """Assign chat to current agent."""
        from apps.chat.models import Conversation, ChatAgent, ConversationStatus
        
        try:
            agent = ChatAgent.objects.get(user=self.user)
            conversation = Conversation.objects.get(id=conversation_id)
            
            conversation.agent = agent
            conversation.status = ConversationStatus.ACTIVE
            conversation.is_bot_handling = False
            conversation.save()
            
            agent.current_chat_count += 1
            agent.save(update_fields=['current_chat_count'])
            
            return True
        except Exception:
            return False

    @database_sync_to_async
    def transfer_chat(self, conversation_id, target_agent_id):
        """Transfer chat to another agent."""
        from apps.chat.models import Conversation, ChatAgent
        
        try:
            target_agent = ChatAgent.objects.get(id=target_agent_id)
            conversation = Conversation.objects.get(id=conversation_id)
            
            # Remove from current agent
            if conversation.agent:
                conversation.agent.current_chat_count -= 1
                conversation.agent.save(update_fields=['current_chat_count'])
            
            # Assign to new agent
            conversation.agent = target_agent
            conversation.save(update_fields=['agent'])
            
            target_agent.current_chat_count += 1
            target_agent.save(update_fields=['current_chat_count'])
            
            return True
        except Exception:
            return False
