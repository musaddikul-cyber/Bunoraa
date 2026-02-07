"""
Chat Models for Bunoraa Live Chat & Support System

Features:
- Conversation management with status tracking
- Message persistence with read receipts
- File/image attachments
- Agent assignment and availability
- AI chatbot integration
- Canned responses for agents
- Customer satisfaction ratings
"""
import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone


class ConversationStatus(models.TextChoices):
    """Conversation status choices."""
    OPEN = 'open', 'Open'
    WAITING = 'waiting', 'Waiting for Agent'
    ACTIVE = 'active', 'Active'
    ON_HOLD = 'on_hold', 'On Hold'
    RESOLVED = 'resolved', 'Resolved'
    CLOSED = 'closed', 'Closed'


class MessageType(models.TextChoices):
    """Message type choices."""
    TEXT = 'text', 'Text'
    IMAGE = 'image', 'Image'
    FILE = 'file', 'File'
    SYSTEM = 'system', 'System Message'
    AI = 'ai', 'AI Response'
    CANNED = 'canned', 'Canned Response'


class ConversationCategory(models.TextChoices):
    """Categories for chat conversations."""
    ORDER_INQUIRY = 'order', 'Order Inquiry'
    PRODUCT_QUESTION = 'product', 'Product Question'
    SHIPPING = 'shipping', 'Shipping & Delivery'
    RETURNS = 'returns', 'Returns & Refunds'
    PAYMENT = 'payment', 'Payment Issue'
    TECHNICAL = 'technical', 'Technical Support'
    GENERAL = 'general', 'General Inquiry'
    COMPLAINT = 'complaint', 'Complaint'
    FEEDBACK = 'feedback', 'Feedback'


class ChatAgent(models.Model):
    """Chat agent profile and availability."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='chat_agent_profile'
    )
    
    # Active status
    is_active = models.BooleanField(default=True, db_index=True)
    
    # Availability
    is_online = models.BooleanField(default=False, db_index=True)
    is_accepting_chats = models.BooleanField(default=True)
    last_active_at = models.DateTimeField(null=True, blank=True)
    
    # Capacity
    max_concurrent_chats = models.PositiveIntegerField(default=5)
    current_chat_count = models.PositiveIntegerField(default=0)
    
    # Skills & specializations
    categories = models.JSONField(
        default=list,
        help_text='List of category codes this agent can handle'
    )
    languages = models.JSONField(
        default=list,
        help_text='Languages the agent can support'
    )
    
    # Performance metrics (updated periodically)
    total_chats_handled = models.PositiveIntegerField(default=0)
    avg_response_time_seconds = models.FloatField(default=0)
    avg_rating = models.DecimalField(max_digits=3, decimal_places=2, default=0)
    
    # Settings
    auto_accept = models.BooleanField(default=False)
    notification_sound = models.BooleanField(default=True)
    desktop_notifications = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Chat Agent'
        verbose_name_plural = 'Chat Agents'
        indexes = [
            models.Index(fields=['is_online', 'is_accepting_chats']),
        ]

    def __str__(self):
        return f"Agent: {self.user.get_full_name() or self.user.email}"

    @property
    def is_available(self):
        """Check if agent is available to take new chats."""
        return (
            self.is_online and 
            self.is_accepting_chats and 
            self.current_chat_count < self.max_concurrent_chats
        )

    def set_online(self):
        """Mark agent as online."""
        self.is_online = True
        self.last_active_at = timezone.now()
        self.save(update_fields=['is_online', 'last_active_at'])

    def set_offline(self):
        """Mark agent as offline."""
        self.is_online = False
        self.last_active_at = timezone.now()
        self.save(update_fields=['is_online', 'last_active_at'])


class Conversation(models.Model):
    """Chat conversation between customer and support."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Participants
    customer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='chat_conversations'
    )
    agent = models.ForeignKey(
        ChatAgent,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='conversations'
    )
    
    # Status & category
    status = models.CharField(
        max_length=20,
        choices=ConversationStatus.choices,
        default=ConversationStatus.OPEN,
        db_index=True
    )
    category = models.CharField(
        max_length=20,
        choices=ConversationCategory.choices,
        default=ConversationCategory.GENERAL
    )
    priority = models.PositiveSmallIntegerField(
        default=3,
        help_text='1=Urgent, 2=High, 3=Normal, 4=Low'
    )
    
    # Subject & context
    subject = models.CharField(max_length=255, blank=True)
    initial_message = models.TextField(blank=True)
    
    # Related order (if applicable)
    order_reference = models.CharField(max_length=50, blank=True, null=True)
    
    # Metadata
    source = models.CharField(
        max_length=20,
        default='website',
        help_text='website, mobile_app, whatsapp, etc.'
    )
    customer_email = models.EmailField(blank=True)
    customer_name = models.CharField(max_length=100, blank=True)
    customer_phone = models.CharField(max_length=20, blank=True)
    
    # AI/Bot handling
    is_bot_handling = models.BooleanField(default=True)
    bot_handoff_requested = models.BooleanField(default=False)
    
    # Timing
    started_at = models.DateTimeField(auto_now_add=True)
    first_response_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    last_message_at = models.DateTimeField(null=True, blank=True)
    
    # Customer satisfaction
    rating = models.PositiveSmallIntegerField(null=True, blank=True)
    rating_comment = models.TextField(blank=True)
    
    # Tags for filtering
    tags = models.JSONField(default=list)
    
    # Internal notes (not visible to customer)
    internal_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-last_message_at', '-created_at']
        indexes = [
            models.Index(fields=['customer', 'status']),
            models.Index(fields=['agent', 'status']),
            models.Index(fields=['status', '-last_message_at']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return f"Chat {self.id} - {self.customer.email}"

    def assign_agent(self, agent: ChatAgent):
        """Assign an agent to this conversation."""
        self.agent = agent
        self.status = ConversationStatus.ACTIVE
        self.is_bot_handling = False
        self.save(update_fields=['agent', 'status', 'is_bot_handling'])
        
        # Update agent's chat count
        agent.current_chat_count += 1
        agent.save(update_fields=['current_chat_count'])

    def resolve(self):
        """Mark conversation as resolved."""
        self.status = ConversationStatus.RESOLVED
        self.resolved_at = timezone.now()
        self.save(update_fields=['status', 'resolved_at'])
        
        if self.agent:
            self.agent.current_chat_count = max(0, self.agent.current_chat_count - 1)
            self.agent.save(update_fields=['current_chat_count'])

    def close(self):
        """Close the conversation."""
        self.status = ConversationStatus.CLOSED
        self.closed_at = timezone.now()
        self.save(update_fields=['status', 'closed_at'])

    def request_human_agent(self):
        """Request handoff from bot to human agent."""
        self.bot_handoff_requested = True
        self.status = ConversationStatus.WAITING
        self.save(update_fields=['bot_handoff_requested', 'status'])

    @property
    def response_time_seconds(self):
        """Calculate first response time in seconds."""
        if self.first_response_at and self.started_at:
            return (self.first_response_at - self.started_at).total_seconds()
        return None


class Message(models.Model):
    """Individual chat message."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    
    # Sender
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='chat_messages'
    )
    is_from_customer = models.BooleanField(default=True)
    is_from_bot = models.BooleanField(default=False)
    
    # Content
    message_type = models.CharField(
        max_length=20,
        choices=MessageType.choices,
        default=MessageType.TEXT
    )
    content = models.TextField()
    
    # For rich content
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text='Additional data like quick replies, buttons, etc.'
    )
    
    # Read status
    is_read = models.BooleanField(default=False, db_index=True)
    read_at = models.DateTimeField(null=True, blank=True)
    
    # Delivery status
    is_delivered = models.BooleanField(default=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    
    # Edit/Delete
    is_edited = models.BooleanField(default=False)
    edited_at = models.DateTimeField(null=True, blank=True)
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    # Reactions
    reactions = models.JSONField(default=dict, blank=True)
    
    # Reply to another message
    reply_to = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='replies'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
            models.Index(fields=['conversation', 'is_read']),
        ]

    def __str__(self):
        return f"Message in {self.conversation_id}"

    def mark_as_read(self):
        """Mark message as read."""
        if not self.is_read:
            self.is_read = True
            self.read_at = timezone.now()
            self.save(update_fields=['is_read', 'read_at'])

    def add_reaction(self, user_id: str, emoji: str):
        """Add a reaction to the message."""
        if emoji not in self.reactions:
            self.reactions[emoji] = []
        if user_id not in self.reactions[emoji]:
            self.reactions[emoji].append(user_id)
            self.save(update_fields=['reactions'])

    def remove_reaction(self, user_id: str, emoji: str):
        """Remove a reaction from the message."""
        if emoji in self.reactions and user_id in self.reactions[emoji]:
            self.reactions[emoji].remove(user_id)
            if not self.reactions[emoji]:
                del self.reactions[emoji]
            self.save(update_fields=['reactions'])


class MessageAttachment(models.Model):
    """File attachment for chat messages."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(
        Message,
        on_delete=models.CASCADE,
        related_name='attachments'
    )
    
    file = models.FileField(upload_to='chat/attachments/%Y/%m/')
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=100)  # MIME type
    file_size = models.PositiveIntegerField()  # bytes
    
    # For images
    thumbnail = models.ImageField(
        upload_to='chat/thumbnails/%Y/%m/',
        null=True,
        blank=True
    )
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return self.file_name

    @property
    def is_image(self):
        return self.file_type.startswith('image/')


class CannedResponse(models.Model):
    """Pre-written responses for quick agent replies."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Ownership
    agent = models.ForeignKey(
        ChatAgent,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='canned_responses',
        help_text='Null = shared across all agents'
    )
    
    # Content
    title = models.CharField(max_length=100)
    shortcut = models.CharField(
        max_length=50,
        help_text='Trigger shortcut, e.g., /greeting'
    )
    content = models.TextField()
    
    # Categorization
    category = models.CharField(
        max_length=20,
        choices=ConversationCategory.choices,
        default=ConversationCategory.GENERAL
    )
    
    # Usage tracking
    usage_count = models.PositiveIntegerField(default=0)
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-usage_count', 'title']
        unique_together = [['agent', 'shortcut']]

    def __str__(self):
        return f"{self.shortcut} - {self.title}"

    def use(self):
        """Increment usage count."""
        self.usage_count += 1
        self.save(update_fields=['usage_count'])


class TypingIndicator(models.Model):
    """Track typing status for real-time indicators."""
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='typing_indicators'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    started_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['conversation', 'user']]

    def __str__(self):
        return f"{self.user} typing in {self.conversation_id}"


class ChatSettings(models.Model):
    """Global chat configuration (singleton)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Availability
    is_chat_enabled = models.BooleanField(default=True)
    offline_message = models.TextField(
        default='We are currently offline. Please leave a message and we will get back to you.'
    )
    
    # Business hours
    business_hours_enabled = models.BooleanField(default=False)
    business_hours = models.JSONField(
        default=dict,
        help_text='{"monday": {"start": "09:00", "end": "18:00"}, ...}'
    )
    timezone = models.CharField(max_length=50, default='Asia/Dhaka')
    
    # Welcome messages
    welcome_message = models.TextField(
        default='Hello! How can we help you today?'
    )
    bot_name = models.CharField(max_length=50, default='Bunoraa Assistant')
    bot_avatar = models.ImageField(
        upload_to='chat/bot/',
        null=True,
        blank=True
    )
    
    # AI Settings
    ai_enabled = models.BooleanField(default=True)
    ai_model = models.CharField(max_length=50, default='gpt-4')
    ai_temperature = models.FloatField(default=0.7)
    ai_max_tokens = models.PositiveIntegerField(default=500)
    ai_system_prompt = models.TextField(
        default='You are a helpful customer support assistant for Bunoraa, a handcrafted artisan marketplace. Be friendly, helpful, and concise.'
    )
    
    # Auto-responses
    auto_reply_delay_seconds = models.PositiveIntegerField(default=1)
    max_ai_responses_before_handoff = models.PositiveIntegerField(default=5)
    
    # Notifications
    notify_on_new_chat = models.BooleanField(default=True)
    notify_on_customer_waiting = models.BooleanField(default=True)
    customer_waiting_threshold_minutes = models.PositiveIntegerField(default=2)
    
    # Limits
    max_message_length = models.PositiveIntegerField(default=5000)
    max_file_size_mb = models.PositiveIntegerField(default=10)
    allowed_file_types = models.JSONField(
        default=list,
        help_text='["image/jpeg", "image/png", "application/pdf"]'
    )
    
    # Customer rating
    request_rating = models.BooleanField(default=True)
    rating_prompt = models.CharField(
        max_length=255,
        default='How would you rate your experience?'
    )
    
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Chat Settings'
        verbose_name_plural = 'Chat Settings'

    def save(self, *args, **kwargs):
        # Ensure singleton
        if not self.pk and ChatSettings.objects.exists():
            raise ValueError('Only one ChatSettings instance is allowed.')
        super().save(*args, **kwargs)

    @classmethod
    def get_settings(cls):
        """Get or create the singleton settings."""
        settings, _ = cls.objects.get_or_create(pk='00000000-0000-0000-0000-000000000001')
        return settings


class ChatAnalytics(models.Model):
    """Daily aggregated chat analytics."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    date = models.DateField(unique=True)
    
    # Volume
    total_conversations = models.PositiveIntegerField(default=0)
    new_conversations = models.PositiveIntegerField(default=0)
    resolved_conversations = models.PositiveIntegerField(default=0)
    
    # Messages
    total_messages = models.PositiveIntegerField(default=0)
    customer_messages = models.PositiveIntegerField(default=0)
    agent_messages = models.PositiveIntegerField(default=0)
    bot_messages = models.PositiveIntegerField(default=0)
    
    # Performance
    avg_first_response_seconds = models.FloatField(default=0)
    avg_resolution_time_seconds = models.FloatField(default=0)
    avg_rating = models.DecimalField(max_digits=3, decimal_places=2, default=0)
    
    # Categories breakdown
    category_breakdown = models.JSONField(default=dict)
    
    # Agent performance
    agent_performance = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']
        verbose_name = 'Chat Analytics'
        verbose_name_plural = 'Chat Analytics'

    def __str__(self):
        return f"Chat Analytics - {self.date}"
