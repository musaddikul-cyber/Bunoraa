"""
Shared permission helpers for chat access control.
"""
from __future__ import annotations

from django.db.models import Q

from apps.chat.models import ChatAgent, Conversation, ConversationStatus


def get_agent_for_user(user):
    """Return active chat agent profile for a user, if any."""
    if not user or not user.is_authenticated:
        return None
    return ChatAgent.objects.filter(user=user, is_active=True).first()


def user_is_agent(user) -> bool:
    """True when user has an active chat agent profile."""
    return get_agent_for_user(user) is not None


def user_can_access_conversation(
    user,
    conversation: Conversation,
    *,
    agent: ChatAgent | None = None,
    allow_waiting_queue: bool = True,
) -> bool:
    """
    Evaluate whether a user can access a conversation.

    Rules:
    - Staff users can access everything.
    - Customers can access their own conversations.
    - Agents can access conversations assigned to them.
    - Agents may access unassigned waiting conversations (queue view).
    """
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
