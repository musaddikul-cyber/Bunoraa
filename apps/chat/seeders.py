from __future__ import annotations

import json
import random
from datetime import timedelta
from typing import Any

from django.contrib.auth import get_user_model
from django.utils import timezone

from core.seed.base import SeedContext, SeedResult, SeedSpec, JSONSeedSpec
from core.seed.registry import register_seed
from apps.chat.models import (
    ChatSettings,
    CannedResponse,
    Conversation,
    Message,
    ChatAgent,
    ConversationStatus,
    ConversationCategory,
)

User = get_user_model()


def _load_json(ctx: SeedContext, path: str) -> dict[str, Any]:
    p = ctx.resolve_path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


class ChatSettingsSeedSpec(SeedSpec):
    name = "chat.settings"
    app_label = "chat"
    kind = "prod"
    description = "Seed ChatSettings singleton"

    def apply(self, ctx: SeedContext) -> SeedResult:
        data = _load_json(ctx, "apps/chat/data/chat_settings.json")
        payload = data.get("settings") or data.get("item") or {}
        if not payload:
            return SeedResult()

        result = SeedResult()
        obj = ChatSettings.objects.first()
        if obj:
            changed = False
            for field, value in payload.items():
                if field == "id":
                    continue
                if getattr(obj, field) != value:
                    if not ctx.dry_run:
                        setattr(obj, field, value)
                    changed = True
            if changed:
                if not ctx.dry_run:
                    obj.save()
                result.updated += 1
        else:
            if ctx.dry_run:
                result.created += 1
            else:
                ChatSettings.objects.create(**payload)
                result.created += 1
        return result


class SharedCannedResponseSeedSpec(JSONSeedSpec):
    """Seed shared canned responses only (agent = null)."""

    def _prune(self, ctx: SeedContext, desired_keys: set[Any]) -> int:
        qs = self.model.objects.filter(agent__isnull=True)
        to_prune = []
        for obj in qs:
            key = self._obj_key(obj)
            if key not in desired_keys:
                to_prune.append(obj)

        if not to_prune:
            return 0

        if ctx.dry_run:
            ctx.log(f"[seed:{self.name}] prune {len(to_prune)} items (dry-run)")
            return 0

        count = 0
        for obj in to_prune:
            if hasattr(obj, "is_deleted"):
                obj.is_deleted = True
                obj.save(update_fields=["is_deleted"])
            elif hasattr(obj, "is_active"):
                obj.is_active = False
                obj.save(update_fields=["is_active"])
            else:
                obj.delete()
            count += 1
        return count


register_seed(ChatSettingsSeedSpec())

register_seed(
    SharedCannedResponseSeedSpec(
        name="chat.canned_responses",
        app_label="chat",
        model=CannedResponse,
        data_path="apps/chat/data/canned_responses.json",
        key_fields=["agent__id", "shortcut"],
        update_fields=["title", "shortcut", "content", "category", "tags", "is_global", "is_active", "agent"],
        description="Seed shared canned responses",
        kind="prod",
    )
)


class ChatDemoSeedSpec(SeedSpec):
    name = "chat.demo"
    app_label = "chat"
    kind = "demo"
    description = "Seed demo chat conversations"

    def apply(self, ctx: SeedContext) -> SeedResult:
        result = SeedResult()
        users = list(User.objects.filter(is_superuser=False))
        agents = list(ChatAgent.objects.filter(is_active=True))

        if not users or not agents:
            result.skipped += 1
            return result

        customer = random.choice(users)
        agent = random.choice(agents)

        if ctx.dry_run:
            if not Conversation.objects.filter(customer=customer, agent=agent, status=ConversationStatus.ACTIVE).exists():
                result.created += 1
            return result

        convo, created = Conversation.objects.get_or_create(
            customer=customer,
            agent=agent,
            status=ConversationStatus.ACTIVE,
            defaults={
                "category": ConversationCategory.GENERAL,
                "priority": 3,
                "subject": "Product inquiry",
                "initial_message": "Hello, I have a question about a product.",
                "source": "website",
                "last_message_at": timezone.now(),
            },
        )
        if created:
            result.created += 1
            Message.objects.create(
                conversation=convo,
                sender=customer,
                is_from_customer=True,
                content="Hello, I have a question about a product.",
            )
            Message.objects.create(
                conversation=convo,
                sender=agent.user,
                is_from_customer=False,
                content="Hi! How can I help you today?",
            )
        else:
            result.updated += 1

        resolved_customer = random.choice(users)
        Conversation.objects.get_or_create(
            customer=resolved_customer,
            status=ConversationStatus.RESOLVED,
            defaults={
                "category": ConversationCategory.ORDER_INQUIRY,
                "priority": 3,
                "subject": "Order issue resolved",
                "initial_message": "I had a question about my order.",
                "source": "website",
                "resolved_at": timezone.now(),
                "last_message_at": timezone.now() - timedelta(days=1),
            },
        )

        return result


register_seed(ChatDemoSeedSpec())
