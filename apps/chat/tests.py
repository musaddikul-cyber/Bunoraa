from __future__ import annotations

from email.message import EmailMessage as MimeEmail
from types import SimpleNamespace

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, SimpleTestCase, override_settings
from rest_framework.test import APITestCase, APIClient

from apps.chat.models import ChatAgent, Conversation, ConversationStatus, ChatSettings
from apps.chat.services import AIService
from ml.services.chat_model_service import ChatModelService

User = get_user_model()
TEST_MIDDLEWARE = [
    middleware
    for middleware in settings.MIDDLEWARE
    if middleware
    not in {
        "debug_toolbar.middleware.DebugToolbarMiddleware",
        "core.middleware.host_canonical.HostCanonicalMiddleware",
        "core.middleware.ensure_trailing.EnsureTrailingSlashMiddleware",
        "core.middleware.api_trailing_slash.ApiTrailingSlashMiddleware",
    }
]


def response_data(response):
    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


@override_settings(DEBUG=False, SECURE_SSL_REDIRECT=False, MIDDLEWARE=TEST_MIDDLEWARE)
class ChatPermissionsTests(APITestCase):
    def setUp(self):
        self.customer_a = User.objects.create_user(email="customer-a@example.com", password="pass12345")
        self.customer_b = User.objects.create_user(email="customer-b@example.com", password="pass12345")

        self.agent_user_a = User.objects.create_user(
            email="agent-a@example.com",
            password="pass12345",
        )
        self.agent_user_b = User.objects.create_user(
            email="agent-b@example.com",
            password="pass12345",
        )
        self.admin_user = User.objects.create_user(
            email="admin@example.com",
            password="pass12345",
            is_staff=True,
            is_superuser=True,
        )

        self.agent_a = ChatAgent.objects.create(user=self.agent_user_a, is_active=True)
        self.agent_b = ChatAgent.objects.create(user=self.agent_user_b, is_active=True)

        self.conv_assigned_a = Conversation.objects.create(
            customer=self.customer_a,
            agent=self.agent_a,
            status=ConversationStatus.ACTIVE,
            source="website",
            is_bot_handling=False,
            customer_email=self.customer_a.email,
            customer_name=self.customer_a.email,
        )
        self.conv_assigned_b = Conversation.objects.create(
            customer=self.customer_b,
            agent=self.agent_b,
            status=ConversationStatus.ACTIVE,
            source="website",
            is_bot_handling=False,
            customer_email=self.customer_b.email,
            customer_name=self.customer_b.email,
        )
        self.conv_waiting = Conversation.objects.create(
            customer=self.customer_b,
            status=ConversationStatus.WAITING,
            source="website",
            is_bot_handling=False,
            customer_email=self.customer_b.email,
            customer_name=self.customer_b.email,
        )

    def test_agent_sees_only_own_and_waiting_conversations(self):
        self.client.force_authenticate(self.agent_user_a)
        response = self.client.get("/api/v1/chat/conversations/")
        self.assertEqual(response.status_code, 200)
        data = response_data(response)
        ids = {item["id"] for item in data}
        self.assertIn(str(self.conv_assigned_a.id), ids)
        self.assertIn(str(self.conv_waiting.id), ids)
        self.assertNotIn(str(self.conv_assigned_b.id), ids)

    def test_agent_cannot_message_unassigned_waiting_conversation(self):
        self.client.force_authenticate(self.agent_user_a)
        response = self.client.post(
            "/api/v1/chat/messages/",
            {
                "conversation": str(self.conv_waiting.id),
                "content": "I can help you now.",
                "message_type": "text",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_agent_cannot_retrieve_other_assigned_conversation(self):
        self.client.force_authenticate(self.agent_user_a)
        response = self.client.get(f"/api/v1/chat/conversations/{self.conv_assigned_b.id}/")
        self.assertEqual(response.status_code, 404)

    def test_staff_can_create_user_specific_conversation(self):
        self.client.force_authenticate(self.admin_user)
        response = self.client.post(
            "/api/v1/chat/conversations/for-user/",
            {
                "user_id": str(self.customer_a.id),
                "subject": "Order follow-up",
                "initial_message": "We are reviewing your order.",
                "force_new": True,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        created = response_data(response)
        self.assertEqual(created["customer"]["id"], str(self.customer_a.id))
        self.assertTrue(
            Conversation.objects.filter(
                id=created["id"],
                customer=self.customer_a,
                source="admin_console",
            ).exists()
        )


@override_settings(DEBUG=False, SECURE_SSL_REDIRECT=False, MIDDLEWARE=TEST_MIDDLEWARE)
class ChatEmailInboundTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def _build_raw_email(self, subject="Test", from_email="user@example.com", to_email="support@example.com"):
        message = MimeEmail()
        message["Subject"] = subject
        message["From"] = from_email
        message["To"] = to_email
        message.set_content("Hello from email")
        return message.as_bytes()

    @override_settings(CHAT_EMAIL_WEBHOOK_SECRET="secret")
    def test_inbound_requires_secret(self):
        raw = self._build_raw_email()
        response = self.client.post(
            "/api/v1/chat/email/inbound/",
            data=raw,
            content_type="message/rfc822",
        )
        self.assertEqual(response.status_code, 403)

    @override_settings(CHAT_EMAIL_WEBHOOK_SECRET="secret")
    def test_inbound_creates_conversation(self):
        raw = self._build_raw_email()
        response = self.client.post(
            "/api/v1/chat/email/inbound/",
            data=raw,
            content_type="message/rfc822",
            **{"HTTP_X_CHAT_EMAIL_SECRET": "secret"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Conversation.objects.count(), 1)


@override_settings(DEBUG=False, SECURE_SSL_REDIRECT=False, MIDDLEWARE=TEST_MIDDLEWARE)
class ChatSettingsTests(TestCase):
    def test_business_hours_disabled(self):
        settings_obj = ChatSettings.get_settings()
        settings_obj.business_hours_enabled = False
        settings_obj.save(update_fields=["business_hours_enabled"])
        self.assertTrue(settings_obj.is_within_business_hours())


class LocalChatModelPolicyTests(SimpleTestCase):
    @override_settings(
        CHAT_AI_ALLOWED_MODELS="Qwen/*,microsoft/Phi-3.5-mini-instruct",
        CHAT_AI_BLOCKED_MODELS="",
        CHAT_AI_DEFAULT_MODEL="Qwen/Qwen2.5-1.5B-Instruct",
        CHAT_AI_MODEL_FALLBACKS="gpt-4,microsoft/Phi-3.5-mini-instruct,Qwen/Qwen2.5-1.5B-Instruct",
    )
    def test_candidate_resolution_filters_disallowed_and_deduplicates(self):
        candidates = ChatModelService._resolve_candidate_model_ids(None)
        self.assertEqual(
            candidates,
            ["Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3.5-mini-instruct"],
        )

    @override_settings(
        CHAT_AI_ALLOWED_MODELS="Qwen/*",
        CHAT_AI_BLOCKED_MODELS="",
    )
    def test_model_allow_policy_blocks_hosted_api_aliases(self):
        self.assertFalse(ChatModelService._is_model_id_allowed("gpt-4"))
        self.assertFalse(ChatModelService._is_model_id_allowed("openai:gpt-4o"))
        self.assertTrue(ChatModelService._is_model_id_allowed("Qwen/Qwen2.5-1.5B-Instruct"))

    @override_settings(
        CHAT_AI_TOP_P=9.9,
        CHAT_AI_TOP_K=999,
        CHAT_AI_REPETITION_PENALTY=9.9,
        CHAT_AI_NO_REPEAT_NGRAM_SIZE=999,
        CHAT_AI_DO_SAMPLE=True,
        CHAT_AI_GENERATION_MAX_TIME_SECONDS=3.5,
        CHAT_AI_USE_KV_CACHE=True,
    )
    def test_generation_config_is_clamped(self):
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
        config = ChatModelService._build_generation_config(
            tokenizer=tokenizer,
            temperature=9.9,
            max_new_tokens=9999,
        )

        self.assertEqual(config["max_new_tokens"], 1024)
        self.assertEqual(config["top_p"], 1.0)
        self.assertEqual(config["top_k"], 200)
        self.assertEqual(config["repetition_penalty"], 2.0)
        self.assertEqual(config["no_repeat_ngram_size"], 8)
        self.assertEqual(config["max_time"], 3.5)
        self.assertTrue(config["use_cache"])


class ChatAIServicePolicyTests(SimpleTestCase):
    @override_settings(CHAT_AI_USE_CHAT_SETTINGS_MODEL=True)
    def test_resolve_model_override_uses_chat_settings_when_enabled(self):
        settings_obj = SimpleNamespace(ai_model="Qwen/Qwen2.5-1.5B-Instruct")
        override = AIService._resolve_model_override(settings_obj)
        self.assertEqual(override, "Qwen/Qwen2.5-1.5B-Instruct")

    @override_settings(CHAT_AI_USE_CHAT_SETTINGS_MODEL=False)
    def test_resolve_model_override_disabled(self):
        settings_obj = SimpleNamespace(ai_model="Qwen/Qwen2.5-1.5B-Instruct")
        override = AIService._resolve_model_override(settings_obj)
        self.assertIsNone(override)

    def test_mask_order_reference(self):
        self.assertEqual(AIService._mask_order_reference("ORD-20260221-ABC123"), "...ABC123")
        self.assertEqual(AIService._mask_order_reference("ABC123"), "ABC123")
