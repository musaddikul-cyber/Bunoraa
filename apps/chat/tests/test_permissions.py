from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework.test import APITestCase

from apps.chat.models import ChatAgent, Conversation, ConversationStatus

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
