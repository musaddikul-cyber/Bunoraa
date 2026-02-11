from email.message import EmailMessage as MimeEmail

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from apps.chat.models import Conversation, ChatSettings


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
            "/api/chat/email/inbound/",
            data=raw,
            content_type="message/rfc822",
        )
        self.assertEqual(response.status_code, 403)

    @override_settings(CHAT_EMAIL_WEBHOOK_SECRET="secret")
    def test_inbound_creates_conversation(self):
        raw = self._build_raw_email()
        response = self.client.post(
            "/api/chat/email/inbound/",
            data=raw,
            content_type="message/rfc822",
            **{"HTTP_X_CHAT_EMAIL_SECRET": "secret"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(Conversation.objects.count(), 1)


class ChatSettingsTests(TestCase):
    def test_business_hours_disabled(self):
        settings_obj = ChatSettings.get_settings()
        settings_obj.business_hours_enabled = False
        settings_obj.save(update_fields=["business_hours_enabled"])
        self.assertTrue(settings_obj.is_within_business_hours())
