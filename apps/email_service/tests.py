import uuid
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.email_service.models import EmailEvent, EmailMessage
from apps.email_service.services import (
    DeliveryEngine,
    DeliveryResult,
    EmailEnvelope,
    QueueManager,
)


class QueueManagerRegressionTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            email="queue-test@example.com",
            password="secure-pass-123",
        )

    def _build_message(self) -> EmailMessage:
        return EmailMessage.objects.create(
            user=self.user,
            api_key=None,
            message_id=f"{uuid.uuid4().hex}@bunoraa.com",
            from_email="noreply@bunoraa.com",
            from_name="Bunoraa",
            to_email="recipient@example.com",
            subject="Queue test",
            html_body="<p>queue</p>",
            text_body="queue",
            status=EmailMessage.Status.QUEUED,
        )

    @override_settings(EMAIL_QUEUE_SYNC_FALLBACK=True)
    @patch("apps.email_service.services.QueueManager.process_queue")
    @patch("apps.email_service.services.QueueManager._has_active_celery_workers", return_value=False)
    @patch("apps.email_service.services.QueueManager._dispatch_async_queue_processor", return_value=True)
    def test_enqueue_processes_inline_when_workers_are_unavailable(
        self,
        _mock_dispatch,
        _mock_has_workers,
        mock_process_queue,
    ):
        message = self._build_message()

        QueueManager.enqueue(message)

        mock_process_queue.assert_called_once_with(batch_size=1, message_ids=[message.id])

    @override_settings(EMAIL_QUEUE_SYNC_FALLBACK=True)
    @patch("apps.email_service.services.QueueManager.process_queue")
    @patch("apps.email_service.services.QueueManager._has_active_celery_workers", return_value=True)
    @patch("apps.email_service.services.QueueManager._dispatch_async_queue_processor", return_value=True)
    def test_enqueue_keeps_async_path_when_workers_are_available(
        self,
        _mock_dispatch,
        _mock_has_workers,
        mock_process_queue,
    ):
        message = self._build_message()

        QueueManager.enqueue(message)

        mock_process_queue.assert_not_called()

    @override_settings(EMAIL_QUEUE_SYNC_FALLBACK=False)
    @patch("apps.email_service.services.QueueManager.process_queue")
    @patch("apps.email_service.services.QueueManager._dispatch_async_queue_processor", return_value=False)
    def test_enqueue_falls_back_when_async_dispatch_fails(
        self,
        _mock_dispatch,
        mock_process_queue,
    ):
        message = self._build_message()

        QueueManager.enqueue(message)

        mock_process_queue.assert_called_once_with(batch_size=1, message_ids=[message.id])

    @patch("apps.email_service.services.DeliveryEngine.send")
    def test_process_queue_claims_message_once(self, mock_send):
        message = self._build_message()
        mock_send.return_value = DeliveryResult(
            success=True,
            message_id=message.message_id,
            response="250 OK",
        )

        first = QueueManager.process_queue(batch_size=1, message_ids=[message.id])
        second = QueueManager.process_queue(batch_size=1, message_ids=[message.id])

        message.refresh_from_db()
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(message.status, EmailMessage.Status.SENT)
        self.assertEqual(message.attempt_count, 1)

    @patch("apps.email_service.tasks.send_webhook_for_event.delay")
    @patch("apps.email_service.services.DeliveryEngine.send")
    def test_process_queue_ignores_webhook_enqueue_failures(self, mock_send, mock_webhook_delay):
        message = self._build_message()
        mock_send.return_value = DeliveryResult(
            success=True,
            message_id=message.message_id,
            response="250 OK",
        )
        mock_webhook_delay.side_effect = ValueError(
            "A rediss:// URL must have parameter ssl_cert_reqs and this must be set to CERT_REQUIRED, CERT_OPTIONAL, or CERT_NONE"
        )

        processed = QueueManager.process_queue(batch_size=1, message_ids=[message.id])

        message.refresh_from_db()
        self.assertEqual(processed, 1)
        self.assertEqual(message.status, EmailMessage.Status.SENT)
        self.assertTrue(
            EmailEvent.objects.filter(
                message=message,
                event_type=EmailEvent.EventType.SENT,
            ).exists()
        )

    @override_settings(EMAIL_SERVICE_TRANSPORT="http")
    @patch("core.utils.email_service.EmailService.send")
    def test_delivery_engine_uses_http_transport_when_configured(self, mock_http_send):
        mock_http_send.return_value = SimpleNamespace(
            success=True,
            error="",
            provider=SimpleNamespace(value="sendgrid"),
        )
        engine = DeliveryEngine()
        envelope = EmailEnvelope(
            message_id=f"{uuid.uuid4().hex}@bunoraa.com",
            from_email="noreply@bunoraa.com",
            from_name="Bunoraa",
            to_email="recipient@example.com",
            subject="Transport test",
            text_body="hello",
        )

        result = engine.send(envelope)

        self.assertTrue(result.success)
        self.assertIn("Delivered via", result.response)
        mock_http_send.assert_called_once()

    @override_settings(
        EMAIL_SERVICE_TRANSPORT="smtp",
        EMAIL_SERVICE_HTTP_FALLBACK_ON_SMTP_FAILURE=True,
    )
    @patch("core.utils.email_service.EmailService.send")
    @patch("apps.email_service.services.SMTPConnection.send", return_value=(False, "Failed to connect to SMTP server"))
    def test_delivery_engine_falls_back_to_http_on_smtp_failure(
        self,
        _mock_smtp_send,
        mock_http_send,
    ):
        mock_http_send.return_value = SimpleNamespace(
            success=True,
            error="",
            provider=SimpleNamespace(value="sendgrid"),
        )
        engine = DeliveryEngine()
        envelope = EmailEnvelope(
            message_id=f"{uuid.uuid4().hex}@bunoraa.com",
            from_email="noreply@bunoraa.com",
            from_name="Bunoraa",
            to_email="recipient@example.com",
            subject="Fallback test",
            text_body="hello",
        )

        result = engine.send(envelope)

        self.assertTrue(result.success)
        self.assertIn("Delivered via", result.response)
        mock_http_send.assert_called_once()
