"""
Email Service Celery Tasks
==========================

Background tasks for email processing, webhooks, and maintenance.
"""

import hashlib
import hmac
import json
import logging
import time
from datetime import timedelta

from celery import shared_task
from django.core.cache import cache
from django.db.models import F
from django.utils import timezone

logger = logging.getLogger('bunoraa.email_service.tasks')


# =============================================================================
# EMAIL QUEUE PROCESSING
# =============================================================================

@shared_task(
    bind=True,
    name='email_service.process_queue',
    max_retries=3,
    default_retry_delay=60,
)
def process_email_queue(self, batch_size=100):
    """
    Process queued emails and send them.
    Runs continuously to process the email queue.
    """
    from .engine import QueueManager
    
    try:
        QueueManager.process_queue(batch_size=batch_size)
        logger.info(f"Processed email queue batch of {batch_size}")
    except Exception as e:
        logger.exception(f"Email queue processing failed: {e}")
        raise self.retry(exc=e)


@shared_task(name='email_service.retry_failed')
def retry_failed_emails():
    """
    Retry failed emails that are due for retry.
    Should run periodically (e.g., every 5 minutes).
    """
    from .engine import QueueManager
    
    QueueManager.retry_failed(max_age_hours=24)
    logger.info("Processed failed email retries")


@shared_task(name='email_service.cleanup_old_messages')
def cleanup_old_messages(days=90):
    """
    Clean up old email messages and events.
    Keeps the database size manageable.
    """
    from .models import EmailMessage, EmailEvent, WebhookLog
    
    cutoff = timezone.now() - timedelta(days=days)
    
    # Delete old webhook logs first
    deleted_logs, _ = WebhookLog.objects.filter(
        created_at__lt=cutoff
    ).delete()
    
    # Delete old events
    deleted_events, _ = EmailEvent.objects.filter(
        timestamp__lt=cutoff
    ).delete()
    
    # Delete old messages
    deleted_messages, _ = EmailMessage.objects.filter(
        created_at__lt=cutoff
    ).delete()
    
    logger.info(
        f"Cleanup complete: {deleted_messages} messages, "
        f"{deleted_events} events, {deleted_logs} webhook logs deleted"
    )


# =============================================================================
# WEBHOOK DELIVERY
# =============================================================================

@shared_task(
    bind=True,
    name='email_service.send_webhook',
    max_retries=5,
    default_retry_delay=60,
    retry_backoff=True,
)
def send_webhook(self, webhook_id, event_type, message_id=None, payload=None):
    """
    Send a webhook notification.
    
    Args:
        webhook_id: ID of the webhook to send to
        event_type: Type of event (opened, clicked, bounced, etc.)
        message_id: ID of the email message (optional)
        payload: Custom payload (optional)
    """
    from .models import Webhook, WebhookLog, EmailMessage
    
    try:
        webhook = Webhook.objects.get(id=webhook_id)
    except Webhook.DoesNotExist:
        logger.error(f"Webhook not found: {webhook_id}")
        return
    
    # Build payload
    if payload is None:
        payload = {
            'event': event_type,
            'timestamp': timezone.now().isoformat(),
        }
        
        if message_id:
            try:
                message = EmailMessage.objects.get(message_id=message_id)
                payload['message'] = {
                    'id': message.message_id,
                    'to': message.to_email,
                    'subject': message.subject,
                    'status': message.status,
                }
            except EmailMessage.DoesNotExist:
                pass
    
    # Send webhook
    result = send_webhook_sync(webhook, event_type, payload)
    
    if not result['success']:
        # Retry with backoff
        raise self.retry(exc=Exception(result.get('error', 'Webhook failed')))


def send_webhook_sync(webhook, event_type, payload):
    """
    Synchronously send a webhook.
    
    Returns:
        dict with 'success', 'status_code', 'response', 'error'
    """
    import urllib.request
    import urllib.error
    
    from .models import WebhookLog, Webhook
    
    start_time = time.time()
    
    # Prepare request
    payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    signature = webhook.sign_payload(payload)
    
    headers = {
        'Content-Type': 'application/json',
        'X-Bunoraa-Signature': signature,
        'X-Bunoraa-Event': event_type,
        'User-Agent': 'Bunoraa-Webhook/1.0',
    }
    
    result = {
        'success': False,
        'status_code': None,
        'response': '',
        'error': '',
    }
    
    try:
        req = urllib.request.Request(
            webhook.url,
            data=payload_str.encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result['status_code'] = response.status
            result['response'] = response.read().decode('utf-8')[:1000]
            result['success'] = 200 <= response.status < 300
            
    except urllib.error.HTTPError as e:
        result['status_code'] = e.code
        result['response'] = e.read().decode('utf-8')[:1000] if e.fp else ''
        result['error'] = str(e)
        
    except urllib.error.URLError as e:
        result['error'] = str(e.reason)
        
    except Exception as e:
        result['error'] = str(e)
    
    # Calculate response time
    response_time = int((time.time() - start_time) * 1000)
    
    # Log the webhook attempt
    WebhookLog.objects.create(
        webhook=webhook,
        event_type=event_type,
        payload=payload,
        response_status=result['status_code'],
        response_body=result['response'],
        response_time_ms=response_time,
        success=result['success'],
        error_message=result['error'],
    )
    
    # Update webhook stats
    if result['success']:
        Webhook.objects.filter(id=webhook.id).update(
            total_sent=F('total_sent') + 1,
            last_sent_at=timezone.now(),
            last_error=''
        )
    else:
        Webhook.objects.filter(id=webhook.id).update(
            total_failed=F('total_failed') + 1,
            last_error=result['error'][:500]
        )
    
    return result


# =============================================================================
# DOMAIN VERIFICATION
# =============================================================================

@shared_task(name='email_service.verify_domains')
def verify_all_domains():
    """
    Verify all pending sender domains.
    Should run periodically (e.g., every hour).
    """
    from .models import SenderDomain
    
    pending_domains = SenderDomain.objects.filter(
        verification_status=SenderDomain.VerificationStatus.PENDING
    )
    
    for domain in pending_domains:
        verify_domain.delay(str(domain.id))


@shared_task(name='email_service.verify_domain')
def verify_domain(domain_id):
    """
    Verify a single domain's DNS records.
    """
    from .models import SenderDomain
    
    try:
        import dns.resolver
    except ImportError:
        logger.error("dnspython not installed, cannot verify domains")
        return
    
    try:
        domain = SenderDomain.objects.get(id=domain_id)
    except SenderDomain.DoesNotExist:
        return
    
    verified = False
    
    # Check verification TXT record
    try:
        verify_host = f'_bunoraa.{domain.domain}'
        expected_value = f'bunoraa-verify={domain.verification_token}'
        
        answers = dns.resolver.resolve(verify_host, 'TXT')
        for rdata in answers:
            if expected_value in str(rdata):
                verified = True
                break
    except Exception as e:
        logger.debug(f"Verification check failed for {domain.domain}: {e}")
    
    if verified:
        domain.verification_status = SenderDomain.VerificationStatus.VERIFIED
        domain.verified_at = timezone.now()
    
    domain.last_checked_at = timezone.now()
    domain.save()
    
    logger.info(f"Domain verification for {domain.domain}: {verified}")


# =============================================================================
# STATISTICS AGGREGATION
# =============================================================================

@shared_task(name='email_service.aggregate_daily_stats')
def aggregate_daily_stats():
    """
    Aggregate daily email statistics.
    Should run at the end of each day.
    """
    from django.db.models import Count, Q
    from .models import EmailMessage, DailyStats
    
    yesterday = (timezone.now() - timedelta(days=1)).date()
    
    # Get all users with messages from yesterday
    user_ids = EmailMessage.objects.filter(
        created_at__date=yesterday
    ).values_list('user_id', flat=True).distinct()
    
    for user_id in user_ids:
        messages = EmailMessage.objects.filter(
            user_id=user_id,
            created_at__date=yesterday
        )
        
        stats, created = DailyStats.objects.get_or_create(
            user_id=user_id,
            date=yesterday
        )
        
        # Count by status
        stats.sent = messages.filter(status__in=['sent', 'delivered', 'opened', 'clicked']).count()
        stats.delivered = messages.filter(status__in=['delivered', 'opened', 'clicked']).count()
        stats.opened = messages.filter(status__in=['opened', 'clicked']).count()
        stats.clicked = messages.filter(status='clicked').count()
        stats.bounced = messages.filter(status='bounced').count()
        stats.dropped = messages.filter(status='dropped').count()
        stats.spam_reports = messages.filter(status='spam').count()
        stats.unsubscribes = messages.filter(status='unsubscribed').count()
        
        # Unique opens/clicks (count distinct recipients)
        stats.unique_opens = messages.filter(
            opened_at__isnull=False
        ).values('to_email').distinct().count()
        
        stats.unique_clicks = messages.filter(
            clicked_at__isnull=False
        ).values('to_email').distinct().count()
        
        stats.save()
    
    logger.info(f"Aggregated daily stats for {yesterday}")


# =============================================================================
# BOUNCE PROCESSING
# =============================================================================

@shared_task(name='email_service.process_bounce')
def process_bounce(message_id, bounce_type, reason):
    """
    Process an email bounce.
    Adds to suppression list for hard bounces.
    """
    from .models import EmailMessage, EmailEvent, Suppression
    
    try:
        message = EmailMessage.objects.get(message_id=message_id)
    except EmailMessage.DoesNotExist:
        return
    
    # Update message
    message.status = EmailMessage.Status.BOUNCED
    message.bounce_type = bounce_type
    message.bounce_reason = reason
    message.save()
    
    # Create event
    EmailEvent.objects.create(
        message=message,
        event_type=EmailEvent.EventType.BOUNCED,
        data={'bounce_type': bounce_type, 'reason': reason}
    )
    
    # Add to suppression list for hard bounces
    if bounce_type == 'hard':
        Suppression.objects.get_or_create(
            user=message.user,
            email=message.to_email.lower(),
            suppression_type=Suppression.SuppressionType.BOUNCE_HARD,
            defaults={
                'source_message': message,
                'reason': reason
            }
        )
        logger.info(f"Added {message.to_email} to suppression list (hard bounce)")


# =============================================================================
# SPAM REPORT PROCESSING
# =============================================================================

@shared_task(name='email_service.process_spam_report')
def process_spam_report(message_id):
    """
    Process a spam report.
    Adds to suppression list.
    """
    from .models import EmailMessage, EmailEvent, Suppression
    
    try:
        message = EmailMessage.objects.get(message_id=message_id)
    except EmailMessage.DoesNotExist:
        return
    
    # Update message
    message.status = EmailMessage.Status.SPAM
    message.save()
    
    # Create event
    EmailEvent.objects.create(
        message=message,
        event_type=EmailEvent.EventType.SPAM_REPORT,
    )
    
    # Add to suppression list
    Suppression.objects.get_or_create(
        user=message.user,
        email=message.to_email.lower(),
        suppression_type=Suppression.SuppressionType.SPAM_REPORT,
        defaults={'source_message': message}
    )
    
    logger.info(f"Processed spam report for {message.to_email}")


# =============================================================================
# EVENT-TRIGGERED WEBHOOKS
# =============================================================================

@shared_task(name='email_service.send_webhook_for_event')
def send_webhook_for_event(event_id):
    """
    Send webhooks for an email event.
    Called when a new EmailEvent is created.
    """
    from .models import EmailEvent, Webhook
    
    try:
        event = EmailEvent.objects.select_related('message', 'message__api_key__user').get(id=event_id)
    except EmailEvent.DoesNotExist:
        logger.error(f"EmailEvent not found: {event_id}")
        return
    
    # Get user from message
    user = None
    if event.message and event.message.api_key:
        user = event.message.api_key.user
    
    if not user:
        return
    
    # Find webhooks that listen for this event type
    webhooks = Webhook.objects.filter(
        user=user,
        is_active=True,
        event_types__contains=[event.event_type]
    )
    
    for webhook in webhooks:
        # Build payload
        payload = {
            'event': event.event_type,
            'timestamp': event.created_at.isoformat(),
            'message_id': event.message.message_id,
            'email': event.message.to_emails[0] if event.message.to_emails else None,
            'subject': event.message.subject,
        }
        
        if event.ip_address:
            payload['ip'] = event.ip_address
        if event.user_agent:
            payload['user_agent'] = event.user_agent
        if event.url:
            payload['url'] = event.url
        if event.bounce_type:
            payload['bounce_type'] = event.bounce_type
        if event.bounce_reason:
            payload['bounce_reason'] = event.bounce_reason
        
        # Queue webhook delivery
        send_webhook.delay(
            webhook_id=str(webhook.id),
            event_type=event.event_type,
            message_id=event.message.message_id,
            payload=payload
        )
