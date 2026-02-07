"""
Email Service Signals
======================

Signal handlers for email service events.
"""

from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.utils import timezone

from .models import (
    EmailMessage, EmailEvent, Suppression, 
    SenderDomain, DailyStats
)


@receiver(post_save, sender=EmailMessage)
def update_message_timestamps(sender, instance, created, **kwargs):
    """
    Update message timestamps based on status changes.
    """
    if not created:
        # Update timestamps based on status
        now = timezone.now()
        
        if instance.status == 'sent' and not instance.sent_at:
            EmailMessage.objects.filter(pk=instance.pk).update(sent_at=now)
        elif instance.status == 'delivered' and not instance.delivered_at:
            EmailMessage.objects.filter(pk=instance.pk).update(delivered_at=now)
        elif instance.status == 'bounced' and not instance.bounced_at:
            EmailMessage.objects.filter(pk=instance.pk).update(bounced_at=now)
        elif instance.status == 'opened' and not instance.opened_at:
            EmailMessage.objects.filter(pk=instance.pk).update(opened_at=now)
        elif instance.status == 'clicked' and not instance.clicked_at:
            EmailMessage.objects.filter(pk=instance.pk).update(clicked_at=now)


@receiver(post_save, sender=EmailEvent)
def handle_email_event(sender, instance, created, **kwargs):
    """
    Handle new email events and update message status/counters.
    """
    if created:
        message = instance.message
        event_type = instance.event_type
        
        # Update message status based on event
        status_map = {
            'delivered': 'delivered',
            'bounced': 'bounced',
            'dropped': 'dropped',
            'deferred': 'deferred',
            'spam_report': 'spam_report',
            'unsubscribed': 'unsubscribed',
        }
        
        if event_type in status_map:
            message.status = status_map[event_type]
            message.save(update_fields=['status', 'updated_at'])
        
        elif event_type == 'opened':
            # Update open tracking
            EmailMessage.objects.filter(pk=message.pk).update(
                open_count=message.open_count + 1,
                opened_at=timezone.now() if not message.opened_at else message.opened_at,
                status='opened' if message.status == 'delivered' else message.status
            )
        
        elif event_type == 'clicked':
            # Update click tracking
            EmailMessage.objects.filter(pk=message.pk).update(
                click_count=message.click_count + 1,
                clicked_at=timezone.now() if not message.clicked_at else message.clicked_at,
                status='clicked' if message.status in ['delivered', 'opened'] else message.status
            )
        
        # Create suppression for bounces and spam reports
        if event_type in ['bounced', 'spam_report']:
            for email in message.to_emails:
                Suppression.objects.get_or_create(
                    user=message.api_key.user if message.api_key else None,
                    email=email,
                    type='bounce' if event_type == 'bounced' else 'spam_report',
                    defaults={
                        'reason': instance.bounce_reason or f'Email {event_type}'
                    }
                )
        
        # Trigger webhook
        from .tasks import send_webhook_for_event
        send_webhook_for_event.delay(instance.id)


@receiver(post_save, sender=SenderDomain)
def handle_domain_verification(sender, instance, created, **kwargs):
    """
    Handle domain verification status changes.
    """
    if not created and instance.verification_status == SenderDomain.VerificationStatus.VERIFIED:
        # Update all sender identities for this domain
        instance.identities.filter(verification_status='pending').update(verification_status='verified')


@receiver(pre_delete, sender=SenderDomain)
def handle_domain_deletion(sender, instance, **kwargs):
    """
    Clean up related data when domain is deleted.
    """
    # Deactivate all identities for this domain
    instance.identities.update(is_verified=False)
