"""Celery tasks for payments: token refresh, recurring billing, reconciliation"""
from __future__ import annotations
from celery import shared_task
from django.utils import timezone
from datetime import timedelta
from .models import BkashCredential, RecurringCharge, PaymentGateway, Payment, PaymentTransaction
from .services import BkashService, PaymentService
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def refresh_bkash_tokens(self):
    """Find gateways with bKash credentials and refresh tokens as needed."""
    now = timezone.now()
    creds = BkashCredential.objects.filter(token_expires_at__lte=now + timedelta(minutes=10))
    for c in creds:
        try:
            BkashService.refresh_token(c)
        except Exception as exc:
            logger.exception("Failed to refresh bKash token for %s: %s", c.gateway, exc)


@shared_task(bind=True)
def run_recurring_billing(self, limit=100):
    """Attempt to bill due subscriptions by creating RecurringCharge and processing payment."""
    due = RecurringCharge.objects.filter(status='pending', attempt_at__lte=timezone.now()).order_by('created_at')[:limit]
    for rc in due:
        try:
            # Create a payment via PaymentService; PaymentService should support gateway selection for recurring
            payment = PaymentService.charge_subscription(rc)
            if payment and payment.status == Payment.STATUS_SUCCEEDED:
                rc.status = 'succeeded'
                rc.processed_at = timezone.now()
                rc.payment = payment
            else:
                rc.status = 'failed'
            rc.save()
        except Exception as exc:
            logger.exception('Recurring billing attempt failed for %s: %s', rc, exc)


@shared_task(bind=True)
def reconcile_transactions(self, hours=24):
    """Basic reconciliation task: aggregate transactions in last `hours` and log summary."""
    since = timezone.now() - timedelta(hours=hours)
    txns = PaymentTransaction.objects.filter(created_at__gte=since)
    total = txns.count()
    succeeded = txns.filter(event_type__icontains='success').count()
    failed = txns.filter(event_type__icontains='fail').count()
    logger.info('Reconciliation: %s txns (%s succeeded, %s failed) in last %s hours', total, succeeded, failed, hours)
    return {'total': total, 'succeeded': succeeded, 'failed': failed}