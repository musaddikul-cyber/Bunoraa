"""
Celery tasks for Accounts app.
Handles background tasks related to user management.
"""
import logging
import json
from typing import Optional
from celery import shared_task
from django.utils import timezone
from datetime import timedelta

logger = logging.getLogger('bunoraa.accounts')


@shared_task
def cleanup_old_interactions(days: int = 730):
    """
    Clean up old user interaction data.
    Keeps the last 2 years by default.
    """
    logger.info(f"Cleaning up user interactions older than {days} days...")
    
    try:
        from apps.accounts.behavior_models import UserInteraction
        
        cutoff = timezone.now() - timedelta(days=days)
        deleted, _ = UserInteraction.objects.filter(created_at__lt=cutoff).delete()
        
        logger.info(f"Deleted {deleted} old user interactions")
        return {'deleted': deleted}
        
    except Exception as e:
        logger.error(f"Failed to cleanup interactions: {e}")
        return {'error': str(e)}


@shared_task
def cleanup_expired_tokens():
    """
    Clean up expired authentication tokens.
    """
    logger.info("Cleaning up expired tokens...")
    
    try:
        from rest_framework_simplejwt.token_blacklist.models import OutstandingToken, BlacklistedToken
        
        # Delete expired outstanding tokens
        expired = OutstandingToken.objects.filter(expires_at__lt=timezone.now())
        count = expired.count()
        
        # First delete associated blacklisted tokens
        BlacklistedToken.objects.filter(token__in=expired).delete()
        expired.delete()
        
        logger.info(f"Cleaned up {count} expired tokens")
        return {'deleted': count}
        
    except ImportError:
        logger.debug("JWT token blacklist not configured")
        return {'skipped': True}
    except Exception as e:
        logger.error(f"Failed to cleanup tokens: {e}")
        return {'error': str(e)}


@shared_task
def update_user_last_seen(user_id: int):
    """
    Update user's last seen timestamp.
    """
    try:
        from apps.accounts.models import User
        
        User.objects.filter(pk=user_id).update(last_seen=timezone.now())
        
    except Exception as e:
        logger.warning(f"Failed to update last seen for user {user_id}: {e}")


@shared_task
def send_welcome_email(user_id: int):
    """
    Send welcome email to new user.
    """
    logger.info(f"Sending welcome email to user {user_id}")
    
    try:
        from apps.accounts.models import User
        from django.core.mail import send_mail
        from django.template.loader import render_to_string
        from django.conf import settings
        
        user = User.objects.get(pk=user_id)
        
        context = {
            'user': user,
            'site_name': 'Bunoraa',
            'site_url': settings.SITE_URL,
        }
        
        html_message = render_to_string('emails/welcome.html', context)
        plain_message = render_to_string('emails/welcome.txt', context)
        
        send_mail(
            subject='Welcome to Bunoraa!',
            message=plain_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=html_message,
            fail_silently=False,
        )
        
        logger.info(f"Welcome email sent to {user.email}")
        return {'sent': True}
        
    except Exception as e:
        logger.error(f"Failed to send welcome email: {e}")
        return {'error': str(e)}


@shared_task
def calculate_user_lifetime_value(user_id: int):
    """
    Calculate and update user's lifetime value.
    """
    try:
        from apps.accounts.models import User
        from apps.accounts.behavior_models import UserBehaviorProfile
        from apps.orders.models import Order
        from django.db.models import Sum
        
        user = User.objects.get(pk=user_id)
        profile, _ = UserBehaviorProfile.objects.get_or_create(user=user)
        
        # Calculate total spent
        total = Order.objects.filter(
            user=user,
            status__in=['completed', 'delivered'],
            is_deleted=False,
        ).aggregate(total=Sum('total'))['total'] or 0
        
        profile.total_spent = total
        profile.save(update_fields=['total_spent'])
        
        return {'user_id': user_id, 'ltv': float(total)}
        
    except Exception as e:
        logger.error(f"Failed to calculate LTV for user {user_id}: {e}")
        return {'error': str(e)}


@shared_task
def generate_data_export(job_id: str):
    """
    Generate a user data export file.
    """
    try:
        from django.core.files.base import ContentFile
        from apps.accounts.models import DataExportJob, Address, User
        from apps.accounts.behavior_models import UserPreferences
        from apps.orders.models import Order
        from apps.payments.models import PaymentMethod, Payment
        from apps.subscriptions.models import Subscription
        from apps.notifications.models import Notification

        job = DataExportJob.objects.select_related('user').get(pk=job_id)
        job.status = DataExportJob.STATUS_PROCESSING
        job.save(update_fields=['status'])

        user = job.user

        data = {
            'profile': {
                'id': str(user.id),
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'full_name': user.get_full_name(),
                'phone': user.phone,
                'is_verified': user.is_verified,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'updated_at': user.updated_at.isoformat() if user.updated_at else None,
            },
            'addresses': list(Address.objects.filter(user=user, is_deleted=False).values()),
            'orders': list(Order.objects.filter(user=user).values()),
            'payments': list(Payment.objects.filter(user=user).values()),
            'payment_methods': list(PaymentMethod.objects.filter(user=user).values()),
            'subscriptions': list(Subscription.objects.filter(user=user, is_deleted=False).values()),
            'notifications': list(Notification.objects.filter(user=user).values()),
        }

        prefs = UserPreferences.objects.filter(user=user).values().first()
        if prefs:
            data['preferences'] = prefs

        payload = json.dumps(data, indent=2, default=str).encode('utf-8')
        filename = f"exports/{user.id}/{job.id}.json"
        job.file.save(filename, ContentFile(payload), save=False)
        job.status = DataExportJob.STATUS_COMPLETED
        job.completed_at = timezone.now()
        job.expires_at = timezone.now() + timedelta(days=7)
        job.save(update_fields=['file', 'status', 'completed_at', 'expires_at'])

        return {'job_id': str(job.id), 'status': 'completed'}

    except Exception as e:
        logger.error(f"Failed to generate data export {job_id}: {e}")
        try:
            from apps.accounts.models import DataExportJob
            DataExportJob.objects.filter(pk=job_id).update(
                status=DataExportJob.STATUS_FAILED,
                error_message=str(e)
            )
        except Exception:
            pass
        return {'error': str(e)}


@shared_task
def cleanup_expired_exports(days: int = 7):
    """
    Remove expired data export files.
    """
    try:
        from django.utils import timezone
        from apps.accounts.models import DataExportJob
        expired = DataExportJob.objects.filter(
            expires_at__lt=timezone.now(),
            status=DataExportJob.STATUS_COMPLETED
        )
        count = expired.count()
        for job in expired:
            if job.file:
                try:
                    job.file.delete(save=False)
                except Exception:
                    pass
        expired.delete()
        return {'deleted': count}
    except Exception as e:
        logger.error(f"Failed to cleanup expired exports: {e}")
        return {'error': str(e)}


@shared_task
def cleanup_old_auth_sessions(days: Optional[int] = None):
    """
    Remove old auth sessions beyond the retention window.
    """
    try:
        from django.conf import settings
        from apps.accounts.behavior_models import UserSession

        retention_days = days or int(getattr(settings, 'AUTH_SESSION_RETENTION_DAYS', 90))
        cutoff = timezone.now() - timedelta(days=retention_days)
        expired = UserSession.objects.filter(
            session_type=UserSession.SESSION_TYPE_AUTH,
            started_at__lt=cutoff
        )
        count = expired.count()
        expired.delete()
        return {'deleted': count}
    except Exception as e:
        logger.error(f"Failed to cleanup old auth sessions: {e}")
        return {'error': str(e)}


@shared_task
def process_account_deletions():
    """
    Process pending account deletion requests.
    """
    try:
        from apps.accounts.models import AccountDeletionRequest
        pending = AccountDeletionRequest.objects.filter(
            status=AccountDeletionRequest.STATUS_PENDING,
            scheduled_for__lte=timezone.now()
        ).select_related('user')
        count = 0
        for req in pending:
            user = req.user
            user.soft_delete()
            req.status = AccountDeletionRequest.STATUS_COMPLETED
            req.processed_at = timezone.now()
            req.save(update_fields=['status', 'processed_at'])
            count += 1
        return {'processed': count}
    except Exception as e:
        logger.error(f"Failed to process account deletions: {e}")
        return {'error': str(e)}
