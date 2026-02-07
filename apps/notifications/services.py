"""
Notifications services
"""
from django.core.mail import send_mail, EmailMultiAlternatives
from django.template import Template, Context
from django.template.loader import render_to_string
from django.conf import settings
from django.utils import timezone

from .models import (
    Notification, NotificationType, NotificationChannel,
    NotificationPreference, EmailTemplate, EmailLog
)


class NotificationService:
    """Service for managing notifications."""
    
    @staticmethod
    def create_notification(
        user,
        notification_type,
        title,
        message,
        url=None,
        reference_type=None,
        reference_id=None,
        metadata=None,
        send_email=True,
        send_push=True
    ):
        """
        Create a notification for a user.
        
        Args:
            user: User instance
            notification_type: NotificationType value
            title: Notification title
            message: Notification message
            url: Optional URL to link to
            reference_type: Optional type of related object (e.g., 'order')
            reference_id: Optional ID of related object
            metadata: Optional additional data
            send_email: Whether to send email notification
            send_push: Whether to send push notification
        
        Returns:
            Notification instance
        """
        notification = Notification.objects.create(
            user=user,
            type=notification_type,
            title=title,
            message=message,
            url=url,
            reference_type=reference_type,
            reference_id=str(reference_id) if reference_id else None,
            metadata=metadata or {}
        )
        
        channels_sent = [NotificationChannel.IN_APP]
        
        # Get user preferences
        prefs = NotificationService._get_preferences(user)
        
        # Send email if enabled
        if send_email and NotificationService._should_send_email(prefs, notification_type):
            email_context = {
                'title': title,
                'message': message,
                'url': url,
            }
            if metadata:
                email_context.update(metadata)
            email_sent = EmailService.send_notification_email(
                user=user,
                notification_type=notification_type,
                context=email_context,
            )
            if email_sent:
                channels_sent.append(NotificationChannel.EMAIL)
        
        # Update channels sent
        notification.channels_sent = [c.value if hasattr(c, 'value') else c for c in channels_sent]
        notification.save(update_fields=['channels_sent'])
        
        return notification
    
    @staticmethod
    def _get_preferences(user):
        """Get or create notification preferences for user."""
        prefs, _ = NotificationPreference.objects.get_or_create(user=user)
        return prefs
    
    @staticmethod
    def _should_send_email(prefs, notification_type):
        """Check if email should be sent based on preferences."""
        email_type_map = {
            NotificationType.ORDER_PLACED: prefs.email_order_updates,
            NotificationType.ORDER_CONFIRMED: prefs.email_order_updates,
            NotificationType.ORDER_SHIPPED: prefs.email_shipping_updates,
            NotificationType.ORDER_DELIVERED: prefs.email_shipping_updates,
            NotificationType.ORDER_CANCELLED: prefs.email_order_updates,
            NotificationType.ORDER_REFUNDED: prefs.email_order_updates,
            NotificationType.PAYMENT_RECEIVED: prefs.email_order_updates,
            NotificationType.PAYMENT_FAILED: prefs.email_order_updates,
            NotificationType.REVIEW_APPROVED: prefs.email_reviews,
            NotificationType.REVIEW_REJECTED: prefs.email_reviews,
            NotificationType.PRICE_DROP: prefs.email_price_drops,
            NotificationType.BACK_IN_STOCK: prefs.email_back_in_stock,
            NotificationType.WISHLIST_SALE: prefs.email_promotions,
            NotificationType.PROMO_CODE: prefs.email_promotions,
        }
        return email_type_map.get(notification_type, True)
    
    @staticmethod
    def get_user_notifications(user, unread_only=False, limit=None):
        """Get notifications for a user."""
        queryset = Notification.objects.filter(user=user)
        if unread_only:
            queryset = queryset.filter(is_read=False)
        if limit:
            queryset = queryset[:limit]
        return queryset
    
    @staticmethod
    def get_unread_count(user):
        """Get count of unread notifications."""
        return Notification.objects.filter(user=user, is_read=False).count()
    
    @staticmethod
    def mark_as_read(notification_id, user):
        """Mark a notification as read."""
        notification = Notification.objects.filter(id=notification_id, user=user).first()
        if notification:
            notification.mark_as_read()
            return True
        return False
    
    @staticmethod
    def mark_all_as_read(user):
        """Mark all notifications as read."""
        Notification.objects.filter(user=user, is_read=False).update(
            is_read=True,
            read_at=timezone.now()
        )
    
    @staticmethod
    def delete_notification(notification_id, user):
        """Delete a notification."""
        return Notification.objects.filter(id=notification_id, user=user).delete()[0] > 0


class EmailService:
    """Service for sending emails."""
    
    @staticmethod
    def send_notification_email(user, notification_type, context):
        """Send notification email to user."""
        template = EmailTemplate.objects.filter(
            notification_type=notification_type,
            is_active=True
        ).first()
        
        if not template:
            # Use default template
            subject = context.get('title', 'Notification from Bunoraa')
            message = context.get('message', '')
        else:
            # Render template
            subject = Template(template.subject).render(Context(context))
            message = Template(template.text_template).render(Context(context))
        
        # Create email log
        log = EmailLog.objects.create(
            recipient_email=user.email,
            recipient_user=user,
            notification_type=notification_type,
            subject=subject,
            reference_type=context.get('reference_type'),
            reference_id=context.get('reference_id')
        )
        
        try:
            if template and template.html_template:
                html_content = Template(template.html_template).render(Context(context))
                email = EmailMultiAlternatives(
                    subject=subject,
                    body=message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[user.email]
                )
                email.attach_alternative(html_content, 'text/html')
                email.send()
            else:
                send_mail(
                    subject=subject,
                    message=message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user.email]
                )
            
            log.status = EmailLog.STATUS_SENT
            log.sent_at = timezone.now()
            log.save(update_fields=['status', 'sent_at'])
            return True
            
        except Exception as e:
            log.status = EmailLog.STATUS_FAILED
            log.error_message = str(e)
            log.save(update_fields=['status', 'error_message'])
            return False
    
    @staticmethod
    def send_order_confirmation(order):
        """Send order confirmation email."""
        context = {
            'title': f'Order Confirmation - {order.order_number}',
            'message': f'Thank you for your order! Your order #{order.order_number} has been received.',
            'order': order,
            'order_number': order.order_number,
            'total': str(order.total_amount),
            'reference_type': 'order',
            'reference_id': str(order.id)
        }
        
        return NotificationService.create_notification(
            user=order.user,
            notification_type=NotificationType.ORDER_PLACED,
            title=context['title'],
            message=context['message'],
            url=f'/account/orders/{order.id}/',
            reference_type='order',
            reference_id=order.id,
            metadata={'order_number': order.order_number}
        )
    
    @staticmethod
    def send_order_shipped(order, tracking_number=None):
        """Send order shipped email."""
        message = f'Your order #{order.order_number} has been shipped!'
        if tracking_number:
            message += f' Tracking number: {tracking_number}'
        
        context = {
            'title': f'Order Shipped - {order.order_number}',
            'message': message,
            'order_number': order.order_number,
            'tracking_number': tracking_number,
            'reference_type': 'order',
            'reference_id': str(order.id)
        }
        
        return NotificationService.create_notification(
            user=order.user,
            notification_type=NotificationType.ORDER_SHIPPED,
            title=context['title'],
            message=context['message'],
            url=f'/account/orders/{order.id}/',
            reference_type='order',
            reference_id=order.id,
            metadata={
                'order_number': order.order_number,
                'tracking_number': tracking_number
            }
        )
    
    @staticmethod
    def send_password_reset(user, reset_url):
        """Send password reset email."""
        context = {
            'title': 'Password Reset Request',
            'message': f'Click the link below to reset your password:\n{reset_url}',
            'reset_url': reset_url,
            'user_name': user.get_full_name() or user.email
        }
        
        return NotificationService.create_notification(
            user=user,
            notification_type=NotificationType.PASSWORD_RESET,
            title=context['title'],
            message=context['message'],
            url=reset_url,
            send_push=False
        )
    
    @staticmethod
    def send_welcome_email(user):
        """Send welcome email to new user."""
        context = {
            'title': 'Welcome to Bunoraa!',
            'message': 'Thank you for creating an account. Start shopping now!',
            'user_name': user.get_full_name() or user.email
        }
        
        return NotificationService.create_notification(
            user=user,
            notification_type=NotificationType.ACCOUNT_CREATED,
            title=context['title'],
            message=context['message'],
            url='/',
            send_push=False
        )


class OrderNotificationService:
    """Service for order-related notifications."""
    
    @staticmethod
    def notify_order_placed(order):
        """Notify user when order is placed."""
        return EmailService.send_order_confirmation(order)
    
    @staticmethod
    def notify_order_confirmed(order):
        """Notify user when order is confirmed."""
        return NotificationService.create_notification(
            user=order.user,
            notification_type=NotificationType.ORDER_CONFIRMED,
            title=f'Order Confirmed - {order.order_number}',
            message=f'Your order #{order.order_number} has been confirmed and is being processed.',
            url=f'/account/orders/{order.id}/',
            reference_type='order',
            reference_id=order.id,
            metadata={'order_number': order.order_number}
        )
    
    @staticmethod
    def notify_order_shipped(order):
        """Notify user when order is shipped."""
        return EmailService.send_order_shipped(order, order.tracking_number)
    
    @staticmethod
    def notify_order_delivered(order):
        """Notify user when order is delivered."""
        return NotificationService.create_notification(
            user=order.user,
            notification_type=NotificationType.ORDER_DELIVERED,
            title=f'Order Delivered - {order.order_number}',
            message=f'Your order #{order.order_number} has been delivered!',
            url=f'/account/orders/{order.id}/',
            reference_type='order',
            reference_id=order.id,
            metadata={'order_number': order.order_number}
        )
    
    @staticmethod
    def notify_order_cancelled(order):
        """Notify user when order is cancelled."""
        return NotificationService.create_notification(
            user=order.user,
            notification_type=NotificationType.ORDER_CANCELLED,
            title=f'Order Cancelled - {order.order_number}',
            message=f'Your order #{order.order_number} has been cancelled.',
            url=f'/account/orders/{order.id}/',
            reference_type='order',
            reference_id=order.id,
            metadata={'order_number': order.order_number}
        )

# =============================================================================
# Email Digest Service
# =============================================================================

import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta

logger = logging.getLogger('bunoraa.notifications.digest')


class DigestFrequency:
    """Digest frequency options."""
    IMMEDIATE = 'immediate'
    HOURLY = 'hourly'
    DAILY = 'daily'
    WEEKLY = 'weekly'
    NEVER = 'never'
    
    CHOICES = [
        (IMMEDIATE, 'Immediate'),
        (HOURLY, 'Hourly'),
        (DAILY, 'Daily'),
        (WEEKLY, 'Weekly'),
        (NEVER, 'Never'),
    ]


class DigestService:
    """
    Service for creating and sending notification digests.
    """
    
    @staticmethod
    def get_pending_digest_users(frequency: str) -> List[int]:
        """
        Get users who should receive a digest based on frequency.
        
        Returns list of user IDs.
        """
        from apps.notifications.models import Notification, NotificationPreference
        from apps.accounts.models import User
        
        now = timezone.now()
        
        # Calculate time window based on frequency
        if frequency == DigestFrequency.HOURLY:
            since = now - timedelta(hours=1)
        elif frequency == DigestFrequency.DAILY:
            since = now - timedelta(days=1)
        elif frequency == DigestFrequency.WEEKLY:
            since = now - timedelta(weeks=1)
        else:
            return []
        
        # Get users with unread notifications in the time window
        users_with_notifications = Notification.objects.filter(
            is_read=False,
            created_at__gte=since
        ).values_list('user_id', flat=True).distinct()
        
        # Filter by digest preference
        # For now, return all users with notifications
        # In production, check NotificationPreference.digest_frequency
        return list(users_with_notifications)
    
    @staticmethod
    def generate_digest(user_id: int, since: timezone.datetime = None) -> Optional[Dict[str, Any]]:
        """
        Generate digest content for a user.
        
        Returns dict with:
        - summary stats
        - notification groups by type
        - recommendations
        """
        from apps.notifications.models import Notification, NotificationType
        from apps.accounts.models import User
        
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
        
        if since is None:
            since = timezone.now() - timedelta(days=1)
        
        notifications = Notification.objects.filter(
            user_id=user_id,
            created_at__gte=since
        ).order_by('-created_at')
        
        if not notifications.exists():
            return None
        
        # Group by type
        by_type = {}
        for notif in notifications:
            if notif.type not in by_type:
                by_type[notif.type] = []
            by_type[notif.type].append({
                'id': str(notif.id),
                'title': notif.title,
                'message': notif.message,
                'url': notif.url,
                'is_read': notif.is_read,
                'created_at': notif.created_at
            })
        
        # Calculate stats
        stats = {
            'total': notifications.count(),
            'unread': notifications.filter(is_read=False).count(),
            'by_type': {k: len(v) for k, v in by_type.items()}
        }
        
        # Priority notifications (orders, payments)
        priority_types = [
            NotificationType.ORDER_PLACED,
            NotificationType.ORDER_SHIPPED,
            NotificationType.ORDER_DELIVERED,
            NotificationType.PAYMENT_RECEIVED,
            NotificationType.PAYMENT_FAILED
        ]
        
        priority_notifications = [
            n for n in notifications
            if n.type in priority_types
        ][:5]
        
        # Build digest
        digest = {
            'user': user,
            'user_name': user.get_full_name() or user.email,
            'user_email': user.email,
            'period_start': since,
            'period_end': timezone.now(),
            'stats': stats,
            'notifications_by_type': by_type,
            'priority_notifications': priority_notifications,
            'all_notifications': notifications[:20],
            'has_more': notifications.count() > 20
        }
        
        return digest
    
    @staticmethod
    def send_digest_email(user_id: int, digest: Dict[str, Any]) -> bool:
        """
        Send digest email to user.
        """
        from apps.notifications.models import EmailLog
        
        user = digest['user']
        
        # Render templates
        try:
            html_content = render_to_string('emails/digest/daily_digest.html', {
                'digest': digest,
                'user': user,
                'site_name': 'Bunoraa',
                'site_url': getattr(settings, 'SITE_URL', 'https://bunoraa.com'),
                'unsubscribe_url': f"{getattr(settings, 'SITE_URL', '')}/account/notifications/unsubscribe/"
            })
        except Exception as e:
            logger.error(f"Failed to render digest template: {e}")
            # Fallback to simple format
            html_content = DigestService._generate_simple_digest_html(digest)
        
        text_content = DigestService._generate_digest_text(digest)
        
        # Build subject
        unread = digest['stats']['unread']
        if unread == 1:
            subject = "You have 1 unread notification"
        else:
            subject = f"You have {unread} unread notifications"
        
        # Create log
        log = EmailLog.objects.create(
            recipient_email=user.email,
            recipient_user=user,
            notification_type='digest',
            subject=subject
        )
        
        try:
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[user.email]
            )
            email.attach_alternative(html_content, 'text/html')
            email.send()
            
            log.status = 'sent'
            log.sent_at = timezone.now()
            log.save(update_fields=['status', 'sent_at'])
            
            logger.info(f"Digest email sent to {user.email}")
            return True
            
        except Exception as e:
            log.status = 'failed'
            log.error_message = str(e)
            log.save(update_fields=['status', 'error_message'])
            
            logger.error(f"Failed to send digest to {user.email}: {e}")
            return False
    
    @staticmethod
    def _generate_simple_digest_html(digest: Dict[str, Any]) -> str:
        """Generate simple HTML digest when template not available."""
        notifications_html = ""
        
        for notif in digest['all_notifications'][:10]:
            notifications_html += f"""
            <tr>
                <td style="padding: 15px; border-bottom: 1px solid #eee;">
                    <h3 style="margin: 0 0 5px; color: #333;">{notif.title}</h3>
                    <p style="margin: 0; color: #666;">{notif.message}</p>
                    <small style="color: #999;">{notif.created_at.strftime('%b %d, %Y %H:%M')}</small>
                </td>
            </tr>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(135deg, #6366F1, #8B5CF6); padding: 30px; text-align: center;">
                    <h1 style="color: #fff; margin: 0;">Bunoraa</h1>
                    <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0;">Your Daily Digest</p>
                </div>
                
                <div style="padding: 30px;">
                    <p style="color: #333; font-size: 16px;">Hi {digest['user_name']},</p>
                    <p style="color: #666;">You have <strong>{digest['stats']['unread']}</strong> unread notifications.</p>
                    
                    <table width="100%" cellpadding="0" cellspacing="0" style="margin: 20px 0;">
                        {notifications_html}
                    </table>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="{getattr(settings, 'SITE_URL', '')}/account/notifications/" 
                           style="display: inline-block; background: #6366F1; color: #fff; padding: 12px 30px; border-radius: 6px; text-decoration: none; font-weight: 500;">
                            View All Notifications
                        </a>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666;">
                    <p>© 2025 Bunoraa. All rights reserved.</p>
                    <p>
                        <a href="{getattr(settings, 'SITE_URL', '')}/account/notifications/preferences/" style="color: #6366F1;">
                            Manage notification preferences
                        </a>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
    
    @staticmethod
    def _generate_digest_text(digest: Dict[str, Any]) -> str:
        """Generate plain text digest."""
        lines = [
            f"Hi {digest['user_name']},",
            "",
            f"You have {digest['stats']['unread']} unread notifications.",
            "",
            "Recent notifications:",
            "-" * 40
        ]
        
        for notif in digest['all_notifications'][:10]:
            lines.extend([
                "",
                f"• {notif.title}",
                f"  {notif.message}",
                f"  {notif.created_at.strftime('%b %d, %Y %H:%M')}"
            ])
        
        lines.extend([
            "",
            "-" * 40,
            "",
            f"View all notifications: {getattr(settings, 'SITE_URL', '')}/account/notifications/",
            "",
            "Best regards,",
            "The Bunoraa Team"
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def process_daily_digests():
        """
        Process and send daily digests.
        Called by Celery beat task.
        """
        users = DigestService.get_pending_digest_users(DigestFrequency.DAILY)
        
        sent = 0
        failed = 0
        
        for user_id in users:
            digest = DigestService.generate_digest(
                user_id,
                since=timezone.now() - timedelta(days=1)
            )
            
            if digest and digest['stats']['unread'] > 0:
                success = DigestService.send_digest_email(user_id, digest)
                if success:
                    sent += 1
                else:
                    failed += 1
        
        logger.info(f"Daily digest processed: {sent} sent, {failed} failed")
        return {'sent': sent, 'failed': failed}
    
    @staticmethod
    def process_weekly_digests():
        """
        Process and send weekly digests.
        Called by Celery beat task.
        """
        users = DigestService.get_pending_digest_users(DigestFrequency.WEEKLY)
        
        sent = 0
        failed = 0
        
        for user_id in users:
            digest = DigestService.generate_digest(
                user_id,
                since=timezone.now() - timedelta(weeks=1)
            )
            
            if digest and digest['stats']['unread'] > 0:
                success = DigestService.send_digest_email(user_id, digest)
                if success:
                    sent += 1
                else:
                    failed += 1
        
        logger.info(f"Weekly digest processed: {sent} sent, {failed} failed")
        return {'sent': sent, 'failed': failed}


class NotificationBatcher:
    """
    Batches notifications to prevent email flood.
    """
    
    @staticmethod
    def should_batch(user_id: int, notification_type: str) -> bool:
        """
        Check if notification should be batched instead of sent immediately.
        
        Returns True if the user has received too many notifications recently.
        """
        from apps.notifications.models import EmailLog
        
        # Don't batch critical notifications
        critical_types = ['order_placed', 'payment_failed', 'password_reset']
        if notification_type in critical_types:
            return False
        
        # Check recent email count
        recent = EmailLog.objects.filter(
            recipient_user_id=user_id,
            created_at__gte=timezone.now() - timedelta(hours=1)
        ).count()
        
        # Batch if more than 5 emails in the last hour
        return recent >= 5
    
    @staticmethod
    def queue_for_digest(user_id: int, notification_type: str, context: dict):
        """
        Queue notification for next digest instead of sending immediately.
        
        The notification will be included in the next scheduled digest.
        """
        from apps.notifications.models import Notification
        
        # Just create the in-app notification
        # It will be picked up by the digest processor
        Notification.objects.create(
            user_id=user_id,
            type=notification_type,
            title=context.get('title', 'Notification'),
            message=context.get('message', ''),
            url=context.get('url'),
            metadata={**context, 'batched': True}
        )
        
        logger.info(f"Notification queued for digest: user={user_id}, type={notification_type}")


# =============================================================================
# Push Notification Services
# =============================================================================

import json

logger_push = logging.getLogger('bunoraa.notifications.push')


class FCMService:
    """
    Firebase Cloud Messaging service for push notifications.
    
    Supports:
    - Android push notifications
    - iOS push notifications (via APNs through FCM)
    - Web push notifications
    """
    
    _client = None
    
    @classmethod
    def get_client(cls):
        """Get or create FCM client."""
        if cls._client is None:
            try:
                import firebase_admin
                from firebase_admin import messaging, credentials
                
                # Initialize Firebase app if not already done
                if not firebase_admin._apps:
                    cred_path = getattr(settings, 'FIREBASE_CREDENTIALS_PATH', None)
                    
                    if cred_path:
                        cred = credentials.Certificate(cred_path)
                    else:
                        # Use environment variable or default
                        cred = credentials.ApplicationDefault()
                    
                    firebase_admin.initialize_app(cred)
                
                cls._client = messaging
                
            except ImportError:
                logger_push.warning("firebase-admin not installed. Run: pip install firebase-admin")
                return None
            except Exception as e:
                logger_push.error(f"Failed to initialize Firebase: {e}")
                return None
        
        return cls._client
    
    @staticmethod
    def send_notification(
        token: str,
        title: str,
        body: str,
        data: Dict[str, Any] = None,
        image_url: str = None,
        click_action: str = None,
        priority: str = 'high',
        ttl_seconds: int = 3600,
        collapse_key: str = None,
        badge: int = None,
        sound: str = 'default'
    ) -> bool:
        """
        Send push notification to a single device.
        
        Args:
            token: FCM device token
            title: Notification title
            body: Notification body
            data: Additional data payload
            image_url: URL for notification image
            click_action: URL to open when notification is clicked
            priority: 'high' or 'normal'
            ttl_seconds: Time to live in seconds
            collapse_key: Key to collapse similar notifications
            badge: Badge count (iOS)
            sound: Sound file (default or custom)
        
        Returns:
            bool: True if sent successfully
        """
        messaging = FCMService.get_client()
        if not messaging:
            logger_push.warning("FCM client not available")
            return False
        
        try:
            # Build notification
            notification = messaging.Notification(
                title=title,
                body=body,
                image=image_url
            )
            
            # Android config
            android_config = messaging.AndroidConfig(
                priority=priority,
                ttl=timezone.timedelta(seconds=ttl_seconds),
                collapse_key=collapse_key,
                notification=messaging.AndroidNotification(
                    icon='ic_notification',
                    color='#6366F1',
                    sound=sound,
                    click_action=click_action
                )
            )
            
            # iOS (APNs) config
            apns_config = messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        alert=messaging.ApsAlert(
                            title=title,
                            body=body
                        ),
                        badge=badge,
                        sound=sound,
                        content_available=True
                    )
                )
            )
            
            # Web push config
            webpush_config = messaging.WebpushConfig(
                notification=messaging.WebpushNotification(
                    title=title,
                    body=body,
                    icon='/static/icons/notification-icon.png',
                    badge='/static/icons/notification-badge.png',
                    image=image_url,
                    renotify=True if collapse_key else False,
                    tag=collapse_key
                ),
                fcm_options=messaging.WebpushFCMOptions(
                    link=click_action
                )
            )
            
            # Create message
            message = messaging.Message(
                notification=notification,
                data={k: str(v) for k, v in (data or {}).items()},
                token=token,
                android=android_config,
                apns=apns_config,
                webpush=webpush_config
            )
            
            # Send
            response = messaging.send(message)
            logger_push.info(f"FCM notification sent: {response}")
            return True
            
        except Exception as e:
            logger_push.error(f"Failed to send FCM notification: {e}")
            return False
    
    @staticmethod
    def send_multicast(
        tokens: List[str],
        title: str,
        body: str,
        data: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send push notification to multiple devices.
        
        Returns dict with success/failure counts and failed tokens.
        """
        messaging = FCMService.get_client()
        if not messaging:
            return {'success': 0, 'failure': len(tokens), 'failed_tokens': tokens}
        
        try:
            notification = messaging.Notification(
                title=title,
                body=body,
                image=kwargs.get('image_url')
            )
            
            message = messaging.MulticastMessage(
                notification=notification,
                data={k: str(v) for k, v in (data or {}).items()},
                tokens=tokens
            )
            
            response = messaging.send_each_for_multicast(message)
            
            # Track failed tokens
            failed_tokens = []
            for idx, send_response in enumerate(response.responses):
                if not send_response.success:
                    failed_tokens.append(tokens[idx])
            
            return {
                'success': response.success_count,
                'failure': response.failure_count,
                'failed_tokens': failed_tokens
            }
            
        except Exception as e:
            logger_push.error(f"Failed to send multicast: {e}")
            return {'success': 0, 'failure': len(tokens), 'error': str(e)}
    
    @staticmethod
    def send_to_topic(
        topic: str,
        title: str,
        body: str,
        data: Dict[str, Any] = None,
        **kwargs
    ) -> bool:
        """Send notification to a topic."""
        messaging = FCMService.get_client()
        if not messaging:
            return False
        
        try:
            notification = messaging.Notification(
                title=title,
                body=body,
                image=kwargs.get('image_url')
            )
            
            message = messaging.Message(
                notification=notification,
                data={k: str(v) for k, v in (data or {}).items()},
                topic=topic
            )
            
            response = messaging.send(message)
            logger_push.info(f"Topic notification sent: {response}")
            return True
            
        except Exception as e:
            logger_push.error(f"Failed to send topic notification: {e}")
            return False
    
    @staticmethod
    def subscribe_to_topic(tokens: List[str], topic: str) -> bool:
        """Subscribe tokens to a topic."""
        messaging = FCMService.get_client()
        if not messaging:
            return False
        
        try:
            response = messaging.subscribe_to_topic(tokens, topic)
            return response.success_count > 0
        except Exception as e:
            logger_push.error(f"Failed to subscribe to topic: {e}")
            return False


class WebPushService:
    """
    Web Push service using VAPID for browser notifications.
    
    Works with service workers for native browser push notifications
    without FCM dependency.
    """
    
    @staticmethod
    def get_vapid_keys():
        """Get VAPID keys from settings."""
        return {
            'public_key': getattr(settings, 'VAPID_PUBLIC_KEY', ''),
            'private_key': getattr(settings, 'VAPID_PRIVATE_KEY', ''),
            'claims': {
                'sub': f"mailto:{getattr(settings, 'VAPID_ADMIN_EMAIL', 'admin@bunoraa.com')}"
            }
        }
    
    @staticmethod
    def send_notification(
        subscription_info: Dict[str, Any],
        title: str,
        body: str,
        icon: str = None,
        badge: str = None,
        image: str = None,
        url: str = None,
        tag: str = None,
        data: Dict[str, Any] = None,
        actions: List[Dict[str, str]] = None,
        require_interaction: bool = False,
        vibrate: List[int] = None,
        ttl: int = 3600
    ) -> bool:
        """
        Send Web Push notification.
        
        Args:
            subscription_info: Push subscription object from browser
                {
                    'endpoint': 'https://...',
                    'keys': {
                        'p256dh': '...',
                        'auth': '...'
                    }
                }
            title: Notification title
            body: Notification body
            icon: URL for notification icon
            badge: URL for badge icon (small icon)
            image: URL for large image
            url: URL to open on click
            tag: Tag for grouping notifications
            data: Additional data
            actions: List of action buttons [{'action': 'view', 'title': 'View'}]
            require_interaction: Keep notification visible until interacted
            vibrate: Vibration pattern [200, 100, 200]
            ttl: Time to live in seconds
        
        Returns:
            bool: True if sent successfully
        """
        try:
            from pywebpush import webpush
        except ImportError:
            logger_push.warning("pywebpush not installed. Run: pip install pywebpush")
            return False
        
        vapid = WebPushService.get_vapid_keys()
        
        if not vapid['private_key']:
            logger_push.warning("VAPID keys not configured")
            return False
        
        # Build payload
        payload = {
            'title': title,
            'body': body,
            'icon': icon or '/static/icons/notification-icon-192.png',
            'badge': badge or '/static/icons/notification-badge-72.png',
            'tag': tag,
            'data': {
                'url': url,
                **(data or {})
            },
            'requireInteraction': require_interaction
        }
        
        if image:
            payload['image'] = image
        
        if actions:
            payload['actions'] = actions
        
        if vibrate:
            payload['vibrate'] = vibrate
        
        try:
            response = webpush(
                subscription_info=subscription_info,
                data=json.dumps(payload),
                vapid_private_key=vapid['private_key'],
                vapid_claims=vapid['claims'],
                ttl=ttl
            )
            
            logger_push.info(f"Web Push sent successfully: {response.status_code}")
            return response.status_code in (200, 201)
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle expired subscriptions
            if '410' in error_msg or 'gone' in error_msg.lower():
                logger_push.info("Web Push subscription expired, should be removed")
                raise ExpiredSubscriptionError("Subscription expired")
            
            logger_push.error(f"Failed to send Web Push: {e}")
            return False
    
    @staticmethod
    def generate_vapid_keys():
        """Generate new VAPID keys (utility function)."""
        try:
            from py_vapid import Vapid
            
            vapid = Vapid()
            vapid.generate_keys()
            
            return {
                'public_key': vapid.public_key,
                'private_key': vapid.private_key
            }
        except ImportError:
            logger_push.warning("py_vapid not installed. Run: pip install py-vapid")
            return None


class ExpiredSubscriptionError(Exception):
    """Raised when a push subscription has expired."""
    pass


class PushNotificationManager:
    """
    High-level manager for push notifications.
    Handles token management, batching, and fallbacks.
    """
    
    @staticmethod
    def send_to_user(
        user_id: int,
        title: str,
        body: str,
        notification_type: str = None,
        priority: str = 'high',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send push notification to all user's registered devices.
        
        Returns summary of sends.
        """
        from apps.notifications.models import PushToken
        
        tokens = PushToken.objects.filter(
            user_id=user_id,
            is_active=True
        ).values_list('token', 'device_type')
        
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'devices': []
        }
        
        for token, device_type in tokens:
            results['total'] += 1
            
            if device_type in ('ios', 'android', 'fcm_web'):
                # Use FCM
                success = FCMService.send_notification(
                    token=token,
                    title=title,
                    body=body,
                    priority=priority,
                    data={'type': notification_type, **(kwargs.get('data', {}))},
                    **{k: v for k, v in kwargs.items() if k != 'data'}
                )
            elif device_type == 'web':
                # Use Web Push
                try:
                    subscription = json.loads(token)
                    success = WebPushService.send_notification(
                        subscription_info=subscription,
                        title=title,
                        body=body,
                        **kwargs
                    )
                except Exception:
                    success = False
            else:
                success = False
            
            if success:
                results['success'] += 1
                results['devices'].append({'type': device_type, 'status': 'sent'})
            else:
                results['failed'] += 1
                results['devices'].append({'type': device_type, 'status': 'failed'})
        
        return results
    
    @staticmethod
    def send_bulk(
        user_ids: List[int],
        title: str,
        body: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send push notification to multiple users efficiently.
        Uses multicast for FCM tokens.
        """
        from apps.notifications.models import PushToken
        
        # Get all tokens
        tokens = PushToken.objects.filter(
            user_id__in=user_ids,
            is_active=True
        ).values_list('token', 'device_type')
        
        # Group by type
        fcm_tokens = []
        web_subscriptions = []
        
        for token, device_type in tokens:
            if device_type in ('ios', 'android', 'fcm_web'):
                fcm_tokens.append(token)
            elif device_type == 'web':
                try:
                    web_subscriptions.append(json.loads(token))
                except Exception:
                    pass
        
        results = {
            'fcm': None,
            'web_push': {'sent': 0, 'failed': 0}
        }
        
        # Send FCM multicast (batches of 500)
        if fcm_tokens:
            all_fcm_results = {'success': 0, 'failure': 0, 'failed_tokens': []}
            
            for i in range(0, len(fcm_tokens), 500):
                batch = fcm_tokens[i:i+500]
                batch_result = FCMService.send_multicast(
                    tokens=batch,
                    title=title,
                    body=body,
                    **kwargs
                )
                all_fcm_results['success'] += batch_result.get('success', 0)
                all_fcm_results['failure'] += batch_result.get('failure', 0)
                all_fcm_results['failed_tokens'].extend(batch_result.get('failed_tokens', []))
            
            results['fcm'] = all_fcm_results
            
            # Deactivate failed tokens
            if all_fcm_results['failed_tokens']:
                PushToken.objects.filter(
                    token__in=all_fcm_results['failed_tokens']
                ).update(is_active=False)
        
        # Send Web Push individually
        for subscription in web_subscriptions:
            try:
                success = WebPushService.send_notification(
                    subscription_info=subscription,
                    title=title,
                    body=body,
                    **kwargs
                )
                if success:
                    results['web_push']['sent'] += 1
                else:
                    results['web_push']['failed'] += 1
            except ExpiredSubscriptionError:
                results['web_push']['failed'] += 1
                # Deactivate expired subscription
                PushToken.objects.filter(
                    token=json.dumps(subscription)
                ).update(is_active=False)
        
        return results
    
    @staticmethod
    def register_token(
        user_id: int,
        token: str,
        device_type: str,
        device_name: str = None
    ):
        """Register or update a push token for a user."""
        from apps.notifications.models import PushToken
        
        obj, created = PushToken.objects.update_or_create(
            token=token,
            defaults={
                'user_id': user_id,
                'device_type': device_type,
                'device_name': device_name,
                'is_active': True
            }
        )
        
        return obj, created
    
    @staticmethod
    def unregister_token(token: str):
        """Unregister a push token."""
        from apps.notifications.models import PushToken
        
        return PushToken.objects.filter(token=token).update(is_active=False)


# =============================================================================
# SMS Notification Service
# =============================================================================

import requests

logger_sms = logging.getLogger('bunoraa.sms')


class SMSService:
    """
    Unified SMS service supporting multiple Bangladesh providers.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """Initialize SMS service with specified or default provider."""
        self.provider = provider or getattr(settings, 'SMS_PROVIDER', 'ssl_wireless')
        
        # Provider configurations
        self.providers = {
            'ssl_wireless': SSLWirelessSMS(),
            'bulksmsbd': BulkSMSBD(),
            'infobip': InfobipSMS(),
        }
    
    def send(
        self,
        phone: str,
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send SMS to a phone number.
        
        Args:
            phone: Phone number (with or without country code)
            message: Message content (max 160 chars for single SMS)
            sender_id: Optional sender ID
            
        Returns:
            Dict with status and message ID
        """
        # Normalize phone number for Bangladesh
        phone = self._normalize_phone(phone)
        
        provider = self.providers.get(self.provider)
        if not provider:
            return {
                'success': False,
                'message': f'Unknown SMS provider: {self.provider}'
            }
        
        try:
            return provider.send(phone, message, sender_id)
        except Exception as e:
            logger_sms.error(f"SMS send error: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def send_bulk(
        self,
        phones: List[str],
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send SMS to multiple phone numbers."""
        phones = [self._normalize_phone(p) for p in phones]
        
        provider = self.providers.get(self.provider)
        if not provider:
            return {
                'success': False,
                'message': f'Unknown SMS provider: {self.provider}'
            }
        
        try:
            return provider.send_bulk(phones, message, sender_id)
        except Exception as e:
            logger_sms.error(f"Bulk SMS error: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to Bangladesh format."""
        phone = ''.join(filter(str.isdigit, phone))
        
        if phone.startswith('880'):
            return phone
        elif phone.startswith('0'):
            return '880' + phone[1:]
        elif len(phone) == 10:
            return '880' + phone
        
        return phone


class SSLWirelessSMS:
    """
    SSL Wireless SMS provider for Bangladesh.
    Documentation: https://sslwireless.com/
    """
    
    API_URL = 'https://smsplus.sslwireless.com/api/v3/send-sms'
    
    def __init__(self):
        self.api_key = getattr(settings, 'SSL_WIRELESS_API_KEY', '')
        self.api_token = getattr(settings, 'SSL_WIRELESS_API_TOKEN', '')
        self.sid = getattr(settings, 'SSL_WIRELESS_SID', 'BUNORAA')
    
    def send(
        self,
        phone: str,
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send single SMS."""
        payload = {
            'api_token': self.api_token,
            'sid': sender_id or self.sid,
            'msisdn': phone,
            'sms': message,
            'csms_id': f'BUNORAA_{phone}_{int(__import__("time").time())}',
        }
        
        try:
            response = requests.post(
                self.API_URL,
                json=payload,
                timeout=30
            )
            
            result = response.json()
            
            if result.get('status') == 'SUCCESS':
                return {
                    'success': True,
                    'message_id': result.get('smsinfo', [{}])[0].get('csms_id'),
                    'raw': result
                }
            else:
                return {
                    'success': False,
                    'message': result.get('status_message', 'Unknown error'),
                    'raw': result
                }
                
        except Exception as e:
            logger_sms.error(f"SSL Wireless SMS error: {e}")
            return {'success': False, 'message': str(e)}
    
    def send_bulk(
        self,
        phones: List[str],
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send bulk SMS."""
        results = []
        for phone in phones:
            result = self.send(phone, message, sender_id)
            results.append({'phone': phone, **result})
        
        success_count = sum(1 for r in results if r.get('success'))
        
        return {
            'success': success_count > 0,
            'total': len(phones),
            'sent': success_count,
            'failed': len(phones) - success_count,
            'results': results
        }


class BulkSMSBD:
    """
    BulkSMS BD provider for Bangladesh.
    Documentation: https://bulksmsbd.com/
    """
    
    API_URL = 'http://bulksmsbd.net/api/smsapi'
    
    def __init__(self):
        self.api_key = getattr(settings, 'BULKSMS_API_KEY', '')
        self.sender_id = getattr(settings, 'BULKSMS_SENDER_ID', 'BUNORAA')
    
    def send(
        self,
        phone: str,
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send single SMS."""
        params = {
            'api_key': self.api_key,
            'type': 'text',
            'number': phone,
            'senderid': sender_id or self.sender_id,
            'message': message,
        }
        
        try:
            response = requests.get(
                self.API_URL,
                params=params,
                timeout=30
            )
            
            result = response.json()
            
            if result.get('response_code') == 202:
                return {
                    'success': True,
                    'message_id': result.get('message_id'),
                    'raw': result
                }
            else:
                return {
                    'success': False,
                    'message': result.get('error_message', 'Unknown error'),
                    'raw': result
                }
                
        except Exception as e:
            logger_sms.error(f"BulkSMS BD error: {e}")
            return {'success': False, 'message': str(e)}
    
    def send_bulk(
        self,
        phones: List[str],
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send bulk SMS."""
        # BulkSMS BD supports comma-separated numbers
        params = {
            'api_key': self.api_key,
            'type': 'text',
            'number': ','.join(phones),
            'senderid': sender_id or self.sender_id,
            'message': message,
        }
        
        try:
            response = requests.get(
                self.API_URL,
                params=params,
                timeout=60
            )
            
            result = response.json()
            
            return {
                'success': result.get('response_code') == 202,
                'total': len(phones),
                'raw': result
            }
            
        except Exception as e:
            logger_sms.error(f"BulkSMS BD bulk error: {e}")
            return {'success': False, 'message': str(e)}


class InfobipSMS:
    """
    Infobip SMS provider (international).
    Documentation: https://www.infobip.com/docs/api
    """
    
    def __init__(self):
        self.api_key = getattr(settings, 'INFOBIP_API_KEY', '')
        self.base_url = getattr(settings, 'INFOBIP_BASE_URL', 'https://api.infobip.com')
        self.sender_id = getattr(settings, 'INFOBIP_SENDER_ID', 'BUNORAA')
    
    def send(
        self,
        phone: str,
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send single SMS."""
        headers = {
            'Authorization': f'App {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'messages': [{
                'from': sender_id or self.sender_id,
                'destinations': [{'to': phone}],
                'text': message,
            }]
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/sms/2/text/advanced',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            result = response.json()
            
            if response.status_code == 200:
                message_result = result.get('messages', [{}])[0]
                return {
                    'success': message_result.get('status', {}).get('groupName') == 'PENDING',
                    'message_id': message_result.get('messageId'),
                    'raw': result
                }
            else:
                return {
                    'success': False,
                    'message': result.get('requestError', {}).get('serviceException', {}).get('text', 'Unknown error'),
                    'raw': result
                }
                
        except Exception as e:
            logger_sms.error(f"Infobip SMS error: {e}")
            return {'success': False, 'message': str(e)}
    
    def send_bulk(
        self,
        phones: List[str],
        message: str,
        sender_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send bulk SMS."""
        headers = {
            'Authorization': f'App {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'messages': [{
                'from': sender_id or self.sender_id,
                'destinations': [{'to': phone} for phone in phones],
                'text': message,
            }]
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/sms/2/text/advanced',
                headers=headers,
                json=payload,
                timeout=60
            )
            
            result = response.json()
            
            return {
                'success': response.status_code == 200,
                'total': len(phones),
                'raw': result
            }
            
        except Exception as e:
            logger_sms.error(f"Infobip bulk SMS error: {e}")
            return {'success': False, 'message': str(e)}


# SMS Templates for common notifications
class SMSTemplates:
    """Pre-defined SMS templates for Bangladesh (Bengali)."""
    
    @staticmethod
    def order_confirmation(order_number: str, total: str) -> str:
        return f"বুনরাআ: আপনার অর্ডার #{order_number} নিশ্চিত করা হয়েছে। মোট: ৳{total}। ধন্যবাদ!"
    
    @staticmethod
    def order_shipped(order_number: str, tracking: str) -> str:
        return f"বুনরাআ: আপনার অর্ডার #{order_number} শিপ করা হয়েছে। ট্র্যাকিং: {tracking}"
    
    @staticmethod
    def order_delivered(order_number: str) -> str:
        return f"বুনরাআ: আপনার অর্ডার #{order_number} ডেলিভারি হয়েছে। ধন্যবাদ!"
    
    @staticmethod
    def otp(code: str) -> str:
        return f"বুনরাআ: আপনার OTP কোড হলো {code}। ৫ মিনিটের মধ্যে ব্যবহার করুন।"
    
    @staticmethod
    def password_reset(code: str) -> str:
        return f"বুনরাআ: পাসওয়ার্ড রিসেট কোড: {code}। কাউকে শেয়ার করবেন না।"
    
    @staticmethod
    def payment_received(order_number: str, amount: str) -> str:
        return f"বুনরাআ: ৳{amount} পেমেন্ট পাওয়া গেছে অর্ডার #{order_number} এর জন্য।"
    
    @staticmethod
    def promotion(message: str) -> str:
        return f"বুনরাআ: {message} bunoraa.com এ ভিজিট করুন!"


# Singleton instance
sms_service = SMSService()