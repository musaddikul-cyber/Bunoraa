"""
Notifications services
"""
from datetime import timedelta
from zoneinfo import ZoneInfo
from django.core.mail import send_mail, EmailMultiAlternatives
from django.core.signing import TimestampSigner, BadSignature, SignatureExpired
from django.db import transaction
from django.template import Template, Context
from django.template.loader import render_to_string
from django.conf import settings
from django.utils import timezone

from .models import (
    Notification,
    NotificationType,
    NotificationChannel,
    NotificationCategory,
    NotificationPriority,
    NotificationStatus,
    DeliveryStatus,
    NotificationPreference,
    NotificationDelivery,
    NotificationTemplate,
    EmailTemplate,
    EmailLog,
)


# =============================================================================
# Notification preference resolution & orchestration
# =============================================================================

CRITICAL_TYPES = {
    NotificationType.PASSWORD_RESET,
    NotificationType.PAYMENT_FAILED,
}

TRANSACTIONAL_TYPES = {
    NotificationType.ORDER_PLACED,
    NotificationType.ORDER_CONFIRMED,
    NotificationType.ORDER_SHIPPED,
    NotificationType.ORDER_DELIVERED,
    NotificationType.ORDER_CANCELLED,
    NotificationType.ORDER_REFUNDED,
    NotificationType.PAYMENT_RECEIVED,
    NotificationType.PAYMENT_FAILED,
    NotificationType.REVIEW_APPROVED,
    NotificationType.REVIEW_REJECTED,
    NotificationType.ACCOUNT_CREATED,
    NotificationType.PASSWORD_RESET,
}

MARKETING_TYPES = {
    NotificationType.PRICE_DROP,
    NotificationType.BACK_IN_STOCK,
    NotificationType.WISHLIST_SALE,
    NotificationType.PROMO_CODE,
}


def resolve_category(notification_type: str) -> str:
    if notification_type in MARKETING_TYPES:
        return NotificationCategory.MARKETING
    if notification_type in TRANSACTIONAL_TYPES:
        return NotificationCategory.TRANSACTIONAL
    return NotificationCategory.SYSTEM


def resolve_priority(notification_type: str) -> str:
    if notification_type in CRITICAL_TYPES:
        return NotificationPriority.URGENT
    if notification_type in {
        NotificationType.ORDER_SHIPPED,
        NotificationType.ORDER_DELIVERED,
        NotificationType.PAYMENT_RECEIVED,
    }:
        return NotificationPriority.HIGH
    return NotificationPriority.NORMAL


class PreferenceResolver:
    """Resolve allowed channels and delivery timing based on preferences."""

    @staticmethod
    def _get_user_timezone(user, prefs: NotificationPreference):
        tz_name = prefs.timezone
        try:
            user_prefs = getattr(user, 'preferences', None)
            if user_prefs and user_prefs.timezone:
                tz_name = user_prefs.timezone
        except Exception:
            pass
        try:
            return ZoneInfo(tz_name or 'UTC')
        except Exception:
            return timezone.get_default_timezone()

    @staticmethod
    def _is_within_quiet_hours(now_local, prefs: NotificationPreference) -> bool:
        if not prefs.quiet_hours_start or not prefs.quiet_hours_end:
            return False
        start = prefs.quiet_hours_start
        end = prefs.quiet_hours_end
        current = now_local.time()

        if start < end:
            return start <= current < end
        # Overnight range (e.g., 22:00 -> 07:00)
        return current >= start or current < end

    @staticmethod
    def _next_quiet_hours_end(now_local, prefs: NotificationPreference):
        if not prefs.quiet_hours_start or not prefs.quiet_hours_end:
            return None
        end = prefs.quiet_hours_end
        today_end = now_local.replace(
            hour=end.hour,
            minute=end.minute,
            second=end.second,
            microsecond=0,
        )
        if today_end > now_local:
            return today_end
        return today_end + timedelta(days=1)

    @staticmethod
    def _type_allowed(prefs: NotificationPreference, notification_type: str, channel: str) -> bool:
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
        sms_type_map = {
            NotificationType.ORDER_PLACED: prefs.sms_order_updates,
            NotificationType.ORDER_CONFIRMED: prefs.sms_order_updates,
            NotificationType.ORDER_SHIPPED: prefs.sms_shipping_updates,
            NotificationType.ORDER_DELIVERED: prefs.sms_shipping_updates,
            NotificationType.ORDER_CANCELLED: prefs.sms_order_updates,
            NotificationType.ORDER_REFUNDED: prefs.sms_order_updates,
            NotificationType.PAYMENT_RECEIVED: prefs.sms_order_updates,
            NotificationType.PAYMENT_FAILED: prefs.sms_order_updates,
            NotificationType.PROMO_CODE: prefs.sms_promotions,
        }
        push_type_map = {
            NotificationType.ORDER_PLACED: prefs.push_order_updates,
            NotificationType.ORDER_CONFIRMED: prefs.push_order_updates,
            NotificationType.ORDER_SHIPPED: prefs.push_order_updates,
            NotificationType.ORDER_DELIVERED: prefs.push_order_updates,
            NotificationType.ORDER_CANCELLED: prefs.push_order_updates,
            NotificationType.PAYMENT_RECEIVED: prefs.push_order_updates,
            NotificationType.PAYMENT_FAILED: prefs.push_order_updates,
            NotificationType.PROMO_CODE: prefs.push_promotions,
        }

        if channel == NotificationChannel.EMAIL:
            return email_type_map.get(notification_type, True)
        if channel == NotificationChannel.SMS:
            return sms_type_map.get(notification_type, True)
        if channel == NotificationChannel.PUSH:
            return push_type_map.get(notification_type, True)
        return True

    @staticmethod
    def resolve(user, notification_type: str, category: str, priority: str, requested_channels=None):
        prefs, _ = NotificationPreference.objects.get_or_create(user=user)

        # Sync from user preferences if available
        try:
            user_prefs = getattr(user, 'preferences', None)
            if user_prefs and user_prefs.updated_at and user_prefs.updated_at > prefs.updated_at:
                NotificationService.sync_from_user_preferences(user_prefs, prefs)
        except Exception:
            pass

        allowed = {NotificationChannel.IN_APP}

        user_prefs = getattr(user, 'preferences', None)

        if prefs.email_enabled and (not user_prefs or user_prefs.email_notifications):
            if PreferenceResolver._type_allowed(prefs, notification_type, NotificationChannel.EMAIL):
                allowed.add(NotificationChannel.EMAIL)
        if prefs.sms_enabled and (not user_prefs or user_prefs.sms_notifications):
            if PreferenceResolver._type_allowed(prefs, notification_type, NotificationChannel.SMS):
                allowed.add(NotificationChannel.SMS)
        if prefs.push_enabled and (not user_prefs or user_prefs.push_notifications):
            if PreferenceResolver._type_allowed(prefs, notification_type, NotificationChannel.PUSH):
                allowed.add(NotificationChannel.PUSH)

        if category == NotificationCategory.MARKETING and not prefs.marketing_opt_in:
            allowed.discard(NotificationChannel.EMAIL)
            allowed.discard(NotificationChannel.SMS)
            allowed.discard(NotificationChannel.PUSH)
        if category == NotificationCategory.TRANSACTIONAL and not prefs.transactional_opt_in:
            allowed = {NotificationChannel.IN_APP}

        # Per-type overrides
        override = (prefs.per_type_overrides or {}).get(notification_type)
        if isinstance(override, dict):
            if override.get('enabled') is False:
                allowed = {NotificationChannel.IN_APP}
            if override.get('channels'):
                channel_values = set()
                for channel in override['channels']:
                    try:
                        channel_values.add(NotificationChannel(channel))
                    except Exception:
                        continue
                allowed = {NotificationChannel.IN_APP} | channel_values
            for ch in ['email', 'sms', 'push', 'in_app']:
                if ch in override:
                    try:
                        channel_enum = NotificationChannel(ch)
                    except Exception:
                        continue
                    if override[ch]:
                        allowed.add(channel_enum)
                    else:
                        allowed.discard(channel_enum)

        if requested_channels:
            requested_set = set()
            for channel in requested_channels:
                if isinstance(channel, str):
                    try:
                        requested_set.add(NotificationChannel(channel))
                    except Exception:
                        continue
                else:
                    requested_set.add(channel)
            allowed = allowed.intersection(requested_set)

        tz = PreferenceResolver._get_user_timezone(user, prefs)
        now_local = timezone.now().astimezone(tz)
        within_quiet = PreferenceResolver._is_within_quiet_hours(now_local, prefs)
        next_send_local = PreferenceResolver._next_quiet_hours_end(now_local, prefs) if within_quiet else None

        schedule = {
            'defer_until': None,
            'batch_email': False,
        }

        if within_quiet and priority != NotificationPriority.URGENT:
            if next_send_local:
                schedule['defer_until'] = next_send_local.astimezone(timezone.utc)

        if prefs.digest_frequency != NotificationPreference.DIGEST_IMMEDIATE:
            if category == NotificationCategory.MARKETING and priority != NotificationPriority.URGENT:
                schedule['batch_email'] = True

        return prefs, allowed, schedule


class NotificationOrchestrator:
    """Create notifications, deliveries, and dispatch tasks."""

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
        requested_channels=None,
        category=None,
        priority=None,
        dedupe_key=None,
        expires_at=None,
    ):
        category = category or resolve_category(notification_type)
        priority = priority or resolve_priority(notification_type)

        dedupe_ttl = int(getattr(settings, 'NOTIFICATION_DEDUPE_TTL_SECONDS', 3600))
        if dedupe_key:
            cutoff = timezone.now() - timedelta(seconds=dedupe_ttl)
            existing = Notification.objects.filter(
                user=user,
                dedupe_key=dedupe_key,
                created_at__gte=cutoff,
            ).first()
            if existing:
                return existing

        prefs, allowed_channels, schedule = PreferenceResolver.resolve(
            user=user,
            notification_type=notification_type,
            category=category,
            priority=priority,
            requested_channels=requested_channels,
        )

        with transaction.atomic():
            notification = Notification.objects.create(
                user=user,
                type=notification_type,
                title=title,
                message=message,
                url=url,
                reference_type=reference_type,
                reference_id=str(reference_id) if reference_id else None,
                metadata=metadata or {},
                category=category,
                priority=priority,
                status=NotificationStatus.PENDING,
                dedupe_key=dedupe_key,
                expires_at=expires_at,
                channels_requested=[c.value if hasattr(c, 'value') else c for c in allowed_channels],
                channels_sent=[],
            )

            # In-app delivery is considered sent immediately
            NotificationDelivery.objects.create(
                notification=notification,
                channel=NotificationChannel.IN_APP,
                status=DeliveryStatus.SENT,
                sent_at=timezone.now()
            )
            notification.channels_sent = [NotificationChannel.IN_APP.value]

            for channel in allowed_channels:
                if channel == NotificationChannel.IN_APP:
                    continue

                status = DeliveryStatus.PENDING
                scheduled_for = None
                if schedule.get('defer_until'):
                    scheduled_for = schedule['defer_until']
                    status = DeliveryStatus.QUEUED
                if channel == NotificationChannel.EMAIL and schedule.get('batch_email'):
                    status = DeliveryStatus.BATCHED

                delivery = NotificationDelivery.objects.create(
                    notification=notification,
                    channel=channel,
                    status=status,
                    scheduled_for=scheduled_for,
                )

                if status in {DeliveryStatus.PENDING, DeliveryStatus.QUEUED}:
                    NotificationOrchestrator.dispatch_delivery(delivery)

            notification.status = NotificationStatus.PROCESSING if len(allowed_channels) > 1 else NotificationStatus.SENT
            notification.save(update_fields=['channels_sent', 'status', 'updated_at'])

        return notification

    @staticmethod
    def dispatch_delivery(delivery: NotificationDelivery):
        from apps.notifications.tasks import send_notification_delivery
        if delivery.scheduled_for and delivery.scheduled_for > timezone.now():
            send_notification_delivery.apply_async(args=[str(delivery.id)], eta=delivery.scheduled_for)
        else:
            send_notification_delivery.delay(str(delivery.id))


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
        send_push=True,
        category=None,
        priority=None,
        dedupe_key=None,
        expires_at=None,
        requested_channels=None
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
        caller_channels = None
        if requested_channels is not None:
            caller_channels = requested_channels
        elif not send_email or not send_push:
            caller_channels = {
                NotificationChannel.IN_APP,
                NotificationChannel.EMAIL,
                NotificationChannel.SMS,
                NotificationChannel.PUSH,
            }
            if not send_email:
                caller_channels.discard(NotificationChannel.EMAIL)
            if not send_push:
                caller_channels.discard(NotificationChannel.PUSH)
            caller_channels = list(caller_channels)

        return NotificationOrchestrator.create_notification(
            user=user,
            notification_type=notification_type,
            title=title,
            message=message,
            url=url,
            reference_type=reference_type,
            reference_id=reference_id,
            metadata=metadata,
            requested_channels=caller_channels,
            category=category,
            priority=priority,
            dedupe_key=dedupe_key,
            expires_at=expires_at,
        )
    
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
    def sync_from_user_preferences(user_prefs, prefs: NotificationPreference = None):
        """Sync NotificationPreference fields from accounts.UserPreferences."""
        if not user_prefs:
            return prefs
        if prefs is None:
            prefs, _ = NotificationPreference.objects.get_or_create(user=user_prefs.user)

        updates = {
            'email_enabled': user_prefs.email_notifications,
            'sms_enabled': user_prefs.sms_notifications,
            'push_enabled': user_prefs.push_notifications,
            'email_order_updates': user_prefs.notify_order_updates,
            'email_shipping_updates': user_prefs.notify_order_updates,
            'email_promotions': user_prefs.notify_promotions,
            'email_price_drops': user_prefs.notify_price_drops,
            'email_back_in_stock': user_prefs.notify_back_in_stock,
            'push_order_updates': user_prefs.notify_order_updates,
            'push_promotions': user_prefs.notify_promotions,
            'sms_order_updates': user_prefs.notify_order_updates,
            'sms_shipping_updates': user_prefs.notify_order_updates,
            'sms_promotions': user_prefs.notify_promotions,
            'timezone': user_prefs.timezone or prefs.timezone,
        }

        NotificationPreference.objects.filter(id=prefs.id).update(**updates)
        for key, value in updates.items():
            setattr(prefs, key, value)
        return prefs

    @staticmethod
    def build_unsubscribe_token(email: str, user_id: str = '') -> str:
        signer = TimestampSigner(getattr(settings, 'NOTIFICATION_UNSUBSCRIBE_SECRET', settings.SECRET_KEY))
        return signer.sign(f"{email}:{user_id}")

    @staticmethod
    def verify_unsubscribe_token(token: str, max_age_seconds: int = 60 * 60 * 24 * 30):
        signer = TimestampSigner(getattr(settings, 'NOTIFICATION_UNSUBSCRIBE_SECRET', settings.SECRET_KEY))
        try:
            value = signer.unsign(token, max_age=max_age_seconds)
            email, user_id = value.split(':', 1)
            return email, user_id or None
        except (BadSignature, SignatureExpired):
            return None, None
    
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
        subject, text_body, html_body = EmailService._render_email_templates(
            user=user,
            notification_type=notification_type,
            context=context,
        )

        unsubscribe_headers = EmailService._build_unsubscribe_headers(user)

        notification = context.get('notification')
        delivery = context.get('delivery')

        # Create email log
        log = EmailLog.objects.create(
            recipient_email=user.email,
            recipient_user=user,
            notification_type=notification_type,
            subject=subject,
            reference_type=context.get('reference_type'),
            reference_id=context.get('reference_id'),
            notification=notification,
            delivery=delivery,
        )

        try:
            if EmailService._use_email_service():
                external_id = EmailService._send_via_email_service(
                    user=user,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                    headers=unsubscribe_headers,
                    metadata={
                        'notification_type': notification_type,
                        'notification_id': str(notification.id) if notification else None,
                        'delivery_id': str(delivery.id) if delivery else None,
                    }
                )
            else:
                email = EmailMultiAlternatives(
                    subject=subject,
                    body=text_body or '',
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to=[user.email],
                    headers=unsubscribe_headers or None,
                )
                if html_body:
                    email.attach_alternative(html_body, 'text/html')
                email.send()
                external_id = None

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
    def _render_email_templates(user, notification_type, context):
        language = 'en'
        try:
            user_prefs = getattr(user, 'preferences', None)
            if user_prefs and user_prefs.language:
                language = user_prefs.language.code if hasattr(user_prefs.language, 'code') else str(user_prefs.language)
        except Exception:
            pass

        context = {
            **context,
            'user': user,
            'site_name': getattr(settings, 'SITE_NAME', 'Bunoraa'),
            'site_url': getattr(settings, 'SITE_URL', 'https://bunoraa.com'),
            'physical_address': getattr(settings, 'NOTIFICATION_PHYSICAL_ADDRESS', ''),
            'unsubscribe_url': EmailService._build_unsubscribe_url(user),
        }

        from apps.i18n.services import LanguageService
        from apps.i18n.tasks import auto_translate_notification_template

        language_codes = LanguageService.get_fallback_chain(language)
        template = None
        matched_language = None
        for lang_code in language_codes:
            template = NotificationTemplate.objects.filter(
                notification_type=notification_type,
                channel=NotificationChannel.EMAIL,
                language=lang_code,
                is_active=True,
            ).first()
            if template:
                matched_language = lang_code
                break

        if template and matched_language and matched_language != language:
            try:
                auto_translate_notification_template.delay(notification_type, language, matched_language)
            except Exception:
                pass

        if template:
            subject = Template(template.subject or context.get('title', '')).render(Context(context))
            text_body = Template(template.text_template or template.body or context.get('message', '')).render(Context(context))
            html_body = Template(template.html_template or '').render(Context(context)) if template.html_template else None
            return subject, text_body, html_body

        # Fallback to legacy EmailTemplate
        legacy = EmailTemplate.objects.filter(
            notification_type=notification_type,
            is_active=True
        ).first()

        if legacy:
            subject = Template(legacy.subject).render(Context(context))
            text_body = Template(legacy.text_template).render(Context(context))
            html_body = Template(legacy.html_template).render(Context(context)) if legacy.html_template else None
            return subject, text_body, html_body

        subject = context.get('title', 'Notification from Bunoraa')
        text_body = context.get('message', '')
        html_body = None
        return subject, text_body, html_body

    @staticmethod
    def _build_unsubscribe_url(user):
        base_url = getattr(settings, 'NOTIFICATION_UNSUBSCRIBE_URL_BASE', '').strip()
        if not base_url:
            base_url = f"{getattr(settings, 'SITE_URL', 'https://bunoraa.com').rstrip('/')}/api/v1/notifications/unsubscribe/"
        token = NotificationService.build_unsubscribe_token(user.email, str(user.id))
        return f"{base_url}?token={token}"

    @staticmethod
    def _build_unsubscribe_headers(user):
        unsubscribe_email = getattr(settings, 'NOTIFICATION_UNSUBSCRIBE_EMAIL', 'unsubscribe@bunoraa.com')
        unsubscribe_url = EmailService._build_unsubscribe_url(user)
        return {
            'List-Unsubscribe': f"<mailto:{unsubscribe_email}>, <{unsubscribe_url}>",
            'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
        }

    @staticmethod
    def _use_email_service():
        return getattr(settings, 'EMAIL_SERVICE_ENABLED', False)

    @staticmethod
    def _send_via_email_service(user, subject, html_body, text_body, headers=None, metadata=None):
        try:
            from apps.email_service.models import EmailMessage
            from apps.email_service.engine import QueueManager
            import uuid

            message = EmailMessage.objects.create(
                api_key=None,
                user=user,
                message_id=str(uuid.uuid4()),
                from_email=settings.DEFAULT_FROM_EMAIL,
                from_name=getattr(settings, 'SITE_NAME', 'Bunoraa'),
                to_email=user.email,
                subject=subject,
                html_body=html_body or '',
                text_body=text_body or '',
                headers=headers or {},
                metadata=metadata or {},
                categories=['notifications'],
                tags=[metadata.get('notification_type')] if metadata else [],
            )
            QueueManager.enqueue(message)
            return message.message_id
        except Exception:
            raise
    
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


class DeliveryService:
    """Send notifications per channel and update delivery status."""

    @staticmethod
    def _render_channel_body(notification, channel: str, user):
        language = 'en'
        try:
            user_prefs = getattr(user, 'preferences', None)
            if user_prefs and user_prefs.language:
                language = user_prefs.language
        except Exception:
            pass

        context = {
            'title': notification.title,
            'message': notification.message,
            'url': notification.url,
            'notification': notification,
            'user': user,
            'metadata': notification.metadata,
        }

        template = NotificationTemplate.objects.filter(
            notification_type=notification.type,
            channel=channel,
            language=language,
            is_active=True
        ).first()

        if template and template.body:
            return Template(template.body).render(Context(context))
        return notification.message

    @staticmethod
    def send(delivery: NotificationDelivery):
        if delivery.channel == NotificationChannel.EMAIL:
            return DeliveryService.send_email(delivery)
        if delivery.channel == NotificationChannel.SMS:
            return DeliveryService.send_sms(delivery)
        if delivery.channel == NotificationChannel.PUSH:
            return DeliveryService.send_push(delivery)
        return {'success': True, 'provider': 'in_app'}

    @staticmethod
    def send_email(delivery: NotificationDelivery):
        notification = delivery.notification
        user = notification.user
        context = {
            'title': notification.title,
            'message': notification.message,
            'url': notification.url,
            'reference_type': notification.reference_type,
            'reference_id': notification.reference_id,
            'notification': notification,
            'delivery': delivery,
        }

        success = EmailService.send_notification_email(
            user=user,
            notification_type=notification.type,
            context=context
        )
        return {
            'success': success,
            'provider': 'email_service' if EmailService._use_email_service() else 'django',
            'external_id': None,
        }

    @staticmethod
    def send_sms(delivery: NotificationDelivery):
        notification = delivery.notification
        user = notification.user

        phone = getattr(user, 'phone_number', None)
        if not phone:
            return {'success': False, 'provider': 'sms', 'error': 'Missing phone number'}

        body = DeliveryService._render_channel_body(notification, NotificationChannel.SMS, user)
        if len(body) > 160:
            body = body[:157] + '...'

        sms_service = SMSService()
        result = sms_service.send(phone=phone, message=body)
        return {
            'success': result.get('success', False),
            'provider': sms_service.provider,
            'external_id': result.get('message_id'),
            'error': result.get('message') if not result.get('success') else None,
        }

    @staticmethod
    def send_push(delivery: NotificationDelivery):
        notification = delivery.notification
        user = notification.user

        body = DeliveryService._render_channel_body(notification, NotificationChannel.PUSH, user)
        title = notification.title or 'Notification'

        result = PushNotificationManager.send_to_user(
            user_id=user.id,
            title=title,
            body=body,
            notification_type=notification.type,
            data={
                'notification_id': str(notification.id),
                'type': notification.type,
            },
            click_action=notification.url,
        )
        success = result.get('success', 0) > 0 or result.get('success', False)
        return {
            'success': success,
            'provider': 'fcm/webpush',
            'external_id': None,
        }


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
        from apps.notifications.models import NotificationDelivery, NotificationPreference, NotificationChannel, DeliveryStatus
        
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
        
        # Only consider users with batched email deliveries in the time window
        users_with_notifications = NotificationDelivery.objects.filter(
            channel=NotificationChannel.EMAIL,
            status=DeliveryStatus.BATCHED,
            created_at__gte=since,
        ).values_list('notification__user_id', flat=True).distinct()
        
        # Filter by digest preference
        eligible_users = NotificationPreference.objects.filter(
            user_id__in=users_with_notifications,
            digest_frequency=frequency
        ).values_list('user_id', flat=True)

        return list(eligible_users)
    
    @staticmethod
    def generate_digest(user_id: int, since: timezone.datetime = None) -> Optional[Dict[str, Any]]:
        """
        Generate digest content for a user.
        
        Returns dict with:
        - summary stats
        - notification groups by type
        - recommendations
        """
        from apps.notifications.models import Notification, NotificationType, NotificationChannel, DeliveryStatus
        from apps.accounts.models import User
        
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
        
        if since is None:
            since = timezone.now() - timedelta(days=1)
        
        notifications = Notification.objects.filter(
            user_id=user_id,
            created_at__gte=since,
            deliveries__channel=NotificationChannel.EMAIL,
            deliveries__status=DeliveryStatus.BATCHED,
        ).distinct().order_by('-created_at')
        
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
            try:
                DigestService.mark_batched_deliveries_sent(
                    user_id=user_id,
                    since=digest.get('period_start'),
                )
            except Exception:
                pass
            return True
            
        except Exception as e:
            log.status = 'failed'
            log.error_message = str(e)
            log.save(update_fields=['status', 'error_message'])
            
            logger.error(f"Failed to send digest to {user.email}: {e}")
            return False

    @staticmethod
    def mark_batched_deliveries_sent(user_id: int, since: timezone.datetime = None):
        """
        Mark batched email deliveries as sent after a digest goes out.
        Also updates notification channels_sent and status.
        """
        from apps.notifications.models import NotificationDelivery, Notification, NotificationChannel, DeliveryStatus

        if since is None:
            since = timezone.now() - timedelta(days=1)

        deliveries = NotificationDelivery.objects.select_related('notification').filter(
            notification__user_id=user_id,
            notification__created_at__gte=since,
            channel=NotificationChannel.EMAIL,
            status=DeliveryStatus.BATCHED,
        )

        if not deliveries.exists():
            return

        now = timezone.now()
        deliveries.update(status=DeliveryStatus.SENT, sent_at=now, updated_at=now)

        notifications = Notification.objects.filter(deliveries__in=deliveries).distinct()
        for notification in notifications:
            channels_sent = set(notification.channels_sent or [])
            channels_sent.add(NotificationChannel.EMAIL.value)
            notification.channels_sent = list(channels_sent)
            notification.save(update_fields=['channels_sent', 'updated_at'])
            try:
                notification.update_status_from_deliveries()
            except Exception:
                continue
    
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

    @staticmethod
    def process_hourly_digests():
        """
        Process and send hourly digests.
        """
        users = DigestService.get_pending_digest_users(DigestFrequency.HOURLY)
        
        sent = 0
        failed = 0
        
        for user_id in users:
            digest = DigestService.generate_digest(
                user_id,
                since=timezone.now() - timedelta(hours=1)
            )
            
            if digest and digest['stats']['unread'] > 0:
                success = DigestService.send_digest_email(user_id, digest)
                if success:
                    sent += 1
                else:
                    failed += 1
        
        logger.info(f"Hourly digest processed: {sent} sent, {failed} failed")
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
