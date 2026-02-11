"""
Notifications API views
"""
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser
from rest_framework.permissions import IsAuthenticated, IsAdminUser, AllowAny
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from ..models import (
    Notification,
    NotificationPreference,
    NotificationDelivery,
    NotificationTemplate,
    PushToken,
)
from ..services import NotificationService
from .serializers import (
    NotificationSerializer,
    NotificationPreferenceSerializer,
    NotificationDeliverySerializer,
    NotificationTemplateSerializer,
    NotificationBroadcastSerializer,
    NotificationUnsubscribeSerializer,
    MarkReadSerializer,
    PushTokenSerializer,
)


class NotificationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for user notifications.
    
    GET /api/v1/notifications/ - List user notifications
    GET /api/v1/notifications/{id}/ - Get notification detail
    DELETE /api/v1/notifications/{id}/ - Delete notification
    GET /api/v1/notifications/unread_count/ - Get unread count
    POST /api/v1/notifications/mark_read/ - Mark notifications as read
    POST /api/v1/notifications/mark_all_read/ - Mark all as read
    """
    serializer_class = NotificationSerializer
    permission_classes = [IsAuthenticated]
    throttle_scope = 'notifications'
    
    def get_queryset(self):
        queryset = Notification.objects.filter(user=self.request.user)

        notif_type = self.request.query_params.get('type')
        if notif_type:
            queryset = queryset.filter(type=notif_type)

        category = self.request.query_params.get('category')
        if category:
            queryset = queryset.filter(category=category)

        status_param = self.request.query_params.get('status')
        if status_param:
            queryset = queryset.filter(status=status_param)

        priority = self.request.query_params.get('priority')
        if priority:
            queryset = queryset.filter(priority=priority)

        return queryset
    
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        
        # Filter by unread
        unread_only = request.query_params.get('unread', '').lower() == 'true'
        if unread_only:
            queryset = queryset.filter(is_read=False)
        
        # Pagination
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            return Response({
                'success': True,
                'message': 'Notifications retrieved successfully',
                'data': serializer.data,
                'meta': {
                    'count': self.paginator.page.paginator.count,
                    'unread_count': NotificationService.get_unread_count(request.user),
                    'next': response.data.get('next'),
                    'previous': response.data.get('previous')
                }
            })
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Notifications retrieved successfully',
            'data': serializer.data,
            'meta': {
                'count': len(serializer.data),
                'unread_count': NotificationService.get_unread_count(request.user)
            }
        })
    
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        
        # Auto-mark as read when viewed
        if not instance.is_read:
            instance.mark_as_read()
        
        serializer = self.get_serializer(instance)
        return Response({
            'success': True,
            'message': 'Notification retrieved successfully',
            'data': serializer.data,
            'meta': {}
        })
    
    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.delete()
        return Response({
            'success': True,
            'message': 'Notification deleted successfully',
            'data': {},
            'meta': {}
        })
    
    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        """Get count of unread notifications."""
        count = NotificationService.get_unread_count(request.user)
        return Response({
            'success': True,
            'message': 'Unread count retrieved successfully',
            'data': {'count': count},
            'meta': {}
        })
    
    @action(detail=False, methods=['post'])
    def mark_read(self, request):
        """Mark specific notifications as read."""
        serializer = MarkReadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        notification_ids = serializer.validated_data.get('notification_ids', [])
        
        if notification_ids:
            Notification.objects.filter(
                id__in=notification_ids,
                user=request.user
            ).update(is_read=True, read_at=timezone.now())
            count = len(notification_ids)
        else:
            count = 0
        
        return Response({
            'success': True,
            'message': f'{count} notifications marked as read',
            'data': {},
            'meta': {'marked_count': count}
        })
    
    @action(detail=False, methods=['post'])
    def mark_all_read(self, request):
        """Mark all notifications as read."""
        NotificationService.mark_all_as_read(request.user)
        return Response({
            'success': True,
            'message': 'All notifications marked as read',
            'data': {},
            'meta': {}
        })
    
    @action(detail=True, methods=['post'])
    def read(self, request, pk=None):
        """Mark a single notification as read."""
        success = NotificationService.mark_as_read(pk, request.user)
        if success:
            return Response({
                'success': True,
                'message': 'Notification marked as read',
                'data': {},
                'meta': {}
            })
        return Response({
            'success': False,
            'message': 'Notification not found',
            'data': {},
            'meta': {}
        }, status=status.HTTP_404_NOT_FOUND)

    @action(
        detail=False,
        methods=['post'],
        permission_classes=[IsAdminUser],
        throttle_scope='notifications_broadcast'
    )
    def broadcast(self, request):
        """Broadcast a notification to multiple users (admin only)."""
        serializer = NotificationBroadcastSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        User = get_user_model()
        user_ids = data.get('user_ids')
        if not user_ids:
            user_ids = list(User.objects.filter(is_active=True).values_list('id', flat=True))

        if not user_ids:
            return Response({
                'success': False,
                'message': 'No users found for broadcast.',
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)

        from apps.notifications.tasks import send_broadcast_notification

        chunk_size = int(getattr(settings, 'NOTIFICATION_BROADCAST_CHUNK_SIZE', 500))
        queued = 0
        for i in range(0, len(user_ids), chunk_size):
            chunk = user_ids[i:i + chunk_size]
            send_broadcast_notification.delay(
                chunk,
                data['notification_type'],
                data['title'],
                data['message'],
                data.get('url'),
                data.get('metadata') or {},
                data.get('category'),
                data.get('priority'),
                data.get('channels'),
                data.get('dedupe_key'),
            )
            queued += len(chunk)

        return Response({
            'success': True,
            'message': 'Broadcast queued successfully.',
            'data': {'queued': queued},
            'meta': {}
        })


class NotificationPreferenceViewSet(viewsets.ViewSet):
    """
    ViewSet for notification preferences.
    
    GET /api/v1/notifications/preferences/ - Get preferences
    PUT /api/v1/notifications/preferences/ - Update preferences
    """
    permission_classes = [IsAuthenticated]
    throttle_scope = 'notifications_preferences'
    
    def list(self, request):
        """Get user's notification preferences."""
        prefs, _ = NotificationPreference.objects.get_or_create(user=request.user)
        serializer = NotificationPreferenceSerializer(prefs)
        return Response({
            'success': True,
            'message': 'Preferences retrieved successfully',
            'data': serializer.data,
            'meta': {}
        })
    
    def update(self, request, pk=None):
        """Update user's notification preferences."""
        prefs, _ = NotificationPreference.objects.get_or_create(user=request.user)
        serializer = NotificationPreferenceSerializer(prefs, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        try:
            from apps.accounts.behavior_models import UserPreferences
            user_prefs, _ = UserPreferences.objects.get_or_create(user=request.user)
            user_prefs.email_notifications = prefs.email_enabled
            user_prefs.sms_notifications = prefs.sms_enabled
            user_prefs.push_notifications = prefs.push_enabled
            user_prefs.notify_order_updates = (
                prefs.email_order_updates or prefs.sms_order_updates or prefs.push_order_updates
            )
            user_prefs.notify_promotions = (
                prefs.email_promotions or prefs.sms_promotions or prefs.push_promotions
            )
            user_prefs.notify_price_drops = prefs.email_price_drops
            user_prefs.notify_back_in_stock = prefs.email_back_in_stock
            if prefs.timezone:
                user_prefs.timezone = prefs.timezone
            user_prefs.save(update_fields=[
                'email_notifications', 'sms_notifications', 'push_notifications',
                'notify_order_updates', 'notify_promotions', 'notify_price_drops',
                'notify_back_in_stock', 'timezone', 'updated_at'
            ])
        except Exception:
            pass
        
        return Response({
            'success': True,
            'message': 'Preferences updated successfully',
            'data': serializer.data,
            'meta': {}
        })


class PushTokenViewSet(viewsets.ViewSet):
    """
    ViewSet for push notification tokens.
    
    POST /api/v1/notifications/push-tokens/ - Register token
    DELETE /api/v1/notifications/push-tokens/{token}/ - Remove token
    """
    permission_classes = [IsAuthenticated]
    throttle_scope = 'notifications_push_tokens'
    
    def create(self, request):
        """Register a push token."""
        serializer = PushTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated = serializer.validated_data
        token, created = PushToken.objects.update_or_create(
            token=validated['token'],
            defaults={
                'user': request.user,
                'device_type': validated['device_type'],
                'device_name': validated.get('device_name'),
                'platform': validated.get('platform'),
                'app_version': validated.get('app_version'),
                'locale': validated.get('locale'),
                'timezone': validated.get('timezone'),
                'browser': validated.get('browser'),
                'user_agent': validated.get('user_agent') or request.META.get('HTTP_USER_AGENT', ''),
                'last_ip': request.META.get('REMOTE_ADDR'),
                'is_active': True,
            }
        )
        
        return Response({
            'success': True,
            'message': 'Token registered successfully',
            'data': {'token_id': str(token.id)},
            'meta': {'created': created}
        }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)
    
    def destroy(self, request, pk=None):
        """Remove a push token."""
        deleted = PushToken.objects.filter(
            token=pk,
            user=request.user
        ).delete()[0] > 0
        
        if deleted:
            return Response({
                'success': True,
                'message': 'Token removed successfully',
                'data': {},
                'meta': {}
            })
        
        return Response({
            'success': False,
            'message': 'Token not found',
            'data': {},
            'meta': {}
        }, status=status.HTTP_404_NOT_FOUND)


class NotificationDeliveryViewSet(viewsets.ReadOnlyModelViewSet):
    """Admin viewset for delivery logs."""
    serializer_class = NotificationDeliverySerializer
    permission_classes = [IsAdminUser]
    throttle_scope = 'notifications_deliveries'

    def get_queryset(self):
        queryset = NotificationDelivery.objects.select_related('notification', 'notification__user')

        status_param = self.request.query_params.get('status')
        if status_param:
            queryset = queryset.filter(status=status_param)

        channel = self.request.query_params.get('channel')
        if channel:
            queryset = queryset.filter(channel=channel)

        user_id = self.request.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(notification__user_id=user_id)

        notification_id = self.request.query_params.get('notification_id')
        if notification_id:
            queryset = queryset.filter(notification_id=notification_id)

        return queryset


class NotificationTemplateViewSet(viewsets.ModelViewSet):
    """Admin viewset for notification templates."""
    serializer_class = NotificationTemplateSerializer
    queryset = NotificationTemplate.objects.all()
    permission_classes = [IsAdminUser]
    throttle_scope = 'notifications_templates'


class NotificationUnsubscribeView(APIView):
    """Handle one-click unsubscribe requests."""
    authentication_classes = []
    permission_classes = [AllowAny]
    parser_classes = [JSONParser, FormParser]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'notifications_unsubscribe'

    def get(self, request):
        return self._handle(request)

    def post(self, request):
        return self._handle(request)

    def _handle(self, request):
        token = request.query_params.get('token') or request.data.get('token')
        serializer = NotificationUnsubscribeSerializer(data={'token': token})
        serializer.is_valid(raise_exception=True)

        email, user_id = NotificationService.verify_unsubscribe_token(serializer.validated_data['token'])
        if not email:
            return Response({
                'success': False,
                'message': 'Invalid or expired unsubscribe token.',
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)

        if user_id:
            prefs, _ = NotificationPreference.objects.get_or_create(user_id=user_id)
            prefs.marketing_opt_in = False
            prefs.email_promotions = False
            prefs.email_newsletter = False
            prefs.sms_promotions = False
            prefs.push_promotions = False
            prefs.save(update_fields=[
                'marketing_opt_in', 'email_promotions', 'email_newsletter',
                'sms_promotions', 'push_promotions', 'updated_at'
            ])

            try:
                from apps.email_service.models import Suppression
                Suppression.objects.get_or_create(
                    user_id=user_id,
                    email=email,
                    suppression_type=Suppression.SuppressionType.UNSUBSCRIBE,
                )
            except Exception:
                pass

        return Response({
            'success': True,
            'message': 'You have been unsubscribed from marketing notifications.',
            'data': {'email': email},
            'meta': {}
        })


class NotificationHealthView(APIView):
    """Admin health check for notification providers."""
    permission_classes = [IsAdminUser]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'notifications_health'

    def get(self, request):
        sms_provider = getattr(settings, 'SMS_PROVIDER', 'ssl_wireless')
        vapid_public = getattr(settings, 'VAPID_PUBLIC_KEY', '')
        vapid_private = getattr(settings, 'VAPID_PRIVATE_KEY', '')
        firebase_credentials = getattr(settings, 'FIREBASE_CREDENTIALS_PATH', None)

        return Response({
            'success': True,
            'message': 'Notification provider status',
            'data': {
                'email_service_enabled': getattr(settings, 'EMAIL_SERVICE_ENABLED', False),
                'sms_provider': sms_provider,
                'sms_configured': bool(getattr(settings, 'SSL_WIRELESS_API_TOKEN', '') or getattr(settings, 'BULKSMS_API_KEY', '') or getattr(settings, 'INFOBIP_API_KEY', '')),
                'vapid_configured': bool(vapid_public and vapid_private),
                'firebase_configured': bool(firebase_credentials),
            },
            'meta': {}
        })
