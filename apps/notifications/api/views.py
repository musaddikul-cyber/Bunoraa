"""
Notifications API views
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from ..models import Notification, NotificationPreference, PushToken
from ..services import NotificationService
from .serializers import (
    NotificationSerializer, NotificationPreferenceSerializer,
    MarkReadSerializer, PushTokenSerializer
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
    
    def get_queryset(self):
        return Notification.objects.filter(user=self.request.user)
    
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
            ).update(is_read=True)
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


class NotificationPreferenceViewSet(viewsets.ViewSet):
    """
    ViewSet for notification preferences.
    
    GET /api/v1/notifications/preferences/ - Get preferences
    PUT /api/v1/notifications/preferences/ - Update preferences
    """
    permission_classes = [IsAuthenticated]
    
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
    
    def create(self, request):
        """Register a push token."""
        serializer = PushTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        token, created = PushToken.objects.update_or_create(
            token=serializer.validated_data['token'],
            defaults={
                'user': request.user,
                'device_type': serializer.validated_data['device_type'],
                'device_name': serializer.validated_data.get('device_name'),
                'is_active': True
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
