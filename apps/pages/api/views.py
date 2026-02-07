"""
Pages API views
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter

from ..models import Page, FAQ, ContactMessage, SiteSettings, Subscriber
from ..services import PageService, FAQService, ContactService, SubscriberService
from .serializers import (
    PageListSerializer, PageDetailSerializer, FAQSerializer,
    ContactMessageSerializer, ContactMessageCreateSerializer,
    SiteSettingsSerializer, SubscriberCreateSerializer,
    UnsubscribeSerializer, MenuPageSerializer
)


class PageViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for pages (public).
    
    GET /api/v1/pages/ - List published pages
    GET /api/v1/pages/{slug}/ - Get page detail
    GET /api/v1/pages/menu/ - Get menu pages
    GET /api/v1/pages/footer/ - Get footer pages
    """
    queryset = Page.objects.filter(is_published=True)
    permission_classes = [AllowAny]
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['title', 'content']
    ordering_fields = ['created_at', 'menu_order']
    ordering = ['menu_order']
    lookup_field = 'slug'
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return PageDetailSerializer
        if self.action in ['menu', 'footer']:
            return MenuPageSerializer
        return PageListSerializer
    
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Pages retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })
    
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response({
            'success': True,
            'message': 'Page retrieved successfully',
            'data': serializer.data,
            'meta': {}
        })
    
    @action(detail=False, methods=['get'])
    def menu(self, request):
        """Get pages for main menu."""
        pages = PageService.get_menu_pages()
        serializer = self.get_serializer(pages, many=True)
        return Response({
            'success': True,
            'message': 'Menu pages retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })
    
    @action(detail=False, methods=['get'])
    def footer(self, request):
        """Get pages for footer."""
        pages = PageService.get_footer_pages()
        serializer = self.get_serializer(pages, many=True)
        return Response({
            'success': True,
            'message': 'Footer pages retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })


class FAQViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for FAQs.
    
    GET /api/v1/faqs/ - List all FAQs
    GET /api/v1/faqs/{id}/ - Get FAQ detail
    GET /api/v1/faqs/grouped/ - Get FAQs grouped by category
    GET /api/v1/faqs/categories/ - Get FAQ categories
    """
    queryset = FAQ.objects.filter(is_active=True).order_by('sort_order')
    serializer_class = FAQSerializer
    permission_classes = [AllowAny]
    filter_backends = [DjangoFilterBackend, SearchFilter]
    filterset_fields = ['category']
    search_fields = ['question', 'answer']
    
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'FAQs retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })
    
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response({
            'success': True,
            'message': 'FAQ retrieved successfully',
            'data': serializer.data,
            'meta': {}
        })
    
    @action(detail=False, methods=['get'])
    def grouped(self, request):
        """Get FAQs grouped by category."""
        faqs = self.get_queryset()
        
        # Group by category
        categories = {}
        for faq in faqs:
            cat = faq.category or 'General'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(faq)
        
        # Format response
        data = []
        for category, faq_list in categories.items():
            data.append({
                'category': category,
                'faqs': FAQSerializer(faq_list, many=True).data
            })
        
        return Response({
            'success': True,
            'message': 'FAQs retrieved successfully',
            'data': data,
            'meta': {'category_count': len(data)}
        })
    
    @action(detail=False, methods=['get'])
    def categories(self, request):
        """Get list of FAQ categories."""
        categories = list(FAQService.get_faq_categories())
        return Response({
            'success': True,
            'message': 'FAQ categories retrieved successfully',
            'data': categories,
            'meta': {'count': len(categories)}
        })


class ContactMessageViewSet(viewsets.ModelViewSet):
    """
    ViewSet for contact messages.
    
    POST /api/v1/contact/ - Submit contact message (public)
    GET /api/v1/contact/ - List messages (admin)
    GET /api/v1/contact/{id}/ - Get message detail (admin)
    POST /api/v1/contact/{id}/mark_read/ - Mark as read (admin)
    POST /api/v1/contact/{id}/reply/ - Reply to message (admin)
    """
    queryset = ContactMessage.objects.all().order_by('-created_at')
    filter_backends = [DjangoFilterBackend, SearchFilter]
    filterset_fields = ['is_read', 'is_replied']
    search_fields = ['name', 'email', 'subject', 'message']
    
    def get_permissions(self):
        if self.action == 'create':
            return [AllowAny()]
        return [IsAdminUser()]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return ContactMessageCreateSerializer
        return ContactMessageSerializer
    
    def create(self, request, *args, **kwargs):
        """Submit a contact message."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        message = serializer.save()
        
        return Response({
            'success': True,
            'message': 'Your message has been sent successfully',
            'data': ContactMessageSerializer(message).data,
            'meta': {}
        }, status=status.HTTP_201_CREATED)
    
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            return Response({
                'success': True,
                'message': 'Contact messages retrieved successfully',
                'data': serializer.data,
                'meta': {
                    'count': self.paginator.page.paginator.count,
                    'next': response.data.get('next'),
                    'previous': response.data.get('previous')
                }
            })
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'message': 'Contact messages retrieved successfully',
            'data': serializer.data,
            'meta': {'count': len(serializer.data)}
        })
    
    @action(detail=True, methods=['post'])
    def mark_read(self, request, pk=None):
        """Mark message as read."""
        ContactService.mark_as_read(pk)
        return Response({
            'success': True,
            'message': 'Message marked as read',
            'data': {},
            'meta': {}
        })
    
    @action(detail=True, methods=['post'])
    def reply(self, request, pk=None):
        """Reply to a message."""
        reply_text = request.data.get('reply', '')
        if not reply_text:
            return Response({
                'success': False,
                'message': 'Reply text is required',
                'data': {},
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)
        
        message = ContactService.reply_to_message(pk, reply_text, request.user)
        if not message:
            return Response({
                'success': False,
                'message': 'Message not found',
                'data': {},
                'meta': {}
            }, status=status.HTTP_404_NOT_FOUND)
        
        return Response({
            'success': True,
            'message': 'Reply sent successfully',
            'data': ContactMessageSerializer(message).data,
            'meta': {}
        })


class SiteSettingsViewSet(viewsets.ViewSet):
    """
    ViewSet for site settings.
    
    GET /api/v1/settings/ - Get public site settings
    """
    permission_classes = [AllowAny]
    
    def list(self, request):
        """Get public site settings."""
        settings = SiteSettings.get_settings()
        serializer = SiteSettingsSerializer(settings)
        return Response({
            'success': True,
            'message': 'Site settings retrieved successfully',
            'data': serializer.data,
            'meta': {}
        })


class SubscriberViewSet(viewsets.ViewSet):
    """
    ViewSet for newsletter subscribers.
    
    POST /api/v1/subscribers/ - Subscribe to newsletter
    POST /api/v1/subscribers/unsubscribe/ - Unsubscribe
    GET /api/v1/subscribers/ - List subscribers (admin)
    """
    def get_permissions(self):
        if self.action in ['create', 'unsubscribe']:
            return [AllowAny()]
        return [IsAdminUser()]
    
    def create(self, request):
        """Subscribe to newsletter."""
        serializer = SubscriberCreateSerializer(data=request.data)
        
        if not serializer.is_valid():
            # Check if already subscribed error
            errors = serializer.errors
            if 'email' in errors:
                for error in errors['email']:
                    if 'already subscribed' in str(error).lower():
                        return Response({
                            'success': False,
                            'message': 'This email is already subscribed',
                            'data': {},
                            'meta': {}
                        }, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({
                'success': False,
                'message': 'Validation error',
                'data': errors,
                'meta': {}
            }, status=status.HTTP_400_BAD_REQUEST)
        
        result = SubscriberService.subscribe(
            email=serializer.validated_data['email'],
            name=serializer.validated_data.get('name')
        )
        
        return Response({
            'success': result['success'],
            'message': result['message'],
            'data': {},
            'meta': {}
        }, status=status.HTTP_201_CREATED if result['success'] else status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def unsubscribe(self, request):
        """Unsubscribe from newsletter."""
        serializer = UnsubscribeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        result = SubscriberService.unsubscribe(serializer.validated_data['email'])
        
        return Response({
            'success': result['success'],
            'message': result['message'],
            'data': {},
            'meta': {}
        })
    
    def list(self, request):
        """List subscribers (admin only)."""
        subscribers = Subscriber.objects.filter(is_active=True).order_by('-created_at')
        data = list(subscribers.values('id', 'email', 'name', 'source', 'created_at'))
        
        return Response({
            'success': True,
            'message': 'Subscribers retrieved successfully',
            'data': data,
            'meta': {'count': len(data)}
        })
