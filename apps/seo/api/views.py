"""
SEO API endpoints.
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser, AllowAny
from django.shortcuts import get_object_or_404
from django.db.models import Sum, Avg, Count
from django.utils import timezone

from apps.seo.models import (
    SEOMetadata, Redirect, SitemapEntry, SearchRanking, Keyword,
    SitemapSubmission, SitemapError
)
from .serializers import (
    SEOMetadataSerializer,
    RedirectSerializer,
    SitemapEntrySerializer,
    SearchRankingSerializer,
    KeywordSerializer,
    SitemapSubmissionSerializer,
    SitemapErrorSerializer,
)



class SEOMetadataViewSet(viewsets.ModelViewSet):
    """SEO Metadata management."""
    queryset = SEOMetadata.objects.all()
    serializer_class = SEOMetadataSerializer
    permission_classes = [IsAdminUser]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        content_type = self.request.query_params.get('content_type')
        if content_type:
            queryset = queryset.filter(content_type__model=content_type)
        return queryset
    
    @action(detail=False, methods=['get'], permission_classes=[AllowAny])
    def for_url(self, request):
        """Get SEO metadata for a specific URL."""
        url = request.query_params.get('url', '')
        if not url:
            return Response({'error': 'URL required'}, status=status.HTTP_400_BAD_REQUEST)
        
        metadata = self.queryset.filter(canonical_url=url).first()
        if metadata:
            return Response(SEOMetadataSerializer(metadata).data)
        return Response({})


class RedirectViewSet(viewsets.ModelViewSet):
    """URL redirect management."""
    queryset = Redirect.objects.filter(is_active=True)
    serializer_class = RedirectSerializer
    permission_classes = [IsAdminUser]
    
    @action(detail=False, methods=['get'], permission_classes=[AllowAny])
    def check(self, request):
        """Check if URL has a redirect."""
        path = request.query_params.get('path', '')
        redirect = self.queryset.filter(old_path=path).first()
        if redirect:
            return Response({
                'redirect': True,
                'new_path': redirect.new_path,
                'type': redirect.redirect_type,
            })
        return Response({'redirect': False})


class SitemapViewSet(viewsets.ReadOnlyModelViewSet):
    """Sitemap entries."""
    queryset = SitemapEntry.objects.filter(is_active=True).order_by('-priority')
    serializer_class = SitemapEntrySerializer
    permission_classes = [AllowAny]


class SearchRankingViewSet(viewsets.ModelViewSet):
    """Search ranking tracking."""
    queryset = SearchRanking.objects.all()
    serializer_class = SearchRankingSerializer
    permission_classes = [IsAdminUser]
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get ranking summary statistics."""
        from django.db.models import Avg, Count
        
        stats = self.queryset.aggregate(
            avg_position=Avg('position'),
            total_keywords=Count('keyword', distinct=True),
        )
        return Response(stats)


class KeywordViewSet(viewsets.ModelViewSet):
    """Keyword management."""
    queryset = Keyword.objects.all()
    serializer_class = KeywordSerializer
    permission_classes = [IsAdminUser]


class SitemapSubmissionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing sitemap submissions"""
    queryset = SitemapSubmission.objects.all()
    serializer_class = SitemapSubmissionSerializer
    permission_classes = [IsAdminUser]
    filterset_fields = ['sitemap_type', 'status', 'submitted_at']
    search_fields = ['url']
    ordering_fields = ['submitted_at', 'last_read', 'created_at']
    ordering = ['-submitted_at']
    
    @action(detail=True, methods=['post'])
    def mark_submitted(self, request, pk=None):
        """Mark a sitemap as submitted to search engines"""
        sitemap = self.get_object()
        sitemap.status = 'submitted'
        sitemap.submitted_at = timezone.now()
        sitemap.save()
        return Response(
            {'status': 'Sitemap marked as submitted'},
            status=status.HTTP_200_OK
        )
    
    @action(detail=True, methods=['post'])
    def update_status(self, request, pk=None):
        """Update sitemap status and discovery metrics"""
        sitemap = self.get_object()
        
        # Update fields from request data
        if 'status' in request.data:
            sitemap.status = request.data['status']
        
        if 'last_read' in request.data:
            sitemap.last_read = request.data['last_read']
        
        if 'discovered_pages' in request.data:
            sitemap.discovered_pages = request.data['discovered_pages']
        
        if 'discovered_videos' in request.data:
            sitemap.discovered_videos = request.data['discovered_videos']
        
        if 'indexed_pages' in request.data:
            sitemap.indexed_pages = request.data['indexed_pages']
        
        if 'errors' in request.data:
            sitemap.errors = request.data['errors']
        
        sitemap.save()
        serializer = self.get_serializer(sitemap)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get a summary of all sitemap submissions"""
        submissions = self.get_queryset()
        
        summary = {
            'total': submissions.count(),
            'by_status': {
                'pending': submissions.filter(status='pending').count(),
                'submitted': submissions.filter(status='submitted').count(),
                'indexed': submissions.filter(status='indexed').count(),
                'error': submissions.filter(status='error').count(),
            },
            'by_type': {
                'static': submissions.filter(sitemap_type='static').count(),
                'products': submissions.filter(sitemap_type='products').count(),
                'categories': submissions.filter(sitemap_type='categories').count(),
                'blog': submissions.filter(sitemap_type='blog').count(),
            },
            'total_discovered_pages': submissions.aggregate(
                total=Sum('discovered_pages')
            )['total'] or 0,
            'total_indexed_pages': submissions.aggregate(
                total=Sum('indexed_pages')
            )['total'] or 0,
        }
        
        return Response(summary)


class SitemapErrorViewSet(viewsets.ModelViewSet):
    """ViewSet for managing sitemap errors"""
    queryset = SitemapError.objects.all()
    serializer_class = SitemapErrorSerializer
    permission_classes = [IsAdminUser]
    filterset_fields = ['severity', 'resolved', 'submission']
    search_fields = ['error_code', 'message']
    ordering_fields = ['created_at', 'severity']
    ordering = ['-created_at']
    
    @action(detail=True, methods=['post'])
    def mark_resolved(self, request, pk=None):
        """Mark an error as resolved"""
        from django.utils import timezone
        error = self.get_object()
        error.resolved = True
        error.resolved_at = timezone.now()
        error.save()
        return Response(
            {'status': 'Error marked as resolved'},
            status=status.HTTP_200_OK
        )