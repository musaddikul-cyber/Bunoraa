from django.contrib import admin
from .models import Keyword, KeywordURLMapping, SERPSnapshot, GSCMetric, ContentBrief, SitemapSubmission, SitemapError


@admin.register(Keyword)
class KeywordAdmin(admin.ModelAdmin):
    list_display = ('term', 'intent', 'monthly_volume', 'is_target', 'created_at')
    search_fields = ('term',)
    list_filter = ('intent', 'is_target')


@admin.register(KeywordURLMapping)
class MappingAdmin(admin.ModelAdmin):
    list_display = ('keyword', 'url', 'score', 'intent', 'created_at')
    search_fields = ('url', 'keyword__term')


@admin.register(SERPSnapshot)
class SERPSnapshotAdmin(admin.ModelAdmin):
    list_display = ('keyword', 'date', 'position', 'url', 'search_engine', 'source')
    list_filter = ('search_engine', 'source', 'date')
    search_fields = ('url', 'keyword__term')


@admin.register(GSCMetric)
class GSCMetricAdmin(admin.ModelAdmin):
    list_display = ('keyword', 'date', 'clicks', 'impressions', 'ctr', 'position')
    search_fields = ('keyword__term',)


@admin.register(ContentBrief)
class ContentBriefAdmin(admin.ModelAdmin):
    list_display = ('keyword', 'created_at', 'recommended_word_count')
    readonly_fields = ('top_urls', 'suggested_headings', 'top_terms', 'notes')
    search_fields = ('keyword__term',)


@admin.register(SitemapSubmission)
class SitemapSubmissionAdmin(admin.ModelAdmin):
    list_display = ('sitemap_type', 'status', 'submitted_at', 'last_read', 'discovered_pages', 'indexed_pages')
    list_filter = ('sitemap_type', 'status', 'submitted_at', 'last_read')
    search_fields = ('url',)
    readonly_fields = ('created_at', 'updated_at', 'url')
    
    fieldsets = (
        ('Sitemap Information', {
            'fields': ('sitemap_type', 'url', 'status')
        }),
        ('Submission Status', {
            'fields': ('submitted_at', 'last_read', 'search_engines')
        }),
        ('Discovery & Indexing', {
            'fields': ('discovered_pages', 'discovered_videos', 'indexed_pages')
        }),
        ('Errors', {
            'fields': ('errors',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(SitemapError)
class SitemapErrorAdmin(admin.ModelAdmin):
    list_display = ('error_code', 'severity', 'submission', 'resolved', 'created_at')
    list_filter = ('severity', 'resolved', 'created_at')
    search_fields = ('error_code', 'message', 'submission__url')
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Error Details', {
            'fields': ('error_code', 'severity', 'message')
        }),
        ('Associated Sitemap', {
            'fields': ('submission',)
        }),
        ('Affected URLs', {
            'fields': ('affected_urls',)
        }),
        ('Resolution', {
            'fields': ('resolved', 'resolved_at')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )