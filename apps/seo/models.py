from django.db import models
from django.utils import timezone


class Keyword(models.Model):
    INTENT_CHOICES = [
        ('informational', 'Informational'),
        ('transactional', 'Transactional'),
        ('navigational', 'Navigational'),
    ]

    term = models.CharField(max_length=255, unique=True)
    intent = models.CharField(max_length=24, choices=INTENT_CHOICES, default='informational')
    monthly_volume = models.IntegerField(null=True, blank=True)
    is_target = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_target', '-monthly_volume', 'term']

    def __str__(self):
        return self.term


class KeywordURLMapping(models.Model):
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE, related_name='mappings')
    url = models.CharField(max_length=2000)
    score = models.FloatField(default=0.0)
    intent = models.CharField(max_length=24, choices=Keyword.INTENT_CHOICES, default='informational')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['keyword', 'url']

    def __str__(self):
        return f"{self.keyword.term} -> {self.url}"


class SERPSnapshot(models.Model):
    """Top results snapshot for a keyword at a timestamp."""
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE, related_name='serp_snapshots')
    date = models.DateField(default=timezone.now)
    search_engine = models.CharField(max_length=50, default='google')
    position = models.PositiveSmallIntegerField()
    url = models.CharField(max_length=2000)
    title = models.CharField(max_length=512, blank=True)
    snippet = models.TextField(blank=True)
    raw = models.JSONField(null=True, blank=True)
    source = models.CharField(max_length=50, default='scrape')

    class Meta:
        ordering = ['keyword', 'date', 'position']

    def __str__(self):
        return f"{self.keyword.term} [{self.date}] #{self.position} -> {self.url}"


class GSCMetric(models.Model):
    """Store aggregated GSC metrics per keyword per day."""
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE, related_name='gsc_metrics')
    date = models.DateField(default=timezone.now)
    clicks = models.IntegerField(default=0)
    impressions = models.IntegerField(default=0)
    ctr = models.FloatField(default=0.0)
    position = models.FloatField(null=True, blank=True)
    raw = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['-date']
        unique_together = ['keyword', 'date']

    def __str__(self):
        return f"GSC {self.keyword.term} {self.date} p:{self.position} clicks:{self.clicks}"


class ContentBrief(models.Model):
    """A generated content brief for a keyword based on current SERP."""
    keyword = models.ForeignKey(Keyword, on_delete=models.CASCADE, related_name='content_briefs')
    created_at = models.DateTimeField(auto_now_add=True)
    generated_by = models.CharField(max_length=100, default='analysis')
    top_urls = models.JSONField(default=list)
    suggested_headings = models.JSONField(default=list)
    top_terms = models.JSONField(default=list)
    recommended_word_count = models.IntegerField(null=True, blank=True)
    notes = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Brief for {self.keyword.term} @ {self.created_at.date()}"


class SitemapSubmission(models.Model):
    """Track submitted sitemaps and their status"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('submitted', 'Submitted'),
        ('indexed', 'Indexed'),
        ('error', 'Error'),
    ]
    
    SITEMAP_TYPE_CHOICES = [
        ('static', 'Static Pages'),
        ('products', 'Products'),
        ('categories', 'Categories'),
        ('blog', 'Blog Posts'),
    ]
    
    sitemap_type = models.CharField(
        max_length=50,
        choices=SITEMAP_TYPE_CHOICES,
        help_text='Type of sitemap'
    )
    
    url = models.URLField(
        help_text='URL of the sitemap',
        unique=True
    )
    
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        help_text='Current submission status'
    )
    
    submitted_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When the sitemap was submitted'
    )
    
    last_read = models.DateTimeField(
        null=True,
        blank=True,
        help_text='When the search engine last read the sitemap'
    )
    
    discovered_pages = models.IntegerField(
        default=0,
        help_text='Number of pages discovered by search engine'
    )
    
    discovered_videos = models.IntegerField(
        default=0,
        help_text='Number of videos discovered by search engine'
    )
    
    indexed_pages = models.IntegerField(
        default=0,
        help_text='Number of pages indexed by search engine'
    )
    
    errors = models.TextField(
        blank=True,
        help_text='Any errors from the search engine'
    )
    
    search_engines = models.JSONField(
        default=list,
        blank=True,
        help_text='List of search engines this sitemap is submitted to'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Sitemap Submission'
        verbose_name_plural = 'Sitemap Submissions'
        ordering = ['-submitted_at', '-created_at']
        indexes = [
            models.Index(fields=['sitemap_type', 'status']),
            models.Index(fields=['-submitted_at']),
        ]
    
    def __str__(self):
        return f"{self.get_sitemap_type_display()} - {self.get_status_display()}"


class SitemapError(models.Model):
    """Track errors and issues with sitemaps"""
    
    SEVERITY_CHOICES = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('critical', 'Critical'),
    ]
    
    submission = models.ForeignKey(
        SitemapSubmission,
        on_delete=models.CASCADE,
        related_name='error_logs',
        null=True,
        blank=True
    )
    
    severity = models.CharField(
        max_length=20,
        choices=SEVERITY_CHOICES,
        default='warning'
    )
    
    error_code = models.CharField(max_length=100)
    message = models.TextField()
    
    affected_urls = models.JSONField(
        default=list,
        blank=True,
        help_text='URLs affected by this error'
    )
    
    resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Sitemap Error'
        verbose_name_plural = 'Sitemap Errors'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['severity', 'resolved']),
        ]
    
    def __str__(self):
        return f"{self.get_severity_display()}: {self.error_code}"