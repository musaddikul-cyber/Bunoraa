"""
Pages models
"""
import uuid
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.utils.text import slugify
from django.contrib.auth import get_user_model

User = get_user_model()


class NewsletterIncentive(models.Model):
    """
    Incentive campaigns for newsletter signups.
    Give discount codes to new subscribers.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Campaign details
    title = models.CharField(_('title'), max_length=255)
    description = models.TextField(_('description'), blank=True)
    
    # Incentive
    discount_percentage = models.PositiveIntegerField(
        _('discount percentage'),
        default=10,
        help_text=_('E.g., 10 for 10% off')
    )
    discount_code = models.CharField(
        _('discount code'),
        max_length=50,
        unique=True,
        help_text=_('E.g., WELCOME10')
    )
    min_order_amount = models.DecimalField(
        _('minimum order amount'),
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text=_('Minimum order amount required to use code')
    )
    max_uses = models.PositiveIntegerField(
        _('maximum uses'),
        null=True,
        blank=True,
        help_text=_('Leave blank for unlimited')
    )
    
    # Active status
    is_active = models.BooleanField(_('active'), default=True)
    valid_from = models.DateTimeField(_('valid from'), auto_now_add=True)
    valid_until = models.DateTimeField(_('valid until'), null=True, blank=True)
    
    # Tracking
    uses_count = models.PositiveIntegerField(default=0)
    signups_count = models.PositiveIntegerField(default=0)
    
    class Meta:
        verbose_name = _('newsletter incentive')
        verbose_name_plural = _('newsletter incentives')
        ordering = ['-valid_from']
    
    def __str__(self):
        return f"{self.title} ({self.discount_code})"
    
    @property
    def available_uses(self):
        """Get remaining uses if limited."""
        if self.max_uses is None:
            return '∞'
        return self.max_uses - self.uses_count


class SubscriberIncentive(models.Model):
    """
    Track which incentive was used when subscriber signed up.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    subscriber = models.OneToOneField(
        'pages.Subscriber',
        on_delete=models.CASCADE,
        related_name='incentive_info'
    )
    incentive = models.ForeignKey(
        NewsletterIncentive,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='subscribers'
    )
    
    discount_code_generated = models.CharField(
        _('discount code'),
        max_length=50,
        unique=True,
        help_text=_('Unique code sent to this subscriber')
    )
    code_used = models.BooleanField(default=False)
    code_used_date = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('subscriber incentive')
        verbose_name_plural = _('subscriber incentives')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.subscriber.email} - {self.discount_code_generated}"


class BlogCategory(models.Model):
    """Blog post categories."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(_('name'), max_length=100)
    slug = models.SlugField(_('slug'), max_length=120, unique=True)
    description = models.TextField(_('description'), blank=True)
    icon = models.CharField(_('icon class'), max_length=50, blank=True)
    
    class Meta:
        verbose_name = _('blog category')
        verbose_name_plural = _('blog categories')
        ordering = ['name']
    
    def __str__(self):
        return self.name


class BlogTag(models.Model):
    """Blog post tags."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(_('name'), max_length=100, unique=True)
    slug = models.SlugField(_('slug'), max_length=120, unique=True)
    
    class Meta:
        verbose_name = _('blog tag')
        verbose_name_plural = _('blog tags')
        ordering = ['name']
    
    def __str__(self):
        return self.name


class BlogPost(models.Model):
    """
    Blog post - Articles about embroidery care, techniques, inspiration.
    
    Featured posts:
    - Embroidery Care Guide
    - How to Choose Embroidery Design
    - Thread Types & Colors
    - Seasonal Embroidery Trends
    - Customer Embroidery Projects
    """
    STATUS_DRAFT = 'draft'
    STATUS_PUBLISHED = 'published'
    STATUS_ARCHIVED = 'archived'
    STATUS_CHOICES = [
        (STATUS_DRAFT, 'Draft'),
        (STATUS_PUBLISHED, 'Published'),
        (STATUS_ARCHIVED, 'Archived'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Content
    title = models.CharField(_('title'), max_length=255)
    slug = models.SlugField(_('slug'), max_length=255, unique=True)
    excerpt = models.CharField(_('excerpt'), max_length=500)
    content = models.TextField(_('content'))
    featured_image = models.ImageField(_('featured image'), upload_to='blog/')
    
    # Metadata
    category = models.ForeignKey(
        BlogCategory,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='posts'
    )
    tags = models.ManyToManyField(
        BlogTag,
        related_name='posts',
        blank=True
    )
    
    author = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='blog_posts'
    )
    
    # Status & Publishing
    status = models.CharField(
        _('status'),
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_DRAFT,
        db_index=True
    )
    published_at = models.DateTimeField(_('published at'), null=True, blank=True)
    
    # Engagement
    view_count = models.PositiveIntegerField(_('view count'), default=0)
    reading_time_minutes = models.PositiveIntegerField(
        _('reading time (minutes)'),
        default=5,
        help_text=_('Estimated reading time')
    )
    
    # SEO
    meta_title = models.CharField(_('meta title'), max_length=255, blank=True)
    meta_description = models.CharField(_('meta description'), max_length=500, blank=True)
    meta_keywords = models.CharField(_('meta keywords'), max_length=500, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('blog post')
        verbose_name_plural = _('blog posts')
        ordering = ['-published_at', '-created_at']
        indexes = [
            models.Index(fields=['status', '-published_at']),
            models.Index(fields=['category']),
        ]
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('blog:post_detail', kwargs={'slug': self.slug})
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        
        # Calculate reading time
        word_count = len(self.content.split())
        self.reading_time_minutes = max(1, word_count // 200)
        
        super().save(*args, **kwargs)


class BlogComment(models.Model):
    """Comments on blog posts."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    post = models.ForeignKey(
        BlogPost,
        on_delete=models.CASCADE,
        related_name='comments'
    )
    
    author = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='blog_comments',
        null=True,
        blank=True
    )
    author_name = models.CharField(max_length=100, blank=True)
    author_email = models.EmailField(blank=True)
    
    content = models.TextField()
    is_approved = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('blog comment')
        verbose_name_plural = _('blog comments')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Comment on {self.post.title}"


class Page(models.Model):
    """
    Static page model for CMS-like pages.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True)
    
    # Content
    content = models.TextField(blank=True)
    excerpt = models.TextField(blank=True, max_length=500)
    
    # SEO
    meta_title = models.CharField(max_length=200, blank=True)
    meta_description = models.TextField(blank=True, max_length=300)
    
    # Featured image
    featured_image = models.ImageField(upload_to='pages/', blank=True, null=True)
    
    # Template
    TEMPLATE_DEFAULT = 'default'
    TEMPLATE_LANDING = 'landing'
    TEMPLATE_CONTACT = 'contact'
    TEMPLATE_ABOUT = 'about'
    TEMPLATE_FAQ = 'faq'
    TEMPLATE_CHOICES = [
        (TEMPLATE_DEFAULT, 'Default'),
        (TEMPLATE_LANDING, 'Landing Page'),
        (TEMPLATE_CONTACT, 'Contact Page'),
        (TEMPLATE_ABOUT, 'About Page'),
        (TEMPLATE_FAQ, 'FAQ Page'),
    ]
    template = models.CharField(
        max_length=20,
        choices=TEMPLATE_CHOICES,
        default=TEMPLATE_DEFAULT
    )
    
    # Menu
    show_in_header = models.BooleanField(default=False)
    show_in_footer = models.BooleanField(default=False)
    menu_order = models.PositiveIntegerField(default=0)
    
    # Status
    is_published = models.BooleanField(default=False)
    published_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['menu_order', 'title']
        verbose_name = 'Page'
        verbose_name_plural = 'Pages'
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return f'/pages/{self.slug}/'


class FAQ(models.Model):
    """
    FAQ item model.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    question = models.CharField(max_length=500)
    answer = models.TextField()
    
    # Category
    category = models.CharField(max_length=100, blank=True)
    
    # Ordering
    sort_order = models.PositiveIntegerField(default=0)
    
    # Status
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['sort_order', 'created_at']
        verbose_name = 'FAQ'
        verbose_name_plural = 'FAQs'
    
    def __str__(self):
        return self.question[:100]


class ContactMessage(models.Model):
    """
    Contact form submission.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True)
    subject = models.CharField(max_length=200)
    message = models.TextField()
    
    # Status
    STATUS_NEW = 'new'
    STATUS_READ = 'read'
    STATUS_REPLIED = 'replied'
    STATUS_CLOSED = 'closed'
    STATUS_CHOICES = [
        (STATUS_NEW, 'New'),
        (STATUS_READ, 'Read'),
        (STATUS_REPLIED, 'Replied'),
        (STATUS_CLOSED, 'Closed'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_NEW
    )
    
    # Admin notes
    admin_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Contact Message'
        verbose_name_plural = 'Contact Messages'
    
    def __str__(self):
        return f"{self.name} - {self.subject}"


class SiteSettings(models.Model):
    """
    Site-wide settings (singleton).
    """
    # Basic info
    site_name = models.CharField(max_length=100, default='Bunoraa')
    site_tagline = models.CharField(max_length=200, blank=True)
    site_description = models.TextField(blank=True)
    
    # Logo
    logo = models.ImageField(upload_to='site/', blank=True, null=True)
    logo_dark = models.ImageField(upload_to='site/', blank=True, null=True)
    favicon = models.ImageField(upload_to='site/', blank=True, null=True)
    
    # Contact info
    contact_email = models.EmailField(blank=True)
    contact_phone = models.CharField(max_length=20, blank=True)
    contact_address = models.TextField(blank=True)
    
    # Social links
    facebook_url = models.URLField(blank=True)
    instagram_url = models.URLField(blank=True)
    twitter_url = models.URLField(blank=True)
    linkedin_url = models.URLField(blank=True)
    youtube_url = models.URLField(blank=True)
    tiktok_url = models.URLField(blank=True)
    
    # SEO defaults
    default_meta_title = models.CharField(max_length=200, blank=True)
    default_meta_description = models.TextField(blank=True, max_length=300)
    
    # E-commerce settings
    currency = models.CharField(max_length=3, default='BDT')
    currency_symbol = models.CharField(max_length=5, default='৳')
    tax_rate = models.DecimalField(max_digits=5, decimal_places=2, default=10)
    

    
    # Footer content
    footer_text = models.TextField(blank=True)
    copyright_text = models.CharField(max_length=200, blank=True)
    
    # Scripts
    google_analytics_id = models.CharField(max_length=50, blank=True)
    facebook_pixel_id = models.CharField(max_length=50, blank=True)
    custom_head_scripts = models.TextField(blank=True)
    custom_body_scripts = models.TextField(blank=True)
    
    # Timestamps
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Site Settings'
        verbose_name_plural = 'Site Settings'
    
    def __str__(self):
        return 'Site Settings'
    
    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)
        # Clear cached site settings
        self._clear_cache()
    
    def delete(self, *args, **kwargs):
        pass  # Prevent deletion
    
    @staticmethod
    def _clear_cache():
        """Clear the site settings cache."""
        from django.core.cache import cache
        cache.delete('site_settings_context')
    
    @classmethod
    def get_settings(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj


from django.core.exceptions import ValidationError


def validate_svg_file(fieldfile_obj):
    """Basic validator for SVG files: checks extension and content type when available."""
    name = getattr(fieldfile_obj, 'name', '')
    if name and not name.lower().endswith('.svg'):
        raise ValidationError('Only SVG files are allowed for SVG icon field.')


class SocialLink(models.Model):
    """
    Social link to show in footer and emails. Managed as part of SiteSettings (inline in admin).
    """
    site = models.ForeignKey('pages.SiteSettings', on_delete=models.CASCADE, related_name='social_links', null=True, blank=True)
    name = models.CharField(max_length=100)
    url = models.URLField()
    def validate_icon_file(fieldfile_obj):
        """Allow common raster images and SVG for icons."""
        name = getattr(fieldfile_obj, 'name', '')
        if name:
            allowed = ('.svg', '.png', '.jpg', '.jpeg', '.gif', '.webp')
            if not any(name.lower().endswith(ext) for ext in allowed):
                raise ValidationError('Icon must be one of: svg, png, jpg, jpeg, gif, webp')

    icon = models.FileField(upload_to='site/social/', validators=[validate_icon_file], blank=True, null=True, help_text='Upload an image or SVG (preferred).')
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['order']
        verbose_name = 'Social Link'
        verbose_name_plural = 'Social Links'

    def __str__(self):
        return self.name

    def get_icon_url(self):
        """Return the stored icon URL (svg or raster)."""
        if self.icon:
            try:
                return self.icon.url
            except Exception:
                pass
        return None

    def save(self, *args, **kwargs):
        # Ensure the SocialLink is associated with the singleton SiteSettings when possible
        if not self.site:
            try:
                self.site = SiteSettings.get_settings()
            except Exception:
                try:
                    from apps.pages.models import SiteSettings
                    self.site = SiteSettings.get_settings()
                except Exception:
                    self.site = None
        super().save(*args, **kwargs)
        # Clear cached site settings
        try:
            SiteSettings._clear_cache()
        except Exception:
            from django.core.cache import cache
            cache.delete('site_settings_context')


class Subscriber(models.Model):
    """
    Newsletter subscriber.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=100, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_verified = models.BooleanField(default=False)
    
    # Source
    source = models.CharField(max_length=50, default='website')
    
    # Timestamps
    subscribed_at = models.DateTimeField(auto_now_add=True)
    unsubscribed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-subscribed_at']
        verbose_name = 'Subscriber'
        verbose_name_plural = 'Subscribers'
    
    def __str__(self):
        return self.email
