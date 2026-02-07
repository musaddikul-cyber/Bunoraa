"""
Contacts Models
"""
import uuid
from django.db import models
from django.conf import settings


class ContactCategory(models.Model):
    """Category for contact inquiries."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    
    # Email routing
    email_recipients = models.TextField(
        blank=True,
        help_text='Comma-separated list of email addresses to receive inquiries'
    )
    
    # Auto-response
    auto_response_subject = models.CharField(max_length=200, blank=True)
    auto_response_message = models.TextField(blank=True)
    
    # Display
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Contact Category'
        verbose_name_plural = 'Contact Categories'
        ordering = ['order', 'name']
    
    def __str__(self):
        return self.name
    
    def get_recipients_list(self):
        """Return list of recipient email addresses."""
        if not self.email_recipients:
            return []
        return [email.strip() for email in self.email_recipients.split(',') if email.strip()]


class ContactInquiry(models.Model):
    """Contact form submission."""
    
    STATUS_CHOICES = [
        ('new', 'New'),
        ('in_progress', 'In Progress'),
        ('responded', 'Responded'),
        ('closed', 'Closed'),
        ('spam', 'Spam'),
    ]
    
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('normal', 'Normal'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Contact info
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=30, blank=True)
    company = models.CharField(max_length=150, blank=True)
    
    # Inquiry details
    category = models.ForeignKey(
        ContactCategory,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='inquiries'
    )
    subject = models.CharField(max_length=200)
    message = models.TextField()
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='normal')
    
    # Order reference (if applicable)
    order_number = models.CharField(max_length=50, blank=True)
    
    # User association
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='contact_inquiries'
    )
    
    # Tracking
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    source_page = models.URLField(blank=True, help_text='Page from which the form was submitted')
    
    # Response tracking
    responded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='responded_inquiries'
    )
    responded_at = models.DateTimeField(null=True, blank=True)
    
    # Auto-response
    auto_response_sent = models.BooleanField(default=False)
    
    # Admin notes
    internal_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Contact Inquiry'
        verbose_name_plural = 'Contact Inquiries'
        ordering = ['-created_at']
    
    def __str__(self):
        return f'{self.subject} - {self.name}'
    
    @property
    def is_new(self):
        return self.status == 'new'
    
    @property
    def is_responded(self):
        return self.status in ['responded', 'closed']


class ContactResponse(models.Model):
    """Response to a contact inquiry."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    inquiry = models.ForeignKey(
        ContactInquiry,
        on_delete=models.CASCADE,
        related_name='responses'
    )
    
    # Response content
    subject = models.CharField(max_length=200)
    message = models.TextField()
    
    # Sender
    sent_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    
    # Email tracking
    sent_at = models.DateTimeField(null=True, blank=True)
    delivered = models.BooleanField(default=False)
    opened = models.BooleanField(default=False)
    opened_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Contact Response'
        verbose_name_plural = 'Contact Responses'
        ordering = ['-created_at']
    
    def __str__(self):
        return f'Response to: {self.inquiry.subject}'



class CustomizationRequest(models.Model):
    """
    A request from a customer for a product customization.
    """
    STATUS_CHOICES = [
        ('new', 'New'),
        ('in_progress', 'In Progress'),
        ('quoted', 'Quoted'),
        ('ordered', 'Ordered'),
        ('closed', 'Closed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='customization_requests'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='customization_requests'
    )
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=30, blank=True)
    message = models.TextField(help_text="Please describe the customization you want in detail.")
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    
    # Tracking
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Admin notes
    internal_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Customization Request'
        verbose_name_plural = 'Customization Requests'
        ordering = ['-created_at']

    def __str__(self):
        return f'Customization request for {self.product.name} from {self.name}'


class ContactAttachment(models.Model):
    """Attachment for contact inquiries."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    inquiry = models.ForeignKey(
        ContactInquiry,
        on_delete=models.CASCADE,
        related_name='attachments'
    )
    
    file = models.FileField(upload_to='contacts/attachments/%Y/%m/')
    filename = models.CharField(max_length=255)
    file_size = models.PositiveIntegerField(default=0)
    content_type = models.CharField(max_length=100, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Contact Attachment'
        verbose_name_plural = 'Contact Attachments'
    
    def __str__(self):
        return self.filename
    
    def save(self, *args, **kwargs):
        if self.file:
            self.filename = self.file.name
            self.file_size = self.file.size
        super().save(*args, **kwargs)


class StoreLocation(models.Model):
    """Physical store location."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    
    # Address
    address_line1 = models.CharField(max_length=255)
    address_line2 = models.CharField(max_length=255, blank=True)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100, blank=True)
    postal_code = models.CharField(max_length=20)
    country = models.CharField(max_length=100)
    
    # Coordinates
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    # Contact info
    phone = models.CharField(max_length=30, blank=True)
    email = models.EmailField(blank=True)
    fax = models.CharField(max_length=30, blank=True)
    
    # Hours
    monday_hours = models.CharField(max_length=50, blank=True)
    tuesday_hours = models.CharField(max_length=50, blank=True)
    wednesday_hours = models.CharField(max_length=50, blank=True)
    thursday_hours = models.CharField(max_length=50, blank=True)
    friday_hours = models.CharField(max_length=50, blank=True)
    saturday_hours = models.CharField(max_length=50, blank=True)
    sunday_hours = models.CharField(max_length=50, blank=True)
    
    # Features
    is_pickup_location = models.BooleanField(default=False)
    is_returns_location = models.BooleanField(default=False)
    
    # Pickup settings (for checkout)
    pickup_fee = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    min_pickup_time_hours = models.PositiveIntegerField(default=2, help_text="Minimum hours before pickup is ready")
    max_hold_days = models.PositiveIntegerField(default=7, help_text="Maximum days to hold for pickup")
    
    # Display
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='contacts/locations/', blank=True)
    order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    is_main = models.BooleanField(default=False, help_text='Main/headquarters location')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Store Location'
        verbose_name_plural = 'Store Locations'
        ordering = ['order', 'name']
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if self.is_main:
            StoreLocation.objects.filter(is_main=True).exclude(pk=self.pk).update(is_main=False)
        super().save(*args, **kwargs)
    
    @property
    def full_address(self):
        parts = [self.address_line1]
        if self.address_line2:
            parts.append(self.address_line2)
        parts.append(f'{self.city}, {self.state} {self.postal_code}'.strip())
        parts.append(self.country)
        return ', '.join(parts)
    
    def get_hours(self):
        """Return hours as a dictionary."""
        return {
            'monday': self.monday_hours,
            'tuesday': self.tuesday_hours,
            'wednesday': self.wednesday_hours,
            'thursday': self.thursday_hours,
            'friday': self.friday_hours,
            'saturday': self.saturday_hours,
            'sunday': self.sunday_hours,
        }


class ContactSettings(models.Model):
    """Global contact settings (singleton)."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # General contact info
    general_email = models.EmailField(blank=True)
    support_email = models.EmailField(blank=True)
    sales_email = models.EmailField(blank=True)
    phone = models.CharField(max_length=30, blank=True)
    
    # Social media
    facebook_url = models.URLField(blank=True)
    twitter_url = models.URLField(blank=True)
    instagram_url = models.URLField(blank=True)
    linkedin_url = models.URLField(blank=True)
    youtube_url = models.URLField(blank=True)
    pinterest_url = models.URLField(blank=True)
    tiktok_url = models.URLField(blank=True)
    
    # Form settings
    allow_attachments = models.BooleanField(default=True)
    max_attachment_size_mb = models.PositiveIntegerField(default=5)
    allowed_file_types = models.CharField(
        max_length=255,
        default='jpg,jpeg,png,pdf,doc,docx',
        help_text='Comma-separated list of allowed file extensions'
    )
    
    # Notifications
    notify_on_new_inquiry = models.BooleanField(default=True)
    notification_emails = models.TextField(
        blank=True,
        help_text='Comma-separated list of email addresses for notifications'
    )
    
    # Auto-response
    enable_auto_response = models.BooleanField(default=True)
    default_auto_response_subject = models.CharField(max_length=200, blank=True)
    default_auto_response_message = models.TextField(blank=True)
    
    # Business hours
    business_hours_note = models.TextField(
        blank=True,
        help_text='Note about response times during business hours'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Contact Settings'
        verbose_name_plural = 'Contact Settings'
    
    def __str__(self):
        return 'Contact Settings'
    
    def save(self, *args, **kwargs):
        # Ensure only one instance exists
        self.pk = ContactSettings.objects.first().pk if ContactSettings.objects.exists() else self.pk
        super().save(*args, **kwargs)
    
    @classmethod
    def get_settings(cls):
        """Get or create settings."""
        settings, _ = cls.objects.get_or_create(defaults={})
        return settings
    
    def get_notification_emails_list(self):
        """Return list of notification email addresses."""
        if not self.notification_emails:
            return []
        return [email.strip() for email in self.notification_emails.split(',') if email.strip()]
    
    def get_allowed_file_types_list(self):
        """Return list of allowed file types."""
        if not self.allowed_file_types:
            return []
        return [ext.strip().lower() for ext in self.allowed_file_types.split(',') if ext.strip()]
