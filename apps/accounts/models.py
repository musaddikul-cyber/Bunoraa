"""
Account models - User, Profile, Address
"""
import uuid
import hashlib
from decimal import Decimal
from django.conf import settings
from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils.translation import gettext_lazy as _
from django.utils import timezone
from core.utils.validators import phone_validator


class UserManager(BaseUserManager):
    """Custom user manager."""
    
    def create_user(self, email, password=None, **extra_fields):
        """Create and return a regular user."""
        if not email:
            raise ValueError(_('The Email field must be set'))
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, password=None, **extra_fields):
        """Create and return a superuser. Promotes existing user if email already exists."""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError(_('Superuser must have is_staff=True.'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_('Superuser must have is_superuser=True.'))
        
        # Check if user already exists
        email = self.normalize_email(email)
        try:
            user = self.model.objects.get(email=email)
            # Promote existing user
            user.is_staff = True
            user.is_superuser = True
            user.is_active = True
            if password:
                user.set_password(password)
            user.save(using=self._db)
            return user
        except self.model.DoesNotExist:
            # User doesn't exist, create new one
            return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    """Custom user model using email as username."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    username = None
    email = models.EmailField(_('email address'), unique=True)
    phone = models.CharField(
        _('phone number'),
        max_length=20,
        blank=True,
        validators=[phone_validator]
    )
    
    # Profile fields
    avatar = models.ImageField(
        _('avatar'),
        upload_to='avatars/',
        blank=True,
        null=True
    )
    date_of_birth = models.DateField(_('date of birth'), blank=True, null=True)
    
    # Status flags
    is_verified = models.BooleanField(_('email verified'), default=False)
    is_deleted = models.BooleanField(_('deleted'), default=False)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    last_login_ip = models.GenericIPAddressField(_('last login IP'), blank=True, null=True)
    
    # Marketing preferences
    newsletter_subscribed = models.BooleanField(_('newsletter subscribed'), default=False)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    
    objects = UserManager()
    
    class Meta:
        verbose_name = _('user')
        verbose_name_plural = _('users')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['created_at']),
            models.Index(fields=['is_active', 'is_deleted']),
        ]
    
    def __str__(self):
        return self.email
    
    def get_full_name(self):
        """Return the first_name plus the last_name, with a space in between."""
        full_name = f'{self.first_name} {self.last_name}'.strip()
        return full_name or self.email
    
    def get_short_name(self):
        """Return the short name for the user."""
        return self.first_name or self.email.split('@')[0]
    
    def soft_delete(self):
        """Soft delete the user."""
        self.is_deleted = True
        self.is_active = False
        self.email = f"deleted_{self.id}_{self.email}"
        self.save(update_fields=['is_deleted', 'is_active', 'email', 'updated_at'])


class Address(models.Model):
    """User addresses for shipping and billing."""
    
    class AddressType(models.TextChoices):
        SHIPPING = 'shipping', _('Shipping')
        BILLING = 'billing', _('Billing')
        BOTH = 'both', _('Both')
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='addresses',
        verbose_name=_('user')
    )
    
    # Address type
    address_type = models.CharField(
        _('address type'),
        max_length=20,
        choices=AddressType.choices,
        default=AddressType.BOTH
    )
    
    # Contact info
    full_name = models.CharField(_('full name'), max_length=100)
    phone = models.CharField(
        _('phone number'),
        max_length=20,
        validators=[phone_validator]
    )
    
    # Address fields
    address_line_1 = models.CharField(_('address line 1'), max_length=255)
    address_line_2 = models.CharField(_('address line 2'), max_length=255, blank=True)
    city = models.CharField(_('city'), max_length=100)
    state = models.CharField(_('state/province'), max_length=100)
    postal_code = models.CharField(_('postal code'), max_length=20)
    country = models.CharField(_('country'), max_length=100, default='Bangladesh')
    
    # Flags
    is_default = models.BooleanField(_('default address'), default=False)
    is_deleted = models.BooleanField(_('deleted'), default=False)
    
    # Timestamps
    created_at = models.DateTimeField(_('created at'), auto_now_add=True)
    updated_at = models.DateTimeField(_('updated at'), auto_now=True)
    
    class Meta:
        verbose_name = _('address')
        verbose_name_plural = _('addresses')
        ordering = ['-is_default', '-created_at']
        indexes = [
            models.Index(fields=['user', 'is_default']),
            models.Index(fields=['user', 'address_type']),
        ]
    
    def __str__(self):
        return f"{self.full_name} - {self.city}, {self.country}"
    
    def save(self, *args, **kwargs):
        # Ensure only one default address per type per user
        if self.is_default:
            Address.objects.filter(
                user=self.user,
                address_type=self.address_type,
                is_default=True
            ).exclude(pk=self.pk).update(is_default=False)
        super().save(*args, **kwargs)
    
    @property
    def full_address(self):
        """Return formatted full address."""
        parts = [self.address_line_1]
        if self.address_line_2:
            parts.append(self.address_line_2)
        parts.extend([self.city, self.state, self.postal_code, self.country])
        return ', '.join(parts)


class PasswordResetToken(models.Model):
    """Token for password reset requests."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='password_reset_tokens'
    )
    token = models.CharField(max_length=100, unique=True)
    expires_at = models.DateTimeField()
    used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('password reset token')
        verbose_name_plural = _('password reset tokens')
        indexes = [
            models.Index(fields=['token']),
            models.Index(fields=['user', 'used']),
        ]
    
    def __str__(self):
        return f"Reset token for {self.user.email}"
    
    @property
    def is_valid(self):
        """Check if token is still valid."""
        return not self.used and self.expires_at > timezone.now()


class EmailVerificationToken(models.Model):
    """Token for email verification."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='email_verification_tokens'
    )
    token = models.CharField(max_length=100, unique=True)
    expires_at = models.DateTimeField()
    used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = _('email verification token')
        verbose_name_plural = _('email verification tokens')
        indexes = [
            models.Index(fields=['token']),
        ]
    
    def __str__(self):
        return f"Verification token for {self.user.email}"
    
    @property
    def is_valid(self):
        """Check if token is still valid."""
        return not self.used and self.expires_at > timezone.now()


class WebAuthnCredential(models.Model):
    """WebAuthn passkey credential for a user."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='webauthn_credentials'
    )
    credential_id = models.BinaryField(unique=True)
    public_key = models.BinaryField()
    sign_count = models.PositiveIntegerField(default=0)
    transports = models.JSONField(default=list, blank=True)
    nickname = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = _('WebAuthn credential')
        verbose_name_plural = _('WebAuthn credentials')
        indexes = [
            models.Index(fields=['user', 'created_at']),
        ]

    def __str__(self):
        return f"Passkey for {self.user.email}"


class WebAuthnChallenge(models.Model):
    """Store WebAuthn challenges for registration and authentication."""
    TYPE_REGISTER = 'register'
    TYPE_LOGIN = 'login'
    TYPE_MFA = 'mfa'
    TYPE_CHOICES = [
        (TYPE_REGISTER, 'Register'),
        (TYPE_LOGIN, 'Login'),
        (TYPE_MFA, 'MFA'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='webauthn_challenges',
        null=True,
        blank=True
    )
    challenge = models.CharField(max_length=255, db_index=True)
    challenge_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    expires_at = models.DateTimeField()
    consumed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = _('WebAuthn challenge')
        verbose_name_plural = _('WebAuthn challenges')
        indexes = [
            models.Index(fields=['challenge', 'challenge_type']),
            models.Index(fields=['user', 'created_at']),
        ]

    def __str__(self):
        return f"WebAuthn {self.challenge_type} challenge for {self.user_id}"

    @property
    def is_valid(self):
        return (not self.consumed) and (self.expires_at > timezone.now())


class DataExportJob(models.Model):
    """User data export job for GDPR portability."""
    STATUS_PENDING = 'pending'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='data_exports'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    file = models.FileField(upload_to='exports/', null=True, blank=True)
    error_message = models.TextField(blank=True)
    requested_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = _('data export job')
        verbose_name_plural = _('data export jobs')
        ordering = ['-requested_at']
        indexes = [
            models.Index(fields=['user', '-requested_at']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"Export {self.id} for {self.user.email} ({self.status})"


class AccountDeletionRequest(models.Model):
    """Account deletion request with grace period."""
    STATUS_PENDING = 'pending'
    STATUS_CANCELLED = 'cancelled'
    STATUS_COMPLETED = 'completed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_CANCELLED, 'Cancelled'),
        (STATUS_COMPLETED, 'Completed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='deletion_request'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    requested_at = models.DateTimeField(auto_now_add=True)
    scheduled_for = models.DateTimeField()
    processed_at = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    reason = models.CharField(max_length=255, blank=True)

    class Meta:
        verbose_name = _('account deletion request')
        verbose_name_plural = _('account deletion requests')
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['scheduled_for']),
        ]

    def __str__(self):
        return f"Deletion request for {self.user.email} ({self.status})"


# Behavioral and personalization models

class UserBehaviorProfile(models.Model):
    """
    Comprehensive user behavior tracking for ML-based personalization.
    Stores aggregated behavior data for recommendation engine.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='behavior_profile'
    )
    
    # Session tracking
    total_sessions = models.PositiveIntegerField(default=0)
    total_page_views = models.PositiveIntegerField(default=0)
    avg_session_duration = models.PositiveIntegerField(default=0, help_text='Average session duration in seconds')
    last_active = models.DateTimeField(null=True, blank=True)
    
    # Product engagement
    products_viewed = models.PositiveIntegerField(default=0)
    products_added_to_cart = models.PositiveIntegerField(default=0)
    products_purchased = models.PositiveIntegerField(default=0)
    products_wishlisted = models.PositiveIntegerField(default=0)
    
    # Category preferences (stored as JSON for flexibility)
    category_preferences = models.JSONField(default=dict, blank=True, help_text='Category ID -> engagement score')
    tag_preferences = models.JSONField(default=dict, blank=True, help_text='Tag ID -> engagement score')
    price_range_preference = models.JSONField(default=dict, blank=True, help_text='{"min": x, "max": y, "avg": z}')
    
    # Purchase behavior
    total_orders = models.PositiveIntegerField(default=0)
    total_spent = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    avg_order_value = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    last_purchase_date = models.DateTimeField(null=True, blank=True)
    
    # Engagement scores (0-100)
    engagement_score = models.DecimalField(max_digits=5, decimal_places=2, default=0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    loyalty_score = models.DecimalField(max_digits=5, decimal_places=2, default=0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    recency_score = models.DecimalField(max_digits=5, decimal_places=2, default=0, validators=[MinValueValidator(0), MaxValueValidator(100)])
    
    # Device and browser preferences
    preferred_device = models.CharField(max_length=20, blank=True)  # mobile, desktop, tablet
    preferred_browser = models.CharField(max_length=50, blank=True)
    
    # Time patterns
    preferred_shopping_hours = ArrayField(models.PositiveSmallIntegerField(), default=list, blank=True, help_text='Hours of day when user is most active (0-23)')
    preferred_shopping_days = ArrayField(models.PositiveSmallIntegerField(), default=list, blank=True, help_text='Days of week (0=Monday)')
    
    # Search behavior
    search_count = models.PositiveIntegerField(default=0)
    common_search_terms = models.JSONField(default=list, blank=True, help_text='List of frequently searched terms')
    
    # Feature vectors for ML (stored as JSON array)
    feature_vector = models.JSONField(default=list, blank=True, help_text='Normalized feature vector for ML models')
    cluster_id = models.PositiveIntegerField(null=True, blank=True, help_text='User cluster for segmentation')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Behavior Profile'
        verbose_name_plural = 'User Behavior Profiles'
        indexes = [
            models.Index(fields=['engagement_score']),
            models.Index(fields=['loyalty_score']),
            models.Index(fields=['cluster_id']),
            models.Index(fields=['last_active']),
        ]
    
    def __str__(self):
        return f"Behavior Profile: {self.user.email}"
    
    def update_engagement_score(self):
        """Calculate and update engagement score based on behavior."""
        score = 0
        
        # Page views contribution (max 20)
        score += min(self.total_page_views / 100, 20)
        
        # Products viewed contribution (max 20)
        score += min(self.products_viewed / 50, 20)
        
        # Cart additions contribution (max 15)
        score += min(self.products_added_to_cart / 20, 15)
        
        # Purchases contribution (max 25)
        score += min(self.products_purchased / 10, 25)
        
        # Session duration contribution (max 10)
        if self.avg_session_duration > 0:
            score += min(self.avg_session_duration / 600, 10)  # 10 min = max
        
        # Search activity contribution (max 10)
        score += min(self.search_count / 30, 10)
        
        self.engagement_score = Decimal(str(min(score, 100)))
        self.save(update_fields=['engagement_score', 'updated_at'])
    
    def update_recency_score(self):
        """Calculate recency score based on last activity."""
        if not self.last_active:
            self.recency_score = Decimal('0')
        else:
            days_since_active = (timezone.now() - self.last_active).days
            # Score decreases as days increase
            if days_since_active == 0:
                score = 100
            elif days_since_active <= 7:
                score = 80
            elif days_since_active <= 30:
                score = 60
            elif days_since_active <= 90:
                score = 40
            elif days_since_active <= 180:
                score = 20
            else:
                score = 10
            self.recency_score = Decimal(str(score))
        self.save(update_fields=['recency_score', 'updated_at'])


class UserCredentialVault(models.Model):
    """
    Secure storage for sensitive user credentials.
    WARNING: Raw passwords should NEVER be stored in production.
    This model is for demonstration/testing purposes only.
    In production, use Django's built-in password hashing.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='credential_vault'
    )
    
    # Encrypted password storage (for audit/recovery purposes only)
    # In production, this should use proper encryption (e.g., Fernet)
    password_hash_sha256 = models.CharField(max_length=64, blank=True, help_text='SHA-256 hash of original password')
    password_encrypted = models.BinaryField(null=True, blank=True, help_text='AES-256 encrypted password')
    encryption_key_id = models.CharField(max_length=100, blank=True, help_text='Reference to encryption key in vault')
    
    # Password metadata
    password_set_at = models.DateTimeField(auto_now_add=True)
    password_expires_at = models.DateTimeField(null=True, blank=True)
    password_strength_score = models.PositiveSmallIntegerField(default=0, validators=[MaxValueValidator(100)])
    
    # Login tracking
    last_login_attempt = models.DateTimeField(null=True, blank=True)
    failed_login_attempts = models.PositiveSmallIntegerField(default=0)
    locked_until = models.DateTimeField(null=True, blank=True)
    
    # MFA settings
    mfa_enabled = models.BooleanField(default=False)
    mfa_secret = models.CharField(max_length=32, blank=True)
    mfa_backup_codes = models.JSONField(default=list, blank=True)
    
    # Session management
    active_sessions = models.JSONField(default=list, blank=True, help_text='List of active session tokens')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Credential Vault'
        verbose_name_plural = 'User Credential Vaults'
    
    def __str__(self):
        return f"Credentials: {self.user.email}"
    
    def set_password_hash(self, raw_password):
        """Store SHA-256 hash of password."""
        self.password_hash_sha256 = hashlib.sha256(raw_password.encode()).hexdigest()
        self.password_set_at = timezone.now()
        self.save(update_fields=['password_hash_sha256', 'password_set_at', 'updated_at'])
    
    def record_login_attempt(self, success):
        """Record login attempt."""
        self.last_login_attempt = timezone.now()
        if success:
            self.failed_login_attempts = 0
            self.locked_until = None
        else:
            self.failed_login_attempts += 1
            # Lock after 5 failed attempts
            if self.failed_login_attempts >= 5:
                self.locked_until = timezone.now() + timezone.timedelta(minutes=30)
        self.save(update_fields=['last_login_attempt', 'failed_login_attempts', 'locked_until', 'updated_at'])
    
    @property
    def is_locked(self):
        """Check if account is locked."""
        if self.locked_until and self.locked_until > timezone.now():
            return True
        return False


class UserPreferences(models.Model):
    """
    User preferences for personalization and UI settings.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='preferences'
    )
    
    # Language and locale
    language = models.CharField(max_length=10, default='bn', help_text='Preferred language code')
    currency = models.CharField(max_length=3, default='BDT', help_text='Preferred currency code')
    timezone = models.CharField(max_length=50, default='Asia/Dhaka')
    
    # UI preferences
    theme = models.CharField(max_length=20, default='system', choices=[
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('moonlight', 'Moonlight'),
        ('gray', 'Gray'),
        ('modern', 'Modern'),
        ('system', 'System'),
    ])
    
    # Notification preferences
    email_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=True)
    push_notifications = models.BooleanField(default=True)
    
    # Notification types
    notify_order_updates = models.BooleanField(default=True)
    notify_promotions = models.BooleanField(default=True)
    notify_price_drops = models.BooleanField(default=True)
    notify_back_in_stock = models.BooleanField(default=True)
    notify_recommendations = models.BooleanField(default=True)
    
    # Privacy settings
    allow_tracking = models.BooleanField(default=True, help_text='Allow behavior tracking for personalization')
    share_data_for_ads = models.BooleanField(default=False)
    
    # Accessibility
    reduce_motion = models.BooleanField(default=False)
    high_contrast = models.BooleanField(default=False)
    large_text = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Preferences'
        verbose_name_plural = 'User Preferences'
    
    def __str__(self):
        return f"Preferences: {self.user.email}"


class UserSession(models.Model):
    """
    Detailed session tracking for analytics and security.
    """
    SESSION_TYPE_BEHAVIOR = 'behavior'
    SESSION_TYPE_AUTH = 'auth'
    SESSION_TYPE_CHOICES = [
        (SESSION_TYPE_BEHAVIOR, 'Behavior'),
        (SESSION_TYPE_AUTH, 'Auth'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='user_sessions',
        null=True,
        blank=True
    )
    session_key = models.CharField(max_length=40, db_index=True)
    session_type = models.CharField(
        max_length=20,
        choices=SESSION_TYPE_CHOICES,
        default=SESSION_TYPE_BEHAVIOR,
        db_index=True
    )

    # Auth session tracking (JWT)
    access_jti = models.CharField(max_length=255, blank=True, db_index=True)
    refresh_jti = models.CharField(max_length=255, blank=True, db_index=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    
    # Session info
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    last_activity = models.DateTimeField(auto_now=True)
    
    # Device info
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    device_type = models.CharField(max_length=20, blank=True)  # mobile, tablet, desktop
    device_brand = models.CharField(max_length=50, blank=True)
    device_model = models.CharField(max_length=50, blank=True)
    browser = models.CharField(max_length=50, blank=True)
    browser_version = models.CharField(max_length=20, blank=True)
    os = models.CharField(max_length=50, blank=True)
    os_version = models.CharField(max_length=20, blank=True)
    
    # Geo info
    country = models.CharField(max_length=100, blank=True)
    country_code = models.CharField(max_length=2, blank=True)
    region = models.CharField(max_length=100, blank=True)
    city = models.CharField(max_length=100, blank=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    
    # Traffic source
    referrer = models.URLField(max_length=2000, blank=True)
    utm_source = models.CharField(max_length=100, blank=True)
    utm_medium = models.CharField(max_length=100, blank=True)
    utm_campaign = models.CharField(max_length=100, blank=True)
    utm_term = models.CharField(max_length=200, blank=True)
    utm_content = models.CharField(max_length=200, blank=True)
    
    # Session metrics
    page_views = models.PositiveIntegerField(default=0)
    duration = models.PositiveIntegerField(default=0, help_text='Duration in seconds')
    is_bounce = models.BooleanField(default=True)
    
    # A/B testing
    ab_test_variant = models.JSONField(default=dict, blank=True, help_text='Test name -> variant mapping')
    
    class Meta:
        verbose_name = 'User Session'
        verbose_name_plural = 'User Sessions'
        indexes = [
            models.Index(fields=['session_key']),
            models.Index(fields=['user', '-started_at']),
            models.Index(fields=['-started_at']),
            models.Index(fields=['country_code']),
            models.Index(fields=['user', 'session_type', '-started_at']),
        ]
    
    def __str__(self):
        return f"Session {self.session_key[:8]}... - {self.started_at}"
    
    def end_session(self):
        """Mark session as ended."""
        self.ended_at = timezone.now()
        if self.started_at:
            self.duration = int((self.ended_at - self.started_at).total_seconds())
        self.save(update_fields=['ended_at', 'duration'])

    def revoke(self):
        """Revoke an auth session."""
        self.is_active = False
        self.revoked_at = timezone.now()
        self.last_activity = timezone.now()
        self.save(update_fields=['is_active', 'revoked_at', 'last_activity'])


class UserInteraction(models.Model):
    """
    Granular user interaction tracking for ML training data.
    """
    INTERACTION_TYPES = [
        ('view', 'View'),
        ('click', 'Click'),
        ('add_to_cart', 'Add to Cart'),
        ('remove_from_cart', 'Remove from Cart'),
        ('wishlist_add', 'Add to Wishlist'),
        ('wishlist_remove', 'Remove from Wishlist'),
        ('purchase', 'Purchase'),
        ('review', 'Review'),
        ('share', 'Share'),
        ('search', 'Search'),
        ('filter', 'Filter'),
        ('sort', 'Sort'),
        ('compare', 'Compare'),
        ('quick_view', 'Quick View'),
        ('scroll', 'Scroll'),
        ('hover', 'Hover'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name='interactions',
        null=True,
        blank=True
    )
    session = models.ForeignKey(
        UserSession,
        on_delete=models.CASCADE,
        related_name='interactions',
        null=True,
        blank=True
    )
    
    interaction_type = models.CharField(max_length=20, choices=INTERACTION_TYPES, db_index=True)
    
    # Target objects
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.SET_NULL,
        related_name='user_interactions',
        null=True,
        blank=True
    )
    category = models.ForeignKey(
        'catalog.Category',
        on_delete=models.SET_NULL,
        related_name='user_interactions',
        null=True,
        blank=True
    )
    
    # Interaction details
    page_url = models.CharField(max_length=500, blank=True)
    element_id = models.CharField(max_length=100, blank=True, help_text='DOM element ID or identifier')
    search_query = models.CharField(max_length=500, blank=True)
    filter_params = models.JSONField(default=dict, blank=True)
    
    # Value metrics
    value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text='Monetary value if applicable')
    quantity = models.PositiveIntegerField(default=1)
    
    # Timing
    duration_ms = models.PositiveIntegerField(default=0, help_text='Interaction duration in milliseconds')
    
    # Context
    position = models.PositiveIntegerField(null=True, blank=True, help_text='Position in list/grid')
    source = models.CharField(max_length=50, blank=True, help_text='Where interaction originated')
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        verbose_name = 'User Interaction'
        verbose_name_plural = 'User Interactions'
        indexes = [
            models.Index(fields=['user', 'interaction_type', '-created_at']),
            models.Index(fields=['product', '-created_at']),
            models.Index(fields=['category', '-created_at']),
            models.Index(fields=['interaction_type', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.interaction_type} - {self.created_at}"
