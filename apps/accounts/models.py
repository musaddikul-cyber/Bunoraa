"""
Account models - User, Profile, Address
"""
import uuid
from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
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


# Import behavior models so they're registered with Django
from .behavior_models import (  # noqa: F401, E402
    UserBehaviorProfile,
    UserCredentialVault,
    UserPreferences,
    UserSession,
)
