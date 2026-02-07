import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.exceptions import ValidationError

class ReferralCode(models.Model):
    """
    Represents a unique referral code associated with a user.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='referral_code',
        null=True, # Allow codes not yet assigned to a user (e.g., for new sign-ups)
        blank=True
    )
    code = models.CharField(max_length=20, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = 'Referral Code'
        verbose_name_plural = 'Referral Codes'
        ordering = ['-created_at']

    def __str__(self):
        return self.code

    @property
    def is_expired(self):
        return self.expires_at and self.expires_at < timezone.now()

    def generate_unique_code(self):
        import string
        import random
        length = 8
        characters = string.ascii_uppercase + string.digits
        while True:
            code = ''.join(random.choice(characters) for i in range(length))
            if not ReferralCode.objects.filter(code=code).exists():
                self.code = code
                break

    def save(self, *args, **kwargs):
        if not self.code:
            self.generate_unique_code()
        super().save(*args, **kwargs)


class ReferralReward(models.Model):
    """
    Tracks rewards earned by referrer and referee.
    """
    REWARD_TYPES = [
        ('discount', 'Discount'),
        ('store_credit', 'Store Credit'),
        ('free_product', 'Free Product'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('earned', 'Earned'),
        ('applied', 'Applied'),
        ('cancelled', 'Cancelled'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    referral_code = models.ForeignKey(
        ReferralCode,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='rewards_generated',
        help_text='The referral code that generated this reward.'
    )
    referrer_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='referral_rewards_given',
        help_text='The user who referred (earned the reward).'
    )
    referee_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='referral_rewards_received',
        help_text='The new user who was referred (received the reward).'
    )
    reward_type = models.CharField(max_length=20, choices=REWARD_TYPES)
    value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    description = models.TextField(blank=True, help_text='Details of the reward (e.g., 10% off, $10 credit).')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    earned_at = models.DateTimeField(null=True, blank=True)
    applied_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Referral Reward'
        verbose_name_plural = 'Referral Rewards'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.reward_type} reward for {self.referrer_user or self.referee_user}"

    def clean(self):
        if not self.referrer_user and not self.referee_user:
            raise ValidationError("Either a referrer user or a referee user must be associated with the reward.")
        if self.referral_code and not self.referrer_user:
            raise ValidationError("A referral code must be linked to a referrer user.")