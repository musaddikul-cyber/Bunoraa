# apps/subscriptions/models.py
"""Production-ready subscription models: Plan and Subscription with timestamps, UUID PKs,
soft delete, and fields needed for recurring billing and Stripe integration."""
from __future__ import annotations

import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.exceptions import ValidationError


class TimeStampedModel(models.Model):
    """Abstract model that provides created/updated timestamps."""

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Plan(TimeStampedModel):
    """Productizable subscription plan that maps to Stripe Price."""

    INTERVAL_MONTH = "month"
    INTERVAL_YEAR = "year"
    INTERVAL_CHOICES = [
        (INTERVAL_MONTH, "Monthly"),
        (INTERVAL_YEAR, "Yearly"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    stripe_price_id = models.CharField(max_length=255, blank=True, db_index=True, help_text="Stripe Price ID (e.g. price_XXX)")
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    interval = models.CharField(max_length=10, choices=INTERVAL_CHOICES)
    price_amount = models.DecimalField(max_digits=12, decimal_places=2, help_text="Decimal unit amount (e.g. 9.99)")
    currency = models.CharField(max_length=3, default="BDT")
    active = models.BooleanField(default=True)
    trial_period_days = models.PositiveIntegerField(default=0, help_text="Number of trial days offered")
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["price_amount"]
        verbose_name = "Plan"
        verbose_name_plural = "Plans"

    def __str__(self):
        return f"{self.name} ({self.interval})"


class SubscriptionQuerySet(models.QuerySet):
    def active(self):
        return self.filter(status=Subscription.STATUS_ACTIVE, is_deleted=False)


class SubscriptionManager(models.Manager):
    def get_queryset(self):
        return SubscriptionQuerySet(self.model, using=self._db)

    def active(self):
        return self.get_queryset().active()


class Subscription(TimeStampedModel):
    """A user's subscription to a plan with Stripe integration metadata and billing fields."""

    STATUS_TRIALING = "trialing"
    STATUS_ACTIVE = "active"
    STATUS_PAST_DUE = "past_due"
    STATUS_CANCELED = "canceled"
    STATUS_UNPAID = "unpaid"

    STATUS_CHOICES = [
        (STATUS_TRIALING, "Trialing"),
        (STATUS_ACTIVE, "Active"),
        (STATUS_PAST_DUE, "Past Due"),
        (STATUS_CANCELED, "Canceled"),
        (STATUS_UNPAID, "Unpaid"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="subscriptions")
    plan = models.ForeignKey(Plan, on_delete=models.PROTECT, related_name="subscriptions")

    # Stripe identifiers
    stripe_subscription_id = models.CharField(max_length=255, blank=True, null=True, db_index=True, help_text="External Stripe subscription reference")

    # Lifecycle fields
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_ACTIVE, db_index=True)
    is_deleted = models.BooleanField(default=False, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    # Periods
    current_period_start = models.DateTimeField(null=True, blank=True)
    current_period_end = models.DateTimeField(null=True, blank=True)

    # Cancellation and end dates
    canceled_at = models.DateTimeField(null=True, blank=True, help_text="When user requested cancellation")
    ended_at = models.DateTimeField(null=True, blank=True, help_text="When subscription actually ended")

    # Billing
    next_billing_at = models.DateTimeField(null=True, blank=True, db_index=True)
    quantity = models.PositiveIntegerField(default=1)

    # Trial
    trial_ends = models.DateTimeField(null=True, blank=True)

    # Metadata
    metadata = models.JSONField(default=dict, blank=True)

    objects = SubscriptionManager()

    class Meta:
        verbose_name = "Subscription"
        verbose_name_plural = "Subscriptions"
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["status"]),
            models.Index(fields=["stripe_subscription_id"]),
        ]
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} -> {self.plan} ({self.status})"

    def soft_delete(self):
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save(update_fields=["is_deleted", "deleted_at"])

    def restore(self):
        self.is_deleted = False
        self.deleted_at = None
        self.save(update_fields=["is_deleted", "deleted_at"])

    def clean(self):
        # Ensure trial_ends is later than created_at if set
        if self.trial_ends and self.trial_ends <= self.created_at:
            raise ValidationError("trial_ends must be after creation time")

    @property
    def is_active(self):
        return self.status == self.STATUS_ACTIVE and not self.is_deleted

    def mark_canceled(self, at=None):
        self.canceled_at = at or timezone.now()
        self.status = self.STATUS_CANCELED
        self.save(update_fields=["canceled_at", "status"])