"""
Canonical review models.

Review persistence is owned by the ``reviews`` app and linked to catalog
products via foreign keys.
"""
from __future__ import annotations

import uuid

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models import Count, Q
from django.utils import timezone


class ReviewQuerySet(models.QuerySet):
    def approved(self):
        return self.filter(moderation_status=Review.MODERATION_APPROVED)

    def pending(self):
        return self.filter(moderation_status=Review.MODERATION_PENDING)

    def rejected(self):
        return self.filter(moderation_status=Review.MODERATION_REJECTED)

    def featured(self):
        return self.filter(is_featured=True)

    def for_product(self, product):
        return self.filter(product=product)


class ReviewManager(models.Manager):
    def get_queryset(self):
        return ReviewQuerySet(self.model, using=self._db)

    def approved(self):
        return self.get_queryset().approved()

    def pending(self):
        return self.get_queryset().pending()


class Review(models.Model):
    MODERATION_PENDING = "pending"
    MODERATION_APPROVED = "approved"
    MODERATION_REJECTED = "rejected"
    MODERATION_CHOICES = (
        (MODERATION_PENDING, "Pending"),
        (MODERATION_APPROVED, "Approved"),
        (MODERATION_REJECTED, "Rejected"),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="reviews")
    product = models.ForeignKey("catalog.Product", on_delete=models.CASCADE, related_name="reviews")
    rating = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    title = models.CharField(max_length=200, blank=True)
    body = models.TextField(blank=True)
    pros = models.TextField(blank=True)
    cons = models.TextField(blank=True)
    would_recommend = models.BooleanField(null=True, blank=True)
    is_anonymous = models.BooleanField(default=False)
    verified_purchase = models.BooleanField(default=False, db_index=True)
    helpful_votes = models.PositiveIntegerField(default=0)
    not_helpful_votes = models.PositiveIntegerField(default=0)
    report_count = models.PositiveIntegerField(default=0)
    is_featured = models.BooleanField(default=False, db_index=True)
    moderation_status = models.CharField(
        max_length=20,
        choices=MODERATION_CHOICES,
        default=MODERATION_PENDING,
        db_index=True,
    )
    moderation_notes = models.TextField(blank=True)
    moderated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="catalog_reviews_moderated",
    )
    moderated_at = models.DateTimeField(null=True, blank=True)
    published_at = models.DateTimeField(null=True, blank=True)
    edited_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = ReviewManager()

    class Meta:
        db_table = "catalog_review"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["product", "moderation_status", "-created_at"], name="catalog_rev_product_22d09c_idx"),
            models.Index(fields=["moderation_status", "-created_at"], name="catalog_rev_moderat_b3cb91_idx"),
            models.Index(fields=["product", "is_featured", "-created_at"], name="catalog_rev_product_4b86f5_idx"),
            models.Index(fields=["verified_purchase", "rating"], name="catalog_rev_verifie_19fa73_idx"),
        ]
        constraints = [
            models.UniqueConstraint(fields=["user", "product"], name="catalog_review_unique_user_product"),
            models.CheckConstraint(
                condition=Q(rating__gte=1) & Q(rating__lte=5),
                name="catalog_review_rating_between_1_5",
            ),
        ]

    def __str__(self):
        return f"Review {self.rating} for {self.product} by {self.user}"

    @property
    def helpfulness_score(self):
        total_votes = self.helpful_votes + self.not_helpful_votes
        if total_votes <= 0:
            return 0.0
        return round(self.helpful_votes / total_votes, 4)

    @property
    def total_votes(self):
        return self.helpful_votes + self.not_helpful_votes

    def clean(self):
        title = (self.title or "").strip()
        body = (self.body or "").strip()
        if not title and not body:
            raise ValidationError("A review must include a title or body.")

    def save(self, *args, **kwargs):
        if self.pk:
            previous = (
                Review.objects.filter(pk=self.pk)
                .values(
                    "rating",
                    "title",
                    "body",
                    "pros",
                    "cons",
                    "would_recommend",
                    "is_anonymous",
                    "moderation_status",
                )
                .first()
            )
            if previous:
                content_changed = any(
                    (
                        previous["rating"] != self.rating,
                        previous["title"] != self.title,
                        previous["body"] != self.body,
                        previous["pros"] != self.pros,
                        previous["cons"] != self.cons,
                        previous["would_recommend"] != self.would_recommend,
                        previous["is_anonymous"] != self.is_anonymous,
                    )
                )
                if content_changed:
                    self.edited_at = timezone.now()
                    if previous["moderation_status"] == self.MODERATION_APPROVED:
                        self.moderation_status = self.MODERATION_PENDING
        super().save(*args, **kwargs)

    def mark_approved(self, moderator=None, notes=""):
        now = timezone.now()
        self.moderation_status = self.MODERATION_APPROVED
        self.moderation_notes = notes or self.moderation_notes
        self.moderated_by = moderator
        self.moderated_at = now
        if not self.published_at:
            self.published_at = now
        self.save(
            update_fields=[
                "moderation_status",
                "moderation_notes",
                "moderated_by",
                "moderated_at",
                "published_at",
                "updated_at",
            ]
        )

    def mark_rejected(self, moderator=None, notes=""):
        self.moderation_status = self.MODERATION_REJECTED
        self.moderation_notes = notes or self.moderation_notes
        self.moderated_by = moderator
        self.moderated_at = timezone.now()
        self.is_featured = False
        self.save(
            update_fields=[
                "moderation_status",
                "moderation_notes",
                "moderated_by",
                "moderated_at",
                "is_featured",
                "updated_at",
            ]
        )

    def refresh_engagement_counters(self):
        votes = self.votes.aggregate(
            helpful=Count("id", filter=Q(is_helpful=True)),
            not_helpful=Count("id", filter=Q(is_helpful=False)),
        )
        reports = self.reports.filter(status=ReviewReport.STATUS_OPEN).count()
        self.helpful_votes = votes.get("helpful") or 0
        self.not_helpful_votes = votes.get("not_helpful") or 0
        self.report_count = reports
        self.save(update_fields=["helpful_votes", "not_helpful_votes", "report_count", "updated_at"])


class ReviewImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    review = models.ForeignKey(Review, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to="catalog/review_images/")
    caption = models.CharField(max_length=200, blank=True)
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        db_table = "catalog_reviewimage"
        ordering = ["sort_order", "created_at"]
        indexes = [models.Index(fields=["review", "sort_order"], name="catalog_rev_review__e82a6f_idx")]

    def __str__(self):
        return f"Image for {self.review}"


class ReviewVote(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    review = models.ForeignKey(Review, on_delete=models.CASCADE, related_name="votes")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="catalog_review_votes",
    )
    is_helpful = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "catalog_reviewvote"
        constraints = [
            models.UniqueConstraint(fields=["review", "user"], name="catalog_review_vote_unique"),
        ]
        indexes = [
            models.Index(fields=["review", "is_helpful"], name="catalog_rev_review__37b464_idx"),
            models.Index(fields=["user"], name="catalog_rev_user_id_5bf3c6_idx"),
        ]

    def __str__(self):
        return f"Vote for {self.review_id} by {self.user_id}"


class ReviewReport(models.Model):
    REASON_SPAM = "spam"
    REASON_ABUSE = "abuse"
    REASON_OFF_TOPIC = "off_topic"
    REASON_MISLEADING = "misleading"
    REASON_OTHER = "other"
    REASON_CHOICES = (
        (REASON_SPAM, "Spam"),
        (REASON_ABUSE, "Abusive Content"),
        (REASON_OFF_TOPIC, "Off Topic"),
        (REASON_MISLEADING, "Misleading"),
        (REASON_OTHER, "Other"),
    )

    STATUS_OPEN = "open"
    STATUS_RESOLVED = "resolved"
    STATUS_DISMISSED = "dismissed"
    STATUS_CHOICES = (
        (STATUS_OPEN, "Open"),
        (STATUS_RESOLVED, "Resolved"),
        (STATUS_DISMISSED, "Dismissed"),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    review = models.ForeignKey(Review, on_delete=models.CASCADE, related_name="reports")
    reporter = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="catalog_review_reports",
    )
    reason = models.CharField(max_length=20, choices=REASON_CHOICES)
    details = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_OPEN, db_index=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="catalog_review_reports_resolved",
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "catalog_reviewreport"
        constraints = [
            models.UniqueConstraint(
                fields=["review", "reporter"],
                condition=Q(status="open"),
                name="catalog_review_open_report_unique",
            ),
        ]
        indexes = [
            models.Index(fields=["review", "status"], name="catalog_rev_review__82e786_idx"),
            models.Index(fields=["status", "created_at"], name="catalog_rev_status_005831_idx"),
        ]

    def __str__(self):
        return f"Report for {self.review_id} by {self.reporter_id}"


__all__ = [
    "Review",
    "ReviewImage",
    "ReviewVote",
    "ReviewReport",
]
