"""Recommendation models.

This module provides lightweight, performant models for interaction events and
product-to-product recommendations. It follows the project's conventions for
UUID PKs, soft-delete support and indexed queries optimized for "top N"
recommendation lookups.
"""

import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone

from apps.catalog.managers import SoftDeleteManager


class Interaction(models.Model):
    """Represents a user or session-level interaction with a product.

    - Supports logged-in users via ``user`` and anonymous visitors via ``session_key``.
    - ``value`` is a flexible weight (e.g., 1.0 for a view, 5.0 for a purchase).
    """

    EVENT_VIEW = "view"
    EVENT_PURCHASE = "purchase"
    EVENT_CART_ADD = "cart_add"

    EVENT_CHOICES = (
        (EVENT_VIEW, "View"),
        (EVENT_PURCHASE, "Purchase"),
        (EVENT_CART_ADD, "Cart Add"),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="recommendation_interactions",
    )
    session_key = models.CharField(
        max_length=40, null=True, blank=True, db_index=True,
        help_text="Session key for anonymous visitors (if user isn't authenticated).",
    )
    product = models.ForeignKey(
        "catalog.Product", on_delete=models.CASCADE, related_name="interactions"
    )
    event = models.CharField(max_length=20, choices=EVENT_CHOICES)
    value = models.FloatField(
        default=1.0,
        help_text="Weight or value for the event (e.g. 1.0 for view, higher for purchase).",
    )
    occurred_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-occurred_at"]
        indexes = [
            models.Index(fields=["event", "occurred_at"]),
            models.Index(fields=["product"]),
            models.Index(fields=["user"]),
            models.Index(fields=["session_key"]),
            models.Index(fields=["product", "event"]),
        ]

    def __str__(self):
        who = self.user or self.session_key or "anonymous"
        return f"Interaction({self.event}) {self.product} by {who} at {self.occurred_at.isoformat()}"


class Recommendation(models.Model):
    """A product-to-product recommendation entry.

    - ``score`` is normalized to [0.0, 1.0].
    - Soft-delete is supported via ``is_deleted`` and ``deleted_at`` and the
      default manager only returns alive recommendations.
    """

    TYPE_FBT = "frequently_bought_together"
    TYPE_SIMILAR = "similar"
    TYPE_PERSONAL = "personal"
    TYPE_TRENDING = "trending"

    TYPE_CHOICES = (
        (TYPE_FBT, "Frequently Bought Together"),
        (TYPE_SIMILAR, "Similar"),
        (TYPE_PERSONAL, "Personal"),
        (TYPE_TRENDING, "Trending"),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # product -> recommended_product
    product = models.ForeignKey(
        "catalog.Product", related_name="recommendations", on_delete=models.CASCADE
    )
    recommended_product = models.ForeignKey(
        "catalog.Product", related_name="recommended_by", on_delete=models.CASCADE
    )

    type = models.CharField(max_length=50, choices=TYPE_CHOICES)

    score = models.FloatField(
        default=0.0,
        db_index=True,
        help_text="Normalized relevance score between 0.0 and 1.0 (higher is better).",
    )

    algorithm = models.CharField(
        max_length=100,
        default="rule_based",
        help_text="Algorithm / model that produced this recommendation (e.g. 'rule_based', 'collab_filter').",
    )

    is_active = models.BooleanField(default=True, db_index=True)

    # soft-delete fields to match catalog app conventions
    is_deleted = models.BooleanField(default=False, db_index=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Managers
    objects = SoftDeleteManager()
    all_objects = models.Manager()

    def soft_delete(self):
        """Soft-delete this recommendation. Marks it deleted and deactivates it."""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.is_active = False
        self.save(update_fields=["is_deleted", "deleted_at", "is_active", "updated_at"])

    def __str__(self):
        return (
            f"Recommendation({self.product} -> {self.recommended_product})"
            f" [{self.get_type_display()}] score={self.score:.3f}"
        )

    class Meta:
        unique_together = ("product", "recommended_product", "type")
        indexes = [
            # Top N lookups per product, sorted by descending score
            models.Index(fields=["product", "type", "-score"], name="rec_product_type_score_idx"),
            # Lookup by the recommended product
            models.Index(fields=["recommended_product"], name="rec_recommended_product_idx"),
            # Active recommendations (fast top list)
            models.Index(fields=["is_active", "-score"], name="rec_active_score_idx"),
            # Helpful additional index
            models.Index(fields=["type", "score"], name="rec_type_score_idx"),
        ]
        constraints = [
            models.CheckConstraint(check=models.Q(score__gte=0.0) & models.Q(score__lte=1.0), name="recommendation_score_range"),
        ]
        verbose_name = "Recommendation"
        verbose_name_plural = "Recommendations"

