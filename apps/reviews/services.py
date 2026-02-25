"""
Reviews domain services.

This module intentionally uses catalog review models as the canonical source of
truth to avoid split data across multiple review tables.
"""
from __future__ import annotations

from typing import Any

from django.db import transaction
from django.db.models import Avg, Count, F, Q
from django.utils import timezone

from apps.reviews.models import Review, ReviewReport, ReviewVote
from apps.orders.models import Order, OrderItem


class ReviewService:
    """Service layer for review creation, moderation, voting, and reporting."""

    VALID_ORDERINGS = {
        "-created_at",
        "created_at",
        "-rating",
        "rating",
        "-helpful_votes",
        "helpful_votes",
        "-report_count",
        "report_count",
    }

    @staticmethod
    def has_verified_purchase(product, user) -> bool:
        """Return True when the user has a delivered order for the product."""
        if not getattr(user, "is_authenticated", False):
            return False
        return OrderItem.objects.filter(
            order__user=user,
            product=product,
            order__status=Order.STATUS_DELIVERED,
        ).exists()

    @staticmethod
    def can_review(product, user) -> tuple[bool, str]:
        """Check if a user can submit a review for a product."""
        if not getattr(user, "is_authenticated", False):
            return False, "Please log in to write a review"
        if Review.objects.filter(product=product, user=user).exists():
            return False, "You have already reviewed this product"
        return True, "You can review this product"

    @classmethod
    @transaction.atomic
    def create_review(
        cls,
        *,
        product,
        user,
        rating: int,
        title: str = "",
        body: str = "",
        pros: str = "",
        cons: str = "",
        would_recommend: bool | None = None,
        is_anonymous: bool = False,
    ) -> tuple[Review | None, str]:
        """Create a pending review for moderation."""
        can_review, reason = cls.can_review(product, user)
        if not can_review:
            return None, reason

        verified = cls.has_verified_purchase(product, user)
        review = Review.objects.create(
            product=product,
            user=user,
            rating=rating,
            title=title,
            body=body,
            pros=pros,
            cons=cons,
            would_recommend=would_recommend,
            is_anonymous=is_anonymous,
            verified_purchase=verified,
            moderation_status=Review.MODERATION_PENDING,
        )
        return review, "Review submitted successfully and is pending moderation"

    @classmethod
    def get_product_reviews(
        cls,
        *,
        product,
        status: str = Review.MODERATION_APPROVED,
        ordering: str = "-created_at",
        rating: int | None = None,
        verified_only: bool = False,
    ):
        """Get product reviews with filtering and safe ordering."""
        queryset = (
            Review.objects.filter(product=product)
            .select_related("user")
            .prefetch_related("images", "votes")
        )
        if status:
            queryset = queryset.filter(moderation_status=status)
        if rating is not None:
            queryset = queryset.filter(rating=rating)
        if verified_only:
            queryset = queryset.filter(verified_purchase=True)
        if ordering not in cls.VALID_ORDERINGS:
            ordering = "-created_at"
        return queryset.order_by(ordering)

    @staticmethod
    def get_user_reviews(user):
        """Get reviews authored by a user."""
        return (
            Review.objects.filter(user=user)
            .select_related("product", "user")
            .prefetch_related("images", "votes", "reports")
            .order_by("-created_at")
        )

    @staticmethod
    def get_review_statistics(product) -> dict[str, Any]:
        """Return aggregate review statistics for a product."""
        reviews = Review.objects.filter(
            product=product,
            moderation_status=Review.MODERATION_APPROVED,
        )
        stats = reviews.aggregate(
            average_rating=Avg("rating"),
            total_count=Count("id"),
            verified_count=Count("id", filter=Q(verified_purchase=True)),
            recommend_count=Count("id", filter=Q(would_recommend=True)),
            recommend_answered=Count("id", filter=Q(would_recommend__isnull=False)),
        )
        distribution = {
            rating: reviews.filter(rating=rating).count() for rating in range(1, 6)
        }
        recommend_answered = stats.get("recommend_answered") or 0
        recommendation_rate = 0.0
        if recommend_answered > 0:
            recommendation_rate = round(
                ((stats.get("recommend_count") or 0) / recommend_answered) * 100,
                2,
            )
        return {
            "average_rating": round(stats.get("average_rating") or 0, 1),
            "total_count": stats.get("total_count") or 0,
            "verified_count": stats.get("verified_count") or 0,
            "distribution": distribution,
            "recommendation_rate": recommendation_rate,
        }

    @staticmethod
    @transaction.atomic
    def update_review(review: Review, user, **changes) -> tuple[bool, str, Review]:
        """Update mutable review fields and trigger re-moderation when needed."""
        if review.user_id != getattr(user, "id", None) and not getattr(user, "is_staff", False):
            return False, "Permission denied", review

        allowed_fields = {
            "rating",
            "title",
            "body",
            "pros",
            "cons",
            "would_recommend",
            "is_anonymous",
        }
        for field, value in changes.items():
            if field in allowed_fields:
                setattr(review, field, value)
        review.save()
        return True, "Review updated", review

    @staticmethod
    @transaction.atomic
    def delete_review(review: Review):
        """Delete a review."""
        review.delete()

    @staticmethod
    @transaction.atomic
    def vote_review(*, review: Review, user, is_helpful: bool) -> tuple[bool, str]:
        """Toggle or update helpful vote for a review."""
        if not getattr(user, "is_authenticated", False):
            return False, "Authentication required"
        if review.user_id == user.id:
            return False, "You cannot vote on your own review"

        vote = ReviewVote.objects.filter(review=review, user=user).first()
        if vote and vote.is_helpful == is_helpful:
            vote.delete()
            if is_helpful:
                Review.objects.filter(pk=review.pk).update(helpful_votes=F("helpful_votes") - 1)
            else:
                Review.objects.filter(pk=review.pk).update(not_helpful_votes=F("not_helpful_votes") - 1)
            review.refresh_engagement_counters()
            return True, "Vote removed"

        if vote:
            previous_helpful = vote.is_helpful
            vote.is_helpful = is_helpful
            vote.save(update_fields=["is_helpful"])
            if previous_helpful and not is_helpful:
                Review.objects.filter(pk=review.pk).update(
                    helpful_votes=F("helpful_votes") - 1,
                    not_helpful_votes=F("not_helpful_votes") + 1,
                )
            elif not previous_helpful and is_helpful:
                Review.objects.filter(pk=review.pk).update(
                    helpful_votes=F("helpful_votes") + 1,
                    not_helpful_votes=F("not_helpful_votes") - 1,
                )
            review.refresh_engagement_counters()
            return True, "Vote updated"

        ReviewVote.objects.create(review=review, user=user, is_helpful=is_helpful)
        if is_helpful:
            Review.objects.filter(pk=review.pk).update(helpful_votes=F("helpful_votes") + 1)
        else:
            Review.objects.filter(pk=review.pk).update(not_helpful_votes=F("not_helpful_votes") + 1)
        review.refresh_engagement_counters()
        return True, "Vote recorded"

    @staticmethod
    @transaction.atomic
    def report_review(*, review: Review, reporter, reason: str, details: str = "") -> tuple[bool, str]:
        """Create an open abuse report for a review."""
        if not getattr(reporter, "is_authenticated", False):
            return False, "Authentication required"
        if review.user_id == reporter.id:
            return False, "You cannot report your own review"
        open_report_exists = ReviewReport.objects.filter(
            review=review,
            reporter=reporter,
            status=ReviewReport.STATUS_OPEN,
        ).exists()
        if open_report_exists:
            return False, "You already have an open report for this review"

        ReviewReport.objects.create(
            review=review,
            reporter=reporter,
            reason=reason,
            details=details,
            status=ReviewReport.STATUS_OPEN,
        )
        review.refresh_engagement_counters()
        return True, "Report submitted"

    @staticmethod
    @transaction.atomic
    def moderate_review(*, review: Review, moderator, approve: bool, notes: str = "") -> Review:
        """Approve or reject a review."""
        if approve:
            review.mark_approved(moderator=moderator, notes=notes)
        else:
            review.mark_rejected(moderator=moderator, notes=notes)
        return review

    @staticmethod
    @transaction.atomic
    def feature_review(*, review: Review, featured: bool) -> Review:
        """Set featured status for a review."""
        review.is_featured = featured and review.moderation_status == Review.MODERATION_APPROVED
        review.save(update_fields=["is_featured", "updated_at"])
        return review

    @staticmethod
    @transaction.atomic
    def resolve_report(*, report: ReviewReport, moderator, dismissed: bool = False) -> ReviewReport:
        """Resolve or dismiss a report."""
        report.status = ReviewReport.STATUS_DISMISSED if dismissed else ReviewReport.STATUS_RESOLVED
        report.resolved_by = moderator
        report.resolved_at = timezone.now()
        report.save(update_fields=["status", "resolved_by", "resolved_at", "updated_at"])
        report.review.refresh_engagement_counters()
        return report
