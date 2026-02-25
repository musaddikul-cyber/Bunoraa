"""
Reviews API serializers.
"""
from __future__ import annotations

from rest_framework import serializers

from apps.catalog.models import Product
from apps.reviews.models import Review, ReviewImage, ReviewReport


class ReviewImageSerializer(serializers.ModelSerializer):
    """Serializer for review image attachments."""

    class Meta:
        model = ReviewImage
        fields = ["id", "image", "caption", "sort_order", "created_at"]


class ReviewSerializer(serializers.ModelSerializer):
    """Serializer for product reviews."""

    images = ReviewImageSerializer(many=True, read_only=True)
    user_name = serializers.SerializerMethodField()
    helpfulness_score = serializers.ReadOnlyField()
    user_vote = serializers.SerializerMethodField()

    # Backward-compatible aliases for existing consumers.
    content = serializers.CharField(source="body", read_only=True)
    is_verified_purchase = serializers.BooleanField(source="verified_purchase", read_only=True)
    helpful_count = serializers.IntegerField(source="helpful_votes", read_only=True)
    not_helpful_count = serializers.IntegerField(source="not_helpful_votes", read_only=True)
    status = serializers.CharField(source="moderation_status", read_only=True)

    class Meta:
        model = Review
        fields = [
            "id",
            "product",
            "rating",
            "title",
            "body",
            "content",
            "pros",
            "cons",
            "would_recommend",
            "is_anonymous",
            "user_name",
            "verified_purchase",
            "is_verified_purchase",
            "helpful_votes",
            "helpful_count",
            "not_helpful_votes",
            "not_helpful_count",
            "helpfulness_score",
            "report_count",
            "is_featured",
            "moderation_status",
            "status",
            "user_vote",
            "images",
            "created_at",
            "updated_at",
            "published_at",
            "edited_at",
        ]
        read_only_fields = [
            "product",
            "verified_purchase",
            "helpful_votes",
            "not_helpful_votes",
            "helpfulness_score",
            "report_count",
            "moderation_status",
            "created_at",
            "updated_at",
            "published_at",
            "edited_at",
        ]

    def get_user_name(self, obj):
        if obj.is_anonymous:
            return "Anonymous"
        if obj.user:
            first_name = (obj.user.first_name or "").strip()
            last_name = (obj.user.last_name or "").strip()
            if first_name and last_name:
                return f"{first_name} {last_name[0]}."
            if first_name:
                return first_name
            if obj.user.email:
                return obj.user.email.split("@")[0]
        return "Customer"

    def get_user_vote(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            vote = obj.votes.filter(user=request.user).first()
            if vote:
                return "helpful" if vote.is_helpful else "not_helpful"
        return None


class CreateReviewSerializer(serializers.Serializer):
    """Serializer for creating a review."""

    product_id = serializers.UUIDField()
    rating = serializers.IntegerField(min_value=1, max_value=5)
    title = serializers.CharField(max_length=200, required=False, allow_blank=True)
    body = serializers.CharField(required=False, allow_blank=True, max_length=8000)
    content = serializers.CharField(required=False, allow_blank=True, write_only=True, max_length=8000)
    pros = serializers.CharField(required=False, allow_blank=True, max_length=8000)
    cons = serializers.CharField(required=False, allow_blank=True, max_length=8000)
    would_recommend = serializers.BooleanField(required=False, allow_null=True)
    is_anonymous = serializers.BooleanField(required=False, default=False)

    def validate_product_id(self, value):
        try:
            product = Product.objects.get(id=value, is_active=True, is_deleted=False)
        except Product.DoesNotExist as exc:
            raise serializers.ValidationError("Product not found") from exc
        self.context["product"] = product
        return value

    def validate(self, attrs):
        body = (attrs.get("body") or attrs.get("content") or "").strip()
        title = (attrs.get("title") or "").strip()
        if not body and not title:
            raise serializers.ValidationError("A review must include a title or body.")
        attrs["body"] = body
        return attrs


class UpdateReviewSerializer(serializers.Serializer):
    """Serializer for updating an existing review."""

    rating = serializers.IntegerField(min_value=1, max_value=5, required=False)
    title = serializers.CharField(max_length=200, required=False, allow_blank=True)
    body = serializers.CharField(required=False, allow_blank=True, max_length=8000)
    content = serializers.CharField(required=False, allow_blank=True, write_only=True, max_length=8000)
    pros = serializers.CharField(required=False, allow_blank=True, max_length=8000)
    cons = serializers.CharField(required=False, allow_blank=True, max_length=8000)
    would_recommend = serializers.BooleanField(required=False, allow_null=True)
    is_anonymous = serializers.BooleanField(required=False)

    def validate(self, attrs):
        if not attrs:
            raise serializers.ValidationError("No fields provided for update.")
        if "content" in attrs and "body" not in attrs:
            attrs["body"] = attrs["content"]
        return attrs


class VoteReviewSerializer(serializers.Serializer):
    """Serializer for helpful/not-helpful vote payloads."""

    is_helpful = serializers.BooleanField()


class ReportReviewSerializer(serializers.Serializer):
    """Serializer for reporting a review."""

    reason = serializers.ChoiceField(choices=[choice[0] for choice in ReviewReport.REASON_CHOICES])
    details = serializers.CharField(required=False, allow_blank=True, max_length=4000)


class ModerateReviewSerializer(serializers.Serializer):
    """Serializer for moderation actions."""

    action = serializers.ChoiceField(choices=["approve", "reject"])
    notes = serializers.CharField(required=False, allow_blank=True, max_length=4000)


class FeatureReviewSerializer(serializers.Serializer):
    """Serializer for featuring a review."""

    is_featured = serializers.BooleanField()


class ReviewStatisticsSerializer(serializers.Serializer):
    """Serializer for public review statistics."""

    average_rating = serializers.FloatField()
    total_count = serializers.IntegerField()
    verified_count = serializers.IntegerField()
    recommendation_rate = serializers.FloatField()
    distribution = serializers.DictField(child=serializers.IntegerField())
    can_review = serializers.BooleanField(required=False)
    can_review_reason = serializers.CharField(required=False, allow_blank=True)
