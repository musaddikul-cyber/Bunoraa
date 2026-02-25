"""
Reviews API views.
"""
from __future__ import annotations

from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAdminUser, IsAuthenticated
from rest_framework.response import Response

from apps.catalog.models import Product
from apps.reviews.models import Review
from apps.reviews.services import ReviewService
from core.pagination import StandardResultsSetPagination

from .serializers import (
    CreateReviewSerializer,
    FeatureReviewSerializer,
    ModerateReviewSerializer,
    ReportReviewSerializer,
    ReviewSerializer,
    ReviewStatisticsSerializer,
    UpdateReviewSerializer,
    VoteReviewSerializer,
)


class ReviewViewSet(viewsets.ModelViewSet):
    """
    Review operations.

    Key endpoints:
    - `GET /api/v1/reviews/`: current user's reviews
    - `POST /api/v1/reviews/`: create review
    - `PATCH /api/v1/reviews/{id}/`: update own review
    - `DELETE /api/v1/reviews/{id}/`: delete own review
    - `POST /api/v1/reviews/{id}/vote/`: helpful vote
    - `POST /api/v1/reviews/{id}/report/`: report abuse
    - `POST /api/v1/reviews/{id}/moderate/`: admin approve/reject
    - `POST /api/v1/reviews/{id}/feature/`: admin feature toggle
    - `GET /api/v1/reviews/product/{product_id}/`: public product reviews
    - `GET /api/v1/reviews/product/{product_id}/statistics/`: public review stats
    - `GET /api/v1/reviews/testimonials/`: public approved reviews showcase
    """

    serializer_class = ReviewSerializer
    pagination_class = StandardResultsSetPagination
    throttle_scope = "reviews"

    def get_permissions(self):
        if self.action in {"retrieve", "product_reviews", "statistics", "testimonials"}:
            return [AllowAny()]
        if self.action in {"moderate", "feature"}:
            return [IsAdminUser()]
        return [IsAuthenticated()]

    def get_queryset(self):
        queryset = Review.objects.select_related("product", "user").prefetch_related("images", "votes")
        if self.action in {"list"}:
            return queryset.filter(user=self.request.user)
        return queryset

    def get_serializer_class(self):
        if self.action == "create":
            return CreateReviewSerializer
        if self.action in {"update", "partial_update"}:
            return UpdateReviewSerializer
        if self.action == "vote":
            return VoteReviewSerializer
        if self.action == "report":
            return ReportReviewSerializer
        if self.action == "moderate":
            return ModerateReviewSerializer
        if self.action == "feature":
            return FeatureReviewSerializer
        return ReviewSerializer

    def list(self, request, *args, **kwargs):
        queryset = ReviewService.get_user_reviews(request.user)
        page = self.paginate_queryset(queryset)
        serializer = ReviewSerializer(page if page is not None else queryset, many=True, context={"request": request})
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        product = serializer.context["product"]
        review, message = ReviewService.create_review(
            product=product,
            user=request.user,
            rating=serializer.validated_data["rating"],
            title=serializer.validated_data.get("title", ""),
            body=serializer.validated_data.get("body", ""),
            pros=serializer.validated_data.get("pros", ""),
            cons=serializer.validated_data.get("cons", ""),
            would_recommend=serializer.validated_data.get("would_recommend"),
            is_anonymous=serializer.validated_data.get("is_anonymous", False),
        )
        if not review:
            return Response({"success": False, "message": message}, status=status.HTTP_400_BAD_REQUEST)
        data = ReviewSerializer(review, context={"request": request}).data
        return Response({"success": True, "message": message, "data": data}, status=status.HTTP_201_CREATED)

    def _get_editable_review(self, request, pk):
        review = get_object_or_404(Review, pk=pk)
        if review.user_id != request.user.id and not request.user.is_staff:
            return None
        return review

    def retrieve(self, request, pk=None):
        review = get_object_or_404(Review, pk=pk)
        is_owner = request.user.is_authenticated and review.user_id == request.user.id
        if not is_owner and not request.user.is_staff and review.moderation_status != Review.MODERATION_APPROVED:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(ReviewSerializer(review, context={"request": request}).data)

    def update(self, request, pk=None):
        review = self._get_editable_review(request, pk)
        if not review:
            return Response({"success": False, "message": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        success, message, review = ReviewService.update_review(
            review,
            request.user,
            **serializer.validated_data,
        )
        if not success:
            return Response({"success": False, "message": message}, status=status.HTTP_403_FORBIDDEN)
        return Response(
            {
                "success": True,
                "message": message,
                "data": ReviewSerializer(review, context={"request": request}).data,
            }
        )

    def partial_update(self, request, pk=None):
        return self.update(request, pk=pk)

    def destroy(self, request, pk=None):
        review = self._get_editable_review(request, pk)
        if not review:
            return Response({"success": False, "message": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
        ReviewService.delete_review(review)
        return Response({"success": True, "message": "Review deleted"}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"], url_path="vote")
    def vote(self, request, pk=None):
        review = get_object_or_404(Review, pk=pk, moderation_status=Review.MODERATION_APPROVED)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        success, message = ReviewService.vote_review(
            review=review,
            user=request.user,
            is_helpful=serializer.validated_data["is_helpful"],
        )
        status_code = status.HTTP_200_OK if success else status.HTTP_400_BAD_REQUEST
        review.refresh_from_db()
        return Response(
            {
                "success": success,
                "message": message,
                "data": ReviewSerializer(review, context={"request": request}).data if success else None,
            },
            status=status_code,
        )

    @action(detail=True, methods=["post"], url_path="report")
    def report(self, request, pk=None):
        review = get_object_or_404(Review, pk=pk, moderation_status=Review.MODERATION_APPROVED)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        success, message = ReviewService.report_review(
            review=review,
            reporter=request.user,
            reason=serializer.validated_data["reason"],
            details=serializer.validated_data.get("details", ""),
        )
        status_code = status.HTTP_200_OK if success else status.HTTP_400_BAD_REQUEST
        return Response({"success": success, "message": message}, status=status_code)

    @action(detail=True, methods=["post"], url_path="moderate")
    def moderate(self, request, pk=None):
        review = get_object_or_404(Review, pk=pk)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        action_value = serializer.validated_data["action"]
        notes = serializer.validated_data.get("notes", "")
        review = ReviewService.moderate_review(
            review=review,
            moderator=request.user,
            approve=action_value == "approve",
            notes=notes,
        )
        return Response(
            {
                "success": True,
                "message": f"Review {action_value}d",
                "data": ReviewSerializer(review, context={"request": request}).data,
            }
        )

    @action(detail=True, methods=["post"], url_path="feature")
    def feature(self, request, pk=None):
        review = get_object_or_404(Review, pk=pk)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        review = ReviewService.feature_review(
            review=review,
            featured=serializer.validated_data["is_featured"],
        )
        return Response(
            {
                "success": True,
                "message": "Feature state updated",
                "data": ReviewSerializer(review, context={"request": request}).data,
            }
        )

    @action(detail=False, methods=["get"], url_path=r"product/(?P<product_id>[^/.]+)")
    def product_reviews(self, request, product_id=None):
        product = get_object_or_404(Product, id=product_id, is_active=True, is_deleted=False)

        ordering = request.query_params.get("ordering", "-created_at")
        rating_value = request.query_params.get("rating")
        rating = None
        if rating_value is not None:
            try:
                rating = int(rating_value)
            except (TypeError, ValueError):
                rating = None

        verified_only = str(request.query_params.get("verified_only", "")).strip().lower() in {"1", "true", "yes"}
        reviews = ReviewService.get_product_reviews(
            product=product,
            ordering=ordering,
            rating=rating,
            verified_only=verified_only,
        )
        page = self.paginate_queryset(reviews)
        serializer = ReviewSerializer(page if page is not None else reviews, many=True, context={"request": request})
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)

    @action(detail=False, methods=["get"], url_path=r"product/(?P<product_id>[^/.]+)/statistics")
    def statistics(self, request, product_id=None):
        product = get_object_or_404(Product, id=product_id, is_active=True, is_deleted=False)
        stats = ReviewService.get_review_statistics(product)
        can_review = False
        can_review_reason = ""
        if request.user.is_authenticated:
            can_review, can_review_reason = ReviewService.can_review(product, request.user)
        stats["can_review"] = can_review
        stats["can_review_reason"] = can_review_reason
        payload = ReviewStatisticsSerializer(stats).data
        return Response(payload)

    @action(detail=False, methods=["get"], url_path="testimonials")
    def testimonials(self, request):
        queryset = Review.objects.filter(
            moderation_status=Review.MODERATION_APPROVED,
            is_featured=True,
        ).select_related("user", "product")
        page = self.paginate_queryset(queryset.order_by("-helpful_votes", "-created_at"))
        serializer = ReviewSerializer(page if page is not None else queryset, many=True, context={"request": request})
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)
