"""
Reviews services
"""
from django.db.models import Avg, Count, Q
from django.utils import timezone

from .models import Review, ReviewVote, ReviewReply


class ReviewService:
    """Service for review operations."""
    
    @staticmethod
    def create_review(product, user, rating, content, title=''):
        """
        Create a new review.
        
        Args:
            product: Product instance
            user: User instance
            rating: Rating 1-5
            content: Review text
            title: Optional title
            
        Returns:
            Review instance or None if already reviewed
        """
        # Check if already reviewed
        if Review.objects.filter(product=product, user=user, is_deleted=False).exists():
            return None, "You have already reviewed this product"
        
        # Check if verified purchase
        from apps.orders.models import Order, OrderItem
        is_verified = OrderItem.objects.filter(
            order__user=user,
            product=product,
            order__status=Order.STATUS_DELIVERED
        ).exists()
        
        review = Review.objects.create(
            product=product,
            user=user,
            rating=rating,
            title=title,
            content=content,
            is_verified_purchase=is_verified,
            status=Review.STATUS_PENDING  # Requires moderation
        )
        
        return review, "Review submitted successfully"
    
    @staticmethod
    def get_product_reviews(product, status=Review.STATUS_APPROVED, ordering='-created_at'):
        """
        Get reviews for a product.
        
        Args:
            product: Product instance
            status: Filter by status
            ordering: Sort order
            
        Returns:
            QuerySet of reviews
        """
        queryset = Review.objects.filter(
            product=product,
            is_deleted=False
        )
        
        if status:
            queryset = queryset.filter(status=status)
        
        return queryset.select_related('user').prefetch_related('images', 'reply').order_by(ordering)
    
    @staticmethod
    def get_review_statistics(product):
        """
        Get review statistics for a product.
        
        Args:
            product: Product instance
            
        Returns:
            Dictionary with statistics
        """
        reviews = Review.objects.filter(
            product=product,
            status=Review.STATUS_APPROVED,
            is_deleted=False
        )
        
        stats = reviews.aggregate(
            average_rating=Avg('rating'),
            total_count=Count('id'),
            verified_count=Count('id', filter=Q(is_verified_purchase=True))
        )
        
        # Rating distribution
        distribution = {}
        for i in range(1, 6):
            distribution[i] = reviews.filter(rating=i).count()
        
        return {
            'average_rating': round(stats['average_rating'] or 0, 1),
            'total_count': stats['total_count'],
            'verified_count': stats['verified_count'],
            'distribution': distribution,
        }
    
    @staticmethod
    def vote_review(review, user, is_helpful):
        """
        Vote on a review.
        
        Args:
            review: Review instance
            user: User instance
            is_helpful: Boolean
            
        Returns:
            Tuple (success, message)
        """
        # Check if user is review author
        if review.user == user:
            return False, "You cannot vote on your own review"
        
        # Check existing vote
        existing_vote = ReviewVote.objects.filter(review=review, user=user).first()
        
        if existing_vote:
            if existing_vote.is_helpful == is_helpful:
                # Remove vote
                if existing_vote.is_helpful:
                    review.helpful_count -= 1
                else:
                    review.not_helpful_count -= 1
                existing_vote.delete()
                review.save()
                return True, "Vote removed"
            else:
                # Change vote
                if existing_vote.is_helpful:
                    review.helpful_count -= 1
                    review.not_helpful_count += 1
                else:
                    review.not_helpful_count -= 1
                    review.helpful_count += 1
                existing_vote.is_helpful = is_helpful
                existing_vote.save()
                review.save()
                return True, "Vote updated"
        else:
            # New vote
            ReviewVote.objects.create(
                review=review,
                user=user,
                is_helpful=is_helpful
            )
            if is_helpful:
                review.helpful_count += 1
            else:
                review.not_helpful_count += 1
            review.save()
            return True, "Vote recorded"
    
    @staticmethod
    def approve_review(review, moderator=None):
        """
        Approve a review.
        
        Args:
            review: Review instance
            moderator: User who approved
            
        Returns:
            Updated review
        """
        review.status = Review.STATUS_APPROVED
        review.save()
        return review
    
    @staticmethod
    def reject_review(review, reason='', moderator=None):
        """
        Reject a review.
        
        Args:
            review: Review instance
            reason: Rejection reason
            moderator: User who rejected
            
        Returns:
            Updated review
        """
        review.status = Review.STATUS_REJECTED
        review.moderation_notes = reason
        review.save()
        return review
    
    @staticmethod
    def add_reply(review, content, replied_by):
        """
        Add admin reply to review.
        
        Args:
            review: Review instance
            content: Reply content
            replied_by: User who replied
            
        Returns:
            ReviewReply instance
        """
        reply, created = ReviewReply.objects.update_or_create(
            review=review,
            defaults={
                'content': content,
                'replied_by': replied_by,
            }
        )
        return reply
    
    @staticmethod
    def delete_review(review, deleted_by=None):
        """
        Soft delete a review.
        
        Args:
            review: Review instance
            deleted_by: User who deleted
            
        Returns:
            Updated review
        """
        review.is_deleted = True
        review.deleted_at = timezone.now()
        review.save()
        return review
    
    @staticmethod
    def get_user_reviews(user):
        """
        Get reviews by a user.
        
        Args:
            user: User instance
            
        Returns:
            QuerySet of reviews
        """
        return Review.objects.filter(
            user=user,
            is_deleted=False
        ).select_related('product').prefetch_related('images')
    
    @staticmethod
    def can_review(product, user):
        """
        Check if user can review a product.
        
        Args:
            product: Product instance
            user: User instance
            
        Returns:
            Tuple (can_review, reason)
        """
        if not user.is_authenticated:
            return False, "Please log in to write a review"
        
        # Check existing review
        if Review.objects.filter(product=product, user=user, is_deleted=False).exists():
            return False, "You have already reviewed this product"
        
        return True, "You can review this product"
