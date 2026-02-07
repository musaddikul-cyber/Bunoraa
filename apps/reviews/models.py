"""
Reviews models
"""
import uuid
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator


class Review(models.Model):
    """
    Product review model.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    product = models.ForeignKey(
        'catalog.Product',
        on_delete=models.CASCADE,
        related_name='product_reviews'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='product_reviews'
    )
    
    # Rating
    rating = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    
    # Content
    title = models.CharField(max_length=200, blank=True)
    content = models.TextField()
    
    # Verified purchase
    is_verified_purchase = models.BooleanField(default=False)
    
    # Helpful votes
    helpful_count = models.PositiveIntegerField(default=0)
    not_helpful_count = models.PositiveIntegerField(default=0)
    
    # Moderation
    STATUS_PENDING = 'pending'
    STATUS_APPROVED = 'approved'
    STATUS_REJECTED = 'rejected'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending Review'),
        (STATUS_APPROVED, 'Approved'),
        (STATUS_REJECTED, 'Rejected'),
    ]
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    moderation_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Soft delete
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Review'
        verbose_name_plural = 'Reviews'
        unique_together = ['product', 'user']
        indexes = [
            models.Index(fields=['product', 'status']),
            models.Index(fields=['user']),
        ]
    
    def __str__(self):
        return f"{self.user.email} - {self.product.name} ({self.rating}★)"
    
    @property
    def helpfulness_score(self):
        """Calculate helpfulness score."""
        total = self.helpful_count + self.not_helpful_count
        if total == 0:
            return 0
        return self.helpful_count / total


class ReviewImage(models.Model):
    """
    Image attached to a review.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    review = models.ForeignKey(
        Review,
        on_delete=models.CASCADE,
        related_name='images'
    )
    
    image = models.ImageField(upload_to='reviews/')
    caption = models.CharField(max_length=200, blank=True)
    
    sort_order = models.PositiveIntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['sort_order']
        verbose_name = 'Review Image'
        verbose_name_plural = 'Review Images'


class ReviewVote(models.Model):
    """
    Track user votes on reviews.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    review = models.ForeignKey(
        Review,
        on_delete=models.CASCADE,
        related_name='votes'
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='review_votes'
    )
    
    is_helpful = models.BooleanField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['review', 'user']
        verbose_name = 'Review Vote'
        verbose_name_plural = 'Review Votes'


class ReviewReply(models.Model):
    """
    Admin reply to a review.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    review = models.OneToOneField(
        Review,
        on_delete=models.CASCADE,
        related_name='reply'
    )
    
    content = models.TextField()
    
    replied_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Review Reply'
        verbose_name_plural = 'Review Replies'
