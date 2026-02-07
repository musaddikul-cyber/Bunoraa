"""
Reviews API serializers
"""
from rest_framework import serializers
from ..models import Review, ReviewImage, ReviewReply


class ReviewImageSerializer(serializers.ModelSerializer):
    """Serializer for review image."""
    
    class Meta:
        model = ReviewImage
        fields = ['id', 'image', 'caption']


class ReviewReplySerializer(serializers.ModelSerializer):
    """Serializer for review reply."""
    replied_by_name = serializers.SerializerMethodField()
    
    class Meta:
        model = ReviewReply
        fields = ['id', 'content', 'replied_by_name', 'created_at']
    
    def get_replied_by_name(self, obj):
        if obj.replied_by:
            return obj.replied_by.get_full_name() or 'Store Admin'
        return 'Store Admin'


class ReviewSerializer(serializers.ModelSerializer):
    """Serializer for review."""
    images = ReviewImageSerializer(many=True, read_only=True)
    reply = ReviewReplySerializer(read_only=True)
    user_name = serializers.SerializerMethodField()
    helpfulness_score = serializers.ReadOnlyField()
    user_vote = serializers.SerializerMethodField()
    
    class Meta:
        model = Review
        fields = [
            'id', 'rating', 'title', 'content',
            'user_name', 'is_verified_purchase',
            'helpful_count', 'not_helpful_count', 'helpfulness_score',
            'images', 'reply', 'user_vote',
            'created_at'
        ]
    
    def get_user_name(self, obj):
        if obj.user:
            first_name = obj.user.first_name or ''
            last_name = obj.user.last_name or ''
            if first_name and last_name:
                return f"{first_name} {last_name[0]}."
            return first_name or obj.user.email.split('@')[0]
        return 'Anonymous'
    
    def get_user_vote(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            vote = obj.votes.filter(user=request.user).first()
            if vote:
                return 'helpful' if vote.is_helpful else 'not_helpful'
        return None


class CreateReviewSerializer(serializers.Serializer):
    """Serializer for creating review."""
    product_id = serializers.UUIDField()
    rating = serializers.IntegerField(min_value=1, max_value=5)
    title = serializers.CharField(max_length=200, required=False, allow_blank=True)
    content = serializers.CharField(min_length=10, max_length=5000)
    
    def validate_product_id(self, value):
        from apps.catalog.models import Product
        
        try:
            product = Product.objects.get(id=value, is_active=True, is_deleted=False)
            self.context['product'] = product
            return value
        except Product.DoesNotExist:
            raise serializers.ValidationError("Product not found")


class UpdateReviewSerializer(serializers.Serializer):
    """Serializer for updating review."""
    rating = serializers.IntegerField(min_value=1, max_value=5, required=False)
    title = serializers.CharField(max_length=200, required=False, allow_blank=True)
    content = serializers.CharField(min_length=10, max_length=5000, required=False)


class VoteReviewSerializer(serializers.Serializer):
    """Serializer for voting on review."""
    is_helpful = serializers.BooleanField()


class ReviewStatisticsSerializer(serializers.Serializer):
    """Serializer for review statistics."""
    average_rating = serializers.FloatField()
    total_count = serializers.IntegerField()
    verified_count = serializers.IntegerField()
    distribution = serializers.DictField()
