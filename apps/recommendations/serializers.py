from rest_framework import serializers
from .models import Recommendation, Interaction


class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = ["id", "product", "recommended_product", "type", "score", "algorithm"]


class InteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interaction
        fields = ["id", "user", "product", "event", "value", "occurred_at"]
