from rest_framework import viewsets, mixins
from .models import Recommendation, Interaction
from .serializers import RecommendationSerializer, InteractionSerializer


class RecommendationViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Recommendation.objects.all()
    serializer_class = RecommendationSerializer


class InteractionViewSet(mixins.CreateModelMixin, viewsets.GenericViewSet):
    queryset = Interaction.objects.all()
    serializer_class = InteractionSerializer
    # create-only endpoint to record interactions
