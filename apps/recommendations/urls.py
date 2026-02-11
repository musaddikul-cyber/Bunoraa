from django.urls import path, include
from rest_framework.routers import SimpleRouter
from .views import RecommendationViewSet, InteractionViewSet

interaction_router = SimpleRouter()
interaction_router.register(r"interactions", InteractionViewSet, basename="interaction")

urlpatterns = [
    path('', RecommendationViewSet.as_view({'get': 'list'}), name='recommendations-list'),
    path('<uuid:pk>/', RecommendationViewSet.as_view({'get': 'retrieve'}), name='recommendations-detail'),
    path('', include(interaction_router.urls)),
]
