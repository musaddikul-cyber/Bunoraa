from rest_framework.routers import DefaultRouter
from .views import RecommendationViewSet, InteractionViewSet

router = DefaultRouter()
router.register(r"recommendations", RecommendationViewSet, basename="recommendation")
router.register(r"interactions", InteractionViewSet, basename="interaction")

urlpatterns = router.urls
