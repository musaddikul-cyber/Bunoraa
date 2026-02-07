from collections import Counter
from django.db.models import Count
from .models import Interaction, Recommendation
from ml.services.recommendation_service import RecommendationService # Added

def compute_also_bought(limit=20):
    """Simple co-purchase rule-based recommender: for each product, find products purchased in same orders/users."""
    # naive: find products with highest purchase co-occurrence
    purchases = Interaction.objects.filter(event="purchase")
    # group by user and collect product pairs
    co_counts = {}
    for user_id, group in purchases.values_list('user_id').annotate(cnt=Count('id')):
        pass
    # placeholder: clear and leave room for better implementation
    return


def compute_similar_by_category(limit=20):
    """Simple similarity: products in same category ordered by popularity."""
    # placeholder for implementation
    return

def get_visual_recommendations(product_id, num_items=10, exclude_product_ids=None):
    """
    Facade to get visually similar products from the ML recommendation service.
    """
    rec_service = RecommendationService()
    return rec_service.get_visually_similar_products(
        product_id=product_id,
        num_items=num_items,
        exclude_product_ids=exclude_product_ids
    )
