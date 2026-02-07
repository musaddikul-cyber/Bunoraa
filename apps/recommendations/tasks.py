from celery import shared_task
from .services import compute_also_bought, compute_similar_by_category


@shared_task
def recompute_recommendations():
    compute_also_bought()
    compute_similar_by_category()
