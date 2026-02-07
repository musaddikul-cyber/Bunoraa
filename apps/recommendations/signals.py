from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Interaction


@receiver(post_save, sender=Interaction)
def interaction_created(sender, instance, created, **kwargs):
    """Hook: when interactions are created we may enqueue recompute jobs or update counters.

    For now we keep this minimal and let the periodic task recompute recommendations in batch.
    """
    if created:
        # placeholder for queuing a job to update item-item co-occurrence
        pass
