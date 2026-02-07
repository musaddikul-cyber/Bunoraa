"""
Shipping App Signals
"""
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Shipment, ShipmentEvent


@receiver(post_save, sender=Shipment)
def create_initial_shipment_event(sender, instance, created, **kwargs):
    """Create an initial event when a shipment is created."""
    if created:
        ShipmentEvent.objects.create(
            shipment=instance,
            status='created',
            description='Shipment created, awaiting processing'
        )


@receiver(post_save, sender=Shipment)
def update_order_on_shipment_status_change(sender, instance, **kwargs):
    """Update order status when shipment status changes."""
    if instance.order and instance.status == 'delivered':
        # Update order status to delivered if all shipments are delivered
        from apps.orders.models import Order
        order = instance.order
        all_shipments = order.shipments.all()
        
        if all_shipments.exists() and all(s.status == 'delivered' for s in all_shipments):
            order.status = 'delivered'
            order.save(update_fields=['status'])
