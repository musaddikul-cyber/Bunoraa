"""
Signals for the accounts app.
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.conf import settings
from apps.referral.models import ReferralCode

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_referral_code_for_new_user(sender, instance, created, **kwargs):
    """
    Create a referral code for a new user upon account creation.
    """
    if created and not instance.is_superuser:
        ReferralCode.objects.get_or_create(user=instance, defaults={'is_active': True})